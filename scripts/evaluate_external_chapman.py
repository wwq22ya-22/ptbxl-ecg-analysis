"""Run zero-shot external evaluation on Chapman-Shaoxing/Ningbo ECGs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ptbxl import LEAD_NAMES, SUPERCLASSES
from src.external_validation.chapman import (
    DATASET_URL,
    EVALUATED_CLASSES,
    SOURCE_FS_HZ,
    TARGET_FS_HZ,
    convert_to_mv,
    encode_mapped_labels,
    load_snomed_dictionary,
    parse_diagnosis_codes,
    resample_500_to_100,
)
from src.inference import load_predictor
from src.signal_processing.filters import preprocess_ecg


def find_dataset_root(path: Path) -> Path:
    candidates = [path, *[item.parent for item in path.glob("*/ConditionNames_SNOMED-CT.csv")]]
    candidates.extend(item.parent for item in path.glob("*/*/ConditionNames_SNOMED-CT.csv"))
    for candidate in candidates:
        if (candidate / "WFDBRecords").is_dir() and (candidate / "ConditionNames_SNOMED-CT.csv").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Chapman-Shaoxing/Ningbo data not found under {path}. "
        "Download from PhysioNet or run scripts/download_chapman.py."
    )


def _ordered_indices(signal_names: list[str]) -> list[int]:
    by_name = {str(name).strip().casefold(): index for index, name in enumerate(signal_names)}
    missing = [name for name in LEAD_NAMES if name.casefold() not in by_name]
    if missing or len(by_name) != len(LEAD_NAMES):
        raise ValueError(f"Expected twelve standard leads; missing={missing}, found={signal_names}.")
    return [by_name[name.casefold()] for name in LEAD_NAMES]


def _load_and_preprocess(record_path: Path, filter_config) -> np.ndarray:
    record = wfdb.rdrecord(str(record_path))
    if int(record.fs) != SOURCE_FS_HZ:
        raise ValueError(f"Expected {SOURCE_FS_HZ} Hz; record uses {record.fs} Hz.")
    if record.p_signal is None:
        raise ValueError("WFDB record does not contain physical signal values.")
    indices = _ordered_indices(list(record.sig_name or []))
    raw = np.asarray(record.p_signal, dtype=np.float32)[:, indices]
    units = [record.units[index] for index in indices]
    physical_mv = convert_to_mv(raw, units)
    resampled = resample_500_to_100(physical_mv)
    processed = preprocess_ecg(resampled, TARGET_FS_HZ, config=filter_config)
    if processed.shape != (1000, len(LEAD_NAMES)):
        raise ValueError(f"Model preprocessing returned unexpected shape {processed.shape}.")
    return np.ascontiguousarray(processed.T, dtype=np.float32)


def _metric_summary(targets: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> dict:
    per_class: dict[str, dict] = {}
    aucs: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for column, name in enumerate(EVALUATED_CLASSES):
        truth = targets[:, SUPERCLASSES.index(name)].astype(np.uint8)
        scores = probabilities[:, SUPERCLASSES.index(name)]
        predicted = scores >= thresholds[SUPERCLASSES.index(name)]
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predicted, average="binary", zero_division=0
        )
        auc = float(roc_auc_score(truth, scores)) if np.unique(truth).size == 2 else None
        if auc is not None:
            aucs.append(auc)
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))
        per_class[name] = {
            "positive_records": int(truth.sum()),
            "negative_records": int(len(truth) - truth.sum()),
            "threshold_frozen_from_ptbxl_validation": float(thresholds[SUPERCLASSES.index(name)]),
            "auroc": auc,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix_tn_fp_fn_tp": confusion_matrix(truth, predicted, labels=[0, 1]).ravel().tolist(),
        }
    return {
        "records_evaluated": int(len(targets)),
        "classes_evaluated": list(EVALUATED_CLASSES),
        "unsupported_class": "NORM: source dictionary has sinus rhythm, not an explicit normal-morphology label",
        "per_class": per_class,
        "macro_auroc": float(np.mean(aucs)) if aucs else None,
        "macro_precision": float(np.mean(precisions)) if precisions else None,
        "macro_recall": float(np.mean(recalls)) if recalls else None,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "classes_with_defined_auroc": len(aucs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/datasets/ecg-arrhythmia"))
    parser.add_argument("--checkpoint", type=Path, action="append", help="Repeat to evaluate multiple full checkpoints.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-records", type=int, help="Debug only; marks results as partial.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/external_validation/chapman")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.max_records is not None and args.max_records < 1):
        parser.error("batch-size and max-records must be positive.")

    root = find_dataset_root(args.dataset_root.expanduser())
    dictionary = load_snomed_dictionary(root / "ConditionNames_SNOMED-CT.csv")
    record_headers = sorted((root / "WFDBRecords").rglob("*.hea"))
    if args.max_records is not None:
        record_headers = record_headers[: args.max_records]
    if not record_headers:
        raise FileNotFoundError(f"No WFDB record headers found under {root / 'WFDBRecords'}.")

    checkpoints = args.checkpoint or [
        PROJECT_ROOT / "outputs/checkpoints/baseline_best.pt",
        PROJECT_ROOT / "outputs/checkpoints/resnet1d_best.pt",
    ]
    predictors = [(path, load_predictor(path)) for path in checkpoints]
    filter_config = predictors[0][1].filter_config
    if any(predictor.sampling_rate_hz != TARGET_FS_HZ for _, predictor in predictors):
        raise ValueError(f"All checkpoints must expect {TARGET_FS_HZ} Hz inputs.")
    if any(predictor.filter_config != filter_config for _, predictor in predictors):
        raise ValueError("Checkpoints use different preprocessing configurations; run them separately.")
    thresholds = {path: predictor.thresholds for path, predictor in predictors}
    probability_chunks: dict[Path, list[np.ndarray]] = {path: [] for path, _ in predictors}
    valid_labels: list[np.ndarray] = []
    metadata: list[dict] = []
    batch_signals: list[np.ndarray] = []
    batch_labels: list[np.ndarray] = []
    batch_meta: list[dict] = []
    unsupported_codes: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    device = predictors[0][1].device

    def flush_batch() -> None:
        if not batch_signals:
            return
        inputs = torch.from_numpy(np.stack(batch_signals)).to(device)
        with torch.inference_mode():
            for checkpoint_path, predictor in predictors:
                probability_chunks[checkpoint_path].append(torch.sigmoid(predictor.model(inputs)).cpu().numpy())
        valid_labels.extend(batch_labels)
        metadata.extend(batch_meta)
        batch_signals.clear()
        batch_labels.clear()
        batch_meta.clear()

    print(f"Dataset: {root}")
    print(f"Reading {len(record_headers):,} WFDB headers; inference device: {device}", flush=True)
    for index, header_path in enumerate(record_headers, start=1):
        record_path = header_path.with_suffix("")
        try:
            header = wfdb.rdheader(str(record_path))
            diagnosis_codes = parse_diagnosis_codes(header.comments or [])
            labels, complete, acronyms, unknown = encode_mapped_labels(diagnosis_codes, dictionary)
            if not complete:
                if unknown:
                    skipped["unsupported_diagnoses"] += 1
                    unsupported_codes.update(unknown)
                else:
                    skipped["rhythm_only_or_no_morphology"] += 1
                continue
            signal = _load_and_preprocess(record_path, filter_config)
            batch_signals.append(signal)
            batch_labels.append(labels)
            batch_meta.append({
                "record": record_path.relative_to(root).as_posix(),
                "snomed_diagnoses": ";".join(diagnosis_codes),
                "mapped_acronyms": ";".join(acronyms),
            })
            if len(batch_signals) >= args.batch_size:
                flush_batch()
        except Exception as exc:
            skipped[f"record_error: {type(exc).__name__}"] += 1
            unsupported_codes[f"ERROR:{type(exc).__name__}:{str(exc)[:120]}"] += 1
        if index % 1000 == 0:
            print(f"Scanned {index:,}/{len(record_headers):,}; evaluated {len(metadata) + len(batch_meta):,}", flush=True)
    flush_batch()

    if not valid_labels:
        raise RuntimeError("No records with fully mapped morphology labels were available for evaluation.")
    targets = np.stack(valid_labels).astype(np.uint8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial = args.max_records is not None
    for checkpoint_path, predictor in predictors:
        probabilities = np.concatenate(probability_chunks[checkpoint_path], axis=0)
        metrics = _metric_summary(targets, probabilities, thresholds[checkpoint_path])
        metrics.update({
            "dataset": "Chapman-Shaoxing / Ningbo 12-lead ECG",
            "dataset_version": "PhysioNet ecg-arrhythmia 1.0.0",
            "dataset_source": DATASET_URL,
            "source_sampling_rate_hz": SOURCE_FS_HZ,
            "model_sampling_rate_hz": TARGET_FS_HZ,
            "preprocessing": "anti-aliased polyphase 500->100 Hz; checkpoint median-baseline, bandpass, and per-record normalization",
            "checkpoint": str(checkpoint_path.resolve()),
            "model_type": predictor.model_type,
            "evaluation_device": str(predictor.device),
            "threshold_source": "frozen PTB-XL validation-fold checkpoint thresholds; no external retuning",
            "headers_scanned": len(record_headers),
            "records_excluded": int(len(record_headers) - len(targets)),
            "excluded_record_reasons": dict(skipped),
            "partial_debug_run": partial,
            "partial_run_limit": args.max_records,
            "diagnosis_codes_or_errors_not_mapped": unsupported_codes.most_common(100),
        })
        model_name = predictor.model_type
        prefix = "partial_" if partial else ""
        metrics_path = args.output_dir / f"{prefix}{model_name}_metrics.json"
        predictions_path = args.output_dir / f"{prefix}{model_name}_predictions.csv"
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        prediction_frame = pd.DataFrame(metadata)
        for name in EVALUATED_CLASSES:
            class_index = SUPERCLASSES.index(name)
            prediction_frame[f"target_{name}"] = targets[:, SUPERCLASSES.index(name)]
            prediction_frame[f"probability_{name}"] = probabilities[:, class_index]
            prediction_frame[f"predicted_{name}"] = (probabilities[:, class_index] >= thresholds[checkpoint_path][class_index]).astype(np.uint8)
        prediction_frame["probability_NORM_unscored"] = probabilities[:, SUPERCLASSES.index("NORM")]
        prediction_frame.to_csv(predictions_path, index=False)
        print(
            f"{model_name}: n={len(targets):,}; mapped-class macro AUROC={metrics['macro_auroc']:.4f}; "
            f"macro F1={metrics['macro_f1']:.4f}; excluded={metrics['records_excluded']:,}",
            flush=True,
        )
        print(f"Metrics: {metrics_path}")
        print(f"Predictions: {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



