"""Frozen-checkpoint external evaluation on PhysioNet LUDB 1.0.1.

Only explicitly compatible diagnosis fields are scored. Each class has its
own eligibility mask; an uncertain diagnosis is never silently made negative.
"""

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
from src.external_validation.chapman import convert_to_mv, resample_500_to_100
from src.inference import load_predictor
from src.signal_processing.filters import preprocess_ecg

DATASET_URL = "https://physionet.org/content/ludb/1.0.1/"
SOURCE_FS_HZ = 500
TARGET_FS_HZ = 100
SCORED_CLASSES = ("MI", "STTC", "CD", "HYP")
REQUIRED_COLUMNS = {
    "ID", "Conduction abnormalities", "Hypertrophies", "Ischemia",
    "Non-specific repolarization abnormalities", "Other states",
}

CONDUCTION_CODES = frozenset({
    "i degree av block", "iii degree av-block",
    "incomplete right bundle branch block", "incomplete left bundle branch block",
    "complete right bundle branch block", "complete left bundle branch block",
    "left anterior hemiblock", "non-specific intravintricular conduction delay",
})
HYPERTROPHY_CODES = frozenset({
    "left ventricular hypertrophy", "right ventricular hypertrophy",
    "left atrial hypertrophy", "right atrial hypertrophy",
    "left atrial overload", "right atrial overload",
})


def find_dataset_root(path: Path) -> Path:
    candidates = [path, *[child for child in path.iterdir() if child.is_dir()]]
    for candidate in candidates:
        if (candidate / "ludb.csv").is_file() and (candidate / "data").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"LUDB ludb.csv and data/ were not found under {path}.")


def _entries(row: pd.Series | dict, column: str) -> set[str]:
    value = row[column]
    if pd.isna(value):
        return set()
    return {part.strip().casefold() for part in str(value).splitlines() if part.strip()}


def encode_ludb_labels(row: pd.Series | dict) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Return five binary targets, five eligibility flags, and audit reasons.

    A blank category in LUDB's per-record diagnosis table is treated as a
    negative for that category. Unmapped entries affecting a target class
    override this only when there is no explicit positive for that class.
    LUDB has no explicit whole-ECG normal label, so NORM is never scored.
    """
    labels = np.zeros(len(SUPERCLASSES), dtype=np.uint8)
    eligible = np.ones(len(SUPERCLASSES), dtype=bool)
    eligible[SUPERCLASSES.index("NORM")] = False
    reasons: list[str] = []

    def set_label(name: str, positive: bool, uncertain: bool) -> None:
        index = SUPERCLASSES.index(name)
        labels[index] = int(positive)
        eligible[index] = positive or not uncertain

    conduction = _entries(row, "Conduction abnormalities")
    cd_positive = bool(conduction & CONDUCTION_CODES)
    cd_unknown = conduction - CONDUCTION_CODES
    set_label("CD", cd_positive, bool(cd_unknown))
    reasons.extend(f"CD:{value}" for value in sorted(cd_unknown))

    hypertrophies = _entries(row, "Hypertrophies")
    hyp_positive = bool(hypertrophies & HYPERTROPHY_CODES)
    hyp_unknown = hypertrophies - HYPERTROPHY_CODES
    set_label("HYP", hyp_positive, bool(hyp_unknown))
    reasons.extend(f"HYP:{value}" for value in sorted(hyp_unknown))

    mi_positive = False
    mi_uncertain = False
    sttc_positive = False
    sttc_uncertain = False
    for value in _entries(row, "Ischemia"):
        if value.startswith("stemi:"):
            mi_positive = True
            sttc_uncertain = True
        elif value.startswith("ischemia:"):
            sttc_positive = True
        elif value.startswith(("scar formation:", "undefined ischemia/scar/supp.nstemi:")):
            mi_uncertain = True
            sttc_uncertain = True
            reasons.append(f"MI/STTC:{value}")
        else:
            mi_uncertain = True
            sttc_uncertain = True
            reasons.append(f"MI/STTC:unrecognized:{value}")
    for value in _entries(row, "Non-specific repolarization abnormalities"):
        if value.startswith("non-specific repolarization abnormalities:"):
            sttc_positive = True
        else:
            sttc_uncertain = True
            reasons.append(f"STTC:unrecognized:{value}")
    for value in _entries(row, "Other states"):
        if value == "early repolarization syndrome":
            sttc_uncertain = True
            reasons.append(f"STTC:{value}")
        else:
            sttc_uncertain = True
            reasons.append(f"STTC:unrecognized:{value}")
    set_label("MI", mi_positive, mi_uncertain)
    set_label("STTC", sttc_positive, sttc_uncertain)
    return labels, eligible, tuple(sorted(set(reasons)))


def _load_signal(record_path: Path, filter_config) -> np.ndarray:
    record = wfdb.rdrecord(str(record_path))
    if int(record.fs) != SOURCE_FS_HZ or record.sig_len != 5000 or record.p_signal is None:
        raise ValueError("LUDB record must contain a physical 10-second, 500 Hz signal.")
    names = [str(name).casefold() for name in record.sig_name]
    expected = [name.casefold() for name in LEAD_NAMES]
    if len(names) != 12 or len(set(names)) != 12 or set(names) != set(expected):
        raise ValueError(f"Unexpected lead names: {record.sig_name}.")
    indices = [names.index(name) for name in expected]
    values = np.asarray(record.p_signal, dtype=np.float32)[:, indices]
    values_mv = convert_to_mv(values, [record.units[index] for index in indices])
    resampled = resample_500_to_100(values_mv)
    processed = preprocess_ecg(resampled, TARGET_FS_HZ, config=filter_config)
    if processed.shape != (1000, 12) or not np.isfinite(processed).all():
        raise ValueError("Preprocessing produced invalid model input.")
    return np.ascontiguousarray(processed.T, dtype=np.float32)


def calculate_masked_metrics(
    targets: np.ndarray, masks: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray
) -> dict:
    per_class: dict[str, dict] = {}
    aurocs: list[float] = []
    f1s: list[float] = []
    for name in SUPERCLASSES:
        index = SUPERCLASSES.index(name)
        selected = masks[:, index]
        truth = targets[selected, index]
        scores = probabilities[selected, index]
        positive = int(truth.sum())
        negative = int(len(truth) - positive)
        result = {
            "records_evaluable": int(selected.sum()),
            "records_unknown_or_unsupported": int((~selected).sum()),
            "positive_records": positive,
            "negative_records": negative,
            "threshold_frozen_from_ptbxl_validation": float(thresholds[index]),
            "auroc": None, "precision": None, "recall": None, "f1": None,
            "confusion_matrix_tn_fp_fn_tp": None,
        }
        if positive > 0 and negative > 0:
            predicted = scores >= thresholds[index]
            precision, recall, f1, _ = precision_recall_fscore_support(
                truth, predicted, average="binary", zero_division=0
            )
            result.update({
                "auroc": float(roc_auc_score(truth, scores)),
                "precision": float(precision), "recall": float(recall), "f1": float(f1),
                "confusion_matrix_tn_fp_fn_tp": confusion_matrix(
                    truth, predicted, labels=[0, 1]
                ).ravel().tolist(),
            })
            aurocs.append(result["auroc"])
            f1s.append(result["f1"])
        per_class[name] = result
    return {
        "per_class": per_class,
        "classes_with_defined_auroc": [name for name in SUPERCLASSES if per_class[name]["auroc"] is not None],
        "mapped_class_macro_auroc": float(np.mean(aurocs)) if aurocs else None,
        "mapped_class_macro_f1": float(np.mean(f1s)) if f1s else None,
        "ptbxl_five_class_macro_directly_comparable": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/datasets/ludb"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/external_validation/ludb")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")

    root = find_dataset_root(args.dataset_root)
    frame = pd.read_csv(root / "ludb.csv")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"LUDB diagnosis table is missing columns: {sorted(missing)}")
    if frame["ID"].isna().any() or frame["ID"].duplicated().any():
        raise ValueError("LUDB record IDs must be present and unique.")
    checkpoints = [
        PROJECT_ROOT / "outputs/checkpoints/baseline_best.pt",
        PROJECT_ROOT / "outputs/checkpoints/resnet1d_best.pt",
    ]
    predictors = [(path, load_predictor(path)) for path in checkpoints]
    filter_config = predictors[0][1].filter_config
    if any(p.sampling_rate_hz != TARGET_FS_HZ or p.filter_config != filter_config for _, p in predictors):
        raise ValueError("The checkpoints must share 100 Hz input and identical preprocessing.")

    signals: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    metadata: list[dict] = []
    errors: list[dict] = []
    audit_counts: Counter[str] = Counter()
    print(f"LUDB root: {root}; diagnosis rows: {len(frame)}", flush=True)
    for _, row in frame.iterrows():
        record_id = str(int(row["ID"]))
        labels, eligible, reasons = encode_ludb_labels(row)
        try:
            signal = _load_signal(root / "data" / record_id, filter_config)
        except Exception as exc:
            errors.append({"record_id": record_id, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        signals.append(signal)
        targets.append(labels)
        masks.append(eligible)
        metadata.append({"record_id": record_id, "mapping_audit": "; ".join(reasons)})
        audit_counts.update(reasons)
    if not signals:
        raise RuntimeError("No LUDB records passed waveform checks.")

    x = np.stack(signals)
    y = np.stack(targets)
    m = np.stack(masks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint_path, predictor in predictors:
        chunks: list[np.ndarray] = []
        for start in range(0, len(x), args.batch_size):
            batch = torch.from_numpy(x[start:start + args.batch_size]).to(predictor.device)
            with torch.inference_mode():
                chunks.append(torch.sigmoid(predictor.model(batch)).cpu().numpy())
        probabilities = np.concatenate(chunks)
        summary = calculate_masked_metrics(y, m, probabilities, predictor.thresholds)
        summary.update({
            "dataset": "LUDB", "dataset_version": "PhysioNet 1.0.1",
            "dataset_source": DATASET_URL,
            "diagnosis_rows": int(len(frame)), "waveforms_evaluated": int(len(x)),
            "waveforms_excluded": int(len(errors)), "waveform_errors": errors,
            "mapping_uncertainty_examples": audit_counts.most_common(),
            "source_sampling_rate_hz": SOURCE_FS_HZ,
            "model_sampling_rate_hz": TARGET_FS_HZ,
            "preprocessing": "polyphase anti-alias 500->100 Hz; frozen checkpoint baseline/bandpass/z-score",
            "label_policy": "class-wise positive/negative/unknown; NORM unscored; blank diagnosis-category column treated as absent",
            "threshold_source": "frozen PTB-XL validation fold 9; no LUDB retuning",
            "checkpoint": str(checkpoint_path.resolve()),
            "model_type": predictor.model_type,
            "evaluation_device": str(predictor.device),
        })
        stem = predictor.model_type
        metrics_path = args.output_dir / f"{stem}_metrics.json"
        predictions_path = args.output_dir / f"{stem}_predictions.csv"
        metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        prediction_frame = pd.DataFrame(metadata)
        for index, name in enumerate(SUPERCLASSES):
            prediction_frame[f"target_{name}"] = np.where(m[:, index], y[:, index], "")
            prediction_frame[f"eligible_{name}"] = m[:, index].astype(np.uint8)
            prediction_frame[f"probability_{name}"] = probabilities[:, index]
            prediction_frame[f"predicted_{name}"] = (
                probabilities[:, index] >= predictor.thresholds[index]
            ).astype(np.uint8)
        prediction_frame.to_csv(predictions_path, index=False)
        print(
            f"{stem}: records={len(x)}; mapped macro AUROC={summary['mapped_class_macro_auroc']:.4f}; "
            f"mapped macro F1={summary['mapped_class_macro_f1']:.4f}; metrics={metrics_path}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
