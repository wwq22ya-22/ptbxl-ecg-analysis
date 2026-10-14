"""Independent sklearn/count-based audit of the teacher/preprocessing study."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_stm32 import digest
from src.data.ptbxl import SUPERCLASSES, load_ecg
from src.signal_processing.filters import FilterConfig, preprocess_ecg
from src.training.data import _cache_fingerprint, load_labeled_splits

OUT = ROOT / "outputs/teacher_preprocessing"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired_patient_bootstrap(rows, frame, teacher_family):
    """Exploratory paired CI, preserving multiple ECGs from one patient.

    The models and thresholds stay fixed; training randomness is represented
    by the three observed seed pairs, not by retraining within the bootstrap.
    """
    families = [("inception1d", teacher_family[1]), teacher_family]
    patients, inverse = np.unique(frame.patient_id.to_numpy(), return_inverse=True)
    counters = np.zeros((2, 3, len(patients), 5, 3), dtype=np.int64)
    for family_index, (model, preprocessing) in enumerate(families):
        for seed_index, seed in enumerate((42, 43, 44)):
            row = next(r for r in rows if (r["model"], r["preprocessing"], r["seed"]) ==
                       (model, preprocessing, seed))
            with np.load(OUT / row["name"] / "evaluation/test_predictions.npz") as a:
                truth = a["targets"].astype(bool)
                predicted = a["probabilities"] >= a["thresholds"]
                for index, records in enumerate((truth & predicted, ~truth & predicted, truth & ~predicted)):
                    np.add.at(counters[family_index, seed_index, :, :, index], inverse, records.astype(np.int64))

    def differences(weights):
        totals = np.einsum("fspck,p->fsck", counters, weights, optimize=True)
        tp, fp, fn = totals[..., 0], totals[..., 1], totals[..., 2]
        ratio = lambda numerator, denominator: np.divide(numerator, denominator,
                          out=np.zeros_like(numerator, dtype=np.float64), where=denominator > 0)
        precision = ratio(tp, tp + fp)
        recall = ratio(tp, tp + fn)
        f1 = ratio(2 * tp, 2 * tp + fp + fn)
        values = {"macro_precision": precision.mean(axis=(1, 2)),
                  "macro_recall": recall.mean(axis=(1, 2)), "macro_f1": f1.mean(axis=(1, 2)),
                  "hyp_recall": recall[:, :, 4].mean(axis=1)}
        return {key: float(value[1] - value[0]) for key, value in values.items()}

    actual = differences(np.ones(len(patients), dtype=np.int64))
    rng = np.random.default_rng(20261007)
    samples = []
    for _ in range(500):
        weights = np.bincount(rng.integers(0, len(patients), len(patients)), minlength=len(patients))
        samples.append(differences(weights))
    return {"method": "paired patient-cluster bootstrap; fixed trained models and validation thresholds",
            "exploratory": True, "iterations": 500, "seed": 20261007, "patients": len(patients),
            "baseline_family": "/".join(families[0]), "teacher_family": "/".join(families[1]),
            "differences": {key: {"mean_delta": number,
                "percentile_95_interval": np.quantile([s[key] for s in samples], [.025, .975]).tolist()}
                for key, number in actual.items()}}


def main():
    summary = read(OUT / "summary.json")
    protocol = read(OUT / "protocol.json")
    frozen = read(OUT / "frozen_selection.json")
    frames = load_labeled_splits("D:/datasets/ptb-xl", 100, require_diagnostic_label=True)
    for split, frame in frames.items():
        assert len(frame) == protocol["cohort"][split]
        saved = pd.read_csv(OUT / f"cohort_{split}.csv")
        for column in ("ecg_id", "patient_id", "strat_fold", *SUPERCLASSES):
            np.testing.assert_array_equal(saved[column], frame[column])
        assert frame[list(SUPERCLASSES)].sum(axis=1).gt(0).all()
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not set(frames[a].patient_id) & set(frames[b].patient_id)
    for mapping in (protocol["protected_artifacts"], protocol["source_sha256"]):
        for path, expected_sha in mapping.items():
            assert digest(ROOT / path) == expected_sha, path
    assert not protocol["test_used_for_selection"] and not frozen["test_used"]
    assert all("test" not in row for row in frozen["all_validation_results"])
    rows = summary["runs"]
    scores = lambda row: (row["validation"]["macro_f1"], row["validation"]["macro_auroc"])
    screening = [r for r in rows if r["seed"] == 42 and r["model"] == "inception1d"]
    assert max(screening, key=scores)["preprocessing"] == summary["selection"]["preprocessing"]
    teacher_screening = [r for r in rows if r["seed"] == 42 and r["model"] != "inception1d"]
    chosen_family = max(teacher_screening, key=scores)
    assert (chosen_family["model"], chosen_family["preprocessing"]) == (
        summary["selection"]["teacher_model"], summary["selection"]["teacher_preprocessing"])
    replicated_family = [r for r in rows if (r["model"], r["preprocessing"]) ==
                         (chosen_family["model"], chosen_family["preprocessing"])]
    assert {r["seed"] for r in replicated_family} == {42, 43, 44}
    chosen = max(replicated_family, key=scores)
    assert chosen["name"] == frozen["selected_teacher"]["name"] == summary["selected_teacher"]["name"]
    assert digest(OUT / "selected_teacher.pt") == chosen["checkpoint_sha256"]
    moments = {}
    audited = 0
    raw_checks = 0
    for row in rows:
        assert digest(Path(row["checkpoint"])) == row["checkpoint_sha256"]
        cp = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)
        assert cp["training_records"] == 17084 and cp["validation_records"] == 2146
        assert cp["superclasses"] == list(SUPERCLASSES) and cp["sampling_rate_hz"] == 100
        cfg = cp["config"]
        assert cfg["dataset"]["require_diagnostic_label"] is True
        assert cfg["dataset"]["train_folds"] == list(range(1, 9))
        assert cfg["dataset"]["validation_fold"] == 9 and cfg["dataset"]["test_fold"] == 10
        raw_cfg = replace(FilterConfig.from_mapping(cfg["signal_processing"]),
                          normalization="none", training_mean=(), training_std=())
        fingerprint = _cache_fingerprint(frames["train"], 100, raw_cfg)
        raw_path = ROOT / cfg["outputs"]["cache_dir"] / f"train_full_{fingerprint}.npy"
        if fingerprint not in moments:
            raw = np.load(raw_path, mmap_mode="r")
            assert raw.shape == (17084, 12, 1000)
            total = np.zeros(12, dtype=np.float64)
            square_total = total.copy()
            count = 0
            for start in range(0, len(raw), 128):
                chunk = np.asarray(raw[start:start+128], dtype=np.float64)
                total += chunk.sum(axis=(0, 2))
                square_total += np.square(chunk).sum(axis=(0, 2))
                count += chunk.shape[0] * chunk.shape[2]
            mean = total / count
            std = np.maximum(np.sqrt(square_total / count - mean ** 2), 1e-8)
            moments[fingerprint] = mean, std
            for index in (0, 420, 8520, 17083):
                record = load_ecg(cfg["dataset"]["root"], frames["train"].iloc[index].filename_lr,
                                  sampling_rate_hz=100)
                expected = preprocess_ecg(record.signal, 100, config=raw_cfg).T
                np.testing.assert_allclose(raw[index], expected, rtol=0, atol=0)
                raw_checks += 1
        mean, std = moments[fingerprint]
        np.testing.assert_allclose(cfg["signal_processing"]["training_mean"], mean, rtol=0, atol=1e-12)
        np.testing.assert_allclose(cfg["signal_processing"]["training_std"], std, rtol=0, atol=1e-12)
        for split in ("validation", "test"):
            path = OUT / row["name"] / "evaluation" / f"{split}_predictions.npz"
            with np.load(path) as arrays:
                np.testing.assert_array_equal(arrays["ecg_ids"], frames[split].ecg_id)
                np.testing.assert_array_equal(arrays["targets"], frames[split][list(SUPERCLASSES)].to_numpy())
                np.testing.assert_array_equal(arrays["thresholds"], cp["thresholds"])
                truth = arrays["targets"].astype(bool)
                probabilities = arrays["probabilities"]
                predicted = probabilities >= arrays["thresholds"]
                value = row[split]
                class_scores = []
                for index, label in enumerate(SUPERCLASSES):
                    t, p = truth[:, index], predicted[:, index]
                    tp = int((t & p).sum()); tn = int((~t & ~p).sum())
                    fp = int((~t & p).sum()); fn = int((t & ~p).sum())
                    precision = tp / max(1, tp + fp)
                    recall = tp / max(1, tp + fn)
                    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
                    auc = float(roc_auc_score(t, probabilities[:, index]))
                    ap = float(average_precision_score(t, probabilities[:, index]))
                    c = value["per_class"][label]
                    assert c["support"] == int(t.sum()) and c["confusion_matrix"] == [[tn, fp], [fn, tp]]
                    actual = dict(precision=precision, recall=recall, f1=f1, auroc=auc, auprc=ap)
                    for key, number in actual.items():
                        assert abs(c[key] - number) < 1e-12, (row["name"], split, label, key)
                    class_scores.append(actual)
                for key in ("precision", "recall", "f1", "auroc", "auprc"):
                    assert abs(value[f"macro_{key}"] - np.mean([c[key] for c in class_scores])) < 1e-12
            audited += 1
    for family, statistics in summary["seed_statistics"].items():
        model, preprocessing = family.split("/")
        members = sorted([r for r in rows if r["model"] == model and r["preprocessing"] == preprocessing],
                         key=lambda r: r["seed"])
        assert [r["seed"] for r in members] == [42, 43, 44]
        for key, stat in statistics.items():
            numbers = [r["test"]["per_class"]["HYP"]["recall"] if key == "hyp_recall" else r["test"][key]
                       for r in members]
            np.testing.assert_array_equal(numbers, stat["values"])
            assert abs(stat["mean"] - np.mean(numbers)) < 1e-12
            assert abs(stat["std"] - np.std(numbers, ddof=1)) < 1e-12
    result = {"passed": True, "prediction_sets_independently_verified": audited,
              "raw_preprocessing_records_rechecked": raw_checks,
              "normalization_independently_recomputed": len(moments),
              "validation_selection_replayed": True, "patient_isolation_verified": True,
              "protected_artifacts_unchanged": True}
    (OUT / "independent_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bootstrap = paired_patient_bootstrap(rows, frames["test"], (chosen_family["model"], chosen_family["preprocessing"]))
    (OUT / "paired_patient_bootstrap.json").write_text(json.dumps(bootstrap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    print(json.dumps(bootstrap), flush=True)


if __name__ == "__main__":
    main()
