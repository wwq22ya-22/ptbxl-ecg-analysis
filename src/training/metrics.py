"""Multi-label classification metrics and validation-only threshold tuning."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

from src.data.ptbxl import SUPERCLASSES


def _validate_arrays(targets: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(targets)
    scores = np.asarray(probabilities, dtype=np.float64)
    if truth.ndim != 2 or truth.shape[1] != len(SUPERCLASSES) or truth.shape != scores.shape:
        raise ValueError(f"Expected matching target/probability arrays of shape [N, {len(SUPERCLASSES)}].")
    if truth.shape[0] == 0 or not np.isin(truth, [0, 1]).all():
        raise ValueError("Targets must contain binary labels and at least one record.")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Probabilities must be finite values in [0, 1].")
    return truth.astype(np.uint8), scores


def optimize_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    fallback: float = 0.5,
) -> list[float]:
    """Choose one F1-maximizing threshold per label using validation data only."""
    truth, scores = _validate_arrays(targets, probabilities)
    grid = np.arange(0.10, 0.901, 0.05)
    selected: list[float] = []
    for index in range(truth.shape[1]):
        if truth[:, index].sum() == 0:
            selected.append(float(fallback))
            continue
        f1_values = np.asarray(
            [f1_score(truth[:, index], scores[:, index] >= threshold, zero_division=0) for threshold in grid]
        )
        candidates = grid[np.isclose(f1_values, f1_values.max())]
        selected.append(float(candidates[np.argmin(np.abs(candidates - fallback))]))
    return selected


def calculate_multilabel_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Sequence[float] | float = 0.5,
) -> dict:
    """Return JSON-safe per-class and macro metrics for independent labels."""
    truth, scores = _validate_arrays(targets, probabilities)
    cutoffs = np.broadcast_to(np.asarray(thresholds, dtype=np.float64), (len(SUPERCLASSES),))
    if not np.isfinite(cutoffs).all() or np.any((cutoffs <= 0) | (cutoffs >= 1)):
        raise ValueError("Thresholds must be finite and strictly between 0 and 1.")
    predictions = (scores >= cutoffs).astype(np.uint8)
    per_class: dict[str, dict] = {}
    auc_values: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1_values: list[float] = []
    for index, name in enumerate(SUPERCLASSES):
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth[:, index], predictions[:, index], average="binary", zero_division=0
        )
        auroc = float(roc_auc_score(truth[:, index], scores[:, index])) if np.unique(truth[:, index]).size == 2 else None
        if auroc is not None:
            auc_values.append(auroc)
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1_values.append(float(f1))
        per_class[name] = {
            "support": int(truth[:, index].sum()),
            "threshold": float(cutoffs[index]),
            "auroc": auroc,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": confusion_matrix(truth[:, index], predictions[:, index], labels=[0, 1]).tolist(),
        }
    return {
        "records": int(truth.shape[0]),
        "per_class": per_class,
        "macro_auroc": float(np.mean(auc_values)) if auc_values else None,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "classes_with_defined_auroc": len(auc_values),
    }
