"""Multi-label classification metrics and validation-only threshold tuning."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, precision_recall_fscore_support, roc_auc_score

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


def constrained_pr_thresholds(targets, probabilities, *, precision_floor, recall_floor, time_limit=30):
    """Select PR-curve points jointly with macro precision/recall constraints.

    A mixed-integer program chooses exactly one threshold for each class and
    maximizes macro F1. PR points dominated in both precision and recall can be
    removed without changing the optimum. This uses calibration labels only.
    Return solver diagnostics; do not call a time-limited incumbent optimal.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    truth, scores = _validate_arrays(targets, probabilities)
    if not 0 <= precision_floor <= 1 or not 0 <= recall_floor <= 1 or time_limit <= 0:
        raise ValueError('Invalid precision/recall floors or solver time limit.')
    curves = []
    for index in range(5):
        if np.unique(truth[:, index]).size != 2:
            raise ValueError('Constrained calibration requires positives and negatives in every class.')
        p, r, thresholds = precision_recall_curve(truth[:, index], scores[:, index])
        valid = (thresholds > 0) & (thresholds < 1)
        points = [(float(pp), float(rr), float(tt)) for pp, rr, tt in zip(p[:-1][valid], r[:-1][valid], thresholds[valid])]
        # Higher recall first, then higher precision; discard dominated points.
        points.sort(key=lambda point: (-point[1], -point[0], abs(point[2]-0.5)))
        frontier, best_precision = [], -1.0
        for pp, rr, tt in points:
            if pp > best_precision:
                frontier.append((pp, rr, tt, 2*pp*rr/max(pp+rr, 1e-12)))
                best_precision = pp
        if not frontier:
            raise ValueError('No valid threshold candidates for calibration.')
        curves.append(frontier)
    count = sum(map(len, curves))
    constraints = np.zeros((7, count), dtype=np.float64)
    objective = np.zeros(count, dtype=np.float64)
    offset = 0
    for index, curve in enumerate(curves):
        end = offset + len(curve)
        constraints[index, offset:end] = 1
        constraints[5, offset:end] = [point[0] for point in curve]
        constraints[6, offset:end] = [point[1] for point in curve]
        objective[offset:end] = [-point[3]/5 for point in curve]
        offset = end
    lower = np.array([1]*5 + [5*precision_floor, 5*recall_floor], dtype=np.float64)
    upper = np.array([1]*5 + [np.inf, np.inf], dtype=np.float64)
    solution = milp(objective, integrality=np.ones(count), bounds=Bounds(0, 1), constraints=LinearConstraint(constraints, lower, upper), options={'time_limit': float(time_limit), 'mip_rel_gap': 0})
    diagnostics = {'status': int(solution.status), 'message': str(solution.message), 'optimal': bool(solution.success), 'frontier_points': [len(curve) for curve in curves], 'mip_gap': float(solution.mip_gap) if getattr(solution, 'mip_gap', None) is not None else None}
    if solution.x is None:
        return None, diagnostics
    thresholds, offset = [], 0
    for curve in curves:
        segment = solution.x[offset:offset+len(curve)]
        if not np.allclose(segment, np.round(segment), atol=1e-6) or not np.isclose(segment.sum(), 1):
            return None, diagnostics
        thresholds.append(curve[int(np.argmax(segment))][2])
        offset += len(curve)
    actual = calculate_multilabel_metrics(truth, scores, thresholds)
    if actual['macro_precision'] + 1e-9 < precision_floor or actual['macro_recall'] + 1e-9 < recall_floor:
        return None, diagnostics
    diagnostics['calibration_macro_f1'] = actual['macro_f1']
    return thresholds, diagnostics
