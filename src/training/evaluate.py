"""Evaluate a frozen checkpoint once on the official PTB-XL test fold."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.ptbxl import SUPERCLASSES, find_ptbxl_root
from src.config import resolve_project_path
from src.models import MODEL_TYPES, build_model
from src.signal_processing.filters import FilterConfig
from src.training.data import CachedPTBXLDataset, ensure_preprocessed_cache, load_labeled_splits
from src.training.metrics import calculate_multilabel_metrics
from src.visualization.metrics_plot import plot_confusion_matrices, plot_roc_curves


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    project_root: Path,
    dataset_root: Path | None = None,
    allow_smoke: bool = False,
) -> dict:
    """Use fixed model and validation thresholds; never tune on test labels."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("scope") != "full" and not allow_smoke:
        raise ValueError("Only a full-training checkpoint can be reported as a test result.")
    model_type = checkpoint.get("model_type")
    if model_type not in MODEL_TYPES or tuple(checkpoint["superclasses"]) != SUPERCLASSES:
        raise ValueError("Checkpoint model or label order does not match this evaluator.")
    config = checkpoint["config"]
    fs = int(checkpoint["sampling_rate_hz"])
    root = find_ptbxl_root(
        dataset_root or resolve_project_path(config["dataset"]["root"], project_root),
        sampling_rate_hz=fs,
    )
    dataset_cfg = config["dataset"]
    test_frame = load_labeled_splits(
        root, fs, train_folds=dataset_cfg.get("train_folds", tuple(range(1, 9))),
        validation_fold=int(dataset_cfg.get("validation_fold", 9)),
        test_fold=int(dataset_cfg.get("test_fold", 10)), test_folds=dataset_cfg.get("test_folds"),
        require_diagnostic_label=bool(dataset_cfg.get("require_diagnostic_label", False)),
    )["test"]
    filter_config = FilterConfig.from_mapping(config["signal_processing"])
    output_cfg = config["outputs"]
    cache_path = ensure_preprocessed_cache(
        test_frame, root, project_root / output_cfg["cache_dir"],
        split_name="test_full", sampling_rate_hz=fs, config=filter_config,
    )
    dataset = CachedPTBXLDataset(cache_path, test_frame)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_type, dropout=float(config["model"]["dropout"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    loader = DataLoader(
        dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    score_batches = []
    with torch.inference_mode():
        for signals, _ in loader:
            logits = model(signals.to(device, non_blocking=True))
            score_batches.append(torch.sigmoid(logits).cpu().numpy())
    probabilities = np.concatenate(score_batches)
    targets = test_frame.loc[:, list(SUPERCLASSES)].to_numpy(dtype=np.uint8)
    thresholds = np.asarray(checkpoint["thresholds"], dtype=np.float64)
    metrics = calculate_multilabel_metrics(targets, probabilities, thresholds)
    metrics.update({
        "dataset": "PTB-XL 1.0.3",
        "fold": 10,
        "sampling_rate_hz": fs,
        "checkpoint": str(checkpoint_path.resolve()),
        "best_validation_epoch": int(checkpoint["epoch"]),
        "checkpoint_scope": checkpoint["scope"],
        "threshold_source": "validation_fold_9",
        "evaluation_device": str(device),
        "require_diagnostic_label": bool(dataset_cfg.get("require_diagnostic_label", False)),
    })
    suffix = f"{model_type}_smoke" if checkpoint["scope"] != "full" else (
        "full" if model_type == "cnn1d" else f"{model_type}_full"
    )
    metrics_dir = project_root / output_cfg["metrics_dir"]
    predictions_dir = project_root / output_cfg["predictions_dir"]
    figures_dir = project_root / output_cfg["figure_dir"]
    for directory in (metrics_dir, predictions_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"test_metrics_{suffix}.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    prediction_frame = pd.DataFrame({"ecg_id": test_frame["ecg_id"].to_numpy()})
    for index, name in enumerate(SUPERCLASSES):
        prediction_frame[f"true_{name}"] = targets[:, index]
        prediction_frame[f"probability_{name}"] = probabilities[:, index]
        prediction_frame[f"predicted_{name}"] = (probabilities[:, index] >= thresholds[index]).astype(np.uint8)
    prediction_frame.to_csv(predictions_dir / f"test_predictions_{suffix}.csv", index=False)
    plot_roc_curves(targets, probabilities, figures_dir / f"test_roc_{suffix}.png")
    plot_confusion_matrices(targets, probabilities, thresholds, figures_dir / f"test_confusion_{suffix}.png")
    print(f"Test records: {len(test_frame):,}; macro AUROC: {metrics['macro_auroc']:.4f}; macro F1: {metrics['macro_f1']:.4f}", flush=True)
    print(f"Metrics: {metrics_path}", flush=True)
    return metrics
