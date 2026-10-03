"""Train the PTB-XL five-label 1D CNN baseline."""

from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.ptbxl import LEAD_NAMES, SUPERCLASSES, find_ptbxl_root
from src.config import resolve_project_path
from src.models import MODEL_TYPES, build_model
from src.signal_processing.filters import FilterConfig
from src.training.data import CachedPTBXLDataset, ensure_preprocessed_cache, load_labeled_splits, fit_training_normalization
from src.training.metrics import calculate_multilabel_metrics, optimize_thresholds
from src.training.strategies import TrainingAugmentation, build_loss
from src.training.distillation import DistillationDataset, multilabel_distillation_loss
from src.visualization.metrics_plot import plot_class_distribution, plot_training_curves


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    augmentation: TrainingAugmentation | None = None,
    distillation: dict | None = None,
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_records = 0
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for batch in loader:
        signals, labels = batch[:2]
        teacher_logits = batch[2].to(device, non_blocking=True) if len(batch) == 3 else None
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        second_labels, coefficient = labels, 1.0
        if training and augmentation is not None:
            signals, labels, second_labels, coefficient = augmentation(signals, labels)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(signals)
                loss = coefficient * criterion(logits, labels)
                if isinstance(coefficient, torch.Tensor):
                    loss = loss + (1 - coefficient) * criterion(logits, second_labels)
                if training and distillation is not None:
                    if teacher_logits is None:
                        raise ValueError('Distillation requires record-aligned teacher logits.')
                    alpha = float(distillation['alpha'])
                    loss = (1 - alpha) * loss + alpha * multilabel_distillation_loss(
                        logits, teacher_logits, float(distillation['temperature'])
                    )
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        batch_size = len(signals)
        total_loss += float(loss.detach().item()) * batch_size
        total_records += batch_size
        if not training:
            targets.append(labels.detach().cpu().numpy())
            probabilities.append(torch.sigmoid(logits.detach()).float().cpu().numpy())
    if total_records == 0:
        raise ValueError("Training or validation loader is empty.")
    if training:
        return total_loss / total_records, None, None
    return total_loss / total_records, np.concatenate(targets), np.concatenate(probabilities)


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".saving")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_baseline(
    config: dict,
    *,
    project_root: Path,
    dataset_root: Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    max_samples: int | None = None,
    model_type: str | None = None,
) -> Path:
    """Train on configured patient-safe folds; defaults to train 1-8 / validation 9."""
    dataset_cfg = config["dataset"]
    train_cfg = config["training"]
    output_cfg = config["outputs"]
    model_type = model_type or config["model"].get("type", "cnn1d")
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model type {model_type!r}; choose from {MODEL_TYPES}.")
    config = copy.deepcopy(config)
    config["model"]["type"] = model_type
    run_config = config
    fs = int(dataset_cfg["sampling_rate_hz"])
    root = find_ptbxl_root(
        dataset_root or resolve_project_path(dataset_cfg["root"], project_root),
        sampling_rate_hz=fs,
    )
    epochs = int(epochs or train_cfg["epochs"])
    batch_size = int(batch_size or train_cfg["batch_size"])
    learning_rate = float(learning_rate or train_cfg["learning_rate"])
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive.")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive.")
    seed = int(train_cfg["random_seed"])
    set_seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(train_cfg["use_amp"]) and device.type == "cuda"
    print(f"Training device: {device}; AMP: {use_amp}; torch: {torch.__version__}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    splits = load_labeled_splits(
        root, fs, train_folds=dataset_cfg.get("train_folds", tuple(range(1, 9))),
        validation_fold=int(dataset_cfg.get("validation_fold", 9)),
        test_fold=int(dataset_cfg.get("test_fold", 10)), test_folds=dataset_cfg.get("test_folds"),
        require_diagnostic_label=bool(dataset_cfg.get("require_diagnostic_label", False)),
    )
    train_frame = splits["train"]
    validation_frame = splits["validation"]
    scope = "full" if max_samples is None else "smoke"
    if max_samples is not None:
        train_frame = train_frame.head(max_samples).copy()
        validation_frame = validation_frame.head(min(len(validation_frame), max(64, max_samples // 2))).copy()
    cache_dir = project_root / output_cfg["cache_dir"]
    if config["signal_processing"].get("normalization") == "train_global":
        raw_settings = {**config["signal_processing"], "normalization": "none"}
        raw_settings.pop('training_mean', None)
        raw_settings.pop('training_std', None)
        raw_cache = ensure_preprocessed_cache(train_frame, root, cache_dir, split_name=f"train_{scope}", sampling_rate_hz=fs, config=FilterConfig.from_mapping(raw_settings))
        mean, std = fit_training_normalization(raw_cache)
        config["signal_processing"].update(training_mean=mean, training_std=std)
    filter_config = FilterConfig.from_mapping(config["signal_processing"])
    train_cache = ensure_preprocessed_cache(
        train_frame, root, cache_dir, split_name=f"train_{scope}", sampling_rate_hz=fs, config=filter_config
    )
    validation_cache = ensure_preprocessed_cache(
        validation_frame, root, cache_dir, split_name=f"validation_{scope}", sampling_rate_hz=fs, config=filter_config
    )
    train_dataset = CachedPTBXLDataset(train_cache, train_frame)
    distillation = train_cfg.get('distillation')
    if distillation is not None:
        if not 0 < float(distillation['alpha']) < 1:
            raise ValueError('Distillation alpha must be between zero and one.')
        if any(float(train_cfg.get(key, 0)) for key in ('augmentation_gain', 'augmentation_noise', 'mixup_alpha')):
            raise ValueError('Cached teacher logits require unchanged training waveforms.')
        train_dataset = DistillationDataset(
            train_dataset, train_frame.ecg_id.to_numpy(),
            resolve_project_path(distillation['logits_path'], project_root),
        )
    validation_dataset = CachedPTBXLDataset(validation_cache, validation_frame)
    generator = torch.Generator().manual_seed(seed)
    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )

    model = build_model(model_type, dropout=float(config["model"]["dropout"])).to(device)
    train_labels = train_frame.loc[:, list(SUPERCLASSES)].to_numpy(dtype=np.float32)
    positives = train_labels.sum(axis=0)
    pos_weight = np.ones(len(SUPERCLASSES), dtype=np.float32)
    np.divide(len(train_labels) - positives, positives, out=pos_weight, where=positives > 0)
    print(f"Training label positives: {dict(zip(SUPERCLASSES, positives.astype(int).tolist()))}", flush=True)
    weight_power = float(train_cfg.get("pos_weight_power", 1.0))
    if not 0 <= weight_power <= 1:
        raise ValueError("pos_weight_power must be between zero and one")
    pos_weight = pos_weight ** weight_power
    print(f"BCE positive weights: {dict(zip(SUPERCLASSES, pos_weight.round(3).tolist()))}", flush=True)
    criterion = build_loss(train_cfg, pos_weight, device)
    augmentation = TrainingAugmentation(train_cfg)
    print(f"Loss: {train_cfg.get('loss', 'bce')}; augmentation: gain={augmentation.gain}, noise={augmentation.noise}, mixup={augmentation.mixup_alpha}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=float(train_cfg["weight_decay"])
    )
    scheduler = None
    if train_cfg.get("scheduler", "none") == "warmup_cosine":
        warmup = int(train_cfg.get("warmup_epochs", 3))
        import math
        def lr_factor(index):
            if index < warmup:
                return (index + 1) / max(1, warmup)
            progress = (index - warmup) / max(1, epochs - warmup - 1)
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint_dir = project_root / output_cfg["checkpoint_dir"]
    metrics_dir = project_root / output_cfg["metrics_dir"]
    figure_dir = project_root / output_cfg["figure_dir"]
    for directory in (checkpoint_dir, metrics_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / (
        f"{model_type}_best.pt" if scope == "full" else f"{model_type}_smoke.pt"
    )
    plot_class_distribution(train_labels, figure_dir / f"class_distribution_{model_type}_{scope}.png")

    history: list[dict] = []
    best_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    patience = int(train_cfg["early_stopping_patience"])
    default_threshold = float(config["labels"]["default_threshold"])
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        try:
            train_loss, _, _ = _run_epoch(
                model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler, use_amp=use_amp, augmentation=augmentation,
                distillation=distillation
            )
            val_loss, val_targets, val_scores = _run_epoch(
                model, validation_loader, criterion, device, use_amp=use_amp
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"CUDA ran out of memory at batch_size={batch_size}; lower training.batch_size "
                "or pass --batch-size with a smaller value."
            ) from exc
        assert val_targets is not None and val_scores is not None
        val_metrics = calculate_multilabel_metrics(val_targets, val_scores, default_threshold)
        score = val_metrics["macro_auroc"]
        if score is None:
            score = -val_loss
        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_macro_auroc": val_metrics["macro_auroc"],
                "validation_macro_f1_at_default_threshold": val_metrics["macro_f1"],
                "seconds": time.monotonic() - started,
            }
        )
        print(
            f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_macro_auroc={val_metrics['macro_auroc']}, "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}, "
            f"seconds={history[-1]['seconds']:.1f}",
            flush=True,
        )
        if scheduler is not None:
            scheduler.step()
        if score > best_score + 1e-4:
            best_score = float(score)
            best_epoch = epoch
            stale_epochs = 0
            _save_checkpoint(
                checkpoint_path,
                {
                    "model_type": model_type,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "scope": scope,
                    "training_records": len(train_frame),
                    "validation_records": len(validation_frame),
                    "validation_macro_auroc": val_metrics["macro_auroc"],
                    "sampling_rate_hz": fs,
                    "lead_names": list(LEAD_NAMES),
                    "superclasses": list(SUPERCLASSES),
                    "thresholds": [default_threshold] * len(SUPERCLASSES),
                    "pos_weight": pos_weight.tolist(),
                    "config": run_config,
                    "torch_version": str(torch.__version__),
                    "device": str(device),
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs; best epoch was {best_epoch}.", flush=True)
                break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    _, val_targets, val_scores = _run_epoch(model, validation_loader, criterion, device, use_amp=use_amp)
    assert val_targets is not None and val_scores is not None
    thresholds = optimize_thresholds(val_targets, val_scores, fallback=default_threshold)
    checkpoint["thresholds"] = thresholds
    checkpoint["validation_metrics_tuned"] = calculate_multilabel_metrics(val_targets, val_scores, thresholds)
    _save_checkpoint(checkpoint_path, checkpoint)
    history_path = metrics_dir / f"training_history_{model_type}_{scope}.json"
    history_path.write_text(
        json.dumps(
            {"scope": scope, "best_epoch": best_epoch, "best_validation_score": best_score, "history": history},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    plot_training_curves(history, figure_dir / f"training_curve_{model_type}_{scope}.png")
    print(f"Best checkpoint: {checkpoint_path}", flush=True)
    print(f"Validation thresholds: {dict(zip(SUPERCLASSES, thresholds))}", flush=True)
    return checkpoint_path
