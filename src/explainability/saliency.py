"""Gradient-times-input temporal saliency for one ECG superclass logit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from src.data.ptbxl import ECGRecord, SUPERCLASSES
from src.inference import ECGPredictor


@dataclass(frozen=True)
class SaliencyResult:
    class_name: str
    probability: float
    importance: np.ndarray
    sampling_rate_hz: int


def compute_input_saliency(
    predictor: ECGPredictor,
    record: ECGRecord,
    *,
    class_name: str,
) -> SaliencyResult:
    """Average absolute gradient×input over all twelve leads at each sample.

    A short Gaussian smoothing window reduces isolated pixel-scale spikes.
    Scores are normalized within this one record for visualization only; they
    are not calibrated probabilities or evidence of clinical causality.
    """
    if class_name not in SUPERCLASSES:
        raise ValueError(f"Unknown superclass {class_name!r}; choose one of {SUPERCLASSES}.")
    batch = predictor.prepare_tensor(
        record.signal,
        sampling_rate_hz=record.sampling_rate_hz,
        lead_names=record.lead_names,
        units=record.units,
    ).detach().requires_grad_(True)
    predictor.model.eval()
    with torch.enable_grad():
        logits = predictor.model(batch)
        index = SUPERCLASSES.index(class_name)
        gradient = torch.autograd.grad(logits[0, index], batch)[0]
        relevance = (gradient * batch).abs().mean(dim=1)[0]
    values = relevance.detach().float().cpu().numpy()
    values = gaussian_filter1d(values, sigma=max(1.0, 0.04 * predictor.sampling_rate_hz))
    scale = float(np.percentile(values, 99))
    importance = np.clip(values / scale, 0.0, 1.0) if scale > 0 else np.zeros_like(values)
    return SaliencyResult(
        class_name=class_name,
        probability=float(torch.sigmoid(logits[0, index]).detach().cpu()),
        importance=importance.astype(np.float32),
        sampling_rate_hz=predictor.sampling_rate_hz,
    )
