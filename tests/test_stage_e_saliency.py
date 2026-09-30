"""Verify gradient response can be computed and aligned with the ECG time axis."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.ptbxl import ECGRecord, LEAD_NAMES, SUPERCLASSES
from src.explainability import compute_input_saliency
from src.inference.predict import ECGPredictor
from src.models import ECGCNN1D
from src.visualization.saliency_plot import plot_saliency_overlay


def test_saliency_on_twelve_lead_record():
    torch.manual_seed(7)
    model = ECGCNN1D(dropout=0.2)
    predictor = ECGPredictor({
        "scope": "full", "model_type": "cnn1d", "model_state": model.state_dict(),
        "lead_names": list(LEAD_NAMES), "superclasses": list(SUPERCLASSES),
        "sampling_rate_hz": 100, "thresholds": [0.5] * 5,
        "config": {"model": {"dropout": 0.2}, "signal_processing": {
            "baseline_windows_s": [0.2, 0.6], "lowcut_hz": 0.5,
            "highcut_hz": 40.0, "bandpass_order": 4,
            "notch_hz": None, "normalization": "zscore",
        }},
    }, torch.device("cpu"))
    time = np.arange(1000) / 100
    signal = np.stack([np.sin(2 * np.pi * (1 + i / 10) * time) for i in range(12)], axis=1).astype(np.float32)
    record = ECGRecord(signal, 100, LEAD_NAMES, ("mV",) * 12, "synthetic")
    result = compute_input_saliency(predictor, record, class_name="MI")
    assert result.class_name == "MI"
    assert result.importance.shape == (1000,)
    assert np.isfinite(result.importance).all()
    assert np.min(result.importance) >= 0 and np.max(result.importance) <= 1
    figure = plot_saliency_overlay(record, result)
    assert len(figure.axes) == 2
    with pytest.raises(ValueError, match="Unknown superclass"):
        compute_input_saliency(predictor, record, class_name="OTHER")
