"""Single-record inference contracts independent of the PTB-XL download."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.ptbxl import LEAD_NAMES, SUPERCLASSES
from src.inference import load_predictor
from src.models import ECGCNN1D


@pytest.fixture
def predictor(tmp_path):
    model = ECGCNN1D(dropout=0.2)
    checkpoint = {
        "scope": "full",
        "model_type": "cnn1d",
        "lead_names": list(LEAD_NAMES),
        "superclasses": list(SUPERCLASSES),
        "sampling_rate_hz": 100,
        "thresholds": [0.5] * 5,
        "config": {
            "model": {"dropout": 0.2},
            "signal_processing": {
                "baseline_windows_s": [0.2, 0.6], "lowcut_hz": 0.5,
                "highcut_hz": 40.0, "bandpass_order": 4,
                "notch_hz": None, "normalization": "zscore",
            },
        },
        "model_state": model.state_dict(),
    }
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    return load_predictor(path, device="cpu")


def test_single_record_probabilities_and_threshold_labels(predictor):
    time = np.arange(1000) / 100
    signal = np.stack([np.sin(2 * np.pi * (1 + lead / 10) * time) for lead in range(12)], axis=1)
    result = predictor.predict(
        signal, sampling_rate_hz=100, lead_names=LEAD_NAMES, units=("mV",) * 12,
    )
    assert tuple(result.probabilities) == SUPERCLASSES
    assert all(0 <= value <= 1 for value in result.probabilities.values())
    assert result.positive_labels == tuple(
        name for name in SUPERCLASSES if result.probabilities[name] >= result.thresholds[name]
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"sampling_rate_hz": 500}, "100 Hz"),
        ({"lead_names": tuple(reversed(LEAD_NAMES))}, "Lead names"),
        ({"units": ("unknown",) * 12}, "mV"),
        ({"signal": np.zeros((999, 12))}, "shape"),
    ],
)
def test_single_record_rejects_incompatible_inputs(predictor, changes, message):
    inputs = {
        "signal": np.zeros((1000, 12)), "sampling_rate_hz": 100,
        "lead_names": LEAD_NAMES, "units": ("mV",) * 12,
    }
    inputs.update(changes)
    signal = inputs.pop("signal")
    with pytest.raises(ValueError, match=message):
        predictor.predict(signal, **inputs)
