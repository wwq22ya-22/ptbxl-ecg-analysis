"""HTTP contract for the shared single-record prediction service."""

from __future__ import annotations

import torch
from fastapi.testclient import TestClient

from api.main import create_app
from src.data.ptbxl import LEAD_NAMES, SUPERCLASSES
from src.models import ECGCNN1D


def _checkpoint(path):
    model = ECGCNN1D()
    torch.save({
        "scope": "full", "model_type": "cnn1d", "model_state": model.state_dict(),
        "lead_names": list(LEAD_NAMES), "superclasses": list(SUPERCLASSES),
        "sampling_rate_hz": 100, "thresholds": [0.5] * 5,
        "config": {"model": {"dropout": 0.2}, "signal_processing": {
            "baseline_windows_s": [0.2, 0.6], "lowcut_hz": 0.5,
            "highcut_hz": 40.0, "bandpass_order": 4,
            "notch_hz": None, "normalization": "zscore",
        }},
    }, path)


def _payload():
    return {
        "signal": [[0.0] * 12 for _ in range(1000)],
        "sampling_rate_hz": 100,
        "lead_names": list(LEAD_NAMES),
        "units": ["mV"] * 12,
    }


def test_health_and_prediction(tmp_path):
    path = tmp_path / "model.pt"
    _checkpoint(path)
    with TestClient(create_app(path)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        response = client.post("/predict", json=_payload())
        assert response.status_code == 200
        result = response.json()
        assert tuple(result["probabilities"]) == SUPERCLASSES
        assert all(0 <= value <= 1 for value in result["probabilities"].values())
        assert result["positive_labels"] == [
            name for name in SUPERCLASSES
            if result["probabilities"][name] >= result["thresholds"][name]
        ]


def test_invalid_ecg_returns_client_error(tmp_path):
    path = tmp_path / "model.pt"
    _checkpoint(path)
    with TestClient(create_app(path)) as client:
        invalid = _payload()
        invalid["lead_names"] = list(reversed(LEAD_NAMES))
        response = client.post("/predict", json=invalid)
        assert response.status_code == 422
        assert "Lead names" in response.json()["detail"]


def test_missing_model_is_reported_without_fake_prediction(tmp_path):
    with TestClient(create_app(tmp_path / "missing.pt")) as client:
        health = client.get("/health").json()
        assert health["model_loaded"] is False
        assert health["status"] == "model_unavailable"
        response = client.post("/predict", json=_payload())
        assert response.status_code == 503
