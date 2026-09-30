"""Apply the training-time ECG pipeline to one twelve-lead record."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.data.ptbxl import LEAD_NAMES, SUPERCLASSES, ECGRecord
from src.models import MODEL_TYPES, build_model
from src.signal_processing.filters import FilterConfig, preprocess_ecg


@dataclass(frozen=True)
class Prediction:
    probabilities: dict[str, float]
    thresholds: dict[str, float]
    positive_labels: tuple[str, ...]


class ECGPredictor:
    """Read-only model service; checks the waveform before inference."""

    def __init__(self, checkpoint: dict, device: torch.device):
        if checkpoint.get("scope") != "full":
            raise ValueError("Single-record inference requires a full-training checkpoint.")
        if checkpoint.get("model_type") not in MODEL_TYPES:
            raise ValueError("Unsupported checkpoint model type.")
        self.model_type = checkpoint["model_type"]
        if tuple(checkpoint.get("lead_names", ())) != LEAD_NAMES:
            raise ValueError("Checkpoint lead order does not match the standard twelve leads.")
        if tuple(checkpoint.get("superclasses", ())) != SUPERCLASSES:
            raise ValueError("Checkpoint label order does not match the five superclasses.")
        self.sampling_rate_hz = int(checkpoint["sampling_rate_hz"])
        self.thresholds = np.asarray(checkpoint["thresholds"], dtype=np.float64)
        if self.thresholds.shape != (len(SUPERCLASSES),) or not np.isfinite(self.thresholds).all():
            raise ValueError("Checkpoint must contain five finite class thresholds.")
        if np.any((self.thresholds <= 0) | (self.thresholds >= 1)):
            raise ValueError("Checkpoint thresholds must be strictly between zero and one.")
        self.filter_config = FilterConfig.from_mapping(checkpoint["config"]["signal_processing"])
        self.device = device
        self.model = build_model(self.model_type, dropout=float(checkpoint["config"]["model"]["dropout"]))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(device).eval()

    def predict(
        self,
        signal: np.ndarray,
        *,
        sampling_rate_hz: int,
        lead_names: Sequence[str],
        units: Sequence[str],
    ) -> Prediction:
        batch = self.prepare_tensor(
            signal, sampling_rate_hz=sampling_rate_hz, lead_names=lead_names, units=units
        )
        with torch.inference_mode():
            scores = torch.sigmoid(self.model(batch))[0].cpu().numpy()
        probabilities = {name: float(scores[index]) for index, name in enumerate(SUPERCLASSES)}
        thresholds = {name: float(self.thresholds[index]) for index, name in enumerate(SUPERCLASSES)}
        positive = tuple(name for name in SUPERCLASSES if probabilities[name] >= thresholds[name])
        return Prediction(probabilities, thresholds, positive)

    def prepare_tensor(
        self,
        signal: np.ndarray,
        *,
        sampling_rate_hz: int,
        lead_names: Sequence[str],
        units: Sequence[str],
    ) -> torch.Tensor:
        """Validate and preprocess one physical ECG using checkpoint settings."""
        values = np.asarray(signal, dtype=np.float32)
        expected_shape = (10 * self.sampling_rate_hz, len(LEAD_NAMES))
        if values.shape != expected_shape:
            raise ValueError(f"Expected ECG shape {expected_shape}, got {values.shape}.")
        if sampling_rate_hz != self.sampling_rate_hz:
            raise ValueError(f"Expected {self.sampling_rate_hz} Hz ECG, got {sampling_rate_hz} Hz.")
        if tuple(lead_names) != LEAD_NAMES:
            raise ValueError("Lead names or order do not match the training input.")
        if len(units) != len(LEAD_NAMES) or any(str(unit).casefold() != "mv" for unit in units):
            raise ValueError("All twelve leads must use physical mV units.")
        if not np.isfinite(values).all():
            raise ValueError("ECG contains NaN or infinite samples.")
        processed = preprocess_ecg(values, self.sampling_rate_hz, config=self.filter_config)
        return torch.from_numpy(np.ascontiguousarray(processed.T[None])).to(self.device)

    def predict_record(self, record: ECGRecord) -> Prediction:
        return self.predict(
            record.signal,
            sampling_rate_hz=record.sampling_rate_hz,
            lead_names=record.lead_names,
            units=record.units,
        )


def load_predictor(checkpoint_path: str | Path, *, device: str | torch.device | None = None) -> ECGPredictor:
    """Load a trusted local checkpoint using PyTorch's restricted loader."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}. Run scripts/train_baseline.py first.")
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return ECGPredictor(checkpoint, resolved_device)
