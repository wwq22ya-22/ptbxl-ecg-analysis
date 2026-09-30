"""Neural networks for ECG classification."""

from .cnn1d import ECGCNN1D
from .resnet1d import ECGResNet1D

MODEL_TYPES = ("cnn1d", "resnet1d")


def build_model(model_type: str, *, dropout: float = 0.2):
    """Create a supported five-logit model using a shared input contract."""
    if model_type == "cnn1d":
        return ECGCNN1D(dropout=dropout)
    if model_type == "resnet1d":
        return ECGResNet1D(dropout=dropout)
    raise ValueError(f"Unsupported model type {model_type!r}; choose from {MODEL_TYPES}.")


__all__ = ["ECGCNN1D", "ECGResNet1D", "MODEL_TYPES", "build_model"]
