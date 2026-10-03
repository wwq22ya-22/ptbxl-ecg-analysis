"""Neural networks for ECG classification."""

from .cnn1d import ECGCNN1D
from .resnet1d import ECGResNet1D, ECGResNetAvgMax1D
from .alternatives import ECGInception1D, ECGCNNBiGRU

MODEL_TYPES = ("cnn1d", "resnet1d", "inception1d", "cnn_bigru", "resnet_avgmax")


def build_model(model_type: str, *, dropout: float = 0.2):
    """Create a supported five-logit model using a shared input contract."""
    if model_type == "cnn1d":
        return ECGCNN1D(dropout=dropout)
    if model_type == "resnet1d":
        return ECGResNet1D(dropout=dropout)
    if model_type == "resnet_avgmax":
        return ECGResNetAvgMax1D(dropout=dropout)
    if model_type == "inception1d":
        return ECGInception1D(dropout=dropout)
    if model_type == "cnn_bigru":
        return ECGCNNBiGRU(dropout=dropout)
    raise ValueError(f"Unsupported model type {model_type!r}; choose from {MODEL_TYPES}.")


__all__ = ["ECGCNN1D", "ECGResNet1D", "MODEL_TYPES", "build_model"]
