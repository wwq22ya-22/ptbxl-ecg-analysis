"""Neural networks for ECG classification."""

from .cnn1d import ECGCNN1D
from .resnet1d import ECGResNet1D, ECGResNetAvgMax1D
from .alternatives import ECGInception1D, ECGCNNBiGRU
from .teachers import ECGSEResNet18, ECGInceptionTime1D
from .ribeiro_resnet import ECGRibeiroResNet100
from .unet1d import ECGUNet1D, ECGUNetEncoder1D

MODEL_TYPES = ("cnn1d", "resnet1d", "inception1d", "cnn_bigru", "resnet_avgmax", "se_resnet18", "inception_time1d", "ribeiro_resnet100", "unet1d", "unet_encoder1d")


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
    if model_type == "se_resnet18":
        return ECGSEResNet18(dropout=dropout)
    if model_type == "inception_time1d":
        return ECGInceptionTime1D(dropout=dropout)
    if model_type == "ribeiro_resnet100":
        return ECGRibeiroResNet100(dropout=dropout)
    if model_type == "unet1d":
        return ECGUNet1D(dropout=dropout)
    if model_type == "unet_encoder1d":
        return ECGUNetEncoder1D(dropout=dropout)
    raise ValueError(f"Unsupported model type {model_type!r}; choose from {MODEL_TYPES}.")


__all__ = ["ECGCNN1D", "ECGResNet1D", "MODEL_TYPES", "build_model"]

