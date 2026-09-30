"""Single-record ECG classification with a trained checkpoint."""

from .predict import ECGPredictor, Prediction, load_predictor

__all__ = ["ECGPredictor", "Prediction", "load_predictor"]
