"""Read-only probability ensemble of trusted local ECG checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch

from src.data.ptbxl import SUPERCLASSES
from src.inference.predict import Prediction, load_predictor


class ECGEnsemblePredictor:
    def __init__(self, predictors, weights, thresholds):
        self.predictors = tuple(predictors)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        if not self.predictors or self.weights.shape != (len(self.predictors),):
            raise ValueError('One weight is required per ensemble component.')
        if not np.isfinite(self.weights).all() or np.any(self.weights <= 0) or not np.isclose(self.weights.sum(), 1):
            raise ValueError('Positive ensemble weights must sum to one.')
        if self.thresholds.shape != (5,) or not np.isfinite(self.thresholds).all() or np.any((self.thresholds <= 0) | (self.thresholds >= 1)):
            raise ValueError('Five thresholds strictly between zero and one are required.')
        self.sampling_rate_hz = self.predictors[0].sampling_rate_hz
        if any(p.sampling_rate_hz != self.sampling_rate_hz for p in self.predictors):
            raise ValueError('Ensemble sampling rates must agree.')

    def predict(self, signal, *, sampling_rate_hz, lead_names, units):
        # Each component uses its own saved training normalization statistics.
        scores = []
        for predictor in self.predictors:
            prediction = predictor.predict(signal, sampling_rate_hz=sampling_rate_hz, lead_names=lead_names, units=units)
            scores.append([prediction.probabilities[c] for c in SUPERCLASSES])
        average = np.average(np.asarray(scores), axis=0, weights=self.weights)
        probabilities = dict(zip(SUPERCLASSES, average.tolist()))
        thresholds = dict(zip(SUPERCLASSES, self.thresholds.tolist()))
        positive = tuple(c for c in SUPERCLASSES if probabilities[c] >= thresholds[c])
        return Prediction(probabilities, thresholds, positive)

    def predict_record(self, record):
        return self.predict(record.signal, sampling_rate_hz=record.sampling_rate_hz, lead_names=record.lead_names, units=record.units)


def load_ensemble(manifest_path, *, device: str | torch.device | None = None):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('type') != 'probability_ensemble' or tuple(manifest.get('superclasses', ())) != SUPERCLASSES:
        raise ValueError('Invalid ensemble manifest or label order.')
    components = []
    for component in manifest['components']:
        model_path = Path(component['checkpoint'])
        if not model_path.is_absolute():
            model_path = path.parent / model_path
        components.append(load_predictor(model_path, device=device))
    return ECGEnsemblePredictor(components, [c['weight'] for c in manifest['components']], manifest['thresholds'])
