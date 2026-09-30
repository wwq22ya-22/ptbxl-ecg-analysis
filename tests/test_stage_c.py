"""Focused checks for the model and multi-label evaluation contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models import ECGCNN1D, ECGResNet1D, build_model
from src.training.metrics import calculate_multilabel_metrics, optimize_thresholds


def test_cnn_accepts_twelve_leads_and_outputs_independent_logits():
    model = ECGCNN1D(dropout=0)
    signal = torch.randn(3, 12, 1000)
    logits = model(signal)
    assert logits.shape == (3, 5)
    loss = torch.nn.BCEWithLogitsLoss()(logits, torch.randint(0, 2, (3, 5)).float())
    loss.backward()
    assert model.features[0][0].weight.grad is not None
    with pytest.raises(ValueError, match="12"):
        model(torch.randn(3, 11, 1000))


def test_resnet_factory_preserves_the_five_logit_contract():
    model = build_model("resnet1d", dropout=0).eval()
    signal = torch.randn(2, 12, 1000)
    with torch.no_grad():
        logits = model(signal)
    assert isinstance(model, ECGResNet1D)
    assert logits.shape == (2, 5)


def test_metrics_keep_label_order_and_handle_undefined_auc():
    truth = np.array([[1, 0, 1, 0, 0], [0, 1, 1, 0, 0], [1, 0, 1, 0, 0], [0, 1, 1, 0, 0]])
    scores = np.array([[.9, .1, .8, .2, .3], [.2, .8, .7, .2, .3], [.8, .2, .6, .2, .3], [.1, .9, .5, .2, .3]])
    result = calculate_multilabel_metrics(truth, scores)
    assert result["records"] == 4
    assert result["classes_with_defined_auroc"] == 2
    assert result["macro_auroc"] == 1
    assert result["per_class"]["NORM"]["confusion_matrix"] == [[2, 0], [0, 2]]
    assert result["per_class"]["HYP"]["auroc"] is None
    thresholds = optimize_thresholds(truth, scores)
    assert len(thresholds) == 5
    assert thresholds[3:] == [0.5, 0.5]


def test_metrics_reject_invalid_probability_range():
    with pytest.raises(ValueError, match="Probabilities"):
        calculate_multilabel_metrics(np.zeros((2, 5)), np.full((2, 5), 1.1))
