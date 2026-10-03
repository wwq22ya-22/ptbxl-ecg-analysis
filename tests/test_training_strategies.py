import numpy as np
import pytest
import torch
from src.models import build_model
from src.training.strategies import AsymmetricLoss, TrainingAugmentation, build_loss


def test_asl_without_focusing_matches_bce_at_extreme_logits():
    logits = torch.tensor([[-1000., 1000., -2., 3.]], requires_grad=True)
    targets = torch.tensor([[1., 0., 0., 1.]])
    criterion = AsymmetricLoss(gamma_negative=0, gamma_positive=0, clip=0)
    loss = criterion(logits, targets)
    reference = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    torch.testing.assert_close(loss, reference)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] > 0


def test_asl_ignores_easy_negatives_and_keeps_grad_mode():
    logits = torch.tensor([[-20., 20., 0.]], requires_grad=True)
    loss = AsymmetricLoss()(logits, torch.zeros_like(logits))
    loss.backward()
    assert logits.grad[0, 0] == 0
    assert torch.isfinite(logits.grad).all() and logits.grad[0, 2] > 0
    with torch.no_grad():
        assert not AsymmetricLoss()(logits, torch.zeros_like(logits)).requires_grad
        assert not torch.is_grad_enabled()


def test_augmentation_default_is_identity_and_does_not_consume_randomness():
    x, y = torch.randn(4, 12, 100), torch.ones(4, 5)
    before = torch.get_rng_state().clone()
    xx, yy, second, weight = TrainingAugmentation({})(x, y)
    assert xx is x and yy is y and second is y and weight == 1
    assert torch.equal(before, torch.get_rng_state())


def test_gain_preserves_lead_ratios_without_mutating_input():
    x = torch.arange(1, 13).float()[None, :, None].expand(8, 12, 100).clone()
    original = x.clone()
    xx, *_ = TrainingAugmentation({'augmentation_gain': 0.1})(x, torch.zeros(8, 5))
    ratio = xx / original
    torch.testing.assert_close(ratio, ratio[:, :1, :1].expand_as(ratio))
    assert (ratio >= 0.9).all() and (ratio <= 1.1).all()
    torch.testing.assert_close(x, original)


def test_mixup_interpolated_loss_equals_soft_target_bce():
    x, y = torch.randn(6, 12, 100), torch.randint(0, 2, (6, 5)).float()
    original = x.clone()
    xx, first, second, weight = TrainingAugmentation({'mixup_alpha': 0.2})(x, y)
    logits = torch.randn(6, 5, requires_grad=True)
    criterion = build_loss({}, np.arange(1, 6, dtype=np.float32), 'cpu')
    loss = weight * criterion(logits, first) + (1 - weight) * criterion(logits, second)
    torch.testing.assert_close(loss, criterion(logits, weight * first + (1 - weight) * second))
    torch.testing.assert_close(x, original)
    assert xx.shape == x.shape and 0 <= weight <= 1


def test_avgmax_resnet_contract_and_backward():
    torch.set_num_threads(2)
    model = build_model('resnet_avgmax')
    output = model(torch.randn(2, 12, 1000))
    assert output.shape == (2, 5)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError):
        model(torch.randn(2, 11, 1000))


@pytest.mark.parametrize('config', [{'loss': 'unknown'}, {'loss': 'asl', 'asl_gamma_negative': -1}])
def test_invalid_loss_is_rejected(config):
    with pytest.raises(ValueError):
        build_loss(config, np.ones(5, dtype=np.float32), 'cpu')


def test_ensemble_average_rejects_record_or_label_mismatch():
    from scripts.run_further_optimization import aligned_mean
    first = {'ecg_ids': np.array([10, 20]), 'targets': np.zeros((2, 5)), 'probabilities': np.full((2, 5), 0.2)}
    second = {**first, 'probabilities': np.full((2, 5), 0.8)}
    np.testing.assert_allclose(aligned_mean(first, second), 0.5)
    with pytest.raises(ValueError, match='IDs/order'):
        aligned_mean(first, {**second, 'ecg_ids': np.array([20, 10])})
    with pytest.raises(ValueError, match='targets'):
        aligned_mean(first, {**second, 'targets': np.ones((2, 5))})


def test_ensemble_uses_ensemble_thresholds_and_each_component():
    from src.inference.ensemble import ECGEnsemblePredictor
    from src.inference.predict import Prediction
    from src.data.ptbxl import SUPERCLASSES
    class FakePredictor:
        sampling_rate_hz = 100
        def __init__(self, value):
            self.value, self.calls = value, 0
        def predict(self, *args, **kwargs):
            self.calls += 1
            return Prediction(dict.fromkeys(SUPERCLASSES, self.value), dict.fromkeys(SUPERCLASSES, 0.9), ())
    a, b = FakePredictor(0.2), FakePredictor(0.8)
    predictor = ECGEnsemblePredictor([a, b], [0.5, 0.5], [0.4, 0.6, 0.4, 0.6, 0.4])
    r = predictor.predict(np.zeros((1000, 12)), sampling_rate_hz=100, lead_names=(), units=())
    assert r.positive_labels == ('NORM', 'STTC', 'HYP')
    assert a.calls == b.calls == 1
    with pytest.raises(ValueError, match='sum to one'):
        ECGEnsemblePredictor([a, b], [0.5, 0.6], [0.5] * 5)


def test_constrained_threshold_solver_matches_exhaustive_small_problem():
    import itertools
    from src.training.metrics import constrained_pr_thresholds, calculate_multilabel_metrics
    rng = np.random.default_rng(13)
    truth = np.array([[1,0,1,0,1], [0,1,0,1,0], [1,0,0,1,0], [0,1,1,0,1]])
    scores = rng.uniform(0.05, 0.95, truth.shape)
    best = -1
    for thresholds in itertools.product(*[sorted(set(scores[:, i])) for i in range(5)]):
        # Compute directly to keep enumeration small and independent of solver.
        predicted = scores >= thresholds
        tp = (predicted * truth).sum(axis=0)
        precision = np.divide(tp, predicted.sum(axis=0), out=np.zeros(5), where=predicted.sum(axis=0)>0)
        recall = tp / truth.sum(axis=0)
        if precision.mean() >= 0.5 and recall.mean() >= 0.6:
            f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(5), where=(precision+recall)>0)
            best = max(best, f1.mean())
    thresholds, diagnostics = constrained_pr_thresholds(truth, scores, precision_floor=0.5, recall_floor=0.6)
    assert thresholds is not None and diagnostics['optimal']
    actual = calculate_multilabel_metrics(truth, scores, thresholds)
    assert actual['macro_precision'] >= 0.5 and actual['macro_recall'] >= 0.6
    assert actual['macro_f1'] == pytest.approx(best)
    infeasible, diagnostics = constrained_pr_thresholds(truth, scores, precision_floor=1, recall_floor=1)
    assert infeasible is None and diagnostics['status'] == 2
