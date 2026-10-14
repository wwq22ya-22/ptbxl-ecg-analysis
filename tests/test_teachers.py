"""Verify larger teachers can learn independent labels and reload exactly."""
import pytest
import torch

from src.models import build_model


@pytest.mark.parametrize("kind", ["se_resnet18", "inception_time1d", "ribeiro_resnet100"])
def test_teacher_backward_and_checkpoint_roundtrip(kind, tmp_path):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = build_model(kind)
    signals = torch.randn(2, 12, 1000)
    truth = torch.tensor([[1., 0., 1., 0., 0.], [0., 1., 0., 1., 1.]])
    logits = model(signals)
    assert logits.shape == truth.shape
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, truth)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = model.classifier[-1].weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.classifier[-1].weight)
    model.eval()
    with torch.inference_mode():
        expected = model(signals)
    path = tmp_path / "teacher.pt"
    torch.save(model.state_dict(), path)
    restored = build_model(kind).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.inference_mode():
        torch.testing.assert_close(restored(signals), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="Expected ECG"):
        restored(torch.randn(2, 1, 1000))


def test_probability_ensemble_averages_independent_probabilities_and_freezes_members():
    from src.models.probability_ensemble import ProbabilityEnsemble
    members = [torch.nn.Linear(2, 5), torch.nn.Linear(2, 5)]
    with torch.no_grad():
        members[0].weight.zero_(); members[0].bias.copy_(torch.tensor([4., -4., 2., 0., -2.]))
        members[1].weight.zero_(); members[1].bias.copy_(torch.tensor([-1., 1., 1., 2., -3.]))
    ensemble = ProbabilityEnsemble(members)
    x = torch.zeros(1, 2)
    expected = torch.stack([member(x).sigmoid() for member in members]).mean(0)
    torch.testing.assert_close(ensemble(x).sigmoid(), expected, rtol=1e-6, atol=1e-7)
    assert not any(parameter.requires_grad for parameter in ensemble.parameters())
    assert not ensemble.training and all(not member.training for member in ensemble.members)
    assert not torch.allclose(expected, torch.stack([member(x) for member in members]).mean(0).sigmoid())
