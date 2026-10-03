import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from src.training.train import _run_epoch

from src.training.distillation import DistillationDataset, multilabel_distillation_loss


def test_distillation_independent_labels_and_frozen_teacher():
    student = torch.tensor([[0., 0., 0., 0., 0.]], requires_grad=True)
    teacher = torch.tensor([[4., 4., -4., -4., 4.]], requires_grad=True)
    loss = multilabel_distillation_loss(student, teacher, 2.)
    loss.backward()
    assert teacher.grad is None
    assert torch.all(student.grad[0, [0, 1, 4]] < 0)
    assert torch.all(student.grad[0, [2, 3]] > 0)
    matched = teacher.detach().clone().requires_grad_()
    multilabel_distillation_loss(matched, teacher, 2.).backward()
    torch.testing.assert_close(matched.grad, torch.zeros_like(matched))


def test_teacher_record_order_is_checked(tmp_path):
    path = tmp_path / 'teacher.npz'
    np.savez(path, ecg_ids=[20, 10], logits=np.zeros((2, 5)))
    dataset = TensorDataset(torch.zeros(2, 12, 1000), torch.zeros(2, 5))
    with pytest.raises(ValueError, match='exact order'):
        DistillationDataset(dataset, [10, 20], path)
    checked = DistillationDataset(dataset, [20, 10], path)
    assert len(checked[0]) == 3


@pytest.mark.parametrize('temperature', [0, -1, float('nan')])
def test_invalid_temperature_rejected(temperature):
    with pytest.raises(ValueError):
        multilabel_distillation_loss(torch.zeros(1, 5), torch.zeros(1, 5), temperature)


def test_training_step_uses_both_real_labels_and_aligned_teacher():
    model = torch.nn.Linear(2, 5, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    signal = torch.tensor([[1., 0.]])
    truth = torch.tensor([[1., 0., 1., 0., 1.]])
    teacher = torch.tensor([[-4., 4., -4., 4., -4.]])
    loader = DataLoader(TensorDataset(signal, truth, teacher), batch_size=1)
    expected = torch.zeros(1, 5, requires_grad=True)
    loss = .5 * torch.nn.functional.binary_cross_entropy_with_logits(expected, truth)
    loss += .5 * multilabel_distillation_loss(expected, teacher, 2.)
    loss.backward()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    actual, _, _ = _run_epoch(model, loader, torch.nn.BCEWithLogitsLoss(), torch.device('cpu'),
                              optimizer=optimizer, scaler=scaler,
                              distillation={'alpha':.5, 'temperature':2.})
    assert actual == pytest.approx(loss.item())
    torch.testing.assert_close(model.weight[:, 0], -.1 * expected.grad[0])
    torch.testing.assert_close(model.weight[:, 1], torch.zeros(5))
