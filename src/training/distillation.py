"""Independent-label distillation with record-aligned frozen teacher logits."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


class DistillationDataset(Dataset):
    def __init__(self, dataset, ecg_ids, path: Path):
        data = np.load(path, allow_pickle=False)
        if not np.array_equal(data['ecg_ids'], np.asarray(ecg_ids)):
            raise ValueError('Teacher ECG IDs must match training records in exact order.')
        logits = data['logits']
        if logits.shape != (len(dataset), 5) or not np.isfinite(logits).all():
            raise ValueError('Teacher logits must be finite [training_records, 5].')
        self.dataset = dataset
        self.logits = torch.from_numpy(logits.astype(np.float32))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        signal, label = self.dataset[index]
        return signal, label, self.logits[index]


def multilabel_distillation_loss(student_logits, teacher_logits, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Distillation temperature must be finite and positive.')
    # Five Bernoulli distributions; class probabilities need not sum to one.
    targets = torch.sigmoid(teacher_logits.detach().float() / temperature)
    return F.binary_cross_entropy_with_logits(
        student_logits.float() / temperature, targets
    ) * temperature ** 2
