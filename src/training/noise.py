"""Controlled raw-voltage noise and record/view-aligned distillation datasets."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def snr_db(signal, residual, *, axis=0):
    """Per-lead AC signal power / residual power; zero-power leads give NaN."""
    reference = np.asarray(signal, dtype=np.float64)
    noise = np.asarray(residual, dtype=np.float64)
    reference = reference - reference.mean(axis=axis, keepdims=True)
    power = np.mean(reference**2, axis=axis)
    noise_power = np.mean(noise**2, axis=axis)
    output = np.full_like(power, np.nan, dtype=np.float64)
    valid = (power > 1e-20) & (noise_power > 1e-20)
    output[valid] = 10 * np.log10(power[valid] / noise_power[valid])
    return output


def add_raw_gaussian_noise(signal, level_db, *, seed):
    """Add zero-mean Gaussian noise at an exact per-lead AC SNR, in mV.

    Input is [time, lead]. The generator is private and deterministic; no
    cached input or global RNG state is modified. Flat leads stay unchanged.
    """
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Raw ECG must be finite [time, lead] with at least two points.')
    if not np.isfinite(level_db):
        raise ValueError('Noise SNR must be finite.')
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(values.shape)
    noise -= noise.mean(axis=0, keepdims=True)
    signal_rms = np.sqrt(np.mean((values-values.mean(axis=0, keepdims=True))**2, axis=0))
    noise_rms = np.sqrt(np.mean(noise**2, axis=0))
    noise *= (signal_rms / np.maximum(noise_rms, 1e-12) * 10**(-float(level_db)/20))[None]
    corrupted = (values + noise).astype(np.float32)
    return corrupted, corrupted.astype(np.float64)-values


class NoiseViewsDataset(Dataset):
    """Draw one clean/noisy view per record, paired with that exact teacher view."""
    def __init__(self, views, targets, ecg_ids, *, probabilities, seed,
                 teacher_logits=None, teacher_ids=None):
        self.views = [np.load(v, mmap_mode='r') if isinstance(v, (str, Path)) else np.asarray(v) for v in views]
        self.targets = np.asarray(targets, dtype=np.float32)
        self.ecg_ids = np.asarray(ecg_ids, dtype=np.int64)
        expected = (len(self.ecg_ids), 12, 1000)
        if not self.views or self.targets.shape != (len(self.ecg_ids), 5):
            raise ValueError('Expected nonempty ECG views and five-label targets.')
        if any(v.shape != expected or v.dtype != np.float32 for v in self.views):
            raise ValueError('Every view must contain the same float32 [N,12,1000] records.')
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        if (self.probabilities.shape != (len(self.views),) or not np.isfinite(self.probabilities).all()
                or np.any(self.probabilities < 0) or not np.isclose(self.probabilities.sum(), 1)):
            raise ValueError('View probabilities must be finite, nonnegative and sum to one.')
        self.teacher_logits = None
        if teacher_logits is not None:
            if not np.array_equal(self.ecg_ids, np.asarray(teacher_ids)):
                raise ValueError('Teacher ECG IDs must match training records in exact order.')
            logits = np.asarray(teacher_logits, dtype=np.float32)
            if logits.shape != (len(self.views), len(self.ecg_ids), 5) or not np.isfinite(logits).all():
                raise ValueError('Teacher logits must be finite [view,record,5].')
            self.teacher_logits = logits
        self.seed = int(seed)
        self.set_epoch(0)

    def set_epoch(self, epoch):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(epoch)]))
        self.selected_views = rng.choice(len(self.views), len(self.ecg_ids), p=self.probabilities)

    def __len__(self):
        return len(self.ecg_ids)

    def __getitem__(self, index):
        view = int(self.selected_views[index])
        pair = (torch.from_numpy(np.array(self.views[view][index], copy=True)),
                torch.from_numpy(self.targets[index].copy()))
        if self.teacher_logits is not None:
            return (*pair, torch.from_numpy(self.teacher_logits[view, index].copy()))
        return pair
