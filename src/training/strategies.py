"""Optional training strategies; baseline behavior stays the default."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class AsymmetricLoss(nn.Module):
    """ASL with detached focusing weights and mean reduction, computed in FP32.

    Formula reference: Alibaba-MIIL/ASL/src/loss_functions/losses.py.
    Mean reduction keeps the batch/label scale comparable to the BCE baseline.
    Targets passed to this loss are binary; mixup interpolates two losses.
    """

    def __init__(self, gamma_negative=4.0, gamma_positive=1.0, clip=0.05):
        super().__init__()
        if not all(math.isfinite(v) for v in (gamma_negative, gamma_positive, clip)):
            raise ValueError("ASL settings must be finite.")
        if gamma_negative < 0 or gamma_positive < 0 or not 0 <= clip < 1:
            raise ValueError("Invalid ASL focusing or clipping settings.")
        self.gamma_negative = gamma_negative
        self.gamma_positive = gamma_positive
        self.clip = clip

    def forward(self, logits, targets):
        logits, targets = logits.float(), targets.float()
        positive = logits.sigmoid()
        negative = (-logits).sigmoid()
        log_positive = F.logsigmoid(logits)
        if self.clip:
            negative = (negative + self.clip).clamp(max=1)
            log_negative = negative.clamp(min=1e-8).log()
        else:
            log_negative = F.logsigmoid(-logits)
        log_likelihood = targets * log_positive + (1 - targets) * log_negative
        if self.gamma_negative or self.gamma_positive:
            with torch.no_grad():
                probability = positive * targets + negative * (1 - targets)
                gamma = self.gamma_positive * targets + self.gamma_negative * (1 - targets)
                weights = (1 - probability).pow(gamma)
            log_likelihood = log_likelihood * weights
        return -log_likelihood.mean()


def build_loss(config, pos_weight, device):
    kind = config.get('loss', 'bce')
    if kind == 'bce':
        return nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, device=device))
    if kind == 'asl':
        return AsymmetricLoss(
            gamma_negative=float(config.get('asl_gamma_negative', 4)),
            gamma_positive=float(config.get('asl_gamma_positive', 1)),
            clip=float(config.get('asl_clip', 0.05)),
        )
    raise ValueError(f"Unsupported loss: {kind}")


class TrainingAugmentation:
    """Small record-wise gain/noise changes and optional batch mixup.

    Gain is shared across the twelve leads, preserving their relative voltages.
    Noise is scaled by each lead's RMS. Neither transformation edits the cache.
    """

    def __init__(self, config):
        self.gain = float(config.get('augmentation_gain', 0))
        self.noise = float(config.get('augmentation_noise', 0))
        self.mixup_alpha = float(config.get('mixup_alpha', 0))
        self.mixup_probability = float(config.get('mixup_probability', 1))
        if not all(math.isfinite(v) for v in (self.gain, self.noise, self.mixup_alpha, self.mixup_probability)):
            raise ValueError("Augmentation settings must be finite.")
        if not 0 <= self.gain < 1 or self.noise < 0 or self.mixup_alpha < 0 or not 0 <= self.mixup_probability <= 1:
            raise ValueError("Invalid augmentation settings.")

    def __call__(self, signals, labels):
        transformed = signals
        if self.gain:
            gain = 1 + (torch.rand((len(signals), 1, 1), device=signals.device) * 2 - 1) * self.gain
            transformed = transformed * gain
        if self.noise:
            rms = transformed.square().mean(dim=-1, keepdim=True).sqrt()
            transformed = transformed + torch.randn_like(transformed) * rms * self.noise
        if self.mixup_alpha and len(signals) > 1 and torch.rand((), device=signals.device) < self.mixup_probability:
            concentration = torch.tensor(self.mixup_alpha, device=signals.device)
            coefficient = torch.distributions.Beta(concentration, concentration).sample()
            order = torch.randperm(len(signals), device=signals.device)
            transformed = coefficient * transformed + (1 - coefficient) * transformed[order]
            return transformed, labels, labels[order], coefficient
        return transformed, labels, labels, 1.0
