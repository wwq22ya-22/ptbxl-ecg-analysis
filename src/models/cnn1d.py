"""Compact 1D CNN baseline for five independent PTB-XL superclass logits."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    """Convolution, batch normalization, ReLU, and temporal pooling."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *, stride: int = 1):
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )


class ECGCNN1D(nn.Module):
    """Map `[B, 12, T]` ECG tensors to five multi-label logits."""

    def __init__(self, *, num_leads: int = 12, num_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        if num_leads < 1 or num_classes < 1:
            raise ValueError("num_leads and num_classes must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.features = nn.Sequential(
            ConvBlock(num_leads, 32, 15, stride=2),
            ConvBlock(32, 64, 7),
            ConvBlock(64, 128, 5),
            ConvBlock(128, 128, 3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3 or signal.shape[1] != self.features[0][0].in_channels:
            raise ValueError(f"Expected ECG input [batch, {self.features[0][0].in_channels}, time].")
        features = self.features(signal)
        return self.classifier(self.pool(features))
