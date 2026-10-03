"""Compact residual 1D CNN for twelve-lead ECG classification experiments."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    """Two-convolution residual block with projection when shape changes."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(signal) + self.skip(signal))


class ECGResNet1D(nn.Module):
    """Residual alternative with the same `[B, 12, T]` to `[B, 5]` contract."""

    def __init__(self, *, num_leads: int = 12, num_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        if num_leads < 1 or num_classes < 1:
            raise ValueError("num_leads and num_classes must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.num_leads = num_leads
        self.stem = nn.Sequential(
            nn.Conv1d(num_leads, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(32, 32),
            ResidualBlock1D(32, 64, stride=2),
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3 or signal.shape[1] != self.num_leads:
            raise ValueError(f"Expected ECG input [batch, {self.num_leads}, time].")
        return self.classifier(self.pool(self.blocks(self.stem(signal))))


class ECGResNetAvgMax1D(ECGResNet1D):
    """Same residual feature extractor with both mean and peak pooling."""

    def __init__(self, *, num_leads=12, num_classes=5, dropout=0.2):
        super().__init__(num_leads=num_leads, num_classes=num_classes, dropout=dropout)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, signal):
        if signal.ndim != 3 or signal.shape[1] != self.num_leads:
            raise ValueError(f"Expected ECG input [batch, {self.num_leads}, time].")
        features = self.blocks(self.stem(signal))
        return self.classifier(torch.cat([features.mean(dim=-1), features.amax(dim=-1)], dim=1))
