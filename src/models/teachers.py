"""Larger waveform-only teachers; all retain the [B, 12, T] -> [B, 5] contract.

These are project implementations of SE residual blocks and a single
InceptionTime-style network, not pretrained weights or a paper-score replica.
Method references: github.com/hujie-frank/SENet and
github.com/hfawaz/InceptionTime/classifiers/inception.py.
"""
from __future__ import annotations

import torch
from torch import nn


class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, max(1, channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(1, channels // reduction), channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class SEResidualBlock1D(nn.Module):
    def __init__(self, inputs: int, outputs: int, stride: int = 1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(inputs, outputs, 7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(outputs), nn.ReLU(inplace=True),
            nn.Conv1d(outputs, outputs, 7, padding=3, bias=False),
            nn.BatchNorm1d(outputs), SqueezeExcitation1D(outputs),
        )
        self.skip = nn.Identity() if inputs == outputs and stride == 1 else nn.Sequential(
            nn.Conv1d(inputs, outputs, 1, stride=stride, bias=False), nn.BatchNorm1d(outputs),
        )

    def forward(self, x):
        return torch.relu(self.main(x) + self.skip(x))


class ECGSEResNet18(nn.Module):
    """Eight basic blocks with SE, widths 32/64/128/256, avg+max pooling."""
    def __init__(self, *, num_leads=12, num_classes=5, dropout=0.2):
        super().__init__()
        self.num_leads = num_leads
        self.stem = nn.Sequential(
            nn.Conv1d(num_leads, 32, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(3, stride=2, padding=1),
        )
        blocks = []
        inputs = 32
        for index, width in enumerate((32, 64, 128, 256)):
            blocks.extend([SEResidualBlock1D(inputs, width, 1 if index == 0 else 2),
                           SEResidualBlock1D(width, width)])
            inputs = width
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.num_leads:
            raise ValueError(f"Expected ECG input [batch, {self.num_leads}, time].")
        features = self.blocks(self.stem(x))
        return self.classifier(torch.cat([features.mean(-1), features.amax(-1)], dim=1))


class InceptionTimeModule1D(nn.Module):
    def __init__(self, inputs: int, width: int = 32):
        super().__init__()
        self.bottleneck = nn.Conv1d(inputs, 32, 1, bias=False)
        self.branches = nn.ModuleList([
            nn.Conv1d(32, width, kernel, padding="same", bias=False)
            for kernel in (40, 20, 10)
        ])
        self.pool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                                  nn.Conv1d(inputs, width, 1, bias=False))
        self.norm = nn.BatchNorm1d(width * 4)

    def forward(self, x):
        reduced = self.bottleneck(x)
        return torch.relu(self.norm(torch.cat([layer(reduced) for layer in self.branches]
                                             + [self.pool(x)], dim=1)))


class ECGInceptionTime1D(nn.Module):
    """Six modules, 32 filters/branch, residual every three; no downsampling.

    A single network with a multilabel logit head and project dropout. The
    original InceptionTime ensemble and its original training recipe are not
    claimed to be reproduced.
    """
    def __init__(self, *, num_leads=12, num_classes=5, dropout=0.2):
        super().__init__()
        self.num_leads = num_leads
        self.groups = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        inputs = num_leads
        for _ in range(2):
            self.groups.append(nn.Sequential(InceptionTimeModule1D(inputs),
                                              InceptionTimeModule1D(128),
                                              InceptionTimeModule1D(128)))
            self.shortcuts.append(nn.Sequential(nn.Conv1d(inputs, 128, 1, bias=False),
                                                nn.BatchNorm1d(128)))
            inputs = 128
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.num_leads:
            raise ValueError(f"Expected ECG input [batch, {self.num_leads}, time].")
        for group, shortcut in zip(self.groups, self.shortcuts):
            x = torch.relu(group(x) + shortcut(x))
        return self.classifier(x.mean(-1))
