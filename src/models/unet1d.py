"""Lightweight U-shaped waveform classifier, not a pointwise segmentation model.

Inspired by Ronneberger et al. (2015), https://arxiv.org/abs/1505.04597.
Project adaptation: Conv1d, widths 12/24/48/96/192, four down/up stages,
linear interpolation plus 1x1 projection, avg+max pooling and five logits.
No waveform image conversion or dense segmentation supervision is used.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv1D(nn.Sequential):
    def __init__(self, inputs: int, outputs: int):
        super().__init__(nn.Conv1d(inputs, outputs, 5, padding=2, bias=False),
                         nn.BatchNorm1d(outputs), nn.ReLU(inplace=True),
                         nn.Conv1d(outputs, outputs, 5, padding=2, bias=False),
                         nn.BatchNorm1d(outputs), nn.ReLU(inplace=True))


class ECGUNetEncoder(nn.Module):
    def __init__(self, num_leads: int = 12):
        super().__init__()
        self.stem = DoubleConv1D(num_leads, 12)
        self.down = nn.ModuleList([nn.Sequential(nn.MaxPool1d(2), DoubleConv1D(a, b))
                                  for a, b in zip((12, 24, 48, 96), (24, 48, 96, 192))])

    def forward(self, x):
        features = [self.stem(x)]
        for layer in self.down: features.append(layer(features[-1]))
        return features


class UpsampleFuse1D(nn.Module):
    def __init__(self, inputs: int, skip_channels: int):
        super().__init__()
        self.project = nn.Conv1d(inputs, skip_channels, 1, bias=False)
        self.fuse = DoubleConv1D(skip_channels * 2, skip_channels)

    def forward(self, x, skip):
        # Exact target lengths retain alignment at 62 -> 125 -> 250 -> 500 -> 1000.
        up = self.project(F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False))
        return self.fuse(torch.cat([up, skip], dim=1))


class ECGUNet1D(nn.Module):
    def __init__(self, *, num_leads=12, num_classes=5, dropout=.2, decoder=True):
        super().__init__()
        if not 0 <= dropout < 1: raise ValueError('dropout must be in [0, 1).')
        self.num_leads = num_leads
        self.encoder = ECGUNetEncoder(num_leads)
        self.up = nn.ModuleList([UpsampleFuse1D(a, b) for a, b in
                               zip((192, 96, 48, 24), (96, 48, 24, 12))]) if decoder else nn.ModuleList()
        final_channels = 12 if decoder else 192
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * final_channels, num_classes))

    def forward(self, signals):
        if signals.ndim != 3 or signals.shape[1:] != (self.num_leads, 1000):
            raise ValueError(f'Expected ECG input [batch, {self.num_leads}, 1000].')
        encoded = self.encoder(signals)
        x = encoded[-1]
        for up, skip in zip(self.up, reversed(encoded[:-1])): x = up(x, skip)
        return self.classifier(torch.cat([x.mean(-1), x.amax(-1)], dim=1))


class ECGUNetEncoder1D(ECGUNet1D):
    """Same encoder without decoder, for a paired architectural ablation.

The classifier consumes bottleneck channels instead of decoded channels.
Parameter counts differ, so this is not a parameter-matched comparison.
"""
    def __init__(self, *, num_leads=12, num_classes=5, dropout=.2):
        super().__init__(num_leads=num_leads, num_classes=num_classes, dropout=dropout, decoder=False)
