"""Alternative architectures for controlled PTB-XL experiments."""
import torch
from torch import nn


class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, width=24):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, width, 1, bias=False)
        self.branches = nn.ModuleList([
            nn.Conv1d(width, width, k, padding=k // 2, bias=False) for k in (9, 19, 39)
        ])
        self.pool_branch = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(in_channels, width, 1, bias=False))
        self.norm = nn.BatchNorm1d(width * 4)
        self.skip = nn.Identity() if in_channels == width * 4 else nn.Conv1d(in_channels, width * 4, 1, bias=False)

    def forward(self, signal):
        reduced = self.bottleneck(signal)
        combined = torch.cat([branch(reduced) for branch in self.branches] + [self.pool_branch(signal)], dim=1)
        return torch.relu(self.norm(combined) + self.skip(signal))


class ECGInception1D(nn.Module):
    """Compact Inception-inspired network; not the published InceptionTime replica."""
    def __init__(self, *, num_leads=12, num_classes=5, dropout=0.2):
        super().__init__()
        self.num_leads = num_leads
        self.features = nn.Sequential(
            nn.Conv1d(num_leads, 48, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(48), nn.ReLU(),
            InceptionBlock1D(48), InceptionBlock1D(96), nn.AvgPool1d(2),
            InceptionBlock1D(96), InceptionBlock1D(96), nn.AvgPool1d(2),
            InceptionBlock1D(96), InceptionBlock1D(96),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(192, num_classes))

    def forward(self, signal):
        if signal.ndim != 3 or signal.shape[1] != self.num_leads:
            raise ValueError(f"Expected ECG input [batch, {self.num_leads}, time].")
        features = self.features(signal)
        return self.classifier(torch.cat([features.mean(dim=-1), features.amax(dim=-1)], dim=1))


class ECGCNNBiGRU(nn.Module):
    """Downsampled convolutional morphology features followed by bidirectional GRU."""
    def __init__(self, *, num_leads=12, num_classes=5, dropout=0.2):
        super().__init__()
        self.num_leads = num_leads
        layers = []
        channels = num_leads
        for width, kernel in ((32, 15), (64, 7), (96, 5)):
            layers.extend([nn.Conv1d(channels, width, kernel, stride=2, padding=kernel//2, bias=False), nn.BatchNorm1d(width), nn.ReLU()])
            channels = width
        self.features = nn.Sequential(*layers)
        self.gru = nn.GRU(96, 64, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, signal):
        if signal.ndim != 3 or signal.shape[1] != self.num_leads:
            raise ValueError(f"Expected ECG input [batch, {self.num_leads}, time].")
        features, _ = self.gru(self.features(signal).transpose(1, 2))
        return self.classifier(torch.cat([features.mean(dim=1), features.amax(dim=1)], dim=1))
