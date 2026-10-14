"""Ribeiro-style dual-stream residual ECG network adapted to 100 Hz.

Architecture reference: Ribeiro et al., Nature Communications 11, 1760 (2020).
https://github.com/antonior92/automatic-ecg-diagnosis/blob/master/model.py
Adaptations: 1000 samples, kernel 5, strides (4,4,2,2), five output logits.
The author's (processed x, raw residual y) stream semantics are retained.

MIT License
Copyright (c) 2020 Antonio Horta Ribeiro
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def same_padding(length: int, kernel: int, stride: int) -> tuple[int, int]:
    output = (length + stride - 1) // stride
    total = max((output - 1) * stride + kernel - length, 0)
    return total // 2, total - total // 2


class SameConv1D(nn.Module):
    """TensorFlow SAME convention, including asymmetric strided padding."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, bias=False)

    def forward(self, x):
        return self.conv(F.pad(x, same_padding(x.shape[-1], self.kernel_size, self.stride)))


class SameMaxPool1D(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1: return x
        padded = F.pad(x, same_padding(x.shape[-1], self.stride, self.stride), value=float('-inf'))
        return F.max_pool1d(padded, self.stride, self.stride)


def batch_norm(channels):
    # Keras defaults: epsilon .001, running-stat retention .99.
    return nn.BatchNorm1d(channels, eps=.001, momentum=.01)


class RibeiroResidualUnit(nn.Module):
    """Process x but carry the pre-BN/ReLU sum as the next shortcut y."""
    def __init__(self, in_channels, out_channels, stride, dropout=.2, kernel_size=5):
        super().__init__()
        self.main_first = nn.Sequential(SameConv1D(in_channels, out_channels, kernel_size),
                                         batch_norm(out_channels), nn.ReLU(), nn.Dropout(dropout))
        self.main_second = SameConv1D(out_channels, out_channels, kernel_size, stride)
        self.skip_pool = SameMaxPool1D(stride)
        self.skip_projection = (nn.Conv1d(in_channels, out_channels, 1, bias=False)
                                if in_channels != out_channels else nn.Identity())
        self.post_addition = nn.Sequential(batch_norm(out_channels), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, inputs):
        x, y = inputs
        shortcut = self.skip_projection(self.skip_pool(y))
        raw_sum = self.main_second(self.main_first(x)) + shortcut
        return self.post_addition(raw_sum), raw_sum


class ECGRibeiroResNet100(nn.Module):
    """100-Hz teacher preserving four blocks and the author's Flatten head."""
    def __init__(self, *, num_leads=12, num_classes=5, dropout=.2):
        super().__init__()
        if not 0 <= dropout < 1: raise ValueError('dropout must be in [0, 1).')
        self.num_leads = num_leads
        self.stem = nn.Sequential(SameConv1D(num_leads, 64, 5), batch_norm(64), nn.ReLU())
        widths = [64, 128, 196, 256, 320]
        self.blocks = nn.ModuleList([RibeiroResidualUnit(a, b, stride, dropout)
            for a, b, stride in zip(widths[:-1], widths[1:], [4, 4, 2, 2])])
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 320, num_classes))
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None: nn.init.zeros_(module.bias)

    def forward(self, signals):
        if signals.ndim != 3 or signals.shape[1:] != (self.num_leads, 1000):
            raise ValueError(f'Expected ECG input [batch, {self.num_leads}, 1000].')
        x = self.stem(signals)
        streams = (x, x)
        for block in self.blocks: streams = block(streams)
        return self.classifier(streams[0])
