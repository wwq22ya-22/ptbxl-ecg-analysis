"""Plot temporal model response beside a physical ECG lead."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.data.ptbxl import ECGRecord, LEAD_NAMES
from src.explainability import SaliencyResult


def plot_saliency_overlay(record: ECGRecord, result: SaliencyResult, *, lead: str = "II") -> Figure:
    if lead not in LEAD_NAMES:
        raise ValueError(f"Unknown ECG lead {lead!r}.")
    if record.sampling_rate_hz != result.sampling_rate_hz:
        raise ValueError("Signal and saliency sampling rates differ.")
    if record.signal.shape[0] != len(result.importance):
        raise ValueError("Signal and saliency lengths differ.")
    lead_index = LEAD_NAMES.index(lead)
    time = np.arange(record.signal.shape[0]) / record.sampling_rate_hz
    waveform = record.signal[:, lead_index]
    figure, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, height_ratios=[2, 1])
    spread = max(float(np.ptp(waveform)), 1e-3)
    lower = float(np.min(waveform)) - 0.1 * spread
    upper = float(np.max(waveform)) + 0.1 * spread
    axes[0].imshow(
        result.importance[None, :], aspect="auto", origin="lower", cmap="Oranges",
        extent=(0, len(time) / record.sampling_rate_hz, lower, upper),
        vmin=0, vmax=1, alpha=0.35,
    )
    axes[0].plot(time, waveform, color="#274c77", linewidth=0.8)
    axes[0].set_ylabel(f"Lead {lead} ({record.units[lead_index]})")
    axes[0].set_title(f"ECG and model response for {result.class_name} (probability {result.probability:.3f})")
    axes[0].grid(True, alpha=0.2)
    axes[1].fill_between(time, result.importance, color="#e76f51", alpha=0.65)
    axes[1].plot(time, result.importance, color="#c44536", linewidth=0.8)
    axes[1].set(xlabel="Time (s)", ylabel="Relative response", ylim=(0, 1.05))
    axes[1].grid(True, alpha=0.2)
    figure.tight_layout()
    return figure
