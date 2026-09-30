"""Plot single-lead and standard twelve-lead ECG waveforms."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.data.ptbxl import LEAD_NAMES


def _validate_signal(signal: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    values = np.asarray(signal)
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")
    if values.ndim != 2 or values.shape[1] != len(LEAD_NAMES):
        raise ValueError(f"Expected signal shape [samples, 12], got {values.shape}.")
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("Signal must contain finite samples.")
    return values


def plot_single_lead(
    signal: np.ndarray,
    sampling_rate_hz: float,
    lead: str = "II",
    *,
    unit: str = "mV",
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[Figure, plt.Axes]:
    """Plot one named lead from a samples-by-12-leads ECG."""
    values = _validate_signal(signal, sampling_rate_hz)
    matches = [index for index, name in enumerate(LEAD_NAMES) if name.casefold() == lead.casefold()]
    if not matches:
        raise ValueError(f"Unknown lead {lead!r}; expected one of {', '.join(LEAD_NAMES)}.")
    if ax is None:
        figure, ax = plt.subplots(figsize=(10, 3))
    else:
        figure = ax.figure
    time = np.arange(values.shape[0], dtype=np.float64) / sampling_rate_hz
    ax.plot(time, values[:, matches[0]], linewidth=0.8)
    ax.set_title(title or f"Lead {lead}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Amplitude ({unit})")
    ax.grid(True, alpha=0.25)
    figure.tight_layout()
    return figure, ax


def plot_12_leads(
    signal: np.ndarray,
    sampling_rate_hz: float,
    *,
    units: Sequence[str] | str = "mV",
    title: str = "12-lead ECG",
    figsize: tuple[float, float] = (12, 14),
) -> Figure:
    """Plot all standard ECG leads using a shared seconds-based time axis."""
    values = _validate_signal(signal, sampling_rate_hz)
    if isinstance(units, str):
        unit_names = (units,) * len(LEAD_NAMES)
    else:
        unit_names = tuple(units)
        if len(unit_names) != len(LEAD_NAMES):
            raise ValueError("units must be a string or contain one unit per lead.")
    time = np.arange(values.shape[0], dtype=np.float64) / sampling_rate_hz
    figure, axes = plt.subplots(6, 2, figsize=figsize, sharex=True)
    for index, ax in enumerate(axes.flat):
        ax.plot(time, values[:, index], linewidth=0.7)
        ax.set_title(LEAD_NAMES[index])
        ax.set_ylabel(f"({unit_names[index]})")
        ax.grid(True, alpha=0.22)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def _extract_plot_lead(signal: np.ndarray, lead: str) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim == 1:
        lead_values = values
    elif values.ndim == 2 and values.shape[1] == len(LEAD_NAMES):
        matches = [i for i, name in enumerate(LEAD_NAMES) if name.casefold() == lead.casefold()]
        if not matches:
            raise ValueError(f"Unknown lead {lead!r}; expected one of {', '.join(LEAD_NAMES)}.")
        lead_values = values[:, matches[0]]
    else:
        raise ValueError(f"Expected one lead or [samples, 12] ECG; got shape {values.shape}.")
    if lead_values.size == 0 or not np.isfinite(lead_values).all():
        raise ValueError("ECG lead must contain finite samples.")
    return lead_values


def plot_raw_filtered_with_rpeaks(
    raw_signal: np.ndarray,
    filtered_signal: np.ndarray,
    sampling_rate_hz: float,
    *,
    lead: str = "II",
    r_peak_indices: Sequence[int] | None = None,
    unit: str = "mV",
) -> Figure:
    """Plot raw and filtered ECG on a shared seconds axis and mark detected R peaks."""
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")
    raw = _extract_plot_lead(raw_signal, lead)
    filtered = _extract_plot_lead(filtered_signal, lead)
    if raw.size != filtered.size:
        raise ValueError("Raw and filtered signals must have the same sample count.")
    time = np.arange(raw.size, dtype=np.float64) / sampling_rate_hz
    figure, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(time, raw, linewidth=0.8, color="#4263eb")
    axes[0].set_title(f"Raw ECG — Lead {lead}")
    axes[0].set_ylabel(f"Amplitude ({unit})")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(time, filtered, linewidth=0.8, color="#138a72", label="Filtered ECG")
    if r_peak_indices is not None:
        raw_peaks = np.asarray(r_peak_indices)
        if raw_peaks.ndim != 1:
            raise ValueError("R-peak indices must be a 1D array.")
        if not np.issubdtype(raw_peaks.dtype, np.integer):
            if not np.isfinite(raw_peaks).all() or not np.equal(raw_peaks, np.floor(raw_peaks)).all():
                raise ValueError("R-peak indices must be integers.")
        peaks = raw_peaks.astype(np.int64)
        if np.any(peaks < 0) or np.any(peaks >= filtered.size):
            raise ValueError("R-peak indices must be a 1D array within the signal bounds.")
        if peaks.size:
            axes[1].scatter(
                time[peaks], filtered[peaks], color="#d9485f", marker="x", s=35,
                label="Detected R peaks", zorder=3,
            )
            axes[1].legend(loc="upper right")
    axes[1].set_title("Filtered ECG and R peaks")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(f"Amplitude ({unit})")
    axes[1].grid(True, alpha=0.25)
    figure.tight_layout()
    return figure


def plot_rr_intervals(rr_intervals_s: Sequence[float]) -> Figure:
    """Plot valid RR intervals against the ending time of each beat interval."""
    rr = np.asarray(rr_intervals_s, dtype=np.float64)
    if rr.ndim != 1 or not np.isfinite(rr).all() or np.any(rr <= 0):
        raise ValueError("RR intervals must be a 1D array of positive finite seconds.")
    figure, axis = plt.subplots(figsize=(8, 3.5))
    if rr.size:
        interval_ends = np.cumsum(rr)
        axis.plot(interval_ends, rr, marker="o", linewidth=1.0)
    axis.set_title("RR intervals")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("RR interval (s)")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    return figure
