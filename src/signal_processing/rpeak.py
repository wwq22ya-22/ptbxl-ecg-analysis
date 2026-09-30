"""Explainable R-peak detection and RR/heart-rate summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from .filters import bandpass_filter


def detect_r_peaks(
    ecg_signal: np.ndarray,
    sampling_rate_hz: float,
    *,
    minimum_rr_s: float = 0.25,
    integration_window_s: float = 0.12,
    threshold_mad: float = 3.0,
    t_wave_window_s: float = 0.36,
    t_wave_max_ratio: float = 0.75,
) -> np.ndarray:
    """Detect R peaks with a Pan-Tompkins-inspired derivative-energy pipeline.

    The lead is bandpass-filtered to 5-20 Hz, differentiated, squared, and
    moving-window integrated to emphasize QRS complexes. Adaptive thresholding
    and a refractory distance find candidates. A low-amplitude candidate within
    the following 360 ms is treated as a possible T wave and rejected. Each
    remaining candidate is refined to the largest absolute filtered sample
    in a short neighborhood. The output contains
    sample indices, not times. Empty output means no reliable candidates.
    """
    values = np.asarray(ecg_signal, dtype=np.float64)
    fs = float(sampling_rate_hz)
    if values.ndim != 1:
        raise ValueError(f"R-peak detection expects one lead as a 1D array; got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("ECG signal contains NaN or infinite values.")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("sampling_rate_hz must be positive and finite.")
    if fs <= 10:
        raise ValueError("R-peak detection requires a sampling rate above 10 Hz.")
    if not np.isfinite(minimum_rr_s) or minimum_rr_s <= 0:
        raise ValueError("minimum_rr_s must be positive and finite.")
    if not np.isfinite(integration_window_s) or integration_window_s <= 0:
        raise ValueError("integration_window_s must be positive and finite.")
    if not np.isfinite(threshold_mad) or threshold_mad < 0:
        raise ValueError("threshold_mad must be non-negative and finite.")
    if not np.isfinite(t_wave_window_s) or t_wave_window_s <= 0:
        raise ValueError("t_wave_window_s must be positive and finite.")
    if not np.isfinite(t_wave_max_ratio) or not 0 < t_wave_max_ratio < 1:
        raise ValueError("t_wave_max_ratio must be between 0 and 1.")
    if values.size < 8:
        return np.asarray([], dtype=np.int64)

    qrs_band = bandpass_filter(values, fs, lowcut_hz=5.0, highcut_hz=min(20.0, fs / 2 - 1e-6), order=2)
    derivative = np.gradient(qrs_band) * fs
    energy = derivative * derivative
    integration_samples = max(1, int(round(integration_window_s * fs)))
    envelope = uniform_filter1d(energy, size=integration_samples, mode="nearest")
    baseline = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - baseline)))
    signal_level = float(np.percentile(envelope, 99.5))
    dynamic_floor = max(threshold_mad * mad, 0.05 * max(signal_level - baseline, 0.0))
    height = baseline + dynamic_floor
    if not np.isfinite(height) or signal_level <= baseline or dynamic_floor <= 0:
        return np.asarray([], dtype=np.int64)

    refractory_samples = max(1, int(round(minimum_rr_s * fs)))
    candidates, _ = find_peaks(envelope, height=height, distance=refractory_samples)
    if candidates.size == 0:
        return np.asarray([], dtype=np.int64)

    refine_radius = max(1, int(round(0.08 * fs)))
    t_wave_samples = max(1, int(round(t_wave_window_s * fs)))
    refined: list[int] = []
    for candidate in candidates:
        start = max(0, int(candidate) - refine_radius)
        stop = min(values.size, int(candidate) + refine_radius + 1)
        local = start + int(np.argmax(np.abs(qrs_band[start:stop])))
        if refined and local - refined[-1] < t_wave_samples:
            if abs(qrs_band[local]) < t_wave_max_ratio * abs(qrs_band[refined[-1]]):
                continue
        if not refined or local - refined[-1] >= refractory_samples:
            refined.append(local)
        elif abs(qrs_band[local]) > abs(qrs_band[refined[-1]]):
            refined[-1] = local
    return np.asarray(refined, dtype=np.int64)


def rr_intervals_from_peaks(r_peak_indices: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    """Convert strictly increasing R-peak sample indices into seconds."""
    peaks = np.asarray(r_peak_indices)
    fs = float(sampling_rate_hz)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("sampling_rate_hz must be positive and finite.")
    if peaks.ndim != 1:
        raise ValueError("r_peak_indices must be a one-dimensional array.")
    if not np.issubdtype(peaks.dtype, np.integer):
        if not np.isfinite(peaks).all() or not np.equal(peaks, np.floor(peaks)).all():
            raise ValueError("R-peak indices must be integers.")
    peaks = peaks.astype(np.int64)
    if peaks.size < 2:
        return np.asarray([], dtype=np.float64)
    differences = np.diff(peaks)
    if np.any(differences <= 0):
        raise ValueError("R-peak indices must be strictly increasing.")
    return differences.astype(np.float64) / fs


@dataclass(frozen=True)
class HeartRateResult:
    """RR intervals and basic heart-rate statistics from detected peaks."""

    rr_intervals_s: np.ndarray
    valid_rr_intervals_s: np.ndarray
    mean_hr_bpm: float | None
    min_hr_bpm: float | None
    max_hr_bpm: float | None
    excluded_rr_count: int
    message: str | None = None


def analyze_heart_rate(
    r_peak_indices: np.ndarray,
    sampling_rate_hz: float,
    *,
    min_valid_rr_s: float = 0.3,
    max_valid_rr_s: float = 2.0,
) -> HeartRateResult:
    """Calculate HR using 60/RR and exclude only implausible RR durations.

    The 0.3-2.0 s default is a broad signal-quality guardrail (30-200 bpm),
    not a diagnostic range. The result retains raw RR intervals and reports
    how many values were excluded instead of silently hiding detections.
    """
    if not (np.isfinite(min_valid_rr_s) and np.isfinite(max_valid_rr_s)):
        raise ValueError("RR limits must be finite.")
    if min_valid_rr_s <= 0 or min_valid_rr_s >= max_valid_rr_s:
        raise ValueError("RR limits must satisfy 0 < min_valid_rr_s < max_valid_rr_s.")
    rr = rr_intervals_from_peaks(r_peak_indices, sampling_rate_hz)
    valid = rr[(rr >= min_valid_rr_s) & (rr <= max_valid_rr_s)]
    excluded = int(rr.size - valid.size)
    if valid.size == 0:
        message = "At least two R peaks with plausible RR intervals are required to calculate heart rate."
        return HeartRateResult(rr, valid, None, None, None, excluded, message)
    rates = 60.0 / valid
    message = f"Excluded {excluded} implausible RR interval(s)." if excluded else None
    return HeartRateResult(
        rr_intervals_s=rr,
        valid_rr_intervals_s=valid,
        mean_hr_bpm=float(np.mean(rates)),
        min_hr_bpm=float(np.min(rates)),
        max_hr_bpm=float(np.max(rates)),
        excluded_rr_count=excluded,
        message=message,
    )
