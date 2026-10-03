"""Configurable ECG baseline, bandpass, notch, and normalization filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

Normalization = Literal["none", "zscore", "minmax", "train_global"]


@dataclass(frozen=True)
class FilterConfig:
    """Default signal-processing parameters for PTB-XL ECGs.

    The two median windows (200 and 600 ms by default) estimate slow baseline
    drift. A fourth-order 0.5-40 Hz Butterworth bandpass then retains the
    principal diagnostic/QRS bandwidth for the 100 Hz PTB-XL records.
    Normalization is per lead and per record; use ``none`` for plots whose
    amplitude axis must retain physical units.
    """

    baseline_windows_s: tuple[float, float] = (0.2, 0.6)
    lowcut_hz: float = 0.5
    highcut_hz: float = 40.0
    bandpass_order: int = 4
    notch_hz: float | None = None
    notch_quality_factor: float = 30.0
    normalization: Normalization = "zscore"
    training_mean: tuple[float, ...] = ()
    training_std: tuple[float, ...] = ()
    baseline_removal: bool = True

    @classmethod
    def from_mapping(cls, settings: Mapping[str, Any]) -> "FilterConfig":
        """Build filter settings from the ``signal_processing`` YAML section."""
        windows = settings.get("baseline_windows_s", (0.2, 0.6))
        if not isinstance(windows, (list, tuple)) or len(windows) != 2:
            raise ValueError("baseline_windows_s must contain two durations.")
        return cls(
            baseline_windows_s=(float(windows[0]), float(windows[1])),
            lowcut_hz=float(settings.get("lowcut_hz", settings.get("bandpass_lowcut_hz", 0.5))),
            highcut_hz=float(settings.get("highcut_hz", settings.get("bandpass_highcut_hz", 40.0))),
            bandpass_order=int(settings.get("bandpass_order", 4)),
            notch_hz=None if settings.get("notch_hz") is None else float(settings["notch_hz"]),
            notch_quality_factor=float(settings.get("notch_quality_factor", 30.0)),
            normalization=settings.get("normalization", "zscore"),
            training_mean=tuple(settings.get("training_mean", ())),
            training_std=tuple(settings.get("training_std", ())),
            baseline_removal=bool(settings.get("baseline_removal", True)),
        )


def _as_signal(signal: np.ndarray) -> tuple[np.ndarray, int | None]:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim not in (1, 2):
        raise ValueError(f"Expected a 1D lead or [samples, leads] signal; got shape {values.shape}.")
    if values.shape[0] < 1:
        raise ValueError("Signal must contain at least one sample.")
    if values.ndim == 2 and values.shape[1] < 1:
        raise ValueError("Signal must contain at least one lead.")
    if not np.isfinite(values).all():
        raise ValueError("Signal contains NaN or infinite values.")
    return values, (0 if values.ndim == 2 else None)


def _validate_fs(sampling_rate_hz: float) -> float:
    fs = float(sampling_rate_hz)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("sampling_rate_hz must be a positive finite number.")
    return fs


def remove_baseline_wander(
    signal: np.ndarray,
    sampling_rate_hz: float,
    *,
    windows_s: tuple[float, float] = (0.2, 0.6),
) -> np.ndarray:
    """Remove slow drift with two cascaded median baseline estimates.

    The windows approximate the common 200 ms / 600 ms median-filter method.
    Each window is converted to an odd number of samples and safely shortened
    for shorter records. The output has the same dimensions as the input.
    """
    values, axis = _as_signal(signal)
    fs = _validate_fs(sampling_rate_hz)
    if len(windows_s) != 2 or any(not np.isfinite(w) or w <= 0 for w in windows_s):
        raise ValueError("windows_s must contain two positive window durations in seconds.")
    if values.shape[0] < 3:
        raise ValueError("At least 3 samples are required for baseline removal.")

    baseline = values
    max_odd = values.shape[0] if values.shape[0] % 2 else values.shape[0] - 1
    for window_s in windows_s:
        size = max(3, int(round(window_s * fs)))
        if size % 2 == 0:
            size += 1
        size = min(size, max_odd)
        if size < 3:
            raise ValueError("Record is too short for the requested baseline windows.")
        filter_size: int | tuple[int, int] = size if axis is None else (size, 1)
        baseline = median_filter(baseline, size=filter_size, mode="reflect")
    return (values - baseline).astype(np.float32)


def bandpass_filter(
    signal: np.ndarray,
    sampling_rate_hz: float,
    *,
    lowcut_hz: float = 0.5,
    highcut_hz: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass using second-order sections."""
    values, axis = _as_signal(signal)
    fs = _validate_fs(sampling_rate_hz)
    nyquist = fs / 2
    if not (0 < lowcut_hz < highcut_hz < nyquist):
        raise ValueError(
            f"Require 0 < lowcut_hz < highcut_hz < Nyquist ({nyquist:g} Hz); "
            f"got {lowcut_hz:g}-{highcut_hz:g} Hz at fs={fs:g} Hz."
        )
    if not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer.")
    sos = butter(order, (lowcut_hz, highcut_hz), btype="bandpass", fs=fs, output="sos")
    try:
        filtered = sosfiltfilt(sos, values, axis=0 if axis is None else axis)
    except ValueError as exc:
        raise ValueError(f"Signal is too short for zero-phase bandpass filtering: {exc}") from exc
    return filtered.astype(np.float32)


def notch_filter(
    signal: np.ndarray,
    sampling_rate_hz: float,
    frequency_hz: float,
    *,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Suppress 50/60 Hz mains interference if it lies below Nyquist.

    A 50 Hz notch is invalid for a 100 Hz signal because 50 Hz is exactly the
    Nyquist frequency. For PTB-XL 100 Hz records, omit the notch; for 50/60 Hz
    suppression, process the 500 Hz records before any downsampling.
    """
    values, axis = _as_signal(signal)
    fs = _validate_fs(sampling_rate_hz)
    if not np.isfinite(frequency_hz) or frequency_hz <= 0 or frequency_hz >= fs / 2:
        raise ValueError(
            f"Notch frequency must be between 0 and Nyquist ({fs / 2:g} Hz), exclusive; "
            f"{frequency_hz:g} Hz is invalid at fs={fs:g} Hz."
        )
    if not np.isfinite(quality_factor) or quality_factor <= 0:
        raise ValueError("quality_factor must be positive and finite.")
    b, a = iirnotch(frequency_hz, quality_factor, fs=fs)
    try:
        filtered = filtfilt(b, a, values, axis=0 if axis is None else axis)
    except ValueError as exc:
        raise ValueError(f"Signal is too short for zero-phase notch filtering: {exc}") from exc
    return filtered.astype(np.float32)


def normalize_signal(signal: np.ndarray, method: Normalization = "zscore") -> np.ndarray:
    """Normalize each lead independently along time.

    ``zscore`` is the default. Constant leads become zeros rather than NaNs.
    ``minmax`` scales each lead to [0, 1]; ``none`` preserves amplitude.
    """
    values, axis = _as_signal(signal)
    if method == "none":
        result = values
    elif method == "zscore":
        mean = np.mean(values, axis=axis, keepdims=True)
        std = np.std(values, axis=axis, keepdims=True)
        result = (values - mean) / np.maximum(std, 1e-8)
    elif method == "minmax":
        minimum = np.min(values, axis=axis, keepdims=True)
        maximum = np.max(values, axis=axis, keepdims=True)
        result = (values - minimum) / np.maximum(maximum - minimum, 1e-8)
    else:
        raise ValueError("method must be one of: 'none', 'zscore', 'minmax'.")
    return result.astype(np.float32)


def preprocess_ecg(
    signal: np.ndarray,
    sampling_rate_hz: float,
    *,
    config: FilterConfig | None = None,
    normalization: Normalization | None = None,
) -> np.ndarray:
    """Run baseline removal, bandpass, optional notch, and normalization."""
    cfg = config or FilterConfig()
    filtered = (remove_baseline_wander(signal, sampling_rate_hz, windows_s=cfg.baseline_windows_s)
                if cfg.baseline_removal else np.asarray(signal, dtype=np.float32))
    filtered = bandpass_filter(
        filtered,
        sampling_rate_hz,
        lowcut_hz=cfg.lowcut_hz,
        highcut_hz=cfg.highcut_hz,
        order=cfg.bandpass_order,
    )
    if cfg.notch_hz is not None:
        filtered = notch_filter(
            filtered,
            sampling_rate_hz,
            cfg.notch_hz,
            quality_factor=cfg.notch_quality_factor,
        )
    method = normalization or cfg.normalization
    if method == "train_global":
        mean = np.asarray(cfg.training_mean, dtype=np.float64)
        std = np.asarray(cfg.training_std, dtype=np.float64)
        if mean.shape != (filtered.shape[1],) or std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("train_global requires finite training-only lead means and positive standard deviations.")
        return ((filtered - mean) / std).astype(np.float32)
    return normalize_signal(filtered, method)
