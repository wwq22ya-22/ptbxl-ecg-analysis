"""Basic time-domain HRV metrics with short-record limitations made explicit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HRV_SHORT_RECORD_WARNING = (
    "当前 ECG 记录长度有限，该结果仅用于算法演示，不应作为临床长期 HRV 评价。"
)


@dataclass(frozen=True)
class HRVMetrics:
    """Time-domain HRV values, sample count, and interpretation warning."""

    mean_rr_ms: float | None
    sdnn_ms: float | None
    rmssd_ms: float | None
    n_rr_intervals: int
    warning: str | None
    message: str | None = None


def calculate_hrv(
    rr_intervals_s: np.ndarray,
    *,
    record_duration_s: float | None = None,
    minimum_intervals: int = 2,
) -> HRVMetrics:
    """Calculate Mean RR, sample SDNN, and RMSSD from valid RR intervals.

    SDNN and RMSSD are left undefined unless there are at least
    ``minimum_intervals``. Records shorter than 5 minutes, including PTB-XL's
    10-second excerpts, carry an explicit educational-use warning.
    """
    rr = np.asarray(rr_intervals_s, dtype=np.float64)
    if rr.ndim != 1:
        raise ValueError("rr_intervals_s must be one-dimensional.")
    if not np.isfinite(rr).all() or np.any(rr <= 0):
        raise ValueError("RR intervals must be positive finite durations in seconds.")
    if not isinstance(minimum_intervals, int) or minimum_intervals < 2:
        raise ValueError("minimum_intervals must be an integer of at least 2.")
    if record_duration_s is not None and (
        not np.isfinite(record_duration_s) or record_duration_s < 0
    ):
        raise ValueError("record_duration_s must be a non-negative finite duration.")

    count = int(rr.size)
    short_record = record_duration_s is None or record_duration_s < 300.0
    warning = HRV_SHORT_RECORD_WARNING if short_record else None
    if count == 0:
        return HRVMetrics(None, None, None, 0, warning, "No valid RR intervals are available.")
    mean_rr_ms = float(np.mean(rr) * 1000.0)
    if count < minimum_intervals:
        return HRVMetrics(
            mean_rr_ms=mean_rr_ms,
            sdnn_ms=None,
            rmssd_ms=None,
            n_rr_intervals=count,
            warning=warning,
            message=f"At least {minimum_intervals} valid RR intervals are required for SDNN and RMSSD.",
        )

    differences_ms = np.diff(rr) * 1000.0
    return HRVMetrics(
        mean_rr_ms=mean_rr_ms,
        sdnn_ms=float(np.std(rr * 1000.0, ddof=1)),
        rmssd_ms=float(np.sqrt(np.mean(np.square(differences_ms)))),
        n_rr_intervals=count,
        warning=warning,
    )
