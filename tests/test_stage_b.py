from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import yaml

from src.signal_processing.filters import (
    FilterConfig,
    bandpass_filter,
    normalize_signal,
    notch_filter,
    preprocess_ecg,
    remove_baseline_wander,
)
from src.signal_processing.hrv import HRV_SHORT_RECORD_WARNING, calculate_hrv
from src.signal_processing.rpeak import analyze_heart_rate, detect_r_peaks, rr_intervals_from_peaks
from src.visualization.ecg_plot import plot_raw_filtered_with_rpeaks, plot_rr_intervals


def qrs_train(fs: int = 100, duration_s: int = 10) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(fs * duration_s) / fs
    signal = 0.15 * np.sin(2 * np.pi * 0.25 * time)
    peaks = np.arange(fs, fs * duration_s, fs)
    for peak in peaks:
        signal += np.exp(-0.5 * ((np.arange(signal.size) - peak) / (0.025 * fs)) ** 2)
    signal += np.random.default_rng(4).normal(0, 0.005, signal.size)
    return signal.astype(np.float32), peaks


def test_baseline_removal_suppresses_slow_drift_and_preserves_shape():
    fs = 100
    time = np.arange(1000) / fs
    drift = 0.4 * np.sin(2 * np.pi * 0.2 * time)
    clean = remove_baseline_wander(drift, fs)
    assert clean.shape == drift.shape
    assert np.std(clean) < np.std(drift) * 0.25


def test_bandpass_retains_ecg_band_and_removes_out_of_band_signal():
    fs = 500
    time = np.arange(5000) / fs
    mixed = np.sin(2 * np.pi * 10 * time) + np.sin(2 * np.pi * 60 * time)
    filtered = bandpass_filter(mixed, fs, lowcut_hz=0.5, highcut_hz=40)
    assert np.std(filtered) == pytest.approx(1 / np.sqrt(2), rel=0.08)
    with pytest.raises(ValueError, match="Nyquist"):
        bandpass_filter(mixed, 100, lowcut_hz=0.5, highcut_hz=50)


def test_notch_respects_nyquist_and_suppresses_50_hz_at_500_hz():
    fs = 500
    time = np.arange(5000) / fs
    mixed = np.sin(2 * np.pi * 10 * time) + np.sin(2 * np.pi * 50 * time)
    filtered = notch_filter(mixed, fs, 50)
    assert np.std(filtered) < np.std(mixed) * 0.8
    with pytest.raises(ValueError, match="Nyquist"):
        notch_filter(mixed[:1000], 100, 50)


def test_normalization_and_preprocessing_are_finite_and_configurable():
    signal = np.column_stack([np.arange(1000, dtype=float), np.ones(1000)])
    normalized = normalize_signal(signal)
    assert normalized.shape == signal.shape
    assert np.mean(normalized[:, 0]) == pytest.approx(0, abs=1e-6)
    assert np.std(normalized[:, 0]) == pytest.approx(1, abs=1e-5)
    assert np.all(normalized[:, 1] == 0)

    ecg, _ = qrs_train()
    cfg = FilterConfig(notch_hz=None, normalization="zscore")
    output = preprocess_ecg(np.column_stack([ecg, ecg]), 100, config=cfg)
    assert output.shape == (1000, 2)
    assert np.isfinite(output).all()
    assert np.mean(output[:, 0]) == pytest.approx(0, abs=1e-5)
    with pytest.raises(ValueError, match="Nyquist"):
        preprocess_ecg(ecg, 100, config=FilterConfig(notch_hz=50))

    with (Path(__file__).resolve().parents[1] / "configs" / "default.yaml").open(encoding="utf-8") as config_file:
        configured = yaml.safe_load(config_file)["signal_processing"]
    loaded_config = FilterConfig.from_mapping(configured)
    assert loaded_config.lowcut_hz == pytest.approx(0.5)
    assert loaded_config.highcut_hz == pytest.approx(40.0)
    assert loaded_config.normalization == "zscore"


def test_explainable_rpeak_detector_finds_synthetic_qrs_locations():
    signal, expected = qrs_train()
    detected = detect_r_peaks(signal, 100)
    assert detected.size == expected.size
    assert np.max(np.abs(detected - expected)) <= 3


def test_rpeak_detector_rejects_lower_amplitude_t_waves():
    signal, expected = qrs_train()
    sample_index = np.arange(signal.size)
    for peak in expected:
        t_peak = peak + 30
        signal += 0.5 * np.exp(-0.5 * ((sample_index - t_peak) / 4) ** 2)
    detected = detect_r_peaks(signal, 100)
    assert detected.size == expected.size
    assert np.max(np.abs(detected - expected)) <= 3


def test_rpeak_detector_handles_short_flat_and_invalid_inputs():
    assert detect_r_peaks(np.zeros(5), 100).size == 0
    assert detect_r_peaks(np.zeros(1000), 100).size == 0
    with pytest.raises(ValueError, match="one lead"):
        detect_r_peaks(np.zeros((100, 2)), 100)


def test_rr_and_heart_rate_exclude_implausible_intervals_with_notice():
    peaks = np.asarray([0, 100, 200, 220, 320], dtype=np.int64)
    np.testing.assert_allclose(rr_intervals_from_peaks(peaks, 100), [1, 1, 0.2, 1])
    result = analyze_heart_rate(peaks, 100)
    assert result.mean_hr_bpm == pytest.approx(60)
    assert result.min_hr_bpm == pytest.approx(60)
    assert result.max_hr_bpm == pytest.approx(60)
    assert result.excluded_rr_count == 1
    assert "Excluded 1" in result.message

    insufficient = analyze_heart_rate(np.asarray([10]), 100)
    assert insufficient.mean_hr_bpm is None
    assert "At least two R peaks" in insufficient.message
    with pytest.raises(ValueError, match="strictly increasing"):
        rr_intervals_from_peaks(np.asarray([2, 2, 1]), 100)


def test_hrv_metrics_and_short_record_warning():
    result = calculate_hrv(np.asarray([0.8, 1.0, 1.2]), record_duration_s=10)
    assert result.mean_rr_ms == pytest.approx(1000)
    assert result.sdnn_ms == pytest.approx(200)
    assert result.rmssd_ms == pytest.approx(200)
    assert result.warning == HRV_SHORT_RECORD_WARNING

    insufficient = calculate_hrv(np.asarray([1.0]), record_duration_s=10)
    assert insufficient.mean_rr_ms == pytest.approx(1000)
    assert insufficient.sdnn_ms is None
    assert insufficient.rmssd_ms is None
    assert insufficient.message is not None
    assert calculate_hrv(np.asarray([0.9, 1.0]), record_duration_s=300).warning is None


def test_signal_analysis_plots_markers_and_rr_timeline():
    raw, peaks = qrs_train()
    filtered = preprocess_ecg(raw, 100, normalization="none")
    figure = plot_raw_filtered_with_rpeaks(raw, filtered, 100, r_peak_indices=peaks)
    assert len(figure.axes) == 2
    assert figure.axes[0].get_xlabel() == ""
    assert figure.axes[1].get_xlabel() == "Time (s)"
    assert len(figure.axes[1].collections[0].get_offsets()) == len(peaks)
    figure.clf()

    rr_figure = plot_rr_intervals([1.0, 0.9, 1.1])
    assert rr_figure.axes[0].get_ylabel() == "RR interval (s)"
    rr_figure.clf()


def test_end_to_end_synthetic_signal_analysis():
    raw, expected = qrs_train()
    filtered = preprocess_ecg(raw, 100, normalization="none")
    detected = detect_r_peaks(filtered, 100)
    assert detected.size == expected.size

    heart_rate = analyze_heart_rate(detected, 100)
    assert heart_rate.mean_hr_bpm == pytest.approx(60, abs=1.0)
    hrv = calculate_hrv(
        heart_rate.valid_rr_intervals_s,
        record_duration_s=raw.size / 100,
    )
    assert hrv.mean_rr_ms == pytest.approx(1000, abs=20)
    assert hrv.warning == HRV_SHORT_RECORD_WARNING
