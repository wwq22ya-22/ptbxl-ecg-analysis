"""Inspect one real PTB-XL ECG and save waveform/signal-analysis figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ptbxl import LEAD_NAMES, find_ptbxl_root, load_ecg, load_ptbxl_metadata
from src.config import load_config, resolve_project_path
from src.signal_processing import (
    FilterConfig,
    analyze_heart_rate,
    calculate_hrv,
    detect_r_peaks,
    preprocess_ecg,
)
from src.visualization.ecg_plot import (
    plot_12_leads,
    plot_raw_filtered_with_rpeaks,
    plot_rr_intervals,
)


def _save_figure(figure, path: Path) -> None:
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"Figure: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecg-id", type=int, default=1)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "figures")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    fs = int(config["dataset"]["sampling_rate_hz"])
    root = find_ptbxl_root(
        args.dataset_root or resolve_project_path(config["dataset"]["root"], PROJECT_ROOT),
        sampling_rate_hz=fs,
    )
    metadata = load_ptbxl_metadata(root, sampling_rate_hz=fs)
    matches = metadata.loc[metadata["ecg_id"].eq(args.ecg_id)]
    if matches.empty:
        parser.error(f"ECG ID {args.ecg_id} was not found in {root / 'ptbxl_database.csv'}")
    filename_column = "filename_lr" if fs == 100 else "filename_hr"
    record = load_ecg(root, str(matches.iloc[0][filename_column]), sampling_rate_hz=fs)

    signal_config = config["signal_processing"]
    filter_config = FilterConfig.from_mapping(signal_config)
    filtered = preprocess_ecg(record.signal, fs, config=filter_config, normalization="none")
    lead_ii = LEAD_NAMES.index("II")
    r_peaks = detect_r_peaks(
        filtered[:, lead_ii],
        fs,
        minimum_rr_s=float(signal_config["rpeak_minimum_rr_s"]),
        integration_window_s=float(signal_config["rpeak_integration_window_s"]),
        t_wave_window_s=float(signal_config["rpeak_t_wave_window_s"]),
        t_wave_max_ratio=float(signal_config["rpeak_t_wave_max_ratio"]),
    )
    heart_rate = analyze_heart_rate(
        r_peaks,
        fs,
        min_valid_rr_s=float(signal_config["rr_min_valid_s"]),
        max_valid_rr_s=float(signal_config["rr_max_valid_s"]),
    )
    hrv = calculate_hrv(
        heart_rate.valid_rr_intervals_s,
        record_duration_s=record.signal.shape[0] / fs,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_figure(
        plot_12_leads(record.signal, fs, units=record.units, title=f"PTB-XL ECG {args.ecg_id}"),
        output_dir / f"ecg_{args.ecg_id}_12_leads.png",
    )
    _save_figure(
        plot_raw_filtered_with_rpeaks(
            record.signal,
            filtered,
            fs,
            lead="II",
            r_peak_indices=r_peaks,
            unit=record.units[lead_ii],
        ),
        output_dir / f"ecg_{args.ecg_id}_lead_ii_analysis.png",
    )
    _save_figure(
        plot_rr_intervals(heart_rate.valid_rr_intervals_s),
        output_dir / f"ecg_{args.ecg_id}_rr_intervals.png",
    )

    print(f"ECG ID: {args.ecg_id}; shape: {record.signal.shape}; sampling rate: {fs} Hz")
    print(f"Lead units: {record.units}")
    print(f"Detected R peaks: {len(r_peaks)}; valid RR intervals: {len(heart_rate.valid_rr_intervals_s)}")
    if heart_rate.mean_hr_bpm is None:
        print(f"Heart rate: unavailable ({heart_rate.message})")
    else:
        print(
            f"Heart rate (mean/min/max): {heart_rate.mean_hr_bpm:.1f}/"
            f"{heart_rate.min_hr_bpm:.1f}/{heart_rate.max_hr_bpm:.1f} bpm"
        )
    if heart_rate.message:
        print(heart_rate.message)
    print(
        f"HRV Mean RR/SDNN/RMSSD: {hrv.mean_rr_ms!s}/"
        f"{hrv.sdnn_ms!s}/{hrv.rmssd_ms!s} ms"
    )
    if hrv.message:
        print(hrv.message)
    if hrv.warning:
        print(hrv.warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
