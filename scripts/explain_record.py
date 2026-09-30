"""Save a temporal model-response figure for one PTB-XL record."""

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

from src.data.ptbxl import SUPERCLASSES, find_ptbxl_root, load_ecg, load_ptbxl_metadata
from src.explainability import compute_input_saliency
from src.inference import load_predictor
from src.visualization.saliency_plot import plot_saliency_overlay
from src.config import load_config, resolve_project_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecg-id", type=int, default=9)
    parser.add_argument("--class-name", choices=SUPERCLASSES, default="NORM")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "outputs/checkpoints/baseline_best.pt")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    predictor = load_predictor(args.checkpoint)
    config = load_config(PROJECT_ROOT / "configs/default.yaml")
    fs = predictor.sampling_rate_hz
    root = find_ptbxl_root(
        args.dataset_root or resolve_project_path(config["dataset"]["root"], PROJECT_ROOT),
        sampling_rate_hz=fs,
    )
    metadata = load_ptbxl_metadata(root, sampling_rate_hz=fs)
    matches = metadata.loc[metadata["ecg_id"].eq(args.ecg_id)]
    if matches.empty:
        parser.error(f"ECG ID {args.ecg_id} was not found in PTB-XL metadata.")
    filename_column = "filename_lr" if fs == 100 else "filename_hr"
    record = load_ecg(root, str(matches.iloc[0][filename_column]), sampling_rate_hz=fs)
    result = compute_input_saliency(predictor, record, class_name=args.class_name)
    output = args.output or PROJECT_ROOT / "outputs/figures" / f"ecg_{args.ecg_id}_saliency_{args.class_name.lower()}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_saliency_overlay(record, result)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"Saved {args.class_name} model-response figure: {output}")
    print("This visualization is a model response, not a clinical explanation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
