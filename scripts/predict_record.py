"""Show five-label model probabilities for one PTB-XL ECG ID."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ptbxl import find_ptbxl_root, load_ecg, load_ptbxl_metadata
from src.inference import load_predictor
from src.config import load_config, resolve_project_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecg-id", type=int, default=1)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "outputs/checkpoints/baseline_best.pt")
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
        parser.error(f"ECG ID {args.ecg_id} was not found in the PTB-XL metadata.")
    filename_column = "filename_lr" if fs == 100 else "filename_hr"
    record = load_ecg(root, str(matches.iloc[0][filename_column]), sampling_rate_hz=fs)
    result = predictor.predict_record(record)
    print(json.dumps({
        "ecg_id": args.ecg_id,
        "sampling_rate_hz": fs,
        "probabilities": result.probabilities,
        "thresholds": result.thresholds,
        "positive_labels": result.positive_labels,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
