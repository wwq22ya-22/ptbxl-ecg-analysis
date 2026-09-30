"""Send one real PTB-XL ECG to the locally running prediction API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ptbxl import find_ptbxl_root, load_ecg, load_ptbxl_metadata
from src.config import load_config, resolve_project_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecg-id", type=int, default=9)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()
    config = load_config(PROJECT_ROOT / "configs/default.yaml")
    fs = int(config["dataset"]["sampling_rate_hz"])
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
    base_url = args.base_url.rstrip("/")
    health = requests.get(f"{base_url}/health", timeout=15)
    health.raise_for_status()
    response = requests.post(
        f"{base_url}/predict",
        json={
            "signal": record.signal.tolist(),
            "sampling_rate_hz": record.sampling_rate_hz,
            "lead_names": list(record.lead_names),
            "units": list(record.units),
        },
        timeout=60,
    )
    response.raise_for_status()
    print("Health:", json.dumps(health.json(), ensure_ascii=False))
    print("Prediction:", json.dumps(response.json(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
