"""Validate PTB-XL files and create official patient-safe split manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ptbxl import (
    SUPERCLASSES,
    PTBXLDataError,
    add_superclass_labels,
    create_fold_splits,
    find_ptbxl_root,
    load_ptbxl_metadata,
    load_scp_statements,
)
from src.config import load_config, resolve_project_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--dataset-root", type=Path, help="PTB-XL root or its parent directory")
    parser.add_argument("--sampling-rate", type=int, choices=(100, 500))
    parser.add_argument("--output-dir", type=Path, help="Where to write train/validation/test CSV manifests")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        dataset_cfg = config.get("dataset", {})
        fs = args.sampling_rate or int(dataset_cfg.get("sampling_rate_hz", 100))
        root_arg = args.dataset_root or resolve_project_path(
            dataset_cfg.get("root", "data/ptb-xl"), PROJECT_ROOT
        )
        root = find_ptbxl_root(root_arg, sampling_rate_hz=fs)
        metadata = load_ptbxl_metadata(root, sampling_rate_hz=fs)
        statements = load_scp_statements(root, sampling_rate_hz=fs)
        labeled = add_superclass_labels(metadata, statements, superclasses=SUPERCLASSES)
        split_cfg = dataset_cfg
        splits = create_fold_splits(
            labeled,
            train_folds=tuple(split_cfg.get("train_folds", range(1, 9))),
            validation_fold=int(split_cfg.get("validation_fold", 9)),
            test_fold=int(split_cfg.get("test_fold", 10)),
        )
    except (PTBXLDataError, OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(2, f"Data preparation stopped: {exc}\n")

    output_dir = args.output_dir or PROJECT_ROOT / config.get("outputs", {}).get("split_dir", "outputs/splits")
    output_dir = output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {root}")
    print(f"Sampling rate: {fs} Hz")
    for split_name, frame in splits.items():
        target_path = output_dir / f"{split_name}.csv"
        frame.to_csv(target_path, index=False)
        counts = {name: int(frame[name].sum()) for name in SUPERCLASSES}
        print(f"{split_name}: {len(frame):,} records; superclass positives={counts}")
        print(f"  manifest: {target_path}")
    print("Patient-level fold overlap check: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
