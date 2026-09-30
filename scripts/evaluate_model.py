"""Evaluate a trained checkpoint on official test fold 10."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.evaluate import evaluate_checkpoint
from src.config import load_config, resolve_project_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "outputs/checkpoints/baseline_best.pt")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--allow-smoke", action="store_true", help="Debug only; marks outputs as smoke")
    args = parser.parse_args()
    config = load_config(PROJECT_ROOT / "configs/default.yaml")
    evaluate_checkpoint(
        args.checkpoint, project_root=PROJECT_ROOT,
        dataset_root=args.dataset_root or resolve_project_path(config["dataset"]["root"], PROJECT_ROOT),
        allow_smoke=args.allow_smoke,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
