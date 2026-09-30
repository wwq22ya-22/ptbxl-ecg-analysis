"""Train the five-label PTB-XL 1D CNN baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train import train_baseline
from src.models import MODEL_TYPES
from src.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--model", choices=MODEL_TYPES, help="Model architecture; defaults to configs/default.yaml")
    parser.add_argument("--max-samples", type=int, help="Small pipeline check; saves a separate smoke checkpoint")
    args = parser.parse_args()
    config = load_config(args.config)
    train_baseline(
        config, project_root=PROJECT_ROOT, dataset_root=args.dataset_root,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, max_samples=args.max_samples,
        model_type=args.model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
