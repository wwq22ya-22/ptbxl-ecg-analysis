"""Memory-mapped preprocessed PTB-XL signals for repeatable training epochs."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.ptbxl import SUPERCLASSES, add_superclass_labels, create_fold_splits, load_ecg, load_ptbxl_metadata, load_scp_statements
from src.signal_processing.filters import FilterConfig, preprocess_ecg

PREPROCESS_VERSION = 1


def load_labeled_splits(dataset_root: str | Path, sampling_rate_hz: int) -> dict[str, pd.DataFrame]:
    """Apply the official patient-safe folds and superclass mapping."""
    metadata = load_ptbxl_metadata(dataset_root, sampling_rate_hz=sampling_rate_hz)
    statements = load_scp_statements(dataset_root, sampling_rate_hz=sampling_rate_hz)
    labeled = add_superclass_labels(metadata, statements)
    return create_fold_splits(labeled)


def _cache_fingerprint(frame: pd.DataFrame, sampling_rate_hz: int, config: FilterConfig) -> str:
    filename_column = "filename_lr" if sampling_rate_hz == 100 else "filename_hr"
    payload = {
        "version": PREPROCESS_VERSION,
        "sampling_rate_hz": sampling_rate_hz,
        "filter_config": asdict(config),
        "ecg_ids": frame["ecg_id"].astype(int).tolist(),
        "filenames": frame[filename_column].astype(str).tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def ensure_preprocessed_cache(
    frame: pd.DataFrame,
    dataset_root: str | Path,
    cache_dir: str | Path,
    *,
    split_name: str,
    sampling_rate_hz: int,
    config: FilterConfig,
) -> Path:
    """Preprocess once on CPU; later epochs read batches from a memory map.

    The cache key includes record order, sampling rate, filter parameters and
    an explicit preprocessing version. No full dataset tensor is loaded to GPU.
    """
    if frame.empty:
        raise ValueError(f"Cannot prepare an empty {split_name} split.")
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    fingerprint = _cache_fingerprint(frame, sampling_rate_hz, config)
    cache_path = cache_root / f"{split_name}_{fingerprint}.npy"
    metadata_path = cache_root / f"{split_name}_{fingerprint}.json"
    expected_shape = (len(frame), 12, sampling_rate_hz * 10)
    if cache_path.is_file() and metadata_path.is_file():
        info = json.loads(metadata_path.read_text(encoding="utf-8"))
        if info.get("fingerprint") == fingerprint and info.get("shape") == list(expected_shape):
            cached = np.load(cache_path, mmap_mode="r")
            if cached.shape == expected_shape and cached.dtype == np.float32:
                print(f"Using existing {split_name} cache: {cache_path}", flush=True)
                return cache_path
    if cache_path.exists() or metadata_path.exists():
        raise RuntimeError(f"Incomplete or incompatible cache files for {split_name}: {cache_path}")

    temporary = cache_root / f"{split_name}_{fingerprint}.building"
    samples = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=expected_shape)
    filename_column = "filename_lr" if sampling_rate_hz == 100 else "filename_hr"
    started = time.monotonic()
    try:
        for index, row in enumerate(frame.itertuples(index=False), start=0):
            record_name = getattr(row, filename_column)
            record = load_ecg(dataset_root, str(record_name), sampling_rate_hz=sampling_rate_hz)
            if record.signal.shape != (expected_shape[2], 12):
                raise ValueError(
                    f"Record {record_name} has shape {record.signal.shape}; expected {(expected_shape[2], 12)}"
                )
            filtered = preprocess_ecg(record.signal, sampling_rate_hz, config=config)
            samples[index] = np.ascontiguousarray(filtered.T)
            if (index + 1) % 1000 == 0 or index + 1 == len(frame):
                print(
                    f"Prepared {split_name}: {index + 1:,}/{len(frame):,} records "
                    f"in {(time.monotonic() - started) / 60:.1f} min",
                    flush=True,
                )
        samples.flush()
    except Exception:
        del samples
        temporary.unlink(missing_ok=True)
        raise
    del samples
    temporary.replace(cache_path)
    metadata_path.write_text(
        json.dumps({"fingerprint": fingerprint, "shape": expected_shape, "dtype": "float32"}, indent=2),
        encoding="utf-8",
    )
    return cache_path


class CachedPTBXLDataset(Dataset):
    """Map-style dataset backed by a read-only NumPy memory-mapped cache."""

    def __init__(self, cache_path: str | Path, frame: pd.DataFrame):
        self.signals = np.load(cache_path, mmap_mode="r")
        self.labels = frame.loc[:, list(SUPERCLASSES)].to_numpy(dtype=np.float32, copy=True)
        self.ecg_ids = frame["ecg_id"].to_numpy(dtype=np.int64, copy=True)
        if self.signals.shape[0] != len(frame) or self.signals.shape[1] != 12:
            raise ValueError("Cache does not match its labeled metadata frame.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(np.array(self.signals[index], dtype=np.float32, copy=True)),
            torch.from_numpy(self.labels[index].copy()),
        )
