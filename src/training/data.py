"""Memory-mapped preprocessed PTB-XL signals for repeatable training epochs."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.ptbxl import SUPERCLASSES, add_superclass_labels, create_fold_splits, load_ecg, load_ptbxl_metadata, load_scp_statements
from src.signal_processing.filters import FilterConfig, preprocess_ecg

PREPROCESS_VERSION = 1


def load_labeled_splits(
    dataset_root: str | Path, sampling_rate_hz: int, *,
    train_folds: Sequence[int] = tuple(range(1, 9)),
    validation_fold: int = 9, test_fold: int = 10,
    test_folds: Sequence[int] | None = None,
    require_diagnostic_label: bool = False,
) -> dict[str, pd.DataFrame]:
    """Apply the official patient-safe folds and superclass mapping."""
    metadata = load_ptbxl_metadata(dataset_root, sampling_rate_hz=sampling_rate_hz)
    statements = load_scp_statements(dataset_root, sampling_rate_hz=sampling_rate_hz)
    labeled = add_superclass_labels(metadata, statements)
    if require_diagnostic_label:
        labeled = labeled.loc[labeled[list(SUPERCLASSES)].sum(axis=1).gt(0)].copy()
    return create_fold_splits(labeled, train_folds=train_folds, validation_fold=validation_fold, test_fold=test_fold, test_folds=test_folds)


def _cache_fingerprint(frame: pd.DataFrame, sampling_rate_hz: int, config: FilterConfig) -> str:
    filename_column = "filename_lr" if sampling_rate_hz == 100 else "filename_hr"
    filter_settings = asdict(config)
    for key, default in (("training_mean", ()), ("training_std", ()), ("baseline_removal", True)):
        if filter_settings[key] == default:
            filter_settings.pop(key)
    payload = {
        "version": PREPROCESS_VERSION,
        "sampling_rate_hz": sampling_rate_hz,
        "filter_config": filter_settings,
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
        if config.normalization == "train_global":
            raw_path = ensure_preprocessed_cache(frame, dataset_root, cache_root, split_name=split_name, sampling_rate_hz=sampling_rate_hz, config=replace(config, normalization="none", training_mean=(), training_std=()))
            raw = np.load(raw_path, mmap_mode="r")
            mean = np.asarray(config.training_mean, dtype=np.float64)[None, :, None]
            std = np.asarray(config.training_std, dtype=np.float64)[None, :, None]
            if mean.shape != (1, 12, 1) or std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
                raise ValueError("Invalid training normalization statistics")
            for start in range(0, len(frame), 256):
                samples[start:start+256] = (raw[start:start+256] - mean) / std
        for index, row in enumerate(frame.itertuples(index=False) if config.normalization != "train_global" else (), start=0):
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


def fit_training_normalization(cache_path: str | Path) -> tuple[list[float], list[float]]:
    """Fit per-lead moments from training waveforms only, in bounded chunks."""
    values = np.load(cache_path, mmap_mode="r")
    count = 0
    mean = np.zeros(12, dtype=np.float64)
    m2 = np.zeros(12, dtype=np.float64)
    for start in range(0, len(values), 128):
        chunk = np.asarray(values[start:start+128], dtype=np.float64)
        n = chunk.shape[0] * chunk.shape[2]
        chunk_mean = chunk.mean(axis=(0, 2))
        chunk_m2 = ((chunk - chunk_mean[None, :, None]) ** 2).sum(axis=(0, 2))
        delta = chunk_mean - mean
        m2 += chunk_m2 + delta ** 2 * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
    return mean.tolist(), np.maximum(np.sqrt(m2 / count), 1e-8).tolist()


def ensure_cache_subset(
    source_cache: str | Path, source_frame: pd.DataFrame, frame: pd.DataFrame,
    cache_dir: str | Path, *, split_name: str, sampling_rate_hz: int, config: FilterConfig,
) -> Path:
    """Copy a checked record subset without refiltering raw waveforms.

    Only record-local transforms can be shared across CV folds; a cache scaled
    using full-training statistics must never be reused for fold normalization.
    """
    if config.normalization == "train_global":
        raise ValueError("Subset only record-local caches; fit global statistics on each fold separately.")
    if frame.empty or not source_frame.ecg_id.is_unique or not frame.ecg_id.is_unique:
        raise ValueError("Cache subsets require nonempty frames with unique ECG IDs.")
    source_cache = Path(source_cache)
    source_info = json.loads(source_cache.with_suffix('.json').read_text(encoding='utf-8'))
    if source_info.get('fingerprint') != _cache_fingerprint(source_frame, sampling_rate_hz, config):
        raise ValueError("Source cache preprocessing does not match the requested settings.")
    positions = pd.Index(source_frame.ecg_id).get_indexer(frame.ecg_id)
    if np.any(positions < 0):
        raise ValueError("Requested ECG IDs are not all present in the source cache.")
    column = 'filename_lr' if sampling_rate_hz == 100 else 'filename_hr'
    if source_frame.iloc[positions][column].tolist() != frame[column].tolist():
        raise ValueError("ECG IDs and filenames disagree with the source cache.")
    source = np.load(source_cache, mmap_mode='r')
    if source.shape != (len(source_frame), 12, sampling_rate_hz * 10) or source.dtype != np.float32:
        raise ValueError("Invalid source cache shape or dtype.")
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    fingerprint = _cache_fingerprint(frame, sampling_rate_hz, config)
    path = cache_root / f'{split_name}_{fingerprint}.npy'
    metadata = path.with_suffix('.json')
    shape = (len(frame), 12, sampling_rate_hz * 10)
    if path.is_file() and metadata.is_file():
        info = json.loads(metadata.read_text(encoding='utf-8'))
        existing = np.load(path, mmap_mode='r')
        if info.get('fingerprint') == fingerprint and info.get('shape') == list(shape) and existing.shape == shape and existing.dtype == np.float32:
            return path
        raise ValueError("Existing subset cache is incompatible.")
    if path.exists() or metadata.exists():
        raise ValueError("Incomplete subset cache; preserve and inspect before retrying.")
    temporary = path.with_suffix('.building')
    output = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=shape)
    try:
        for start in range(0, len(frame), 128):
            output[start:start+128] = source[positions[start:start+128]]
        output.flush()
    except Exception:
        del output
        temporary.unlink(missing_ok=True)
        raise
    del output
    temporary.replace(path)
    metadata.write_text(json.dumps({'fingerprint': fingerprint, 'shape': list(shape), 'dtype': 'float32', 'source_cache': str(source_cache)}), encoding='utf-8')
    return path
