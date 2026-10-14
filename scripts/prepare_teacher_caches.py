"""Prewarm the gentle-bandpass train/validation caches with bounded threads.

Produces the same fingerprint, dtype, record order and values as the serial
cache builder. Does not touch any cache that another process is constructing.
No test waveforms or test predictions are prepared here.
"""
from __future__ import annotations

import concurrent.futures
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.ptbxl import load_ecg
from src.signal_processing.filters import FilterConfig, preprocess_ecg
from src.training.data import _cache_fingerprint, load_labeled_splits


def main():
    cp = torch.load(ROOT / "outputs/distillation/distilled_seed44/checkpoint/cnn1d_best.pt",
                    map_location="cpu", weights_only=True)
    settings = copy.deepcopy(cp["config"]["signal_processing"])
    settings.update(normalization="none", baseline_removal=False, lowcut_hz=0.1, highcut_hz=40.0)
    settings.pop("training_mean", None); settings.pop("training_std", None)
    cfg = FilterConfig.from_mapping(settings)
    dataset_root = cp["config"]["dataset"]["root"]
    frames = load_labeled_splits(dataset_root, 100, require_diagnostic_label=True)
    cache_root = ROOT / "outputs/cache"
    for split in ("train", "validation"):
        frame = frames[split]
        fingerprint = _cache_fingerprint(frame, 100, cfg)
        path = cache_root / f"{split}_full_{fingerprint}.npy"
        metadata = path.with_suffix(".json")
        building = path.with_suffix(".building")
        shape = (len(frame), 12, 1000)
        if path.exists() and metadata.exists():
            assert json.loads(metadata.read_text())["fingerprint"] == fingerprint
            assert np.load(path, mmap_mode="r").shape == shape
            continue
        assert not path.exists() and not metadata.exists() and not building.exists(), str(path)
        output = np.lib.format.open_memmap(building, mode="w+", dtype=np.float32, shape=shape)
        rows = list(frame.itertuples(index=False))

        def process(index):
            raw = load_ecg(dataset_root, rows[index].filename_lr, sampling_rate_hz=100)
            return index, preprocess_ecg(raw.signal, 100, config=cfg).T.copy()

        print(f"Parallel P2 {split} raw preprocessing: {len(frame)} records", flush=True)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                for start in range(0, len(frame), 256):
                    for index, prepared in pool.map(process, range(start, min(len(frame), start + 256))):
                        assert prepared.shape == (12, 1000) and np.isfinite(prepared).all()
                        output[index] = prepared
                    if start % 1024 == 0 or start + 256 >= len(frame):
                        print(f"P2 {split}: {min(len(frame), start + 256)}/{len(frame)}", flush=True)
            output.flush()
        except BaseException:
            del output
            building.unlink(missing_ok=True)
            raise
        del output
        building.replace(path)
        metadata.write_text(json.dumps({"fingerprint": fingerprint, "shape": list(shape), "dtype": "float32",
                                       "preparation": "eight threads; record-local deterministic transforms"}), encoding="utf-8")
        print(f"P2 {split} cache ready: {path}", flush=True)


if __name__ == "__main__":
    main()
