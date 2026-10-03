"""Export and validate a frozen PTB-XL single model for STM32Cube AI."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.ptbxl import LEAD_NAMES, SUPERCLASSES
from src.models import MODEL_TYPES, build_model
from src.config import resolve_project_path
from src.signal_processing.filters import FilterConfig
from src.training.data import load_labeled_splits, ensure_preprocessed_cache

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def get_split(name, checkpoint):
    cfg = checkpoint['config']
    dataset_root = resolve_project_path(cfg['dataset']['root'], ROOT)
    fs = int(checkpoint['sampling_rate_hz'])
    frame = load_labeled_splits(
        dataset_root, fs,
        require_diagnostic_label=bool(cfg['dataset'].get('require_diagnostic_label', False)),
    )[name]
    cache = ensure_preprocessed_cache(
        frame, dataset_root, resolve_project_path(cfg['outputs']['cache_dir'], ROOT),
        split_name=f'{name}_full', sampling_rate_hz=fs,
        config=FilterConfig.from_mapping(cfg['signal_processing']),
    )
    metadata = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
    array = np.load(cache, mmap_mode="r")
    if list(array.shape) != metadata.get("shape") or metadata.get("dtype") != "float32":
        raise ValueError(f"Invalid {name} cache metadata.")
    if array.shape != (len(frame), 12, 1000):
        raise ValueError(f"Expected {name} data [N,12,1000], got {array.shape}.")
    if not set(SUPERCLASSES).issubset(frame.columns):
        raise ValueError(f"{name}.csv misses superclass labels.")
    return array, frame

def select(array, frame, count, seed):
    if count < 1 or count > len(frame):
        raise ValueError(f"Requested {count} rows, but split has {len(frame)}.")
    rows = np.sort(np.random.default_rng(seed).choice(len(frame), count, replace=False))
    x = np.ascontiguousarray(array[rows], dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Selected input has non-finite values.")
    selected = frame.iloc[rows]
    y = selected[list(SUPERCLASSES)].to_numpy(dtype=np.uint8)
    ids = selected["ecg_id"].to_numpy(dtype=np.int64)
    return x, y, ids, rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-count", type=int, default=256)
    parser.add_argument("--validation-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / 'outputs/checkpoints/baseline_best.pt')
    parser.add_argument("--output-dir", type=Path, default=ROOT / 'outputs/stm32')
    args = parser.parse_args()
    out = resolve_project_path(args.output_dir, ROOT)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = resolve_project_path(args.checkpoint, ROOT)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    kind = checkpoint.get('model_type')
    if checkpoint.get("scope") != "full" or kind not in MODEL_TYPES:
        raise ValueError("Expected a supported full-training single-model checkpoint.")
    if int(checkpoint['sampling_rate_hz']) != 100:
        raise ValueError('This deployment input contract requires 100 Hz records.')
    if tuple(checkpoint.get("lead_names", ())) != LEAD_NAMES or tuple(checkpoint.get("superclasses", ())) != SUPERCLASSES:
        raise ValueError("Checkpoint lead/class order differs from deployment contract.")
    thresholds = np.asarray(checkpoint.get("thresholds"), dtype=np.float64)
    if thresholds.shape != (5,) or not np.isfinite(thresholds).all() or np.any((thresholds <= 0) | (thresholds >= 1)):
        raise ValueError("Expected five finite validation thresholds.")

    train, train_frame = get_split("train", checkpoint)
    valid, valid_frame = get_split("validation", checkpoint)
    calib_x, _, calib_ids, calib_rows = select(train, train_frame, args.calibration_count, args.seed)
    valid_x, valid_y, valid_ids, valid_rows = select(valid, valid_frame, args.validation_count, args.seed + 1)

    torch.set_num_threads(4)
    model = build_model(kind, dropout=float(checkpoint["config"]["model"]["dropout"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval().cpu()
    params = sum(p.numel() for p in model.parameters())
    onnx_path = out / f"ptbxl_{kind}_float32.onnx"
    torch.onnx.export(
        model, (torch.from_numpy(valid_x[:1]),), str(onnx_path),
        input_names=["ecg"], output_names=["logits"], opset_version=args.opset,
        export_params=True, do_constant_folding=True, dynamo=False,
    )

    import onnx
    import onnxruntime as ort
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph, full_check=True)
    graph = onnx.shape_inference.infer_shapes(graph)
    def dims(v):
        return [d.dim_value if d.HasField("dim_value") else d.dim_param for d in v.type.tensor_type.shape.dim]
    in_shape, out_shape = dims(graph.graph.input[0]), dims(graph.graph.output[0])
    if in_shape != [1, 12, 1000] or out_shape != [1, 5]:
        raise ValueError(f"Unexpected ONNX shapes: {in_shape} -> {out_shape}.")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].name != "ecg" or session.get_outputs()[0].name != "logits":
        raise ValueError("Unexpected ONNX input/output names.")
    with torch.inference_mode():
        pt = np.concatenate([model(torch.from_numpy(row[None])).numpy() for row in valid_x])
    ox = np.concatenate([session.run(["logits"], {"ecg": row[None]})[0] for row in valid_x])
    error = np.abs(pt - ox)
    parity = {
        "status": "passed" if np.allclose(pt, ox, rtol=2e-4, atol=1e-5) else "failed",
        "records_compared": len(valid_x),
        "max_absolute_error": float(error.max(initial=0)),
        "mean_absolute_error": float(error.mean()),
        "tolerance": {"rtol": 2e-4, "atol": 1e-5},
        "onnxruntime_version": ort.__version__,
    }
    np.save(out / "calibration_train.npy", calib_x, allow_pickle=False)
    np.savez_compressed(out / "calibration_train.npz", ecg=calib_x)
    probs = 1.0 / (1.0 + np.exp(-np.clip(pt, -80, 80)))
    np.savez_compressed(out / "validation_reference.npz", ecg=valid_x, targets=valid_y,
                        ecg_id=valid_ids, reference_logits=pt, reference_probabilities=probs)
    weights = out / f"{kind}_weights_only.pt"
    torch.save({k: v.detach().cpu() for k, v in checkpoint["model_state"].items()}, weights)
    (out / "onnx_parity.json").write_text(json.dumps(parity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = [onnx_path, out / "calibration_train.npy", out / "calibration_train.npz",
             out / "validation_reference.npz", weights]
    manifest = {
        "source_checkpoint": str(checkpoint_path), "source_checkpoint_sha256": digest(checkpoint_path),
        "model_type": kind, "trainable_parameter_count": int(params),
        "onnx": {"file": onnx_path.name, "opset": args.opset, "input": {"name": "ecg", "dtype": "float32", "shape": in_shape},
                 "output": {"name": "logits", "dtype": "float32", "shape": out_shape},
                 "operators": sorted({n.op_type for n in graph.graph.node}), "preprocessing_in_graph": False},
        "signal_contract": {"sampling_rate_hz": checkpoint["sampling_rate_hz"], "duration_seconds": 10,
                            "units_before_preprocessing": "mV", "layout": "[batch,lead,time]",
                            "lead_order": list(LEAD_NAMES), "preprocessing": checkpoint["config"]["signal_processing"]},
        "outputs": {"labels": list(SUPERCLASSES), "semantics": "independent logits; sigmoid each output",
                    "thresholds_from_validation_fold_9": thresholds.tolist()},
        "calibration": {"files": ["calibration_train.npy", "calibration_train.npz"], "npz_key": "ecg",
                        "source": "PTB-XL training folds 1-8 only", "records": len(calib_x),
                        "seed": args.seed, "ecg_ids": calib_ids.tolist(), "row_indices": calib_rows.tolist()},
        "host_validation": {"file": "validation_reference.npz", "source": "PTB-XL validation fold 9",
                            "records": len(valid_x), "seed": args.seed + 1,
                            "ecg_ids": valid_ids.tolist(), "row_indices": valid_rows.tolist(),
                            "parity_report": "onnx_parity.json"},
        "test_fold_10_used": False,
        "artifacts": {p.name: {"size_bytes": p.stat().st_size, "sha256": digest(p)} for p in files},
    }
    (out / "deployment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ONNX: {onnx_path} ({onnx_path.stat().st_size:,} bytes)")
    print(f"Shapes: {in_shape} -> {out_shape}; operators: {manifest['onnx']['operators']}; parameters: {params:,}")
    print(f"Parity: {parity}; calibration records: {len(calib_x)}; validation records: {len(valid_x)}")
    print(f"Artifacts: {out}")
    if parity["status"] != "passed":
        raise RuntimeError("ONNX Runtime and PyTorch parity failed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
