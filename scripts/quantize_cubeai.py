"""Quantize an exported ECG model using training-only calibration records."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_stm32 import digest, get_split
from src.data.ptbxl import SUPERCLASSES
from src.training.metrics import calculate_multilabel_metrics


class ECGCalibrationReader(CalibrationDataReader):
    def __init__(self, records):
        self.records = records
        self.rewind()

    def get_next(self):
        return next(self.iterator, None)

    def rewind(self):
        self.iterator = ({"ecg": np.ascontiguousarray(row[None])} for row in self.records)


def probabilities(session, records):
    values = [session.run(None, {"ecg": np.ascontiguousarray(row[None])})[0] for row in records]
    logits = np.concatenate(values)
    return 1 / (1 + np.exp(-np.clip(logits, -80, 80)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.export_dir.resolve()
    manifest = json.loads((out / "deployment_manifest.json").read_text(encoding="utf-8"))
    if manifest["calibration"]["source"] != "PTB-XL training folds 1-8 only":
        raise ValueError("Quantization requires training-only calibration.")
    source = out / manifest["onnx"]["file"]
    prepared = out / "preprocessed_float32.onnx"
    target = out / f"ptbxl_{manifest['model_type']}_int8.onnx"
    # ONNX's Windows shape inference does not handle a Unicode user temp path.
    temporary = out / "quantization_tmp"
    temporary.mkdir(exist_ok=True)
    tempfile.tempdir = str(temporary)
    quant_pre_process(str(source), str(prepared), skip_symbolic_shape=True)
    calibration = np.load(out / "calibration_train.npy", allow_pickle=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    print("Quantizing with 256 training records, per-channel int8 QDQ...", flush=True)
    quantize_static(
        str(prepared), str(target), ECGCalibrationReader(calibration),
        quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8, per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"WeightSymmetric": True, "ActivationSymmetric": False},
        calibration_providers=["CPUExecutionProvider"],
    )
    onnx.checker.check_model(onnx.load(str(target)), full_check=True)
    fp = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
    qp = ort.InferenceSession(str(target), options, providers=["CPUExecutionProvider"])
    checkpoint = torch.load(manifest["source_checkpoint"], map_location="cpu", weights_only=True)
    if digest(Path(manifest["source_checkpoint"])) != manifest["source_checkpoint_sha256"]:
        raise ValueError("Source checkpoint changed since export.")
    records, frame = get_split("validation", checkpoint)
    truth = frame[list(SUPERCLASSES)].to_numpy(dtype=np.uint8)
    thresholds = manifest["outputs"]["thresholds_from_validation_fold_9"]
    print(f"Comparing float32/int8 on all {len(frame)} validation-fold-9 records...", flush=True)
    p_float = probabilities(fp, records)
    p_quant = probabilities(qp, records)
    np.savez_compressed(out / "int8_validation_predictions.npz", ecg_id=frame.ecg_id.to_numpy(),
                        targets=truth, probabilities_float32=p_float, probabilities_int8=p_quant)
    delta = np.abs(p_float - p_quant)
    report = {
        "model_type": manifest["model_type"], "quantization": "static QDQ int8",
        "weights": "signed symmetric per-channel", "activations": "signed asymmetric per-tensor",
        "calibration": manifest["calibration"], "validation_source": "PTB-XL validation fold 9",
        "test_fold_10_used": False, "thresholds_retuned": False,
        "thresholds": thresholds, "onnxruntime_version": ort.__version__,
        "max_probability_error": float(delta.max()), "mean_probability_error": float(delta.mean()),
        "label_agreement": float(np.mean((p_float >= thresholds) == (p_quant >= thresholds))),
        "float32": calculate_multilabel_metrics(truth, p_float, thresholds),
        "int8": calculate_multilabel_metrics(truth, p_quant, thresholds),
        "onnx": {"file": target.name, "size_bytes": target.stat().st_size, "sha256": digest(target)},
    }
    (out / "int8_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in ("float32", "int8"):
        print(name, {k: v for k, v in report[name].items() if k.startswith("macro")}, flush=True)
    print(f"ONNX: {target}; label agreement: {report['label_agreement']:.6f}", flush=True)


if __name__ == "__main__":
    main()
