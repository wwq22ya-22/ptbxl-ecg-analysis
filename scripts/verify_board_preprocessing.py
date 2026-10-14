"""Verify actual board C preprocessing on all 128 frozen fold-9 demo records."""
import json
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_board_preprocess import OUT, OLD
from scripts.refine_student_thresholds import read, save, digest
from src.inference.board_preprocessing import load_library, preprocess_float, preprocess_int8
from src.inference.raw_serial_simulator import RawFirmwareSimulator
from src.inference.serial_link import STM32Link
from src.signal_processing.filters import FilterConfig, preprocess_ecg

def main():
    m = read(OUT / 'serial_manifest.json')
    assert digest(ROOT / 'firmware/ecg_preprocess/ecg_preprocess.c') == m['preprocessing_implementation']['source_sha256']
    lib = load_library()
    assert lib.ecg_preprocess_contract_sha256().decode() == m['preprocessing_sha256']
    assert digest(OUT / m['sample_file']) == m['sample_sha256']
    with np.load(OUT / m['sample_file']) as d: samples = {k: d[k].copy() for k in d.files}
    cfg = FilterConfig.from_mapping(m['preprocessing'])
    options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(OUT / m['model_file']), options, providers=['CPUExecutionProvider'])
    scale = np.float32(m['input_scale']); zero = m['input_zero_point']
    thresholds = np.asarray(m['thresholds'], dtype=np.float32)
    def logits(q):
        x = ((q.astype(np.float32) - zero) * scale).reshape(1, 12, 1000)
        return session.run(None, {session.get_inputs()[0].name: x})[0][0]
    sim = RawFirmwareSimulator(m, logits); link = STM32Link(sim, m); link.handshake()
    normalized, quantized, probabilities = [], [], []
    max_vs_python, max_vs_cache, quant_vs_python, quant_vs_cache, protocol_mismatch = 0., 0., 0, 0, 0
    for i, raw in enumerate(samples['raw_mv']):
        raw32 = np.asarray(raw, dtype=np.float32)
        c = preprocess_float(raw32, lib)
        p = preprocess_ecg(raw32.T, 100, config=cfg).T
        np.testing.assert_allclose(c, p, rtol=2e-6, atol=2e-6)
        max_vs_python = max(max_vs_python, float(np.max(abs(c - p))))
        max_vs_cache = max(max_vs_cache, float(np.max(abs(c - samples['ecg'][i]))))
        q = preprocess_int8(raw32, lib)
        expected_q = np.clip(np.rint(p / scale) + zero, -128, 127).astype(np.int8)
        cached_q = np.clip(np.rint(samples['ecg'][i] / scale) + zero, -128, 127).astype(np.int8)
        quant_vs_python += int((q != expected_q).sum())
        quant_vs_cache += int((q != cached_q).sum())
        np.testing.assert_array_equal(q, expected_q)
        result = link.predict(raw32)
        protocol_mismatch += int((np.ctypeslib.as_array(sim.input).reshape(12, 1000) != q).sum())
        normalized.append(c); quantized.append(q); probabilities.append(result['probabilities'])
        if (i + 1) % 32 == 0: print(f'Checked raw C pipeline and serial protocol: {i+1}/{len(samples["ecg_id"])}', flush=True)
    probs = np.asarray(probabilities, dtype=np.float32)
    reference = samples['reference_probabilities']
    label_agreement = float(np.mean((probs >= thresholds) == (reference >= thresholds)))
    max_probability_error = float(np.max(abs(probs - reference)))
    assert protocol_mismatch == 0 and label_agreement == 1.
    np.savez_compressed(OUT / 'preprocessing_verified.npz', ecg_ids=samples['ecg_id'], targets=samples['targets'],
                        normalized=np.stack(normalized), int8_inputs=np.stack(quantized),
                        probabilities=probs, reference_probabilities=reference)
    report = {'records': len(probs), 'sample_source': m['sample_source'], 'preprocessing_sha256': m['preprocessing_sha256'],
              'model_sha256': m['model_sha256'], 'host_C_vs_python_raw32_max_absolute_error': max_vs_python,
              'host_C_vs_original_training_cache_max_absolute_error': max_vs_cache,
              'int8_different_values_vs_python_raw32': quant_vs_python,
              'int8_different_values_vs_original_cache': quant_vs_cache, 'serial_parser_int8_mismatches': protocol_mismatch,
              'classification_agreement_with_original_host_C': label_agreement,
              'max_probability_difference_from_original_host_C': max_probability_error,
              'workspace_bytes': int(lib.ecg_preprocess_workspace_bytes()), 'raw_buffer_bytes': 48000,
              'preparation_calls': sim.preparation_calls, 'physical_board_test': False,
              'board_latency_measured': False, 'note': 'Same C source on host; not an ARM arithmetic or physical-board validation.'}
    save(OUT / 'preprocessing_validation.json', report)
    print(json.dumps(report, indent=2), flush=True)
if __name__ == '__main__': main()
