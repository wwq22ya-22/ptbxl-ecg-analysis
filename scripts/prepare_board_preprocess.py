"""Freeze seed44 raw-input protocol and exact SciPy preprocessing constants."""
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfilt_zi
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.refine_student_thresholds import read, save, digest
OLD = ROOT / 'outputs/distillation/distilled_export'
OUT = ROOT / 'outputs/distillation/board_preprocess_export'
def prepare():
    m = read(OLD / 'serial_manifest.json')
    p = m['preprocessing']
    assert p['baseline_windows_s'] == [.2, .6] and p['lowcut_hz'] == .5 and p['highcut_hz'] == 40.
    assert p['bandpass_order'] == 4 and p['notch_hz'] is None and p['normalization'] == 'train_global'
    assert m['input_shape'] == [12, 1000]
    assert digest(OLD / m['model_file']) == m['model_sha256']
    assert digest(OLD / m['sample_file']) == m['sample_sha256']
    sos = butter(4, (.5, 40.), fs=100, btype='bandpass', output='sos')
    zi = sosfilt_zi(sos)
    assert sos.shape == (4, 6) and np.array_equal(sos[:, 3], np.ones(4))
    contract = {'sampling_rate_hz': 100, 'input_shape': [12, 1000], 'raw_wire_type': 'float32_mV',
                'preprocessing': p, 'median_widths': [21, 61], 'median_boundary': 'reflect',
                'sos': sos.tolist(), 'sos_initial_state': zi.tolist(), 'odd_padlen': 27,
                'arithmetic': 'float32 median/baseline; float64 SOS; float32 bandpass; float64 normalization -> float32',
                'source_sha256': digest(ROOT / 'firmware/ecg_preprocess/ecg_preprocess.c')}
    sha = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    header = ('/* Generated from frozen seed44 training preprocessing; do not refit. */\n'
              '#ifndef ECG_PREPROCESS_CONFIG_H\n#define ECG_PREPROCESS_CONFIG_H\n'
              f'#define ECG_PREPROCESS_SHA256 "{sha}"\n'
              f'#define ECG_PREPROCESS_INPUT_SCALE {m["input_scale"]:.17g}f\n'
              f'#define ECG_PREPROCESS_INPUT_ZERO ({m["input_zero_point"]})\n')
    for name, values in [('ECG_SOS', sos), ('ECG_SOS_ZI', zi)]:
        header += f'static const double {name}[{values.shape[0]}][{values.shape[1]}] = {{\n'
        header += '\n'.join('    { ' + ', '.join(format(float(v), '.17g') for v in row) + ' },' for row in values)
        header += '\n};\n'
    for name, key in [('ECG_TRAIN_MEAN', 'training_mean'), ('ECG_TRAIN_STD', 'training_std')]:
        header += f'static const double {name}[12] = {{ ' + ', '.join(format(float(v), '.17g') for v in p[key]) + ' };\n'
    header += '#endif\n'
    OUT.mkdir(parents=True, exist_ok=True)
    config = OUT / 'ecg_preprocess_config.h'
    if not config.exists() or config.read_text() != header:
        config.write_text(header, encoding='ascii')
    for name in [m['sample_file'], m['model_file']]:
        destination = OUT / name
        if destination.exists(): assert digest(destination) == digest(OLD / name)
        else: shutil.copy2(OLD / name, destination)
    m.update(protocol=2, preprocessing_location='board', input_kind='raw_mv',
             preprocessing_sha256=sha, pipeline_id='seed44_raw_board_preprocess_v1',
             record_length=1000, preprocessing_implementation=contract,
             reference='Existing seed44 host C-model on host-preprocessed ECG; not board outputs', physical_board_test=False)
    save(OUT / 'serial_manifest.json', m)
    save(OUT / 'preprocessing_contract.json', contract)
    print(f'Raw-input manifest prepared: {OUT}', flush=True)
    return m
if __name__ == '__main__': prepare()
