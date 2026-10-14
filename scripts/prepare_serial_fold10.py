"""Prepare all eligible fold-10 ECGs for the existing seed44 serial firmware."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_stm32 import digest
from scripts.refine_student_thresholds import read, save
from scripts.verify_student_firmware import CLI
from src.config import resolve_project_path
from src.data.ptbxl import LEAD_NAMES, SUPERCLASSES, load_ecg
from src.inference.serial_samples import FOLD10_DIRECTORY, CONTRACT_KEYS, read_samples
from src.inference.board_preprocessing import load_library, preprocess_float, preprocess_int8
from src.inference.cubeai_host_reference import CubeAIHostReference
from src.signal_processing.filters import FilterConfig, preprocess_ecg
from src.training.data import load_labeled_splits

def audit_protected(protected, output=FOLD10_DIRECTORY):
    """Only allow the exact pre-existing workspace differences approved by the user."""
    approval_path = output / 'artifact_guard_approval.json'
    approval = read(approval_path) if approval_path.exists() else {'approved_differences': []}
    accepted = {item['path']: item for item in approval['approved_differences']}
    verified = []
    directory = ROOT / 'outputs/distillation/stm32_student_project_vg/MDK_Project'
    for name, expected in protected.items():
        path = Path(name)
        actual = digest(path)
        if actual == expected: continue
        item = accepted.get(name)
        if (not item or item['expected_sha256'] != expected or item['actual_sha256'] != actual
                or not path.is_relative_to(directory)
                or path.suffix.lower() not in {'.uvprojx', '.uvoptx', '.o', '.axf', '.htm'}):
            raise AssertionError(f'Unapproved protected artifact difference: {path}')
        verified.append(item)
    return verified

def write_data_attribution(output, old, count):
    (output / 'LICENSE_PTBXL.txt').write_bytes((old / 'LICENSE_PTBXL.txt').read_bytes())
    attribution = (old / 'DATA_ATTRIBUTION.txt').read_text(encoding='utf-8').split('\nserial_samples.npz')[0]
    attribution += (f'\nserial_samples.npz contains all {count} eligible records from official fold 10,\n'
                    'excluding records with no positive diagnostic superclass, as in the seed44 training cohort.\n'
                    'ecg_id and patient_id identify source records; strat_fold is 10 for every row.\n'
                    'raw_mv stores original mV waveforms; ecg stores baseline removal, bandpass\n'
                    'filtering and normalization using frozen training-set statistics.\n'
                    'targets stores NORM/MI/STTC/CD/HYP multi-label targets. reference_probabilities\n'
                    'comes from a fresh Cube.AI host C-model validation, not a physical board.\n'
                    'sample_manifest.json binds this bundle to the existing seed44 model and thresholds.\n'
                    'Historical fold-10 results have been used during development; replay is not a new blind test.\n'
                    'The data license does not extend to the ST runtime or imply endorsement by dataset authors.\n')
    (output / 'DATA_ATTRIBUTION.txt').write_text(attribution, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--resume-host-validation', action='store_true',
                        help='Complete packaging after a successful separate Cube.AI validation retry.')
    args = parser.parse_args()
    old = ROOT / 'outputs/distillation/distilled_export'
    manifest = read(old / 'serial_manifest.json')
    base = read(old / 'deployment_manifest.json')
    output = FOLD10_DIRECTORY
    final = output / 'sample_manifest.json'
    if final.exists():
        if digest(old / manifest['model_file']) != manifest['model_sha256']:
            raise ValueError('当前模型文件与清单不一致，不能复用折10样本包。')
        samples = read_samples(manifest, old, 10)
        print(f"折10样本包已准备好并通过校验，共{len(samples['ecg_id'])}条；保留现有文件，不重新生成。")
        print('请运行串口程序（不是本准备脚本）：')
        print('  板端预处理＋推理：.venv\\Scripts\\python.exe scripts/ecg_stm32_raw_viz.py --fold 10')
        print('  仅推理：.venv\\Scripts\\python.exe scripts/ecg_stm32_viz.py --fold 10')
        return
    checkpoint_path = resolve_project_path(base['source_checkpoint'], ROOT)
    assert digest(checkpoint_path) == base['source_checkpoint_sha256']
    assert digest(old / manifest['model_file']) == manifest['model_sha256']
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    assert checkpoint['sampling_rate_hz'] == 100 and tuple(checkpoint['lead_names']) == LEAD_NAMES
    assert tuple(checkpoint['superclasses']) == SUPERCLASSES
    assert checkpoint['config']['signal_processing'] == manifest['preprocessing']
    require_label = bool(checkpoint['config']['dataset'].get('require_diagnostic_label', False))
    dataset = args.dataset_root or resolve_project_path(checkpoint['config']['dataset']['root'], ROOT)
    splits = load_labeled_splits(dataset, 100, require_diagnostic_label=require_label)
    frame = splits['test'].reset_index(drop=True)
    assert set(frame['strat_fold']) == {10} and len(frame) > 0
    for other in ('train', 'validation'):
        assert set(frame['patient_id']).isdisjoint(splits[other]['patient_id'])
    targets = frame[list(SUPERCLASSES)].to_numpy(dtype=np.uint8)
    if require_label: assert np.all(targets.sum(axis=1) > 0)
    protected = read(ROOT / 'outputs/board_release/seed44_int8_stm32f407vg_preprocess/release_manifest.json')['protected_sha256']
    protected = dict(protected)
    for release in ['seed44_int8_stm32f407vg', 'seed44_int8_stm32f407vg_preprocess']:
        for p in (ROOT / 'outputs/board_release' / release).rglob('*'):
            if p.is_file(): protected[str(p)] = digest(p)
    output.mkdir(parents=True, exist_ok=True)
    if args.resume_host_validation:
        protected = read(output / 'protocol.json')['protected_sha256']
    save(output / 'protocol.json', {'model_sha256': manifest['model_sha256'],
                                  'fold': 10, 'records': len(frame), 'require_diagnostic_label': require_label,
                                  'protected_sha256': protected, 'retune_or_train': False,
                                  'evaluation_scope': 'Historical test replay; not a new independent blind test'})
    raw = np.empty((len(frame), 12, 1000), dtype=np.float32)
    prepared = np.empty_like(raw)
    saved_input = None
    saved_raw = None
    if args.resume_host_validation:
        if not (output / 'host_validation/ecg_network_val_io.npz').is_file():
            raise ValueError('Resume requires a completed Cube.AI host validation IO file.')
        saved_input = np.load(output / 'host_input.npy', mmap_mode='r')
        assert saved_input.shape == raw.shape and saved_input.dtype == np.float32
        if (output / 'raw_mv.npy').exists():
            saved_raw = np.load(output / 'raw_mv.npy', mmap_mode='r')
            assert saved_raw.shape == raw.shape and saved_raw.dtype == np.float32
    config = FilterConfig.from_mapping(manifest['preprocessing'])
    lib = load_library()
    raw_manifest = read(ROOT / 'outputs/distillation/board_preprocess_export/serial_manifest.json')
    assert lib.ecg_preprocess_contract_sha256().decode() == raw_manifest['preprocessing_sha256']
    max_error, q_errors = 0., 0
    scale = np.float32(manifest['input_scale'])
    for i, row in enumerate(frame.itertuples(index=False)):
        if saved_raw is not None:
            raw[i] = saved_raw[i]
        else:
            record = load_ecg(dataset, row.filename_lr, sampling_rate_hz=100)
            assert record.signal.shape == (1000, 12) and record.lead_names == LEAD_NAMES
            assert all(unit == 'mV' for unit in record.units)
            raw[i] = record.signal.T
        prepared[i] = saved_input[i] if saved_input is not None else preprocess_ecg(record.signal, 100, config=config).T
        c = preprocess_float(raw[i], lib)
        max_error = max(max_error, float(np.max(abs(c - prepared[i]))))
        np.testing.assert_allclose(c, prepared[i], rtol=2e-6, atol=2e-6)
        q = preprocess_int8(raw[i], lib)
        expected = np.clip(np.rint(prepared[i] / scale) + manifest['input_zero_point'], -128, 127).astype(np.int8)
        q_errors += int((q != expected).sum())
        np.testing.assert_array_equal(q, expected)
        if (i + 1) % 128 == 0 or i + 1 == len(frame):
            print(f'Prepared and checked fold-10 raw/C preprocessing: {i+1}/{len(frame)}', flush=True)
    input_file = output / 'host_input.npy'
    if saved_input is None: np.save(input_file, prepared, allow_pickle=False)
    if saved_raw is None: np.save(output / 'raw_mv.npy', raw, allow_pickle=False)
    arguments = [str(CLI), 'validate', '--model', str(old / manifest['model_file']), '--target', 'stm32f4',
                 '--name', 'ecg_network', '--c-api', 'legacy', '--compression', 'none', '--optimization', 'ram',
                 '--inputs-ch-position', 'chfirst', '--output-data-type', 'float32', '--quiet',
                 '--mode', 'host', '--io-only', '--valinput', str(input_file),
                 '--workspace', str(ROOT / 'outputs/f10host'), '--output', str(output / 'host_validation')]
    if not args.resume_host_validation:
        print('Running fresh Cube.AI host C-model validation for the full fold-10 bundle...', flush=True)
        with (output / 'host_validate.log').open('w', encoding='utf-8') as stream:
            result = subprocess.run(arguments, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW, timeout=900)
        save(output / 'host_command.json', {'arguments': arguments, 'returncode': result.returncode})
        if result.returncode:
            raise RuntimeError(f'Cube.AI host validation failed; inspect {output / "host_validate.log"}')
    with np.load(output / 'host_validation/ecg_network_val_io.npz', allow_pickle=False) as data:
        logits = data['c_outputs_1'].reshape(len(frame), 5).astype(np.float32)
        c_input = data['c_inputs_1'].reshape(prepared.shape)
        np.testing.assert_array_equal(data['m_inputs_1'].reshape(prepared.shape), prepared)
    expected_q = np.clip(np.rint(prepared / scale) + manifest['input_zero_point'], -128, 127).astype(np.int8)
    assert c_input.dtype == np.int8
    differences = np.argwhere(c_input != expected_q)
    corrected = np.unique(differences[:, 0]).tolist()
    bridge_error = 0.
    reference_details = {'CLI_automatic_quantization_different_values': len(differences),
                         'CLI_automatic_quantization_positions': differences.tolist(),
                         'corrected_record_indices': corrected,
                         'corrected_ecg_ids': frame.iloc[corrected]['ecg_id'].astype(int).tolist()}
    if corrected:
        host_base = ROOT / 'outputs/f10host'
        if args.resume_host_validation:
            retry = read(output / 'host_retry_command.json')
            assert retry['returncode'] == 0
            host_base = resolve_project_path(retry['workspace'], ROOT)
        runner = CubeAIHostReference(host_base / 'inspector_ecg_network/workspace', output)
        checks = sorted(set([0, len(frame)-1, *corrected]))
        for i in checks:
            actual = runner.logits(c_input[i])
            bridge_error = max(bridge_error, float(np.max(abs(actual - logits[i]))))
            np.testing.assert_array_equal(actual, logits[i])
        for i in corrected:
            logits[i] = runner.logits(expected_q[i])
        reference_details.update(host_C_DLL=str(runner.dll), host_C_DLL_sha256=digest(runner.dll),
                                 firmware_weight_source_sha256=digest(runner.weights_source),
                                 bridge_vs_CLI_checked_record_indices=checks,
                                 bridge_vs_CLI_max_absolute_error=bridge_error)
    reference_details['exact_firmware_quantized_input_sha256'] = hashlib.sha256(expected_q.tobytes()).hexdigest()
    reference_details['prepared_input_file_sha256'] = digest(input_file)
    reference_details['raw_input_file_sha256'] = digest(output / 'raw_mv.npy')
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    assert np.isfinite(probabilities).all()
    sample_file = output / 'serial_samples.npz'
    np.savez_compressed(sample_file, raw_mv=raw, ecg=prepared, targets=targets,
                        ecg_id=frame['ecg_id'].to_numpy(dtype=np.int64),
                        patient_id=frame['patient_id'].to_numpy(dtype=np.int64),
                        strat_fold=frame['strat_fold'].to_numpy(dtype=np.uint8),
                        reference_probabilities=probabilities)
    write_data_attribution(output, old, len(frame))
    report = {'records': len(frame), 'patients': int(frame['patient_id'].nunique()), 'fold': 10,
              'require_diagnostic_label': require_label, 'all_patient_isolation_checks_passed': True,
              'C_preprocessing_vs_python_max_absolute_error': max_error,
              'C_preprocessing_INT8_different_values': q_errors, 'final_reference_INT8_input_different_values': 0,
              'CubeAI_host_C_reference_generated': True, 'physical_board_test': False}
    report.update(reference_details)
    report['user_approved_preexisting_Keil_workspace_differences'] = audit_protected(protected, output)
    report['models_sources_firmware_releases_and_fold9_samples_unchanged'] = True
    save(output / 'preparation_validation.json', report)
    bundle = {key: manifest[key] for key in CONTRACT_KEYS}
    bundle.update(fold=10, sample_source='PTB-XL historical test fold 10', sample_count=len(frame),
                  sample_file=sample_file.name, sample_sha256=digest(sample_file),
                  require_diagnostic_label=require_label, source_checkpoint_sha256=base['source_checkpoint_sha256'],
                  reference='Cube.AI host C-model; not physical-board outputs',
                  evaluation_scope='Historical test replay; not a new independent blind test')
    save(final, bundle)
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__': main()
