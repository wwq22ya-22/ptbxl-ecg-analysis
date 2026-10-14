"""Controlled three-seed distillation from the new frozen ensemble teacher."""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_stm32 import digest, get_split
from scripts.run_diagnostic_only import predict
from scripts.run_distillation import analysis, command, save
from src.data.ptbxl import SUPERCLASSES
from src.models import build_model
from src.training.data import load_labeled_splits
from src.training.metrics import calculate_multilabel_metrics
from src.training.train import train_baseline

OUT = ROOT / 'outputs/teacher_distillation'
TEACHER = ROOT / 'outputs/teacher_preprocessing/ensemble_followup/recommended_teacher.pt'
LOGITS = TEACHER.parent / 'teacher_training_logits.npz'
KEYS = ['macro_auroc', 'macro_precision', 'macro_recall', 'macro_f1']


def status(message):
    print(message, flush=True)
    save(OUT / 'status.json', {'stage': message, 'time': time.strftime('%Y-%m-%d %H:%M:%S')})


def measured(path, frame, split, folder):
    scores, cp = predict(path, frame, frame, split)
    truth = frame[list(SUPERCLASSES)].to_numpy(dtype=np.uint8)
    thresholds = np.asarray(cp['thresholds'], dtype=float)
    metrics = calculate_multilabel_metrics(truth, scores, thresholds)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f'{split}_predictions.npz', ecg_ids=frame.ecg_id.to_numpy(),
                        targets=truth, probabilities=scores, thresholds=thresholds)
    save(folder / f'{split}_metrics.json', metrics)
    return metrics


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'summary.json').exists():
        status('Completed summary already exists; no retraining or artifact replacement')
        return
    torch.set_num_threads(8)
    original = ROOT / 'outputs/distillation/distilled_seed44/checkpoint/cnn1d_best.pt'
    reference = torch.load(original, map_location='cpu', weights_only=True)
    cfg = copy.deepcopy(reference['config'])
    for name in ['training_mean', 'training_std']:
        cfg['signal_processing'].pop(name, None)
    selected = load_labeled_splits(Path(cfg['dataset']['root']), 100, require_diagnostic_label=True)
    assert {k: len(v) for k, v in selected.items()} == {'train': 17084, 'validation': 2146, 'test': 2158}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not set(selected[a].patient_id) & set(selected[b].patient_id)
    teacher_sha = digest(TEACHER)
    with np.load(LOGITS, allow_pickle=False) as data:
        np.testing.assert_array_equal(data['ecg_ids'], selected['train'].ecg_id.to_numpy())
        assert str(data['teacher_sha256']) == teacher_sha
        assert data['logits'].shape == (17084, 5) and np.isfinite(data['logits']).all()
        np.testing.assert_array_equal(data['superclasses'], SUPERCLASSES)
    teacher = torch.load(TEACHER, map_location='cpu', weights_only=True)
    for stat in ['training_mean', 'training_std']:
        np.testing.assert_allclose(reference['config']['signal_processing'][stat],
                                   teacher['config']['signal_processing'][stat], rtol=0, atol=1e-12)
    protected_paths = [original, ROOT / 'outputs/optimization/optimized_best.pt',
                       ROOT / 'outputs/distillation/distilled_export/ptbxl_cnn1d_int8.onnx',
                       ROOT / 'outputs/board_release/seed44_int8_stm32f407ve/ECG_Student_INT8_STM32F407VE.hex', TEACHER]
    protected = {str(p): digest(p) for p in protected_paths}
    controls = []
    for arm in ['baseline', 'distilled']:
        for seed in [42, 43, 44]:
            path = ROOT / f'outputs/distillation/{arm}_seed{seed}/checkpoint/cnn1d_best.pt'
            cp = torch.load(path, map_location='cpu', weights_only=True)
            assert cp['training_records'] == 17084 and cp['validation_records'] == 2146
            for key in ['dataset', 'model']:
                assert cp['config'][key] == reference['config'][key]
            for stat in ['training_mean', 'training_std']:
                np.testing.assert_allclose(cp['config']['signal_processing'][stat],
                                           reference['config']['signal_processing'][stat], rtol=0, atol=1e-12)
            expected = copy.deepcopy(reference['config']['training'])
            actual = copy.deepcopy(cp['config']['training'])
            for value in [expected, actual]:
                value.pop('random_seed', None)
                value.pop('distillation', None)
            assert actual == expected
            assert cp['config']['training']['random_seed'] == seed
            if arm == 'distilled':
                assert cp['config']['training']['distillation']['alpha'] == .5
                assert cp['config']['training']['distillation']['temperature'] == 2
            protected[str(path)] = digest(path)
            controls.append({'arm': arm, 'seed': seed, 'checkpoint': str(path), 'sha256': digest(path), 'epoch': cp['epoch']})
    protocol = {'date': '2026-10-07', 'seeds': [42, 43, 44], 'teacher': str(TEACHER),
                'teacher_sha256': teacher_sha, 'logits_sha256': digest(LOGITS),
                'student_parameters': sum(p.numel() for p in build_model('cnn1d').parameters()),
                'training': cfg['training'], 'alpha': .5, 'temperature': 2,
                'preprocessing': cfg['signal_processing'], 'cohort': {k: len(v) for k, v in selected.items()},
                'selection': 'new student FP32 validation macro F1 then AUROC; freeze before test',
                'deployment_policy': 'compare selected new INT8 versus original INT8 validation macro F1 then AUROC',
                'thresholds': 'original training AMP validation coarse F1 grid; preserved for FP32 and INT8',
                'test_used_for_selection': False, 'test_previously_reused': True,
                'teacher_ensemble_proposed_after_parent_test_review': True,
                'controls_reused_after_config_and_normalization_checks': controls, 'protected_artifacts': protected}
    protocol['training'].pop('distillation', None)
    protocol_file = OUT / 'protocol.json'
    if protocol_file.exists():
        assert json.loads(protocol_file.read_text(encoding='utf-8')) == protocol
    else:
        save(protocol_file, protocol)
    runs = controls.copy()
    for seed in [42, 43, 44]:
        trial = copy.deepcopy(cfg)
        trial['training']['random_seed'] = seed
        trial['training']['distillation'] = {'alpha': .5, 'temperature': 2,
            'logits_path': str(LOGITS), 'teacher_checkpoint': str(TEACHER), 'teacher_sha256': teacher_sha}
        folder = OUT / f'ensemble_kd_seed{seed}'
        for key, sub in [('checkpoint_dir', 'checkpoint'), ('metrics_dir', 'metrics'),
                         ('figure_dir', 'figure'), ('predictions_dir', 'predictions')]:
            trial['outputs'][key] = str((folder / sub).relative_to(ROOT))
        save(folder / 'config.json', trial)
        path = folder / 'checkpoint/cnn1d_best.pt'
        history = folder / 'metrics/training_history_cnn1d_full.json'
        if path.exists() and not history.exists():
            raise RuntimeError(f'Partial run requires inspection: {folder}')
        if not history.exists():
            status(f'Training ensemble KD seed {seed}; alpha .5, T 2, maximum 30 epochs')
            train_baseline(trial, project_root=ROOT, model_type='cnn1d')
        cp = torch.load(path, map_location='cpu', weights_only=True)
        assert cp['training_records'] == 17084 and cp['validation_records'] == 2146
        for stat in ['training_mean', 'training_std']:
            np.testing.assert_allclose(cp['config']['signal_processing'][stat],
                                       reference['config']['signal_processing'][stat], rtol=0, atol=1e-12)
        runs.append({'arm': 'ensemble_kd', 'seed': seed, 'checkpoint': str(path), 'epoch': cp['epoch'],
                     'sha256': digest(path)})
    status('All three new students trained; measuring all nine validation sets without changing weights or thresholds')
    for r in runs:
        r['validation'] = measured(Path(r['checkpoint']), selected['validation'], 'validation',
                                    OUT / 'comparison' / f'{r["arm"]}_seed{r["seed"]}')
    chosen = max([r for r in runs if r['arm'] == 'ensemble_kd'],
                 key=lambda r: (r['validation']['macro_f1'], r['validation']['macro_auroc']))
    save(OUT / 'frozen_selection.json', {'selected': chosen, 'test_used': False})
    export = OUT / 'selected_export'
    export.mkdir(exist_ok=True)
    status(f'Validation-selected seed {chosen["seed"]}; exporting ONNX and quantizing on training samples only')
    if not (export / 'deployment_manifest.json').exists():
        command([sys.executable, ROOT / 'scripts/prepare_stm32.py', '--checkpoint', chosen['checkpoint'],
                 '--output-dir', export], export / 'export.log')
    if not (export / 'int8_validation_report.json').exists():
        command([sys.executable, ROOT / 'scripts/quantize_cubeai.py', '--export-dir', export], export / 'quantization.log')
    quant = json.loads((export / 'int8_validation_report.json').read_text(encoding='utf-8'))
    old_quant = json.loads((ROOT / 'outputs/distillation/distilled_export/int8_validation_report.json').read_text(encoding='utf-8'))
    assert quant['calibration']['ecg_ids'] == old_quant['calibration']['ecg_ids']
    recommendation = 'ensemble_kd' if (quant['int8']['macro_f1'], quant['int8']['macro_auroc']) > (
        old_quant['int8']['macro_f1'], old_quant['int8']['macro_auroc']) else 'original_seed44'
    save(OUT / 'frozen_deployment_selection.json', {'recommendation': recommendation,
         'new_validation': quant['int8'], 'original_validation': old_quant['int8'], 'test_used': False})
    status('Cube.AI resource analysis; validation recommendation frozen before test measurements')
    analyses = {v: analysis(export, v) for v in ['float32', 'int8']}
    for r in runs:
        status(f'Test measurement {r["arm"]} seed {r["seed"]} with fixed validation thresholds')
        r['test'] = measured(Path(r['checkpoint']), selected['test'], 'test',
                              OUT / 'comparison' / f'{r["arm"]}_seed{r["seed"]}')
    import onnxruntime as ort
    from scripts.quantize_cubeai import probabilities
    cp = torch.load(chosen['checkpoint'], map_location='cpu', weights_only=True)
    array, frame = get_split('test', cp)
    truth = frame[list(SUPERCLASSES)].to_numpy(dtype=np.uint8)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    tests = {}
    for version in ['float32', 'int8']:
        session = ort.InferenceSession(str(export / f'ptbxl_cnn1d_{version}.onnx'), options,
                                       providers=['CPUExecutionProvider'])
        scores = probabilities(session, array)
        tests[version] = calculate_multilabel_metrics(truth, scores, cp['thresholds'])
        np.savez_compressed(export / f'{version}_test_predictions.npz', ecg_ids=frame.ecg_id.to_numpy(),
                            targets=truth, probabilities=scores, thresholds=np.asarray(cp['thresholds']))
    for path, sha in protected.items():
        assert digest(Path(path)) == sha, f'Protected artifact changed: {path}'
    for r in runs:
        assert digest(Path(r['checkpoint'])) == r['sha256']
    save(OUT / 'summary.json', {'protocol': protocol, 'runs': runs, 'selected_seed': chosen['seed'],
         'quantization': quant, 'analysis': analyses, 'test_export': tests,
         'original_int8_validation': old_quant['int8'], 'deployment_recommendation': recommendation,
         'protected_artifacts_unchanged': True, 'test_used_for_selection': False})
    status('Completed three new students, nine-model comparison, INT8 and Cube.AI analysis')


if __name__ == '__main__':
    main()
