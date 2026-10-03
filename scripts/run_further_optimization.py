"""Second-round ablations, fixed architecture ensemble, and four-fold checks."""
from __future__ import annotations

import argparse
import copy
import json
import hashlib
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import load_config
from src.data.ptbxl import SUPERCLASSES
from src.training.data import load_labeled_splits
from src.training.metrics import calculate_multilabel_metrics
from src.training.train import train_baseline, _save_checkpoint
from scripts.run_optimization import exact_thresholds, balanced_thresholds, validation_result
from scripts.run_cross_validation import predict_frame, write_json, HELDOUT_GROUPS

ROOT = PROJECT_ROOT / 'outputs/further_optimization'
REPORT = PROJECT_ROOT / 'outputs/reports/第二轮模型优化记录.md'
OLD_ROOT = PROJECT_ROOT / 'outputs/optimization'
CV_ROOT = PROJECT_ROOT / 'outputs/cross_validation'
KEYS = ('macro_auroc', 'macro_precision', 'macro_recall', 'macro_f1')
CASES = {
    'resnet_small_lr': ('resnet1d', {'learning_rate': 0.0003, 'epochs': 30, 'early_stopping_patience': 8}),
    'resnet_mixup': ('resnet1d', {'mixup_alpha': 0.2, 'mixup_probability': 0.5}),
    'resnet_mild_aug': ('resnet1d', {'augmentation_gain': 0.1, 'augmentation_noise': 0.01}),
    'resnet_asl4': ('resnet1d', {'loss': 'asl', 'asl_gamma_negative': 4, 'asl_gamma_positive': 1, 'asl_clip': 0.05}),
    'resnet_asl2': ('resnet1d', {'loss': 'asl', 'asl_gamma_negative': 2, 'asl_gamma_positive': 0, 'asl_clip': 0.05}),
    'resnet_avgmax': ('resnet_avgmax', {}),
    'inception_mixup': ('inception1d', {'mixup_alpha': 0.2, 'mixup_probability': 0.5}),
}
DESCRIPTIONS = {
    'resnet_small_lr': '学习率0.0003，最多30轮、早停8轮（训练策略包）',
    'resnet_mixup': 'Mixup alpha=0.2，每批50%概率，仅训练启用',
    'resnet_mild_aug': '每记录共享导联增益±10%，噪声为逐导联RMS的1%',
    'resnet_asl4': 'ASL negative gamma=4、positive gamma=1、clip=0.05，不叠加BCE正类权重',
    'resnet_asl2': 'ASL negative gamma=2、positive gamma=0、clip=0.05，不叠加BCE正类权重',
    'resnet_avgmax': 'ResNet特征同时做平均/最大池化，分类层输入128→256',
    'inception_mixup': 'Inception+同一Mixup方案，与原Inception比较',
    'ensemble_equal': '同一训练划分的ResNet与Inception概率固定各占50%，不搜索集成权重',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def config_for(case, directory):
    cfg = load_config(PROJECT_ROOT / 'configs/default.yaml')
    cfg['signal_processing']['normalization'] = 'train_global'
    cfg['training'].update(random_seed=42, epochs=20, early_stopping_patience=5, scheduler='none', pos_weight_power=1.0)
    cfg['model']['type'], overrides = CASES[case]
    cfg['training'].update(overrides)
    relative = directory.relative_to(PROJECT_ROOT).as_posix()
    for key in ('checkpoint_dir', 'metrics_dir', 'figure_dir', 'predictions_dir'):
        cfg['outputs'][key] = f'{relative}/{key.removesuffix("_dir")}'
    return cfg


def aligned_mean(first, second):
    """Refuse to average mismatched record orders or labels."""
    if not np.array_equal(first['ecg_ids'], second['ecg_ids']):
        raise ValueError('Ensemble ECG IDs/order differ.')
    if not np.array_equal(first['targets'], second['targets']):
        raise ValueError('Ensemble targets differ.')
    if first['probabilities'].shape != first['targets'].shape or second['probabilities'].shape != second['targets'].shape:
        raise ValueError('Ensemble probability shapes differ.')
    return 0.5 * first['probabilities'] + 0.5 * second['probabilities']


def score_prediction(truth, scores, reference):
    thresholds = exact_thresholds(truth, scores, [0.5] * 5)
    balanced, feasible = balanced_thresholds(truth, scores, thresholds, reference['macro_recall'], reference['macro_precision'])
    return {
        'thresholds_f1': thresholds,
        'thresholds_balanced': balanced,
        'calibration_f1': calculate_multilabel_metrics(truth, scores, thresholds),
        'calibration_balanced': calculate_multilabel_metrics(truth, scores, balanced),
        'threshold_policy': {'precision_floor': reference['macro_precision'], 'recall_floor': reference['macro_recall'], 'feasible': feasible},
    }


def save_predictions(path, saved, probabilities, scored=None):
    values = {'targets': saved['targets'], 'ecg_ids': saved['ecg_ids'], 'probabilities': probabilities}
    if 'patient_ids' in saved:
        values['patient_ids'] = saved['patient_ids']
    if scored:
        for policy in ('f1', 'balanced'):
            values['predictions_' + policy] = (probabilities >= scored['thresholds_' + policy]).astype(np.uint8)
    np.savez_compressed(path, **values)


def screen():
    reference = read_json(OLD_ROOT / '01_global_norm/result.json')['validation_fine']
    for case in CASES:
        directory = ROOT / 'screen' / case
        cfg = config_for(case, directory)
        result_path = directory / 'result.json'
        if result_path.exists():
            saved = read_json(result_path)
            if saved['run_config'] != cfg:
                raise ValueError(f'Configuration changed for {case}.')
            print('SKIP screen ' + case, flush=True)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / 'config.json', cfg)
        started = time.monotonic()
        print('START screen ' + case, flush=True)
        checkpoint_path = train_baseline(cfg, project_root=PROJECT_ROOT)
        r = validation_result(checkpoint_path, Path(cfg['dataset']['root']), directory)
        with np.load(directory / 'validation_predictions.npz') as saved:
            scored = score_prediction(saved['targets'], saved['probabilities'], reference)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        checkpoint.update(thresholds=scored['thresholds_f1'], validation_metrics_tuned=scored['calibration_f1'], threshold_search='exact_fold9_pr_curve')
        _save_checkpoint(checkpoint_path, checkpoint)
        r.update(scored, name=case, description=DESCRIPTIONS[case], checkpoint=str(checkpoint_path), run_config=cfg, seconds=time.monotonic()-started)
        write_json(result_path, r)
        write_report()
        print(f"DONE screen {case}: F1={r['calibration_f1']['macro_f1']:.6f}, AUC={r['calibration_f1']['macro_auroc']:.6f}", flush=True)
    directory = ROOT / 'screen/ensemble_equal'
    directory.mkdir(parents=True, exist_ok=True)
    with np.load(OLD_ROOT / '01_global_norm/validation_predictions.npz') as a, np.load(OLD_ROOT / '08_global_inception1d/validation_predictions.npz') as b:
        scores = aligned_mean(a, b)
        r = score_prediction(a['targets'], scores, reference)
        save_predictions(directory / 'validation_predictions.npz', a, scores)
    r.update(name='ensemble_equal', description=DESCRIPTIONS['ensemble_equal'], components=[read_json(OLD_ROOT / p / 'result.json')['checkpoint'] for p in ('01_global_norm', '08_global_inception1d')])
    write_json(directory / 'result.json', r)
    # Lock screening selection before examining new heldout predictions.
    singles = [read_json(ROOT / 'screen' / case / 'result.json') for case in CASES]
    eligible = [r for r in singles if r['calibration_f1']['macro_f1'] >= reference['macro_f1'] + 0.001 and r['calibration_f1']['macro_auroc'] >= reference['macro_auroc'] - 0.001]
    eligible.sort(key=lambda r: (r['calibration_f1']['macro_f1'], r['calibration_f1']['macro_auroc']), reverse=True)
    selection = {'rule': 'fold9 F1 >= full ResNetglobal +0.001 and AUROC >= reference -0.001; top two; fixed equal ensemble always checked', 'single_cases': [r['name'] for r in eligible[:2]], 'calibration_reference': reference, 'cv_cases': [r['name'] for r in eligible[:2]] + ['ensemble_equal']}
    path = ROOT / 'screen_selection.json'
    if path.exists() and read_json(path) != selection:
        raise ValueError('Locked screening selection changed.')
    write_json(path, selection)
    write_report()
    print('LOCKED CV candidates: ' + str(selection['cv_cases']), flush=True)


def run_cv():
    selection = read_json(ROOT / 'screen_selection.json')
    for fold, heldout in enumerate(HELDOUT_GROUPS, start=1):
        baseline = read_json(CV_ROOT / f'fold_{fold}/resnet_global/result.json')
        for case in selection['cv_cases']:
            directory = ROOT / f'cv/fold_{fold}' / case
            directory.mkdir(parents=True, exist_ok=True)
            result_path = directory / 'result.json'
            if result_path.exists():
                existing = read_json(result_path)
                if case != 'ensemble_equal':
                    expected = config_for(case, directory)
                    expected['dataset'] = copy.deepcopy(baseline['run_config']['dataset'])
                    expected['outputs']['cache_dir'] = 'outputs/cross_validation/cache'
                    if existing['run_config'] != expected:
                        raise ValueError(f'CV configuration changed for {fold}/{case}.')
                print(f'SKIP CV {fold}/{case}', flush=True)
                continue
            started = time.monotonic()
            print(f'START CV {fold}/4 {case}', flush=True)
            reference = baseline['calibration_f1']
            if case == 'ensemble_equal':
                a_dir = CV_ROOT / f'fold_{fold}/resnet_global'
                b_dir = CV_ROOT / f'fold_{fold}/inception_global'
                with np.load(a_dir / 'calibration_predictions.npz') as a, np.load(b_dir / 'calibration_predictions.npz') as b:
                    scores = aligned_mean(a, b)
                    r = score_prediction(a['targets'], scores, reference)
                    save_predictions(directory / 'calibration_predictions.npz', a, scores)
                with np.load(a_dir / 'heldout_predictions.npz') as a, np.load(b_dir / 'heldout_predictions.npz') as b:
                    heldout_scores = aligned_mean(a, b)
                    heldout_truth = a['targets']
                    save_predictions(directory / 'heldout_predictions.npz', a, heldout_scores, r)
                r.update(components=[read_json(p / 'result.json')['checkpoint'] for p in (a_dir, b_dir)])
            else:
                cfg = config_for(case, directory)
                cfg['dataset'] = copy.deepcopy(baseline['run_config']['dataset'])
                cfg['outputs']['cache_dir'] = 'outputs/cross_validation/cache'
                write_json(directory / 'config.json', cfg)
                dataset_root = Path(cfg['dataset']['root'])
                frames = load_labeled_splits(dataset_root, 100, train_folds=cfg['dataset']['train_folds'], validation_fold=9, test_folds=list(heldout))
                path = train_baseline(cfg, project_root=PROJECT_ROOT, dataset_root=dataset_root)
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                if checkpoint['training_records'] != len(frames['train']) or checkpoint['config']['dataset']['train_folds'] != baseline['train_folds']:
                    raise ValueError('Wrong training partition.')
                scores = predict_frame(checkpoint, frames['validation'], dataset_root, 'validation_full')
                truth = frames['validation'][list(SUPERCLASSES)].to_numpy(dtype=np.uint8)
                r = score_prediction(truth, scores, reference)
                with np.load(CV_ROOT / f'fold_{fold}/resnet_global/calibration_predictions.npz') as ref:
                    if not np.array_equal(ref['targets'], truth) or not np.array_equal(ref['ecg_ids'], frames['validation'].ecg_id):
                        raise ValueError('Calibration order differs.')
                save_predictions(directory / 'calibration_predictions.npz', {'targets': truth, 'ecg_ids': frames['validation'].ecg_id.to_numpy()}, scores)
                checkpoint.update(thresholds=r['thresholds_f1'], validation_metrics_tuned=r['calibration_f1'], threshold_search='exact_fold9_pr_curve', evaluation_role='cross_validation', cv_fold=fold)
                _save_checkpoint(path, checkpoint)
                heldout_scores = predict_frame(checkpoint, frames['test'], dataset_root, f'cv_holdout_{fold}')
                heldout_truth = frames['test'][list(SUPERCLASSES)].to_numpy(dtype=np.uint8)
                save_predictions(directory / 'heldout_predictions.npz', {'targets': heldout_truth, 'ecg_ids': frames['test'].ecg_id.to_numpy(), 'patient_ids': frames['test'].patient_id.to_numpy()}, heldout_scores, r)
                r.update(checkpoint=str(path), run_config=cfg, epoch=checkpoint['epoch'], normalization_statistics={k: checkpoint['config']['signal_processing'][k] for k in ('training_mean', 'training_std')})
            for policy in ('f1', 'balanced'):
                r['heldout_' + policy] = calculate_multilabel_metrics(heldout_truth, heldout_scores, r['thresholds_' + policy])
            r.update(name=case, cv_fold=fold, heldout_folds=list(heldout), train_folds=baseline['train_folds'], training_records=baseline['training_records'], calibration_fold=9, excluded_test_fold=10, run_dir=str(directory), seconds=time.monotonic()-started)
            write_json(result_path, r)
            summarize_cv()
            write_report()
            print(f"DONE CV {fold} {case}: F1={r['heldout_f1']['macro_f1']:.6f}, AUC={r['heldout_f1']['macro_auroc']:.6f}", flush=True)
    summarize_cv()
    write_report()


def summarize_cv():
    selection = read_json(ROOT / 'screen_selection.json')
    summary = {'models': {}, 'baseline': {}, 'completed_runs': 0, 'expected_runs': 4 * len(selection['cv_cases'])}
    for case in ['resnet_global', 'inception_global'] + selection['cv_cases']:
        paths = [CV_ROOT / f'fold_{f}' / case / 'result.json' if case in ('resnet_global', 'inception_global') else ROOT / f'cv/fold_{f}' / case / 'result.json' for f in range(1, 5)]
        runs = [read_json(p) for p in paths if p.exists()]
        if case in ('resnet_global', 'inception_global'):
            for r in runs:
                # Apply the second-round precision/recall floors to old baselines
                # as well; the first round used different original-ResNet floors.
                r['heldout_balanced'] = baseline_balanced_result(case, r['cv_fold'])
        if not runs:
            continue
        item = {'completed_folds': len(runs), 'policies': {}}
        for policy in ('f1', 'balanced'):
            policies = {'macro': {}, 'per_class': {c: {} for c in SUPERCLASSES}}
            for key in KEYS:
                values = [r['heldout_' + policy][key] for r in runs]
                policies['macro'][key] = {'mean': float(np.mean(values)), 'std': float(np.std(values)), 'values': values}
            for c in SUPERCLASSES:
                for key in ('auroc', 'precision', 'recall', 'f1'):
                    values = [r['heldout_' + policy]['per_class'][c][key] for r in runs]
                    policies['per_class'][c][key] = {'mean': float(np.mean(values)), 'std': float(np.std(values)), 'values': values}
            if case not in ('resnet_global', 'inception_global'):
                differences = [r['heldout_' + policy]['macro_f1'] - read_json(CV_ROOT / f"fold_{r['cv_fold']}/resnet_global/result.json")['heldout_' + policy]['macro_f1'] for r in runs]
                policies['paired_f1_vs_resnet_global'] = {'mean_difference': float(np.mean(differences)), 'wins': sum(d > 0 for d in differences), 'differences': differences}
            item['policies'][policy] = policies
        if case in ('resnet_global', 'inception_global'):
            summary['baseline'][case] = item
        else:
            summary['models'][case] = item
            summary['completed_runs'] += len(runs)
    write_json(ROOT / 'cv_summary.json', summary)
    return summary


@lru_cache(maxsize=8)
def baseline_balanced_result(case, fold):
    reference = read_json(CV_ROOT / f'fold_{fold}/resnet_global/result.json')
    if case == 'resnet_global':
        return reference['heldout_f1']
    directory = CV_ROOT / f'fold_{fold}' / case
    r = read_json(directory / 'result.json')
    with np.load(directory / 'calibration_predictions.npz') as calibration:
        thresholds, _ = balanced_thresholds(calibration['targets'], calibration['probabilities'], r['thresholds_f1'], reference['calibration_f1']['macro_recall'], reference['calibration_f1']['macro_precision'])
    with np.load(directory / 'heldout_predictions.npz') as saved:
        return calculate_multilabel_metrics(saved['targets'], saved['probabilities'], thresholds)


def finalize():
    summary = summarize_cv()
    if summary['completed_runs'] != summary['expected_runs']:
        raise ValueError('Finish all locked CV cases before final selection.')
    reference = summary['baseline']['resnet_global']['policies']['f1']['macro']
    eligible = []
    for name, r in summary['models'].items():
        m = r['policies']['f1']
        if (m['macro']['macro_f1']['mean'] >= reference['macro_f1']['mean'] + 0.001
                and m['paired_f1_vs_resnet_global']['wins'] >= 3
                and m['macro']['macro_auroc']['mean'] >= reference['macro_auroc']['mean'] - 0.001):
            eligible.append((m['macro']['macro_f1']['mean'], name))
    final_path = ROOT / 'final_selection.json'
    if final_path.exists():
        frozen = read_json(final_path)
    elif not eligible:
        frozen = {'name': 'retain_previous', 'description': '本轮候选未同时满足四折F1至少+0.001、胜出至少3/4折、AUROC降幅不超过0.001的采用标准；保留第一轮模型，本轮未复测fold10。'}
        write_json(final_path, frozen)
    else:
        _, name = max(eligible)
        directory = ROOT / 'screen' / name
        screen_result = read_json(directory / 'result.json')
        old_policy = read_json(OLD_ROOT / 'selection.json')['threshold_policy']['validation_metrics']
        full_reference = read_json(OLD_ROOT / '01_global_norm/result.json')['validation_fine']
        floors = {'macro_precision': max(old_policy['macro_precision'], full_reference['macro_precision']), 'macro_recall': max(old_policy['macro_recall'], full_reference['macro_recall'])}
        with np.load(directory / 'validation_predictions.npz') as saved:
            scored = score_prediction(saved['targets'], saved['probabilities'], floors)
        policy = 'balanced' if scored['threshold_policy']['feasible'] else 'f1'
        thresholds = scored['thresholds_' + policy]
        sources = screen_result['components'] if name == 'ensemble_equal' else [screen_result['checkpoint']]
        model_directory = ROOT / 'frozen_models'
        model_directory.mkdir(parents=True, exist_ok=True)
        components = []
        for index, source in enumerate(sources):
            checkpoint = torch.load(source, map_location='cpu', weights_only=True)
            path = model_directory / f"{index}_{checkpoint['model_type']}.pt"
            if len(sources) == 1:
                checkpoint.update(thresholds=thresholds, validation_metrics_tuned=scored['calibration_' + policy], threshold_search='fold9_pr_curve_with_precision_recall_floors' if policy == 'balanced' else 'fold9_pr_curve')
            _save_checkpoint(path, checkpoint)
            components.append({'checkpoint': str(path), 'source_checkpoint': source, 'weight': 1 / len(sources), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        frozen = {'name': name, 'type': 'probability_ensemble' if len(components) > 1 else 'single_model', 'superclasses': list(SUPERCLASSES), 'components': components, 'thresholds': thresholds, 'threshold_policy': scored['threshold_policy'], 'validation_metrics': scored['calibration_' + policy], 'cv_selection_rule': 'mean F1 +0.001, wins>=3/4, AUC >= reference-0.001', 'description': f'四折主结果选择{name}。全量模型seed42；最终阈值仅从fold9确定，并以第一轮最终模型和全量标准化ResNet的验证Precision/Recall较高值作下限；找到可行候选：{scored["threshold_policy"]["feasible"]}。不保证测试集也满足下限。'}
        write_json(final_path, frozen)
        if frozen['type'] == 'probability_ensemble':
            write_json(ROOT / 'ensemble_manifest.json', frozen)
    if frozen['name'] == 'retain_previous':
        write_report()
        return
    test_path = ROOT / 'final_test.json'
    if test_path.exists():
        print('SKIP frozen fold10 result', flush=True)
        write_report()
        return
    # Selection manifest and model hashes are saved before fold10 inference.
    checkpoints = [torch.load(c['checkpoint'], map_location='cpu', weights_only=True) for c in frozen['components']]
    frame = load_labeled_splits(Path(checkpoints[0]['config']['dataset']['root']), 100)['test']
    scores = []
    for component, checkpoint in zip(frozen['components'], checkpoints):
        if hashlib.sha256(Path(component['checkpoint']).read_bytes()).hexdigest() != component['sha256']:
            raise ValueError('Frozen component changed before evaluation.')
        scores.append(predict_frame(checkpoint, frame, Path(checkpoint['config']['dataset']['root']), 'test_full'))
    probabilities = np.average(np.asarray(scores), axis=0, weights=[c['weight'] for c in frozen['components']])
    targets = frame[list(SUPERCLASSES)].to_numpy(dtype=np.uint8)
    metrics = calculate_multilabel_metrics(targets, probabilities, frozen['thresholds'])
    metrics.update(name=frozen['name'], test_fold=10, threshold_source='fold9', test_role='reused_test_after_iteration', model_components=frozen['components'])
    save_predictions(ROOT / 'final_test_predictions.npz', {'targets': targets, 'ecg_ids': frame.ecg_id.to_numpy(), 'patient_ids': frame.patient_id.to_numpy()}, probabilities, {'thresholds_f1': frozen['thresholds'], 'thresholds_balanced': frozen['thresholds']})
    write_json(test_path, metrics)
    write_report()
    print(f"FINAL reused fold10 {frozen['name']}: F1={metrics['macro_f1']:.6f}, AUC={metrics['macro_auroc']:.6f}", flush=True)


def audit_and_plot():
    selection = read_json(ROOT / 'screen_selection.json')
    checks = []
    for fold, heldout in enumerate(HELDOUT_GROUPS, start=1):
        reference = read_json(CV_ROOT / f'fold_{fold}/resnet_global/result.json')
        cfg = reference['run_config']
        frames = load_labeled_splits(Path(cfg['dataset']['root']), 100, train_folds=reference['train_folds'], validation_fold=9, test_folds=list(heldout))
        patients = {k: set(v.patient_id) for k, v in frames.items()}
        assert not (patients['train'] & patients['validation'] or patients['train'] & patients['test'] or patients['validation'] & patients['test'])
        for case in selection['cv_cases']:
            directory = ROOT / f'cv/fold_{fold}' / case
            r = read_json(directory / 'result.json')
            assert r['train_folds'] == reference['train_folds']
            with np.load(directory / 'calibration_predictions.npz') as calibration, np.load(directory / 'heldout_predictions.npz') as saved:
                np.testing.assert_array_equal(calibration['ecg_ids'], frames['validation'].ecg_id)
                np.testing.assert_array_equal(calibration['targets'], frames['validation'][list(SUPERCLASSES)])
                np.testing.assert_array_equal(saved['ecg_ids'], frames['test'].ecg_id)
                np.testing.assert_array_equal(saved['targets'], frames['test'][list(SUPERCLASSES)])
                np.testing.assert_array_equal(saved['patient_ids'], frames['test'].patient_id)
                expected_thresholds = exact_thresholds(calibration['targets'], calibration['probabilities'], [0.5]*5)
                np.testing.assert_allclose(expected_thresholds, r['thresholds_f1'], atol=0, rtol=0)
                for policy in ('f1', 'balanced'):
                    recomputed = calculate_multilabel_metrics(saved['targets'], saved['probabilities'], r['thresholds_' + policy])
                    assert recomputed == r['heldout_' + policy]
                    np.testing.assert_array_equal(saved['predictions_' + policy], saved['probabilities'] >= r['thresholds_' + policy])
                if case == 'ensemble_equal':
                    for filename, current in [('calibration_predictions.npz', calibration), ('heldout_predictions.npz', saved)]:
                        with np.load(CV_ROOT / f'fold_{fold}/resnet_global' / filename) as a, np.load(CV_ROOT / f'fold_{fold}/inception_global' / filename) as b:
                            np.testing.assert_allclose(current['probabilities'], aligned_mean(a, b), rtol=0, atol=0)
                else:
                    checkpoint = torch.load(r['checkpoint'], map_location='cpu', weights_only=True)
                    assert checkpoint['config']['dataset']['train_folds'] == reference['train_folds']
                    for key in ('training_mean', 'training_std'):
                        np.testing.assert_allclose(checkpoint['config']['signal_processing'][key], reference['normalization_statistics'][key], rtol=0, atol=0)
            checks.append({'cv_fold': fold, 'case': case, 'order_labels_thresholds_metrics_verified': True, 'train_calibration_heldout_patient_overlap': 0})
    oof_counts = {}
    for case in selection['cv_cases']:
        ids = []
        for fold in range(1, 5):
            with np.load(ROOT / f'cv/fold_{fold}' / case / 'heldout_predictions.npz') as saved:
                ids.extend(saved['ecg_ids'].tolist())
        assert len(ids) == len(set(ids)) == 17418
        oof_counts[case] = len(ids)
    audit = {'checks': checks, 'oof_unique_records': oof_counts, 'gpu_amp_finite_gradients_checked': True}
    final_path = ROOT / 'final_selection.json'
    if final_path.exists() and read_json(final_path)['name'] != 'retain_previous':
        frozen = read_json(final_path)
        for c in frozen['components']:
            assert hashlib.sha256(Path(c['checkpoint']).read_bytes()).hexdigest() == c['sha256']
        assert final_path.stat().st_mtime <= (ROOT / 'final_test.json').stat().st_mtime
        with np.load(ROOT / 'final_test_predictions.npz') as saved:
            metrics = calculate_multilabel_metrics(saved['targets'], saved['probabilities'], frozen['thresholds'])
            for key, value in metrics.items():
                assert value == read_json(ROOT / 'final_test.json')[key]
        from src.data.ptbxl import load_ecg
        from src.inference.predict import load_predictor
        from src.inference.ensemble import load_ensemble
        full_frame = load_labeled_splits(Path(cfg['dataset']['root']), 100)['validation']
        record = load_ecg(Path(cfg['dataset']['root']), full_frame.iloc[0].filename_lr, sampling_rate_hz=100)
        predictor = load_ensemble(ROOT / 'ensemble_manifest.json') if frozen['type'] == 'probability_ensemble' else load_predictor(frozen['components'][0]['checkpoint'])
        prediction = predictor.predict_record(record)
        with np.load(ROOT / 'screen' / frozen['name'] / 'validation_predictions.npz') as saved:
            probability = np.array([prediction.probabilities[c] for c in SUPERCLASSES])
            difference = float(np.max(np.abs(probability - saved['probabilities'][0])))
            np.testing.assert_allclose(probability, saved['probabilities'][0], rtol=1e-4, atol=1e-5)
        audit['single_record_vs_batch_max_difference'] = difference
    write_json(ROOT / 'audit.json', audit)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = summarize_cv()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, label in zip(axes, ('Macro', 'MI', 'HYP')):
        for case, item in {**summary['baseline'], **summary['models']}.items():
            policy = item['policies']['f1']
            values = policy['macro']['macro_f1']['values'] if label == 'Macro' else policy['per_class'][label]['f1']['values']
            ax.plot(range(1, 5), values, marker='o', label=case)
        ax.set_title(label + ' F1')
        ax.set_xlabel('CV holdout group')
        ax.set_ylabel('F1')
        ax.set_xticks(range(1, 5))
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / 'cv_f1_comparison.png', dpi=180)
    plt.close(fig)
    write_report()
    print('AUDIT PASS: patient splits, normalization, 17418 OOF IDs, thresholds, metrics, ensemble alignment, frozen hashes', flush=True)


def write_report():
    lines = ['# 第二轮模型优化记录', '', '日期：2026-09-30。沿用五标签多标签任务；保留原基线、第一轮优化模型和四折模型。', '', '## 本轮协议', '',
        '全量筛选：folds 1–8训练，fold 9选择最佳AUROC轮次及阈值；固定seed42、train_global标准化、batch32、dropout0.2。除明确记录的改动外，保持第一轮最佳ResNet训练设置。每类PR曲线阈值只从fold 9确定。候选未按fold 10选择。', '',
        '预设七个训练候选，筛选规则：相对全量ResNetglobal的验证F1至少+0.001、AUROC不低于-0.001，取验证F1最高的两个单模型做四折复核；固定50/50的ResNet+Inception始终做四折比较，不搜索集成权重。筛选名单在预测新候选的留出组前锁定。Inception Mixup另与原Inception比较。', '',
        '四折复核沿用四组官方留出folds 1–2/3–4/5–6/7–8；当折其余六折训练，fold9校准；每折重新拟合训练集标准化，不混入校准或留出数据。记录主F1阈值与补充Precision/Recall约束阈值；后者以同折标准化ResNet的fold9值作下限，约束可行性在校准集判断。基线的补充阈值也按本轮同一下限重新校准，原有checkpoint与结果文件保留。', '',
        '本轮继续使用已有的内部数据与fold9，四折复核也使用此前训练池，因此属于迭代优化的内部证据，不是完全独立验证。fold10在前一轮已有结果；若本轮复测，将明确标为重复使用的测试集，不当作全新盲测。', '', '## 候选设置', '']
    for name, desc in DESCRIPTIONS.items():
        lines.append(f'- `{name}`：{desc}。')
    lines += ['', 'ASL参考[作者实现](https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py)，本项目使用FP32、mean reduction及局部no_grad；ASL候选替换加权BCE，不叠加其正类权重。Mixup在原二元标签对应的两项损失间插值，仅训练启用。', '', '## 全量训练后的fold9筛选', '', '下表是校准集结果，调过阈值，不能当成测试性能。', '', '| 方案 | AUROC | Precision | Recall | F1 | MI F1 | HYP F1 |', '|---|---:|---:|---:|---:|---:|---:|']
    for name, old in [('resnet_global', '01_global_norm'), ('inception_global', '08_global_inception1d')]:
        r = read_json(OLD_ROOT / old / 'result.json')
        append_metrics(lines, name, r['validation_fine'])
    for p in sorted((ROOT / 'screen').glob('*/result.json')):
        r = read_json(p)
        append_metrics(lines, r['name'], r['calibration_f1'])
    selection_path = ROOT / 'screen_selection.json'
    if selection_path.exists():
        lines += ['', '锁定四折候选：' + ', '.join(read_json(selection_path)['cv_cases']) + '。']
    summary_path = ROOT / 'cv_summary.json'
    if summary_path.exists():
        summary = read_json(summary_path)
        lines += ['', '## 四折留出结果', '', f"完成 {summary['completed_runs']}/{summary['expected_runs']} 个候选评估。均值按四折等权；标准差ddof=0，仅描述折间波动，未作显著性检验。", '']
        for policy in ('f1', 'balanced'):
            lines += [f'### 阈值策略：{policy}', '', '| 方案 | 完成折数 | AUROC | Precision | Recall | F1 | MI F1 | HYP F1 | F1胜出折数 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
            for name, r in {**summary['baseline'], **summary['models']}.items():
                m = r['policies'][policy]
                values = [f"{m['macro'][k]['mean']:.6f} ± {m['macro'][k]['std']:.6f}" for k in KEYS]
                values += [f"{m['per_class'][c]['f1']['mean']:.6f}" for c in ('MI', 'HYP')]
                wins = m.get('paired_f1_vs_resnet_global', {}).get('wins', '-')
                lines.append(f"| {name} | {r['completed_folds']} | " + ' | '.join(values) + f' | {wins} |')
        lines += ['', '### 每折主结果', '', '| 方案 | 折 | AUROC | Precision | Recall | F1 | MI F1 | HYP F1 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for p in sorted((ROOT / 'cv').glob('fold_*/*/result.json')):
            r = read_json(p)
            append_metrics(lines, f"{r['name']} | {r['cv_fold']}", r['heldout_f1'])
    final_path = ROOT / 'final_selection.json'
    if final_path.exists():
        selection = read_json(final_path)
        lines += ['', '## 冻结选择', '', selection['description'], '', '配置/模型清单：`outputs/further_optimization/final_selection.json`。']
    test_path = ROOT / 'final_test.json'
    if test_path.exists():
        r = read_json(test_path)
        previous = read_json(OLD_ROOT / 'final_test.json')
        lines += ['', '## 集成第一版fold10复测（该测试集此前已使用）', '', '候选和阈值在复测前已冻结；本轮只评估选中的新方案，未用fold10调阈值/集成权重。迭代过程已知道旧测试结果，不能称完全独立盲测。第一版的有限候选搜索未找到校准约束可行解，退回F1阈值；其召回率下降，完整结果保留。后续联合约束搜索修正见下一节。', '', '| 方案 | AUROC | Precision | Recall | F1 | MI F1 | HYP F1 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        append_metrics(lines, 'previous_inception_seed44', previous)
        append_metrics(lines, r['name'], r)
        lines += ['', '| 类别 | Precision | Recall | F1 | 旧模型F1 | 差值 |', '|---|---:|---:|---:|---:|---:|']
        for c in SUPERCLASSES:
            m, old = r['per_class'][c], previous['per_class'][c]
            lines.append(f"| {c} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {old['f1']:.6f} | {m['f1']-old['f1']:+.6f} |")
    refined_path = ROOT / 'threshold_refinement/test_metrics.json'
    if refined_path.exists():
        refined = read_json(refined_path)
        refined_summary = read_json(ROOT / 'threshold_refinement/cv_summary.json')
        manifest = read_json(ROOT / 'threshold_refinement/ensemble_manifest.json')
        lines += ['', '## 联合约束阈值修正：当前方案', '',
            '修正原因：原阈值辅助函数只沿单个召回权重搜索一组候选，返回“未找到可行候选”不意味着约束无解。以相同五类PR曲线建立整数规划，每类选择一个阈值，联合最大化F1并满足宏Precision/Recall下限；删除被同时更高Precision/Recall支配的点不会损失最优解。四折与全量校准均返回最优解（solver gap=0）。', '',
            f"模型和50/50权重保持已冻结值，完整保留第一版结果。全量校准下限沿用第一版设定：Precision≥{manifest['threshold_policy']['precision_floor']:.6f}、Recall≥{manifest['threshold_policy']['recall_floor']:.6f}；现校准Precision={manifest['validation_metrics']['macro_precision']:.6f}、Recall={manifest['validation_metrics']['macro_recall']:.6f}、F1={manifest['validation_metrics']['macro_f1']:.6f}，满足约束。阈值求解没有读取fold10标签。", '',
            '四折使用各折标准化ResNet校准指标作相同下限，按此方法重新求集成阈值；网络不重训。注意：修正发生在第一版复测之后，后续fold10仍属于迭代复测，不能称独立盲测。', '', '| 指标 | 标准化ResNet四折均值 | 联合约束集成四折均值 | 差值 |', '|---|---:|---:|---:|']
        for key in KEYS:
            m = refined_summary['macro'][key]
            reference_value = m['mean'] - m['delta_vs_resnet_global']
            lines.append(f"| {key} | {reference_value:.6f} | {m['mean']:.6f} ± {m['std']:.6f} | {m['delta_vs_resnet_global']:+.6f} |")
        lines += ['', f"F1胜出 {refined_summary['f1_wins']}/4 折。MI平均F1={refined_summary['per_class']['MI']['mean_f1']:.6f}；HYP平均F1={refined_summary['per_class']['HYP']['mean_f1']:.6f}。", '', '当前方案fold10复测：', '', '| 方案 | AUROC | Precision | Recall | F1 | MI F1 | HYP F1 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        previous = read_json(OLD_ROOT / 'final_test.json')
        append_metrics(lines, 'previous_inception_seed44', previous)
        append_metrics(lines, refined['name'], refined)
        lines += ['', '| 类别 | 当前F1 | 旧模型F1 | 差值 |', '|---|---:|---:|---:|']
        for c in SUPERCLASSES:
            new_f1, old_f1 = refined['per_class'][c]['f1'], previous['per_class'][c]['f1']
            lines.append(f'| {c} | {new_f1:.6f} | {old_f1:.6f} | {new_f1-old_f1:+.6f} |')
        lines += ['', '当前集成清单：`outputs/further_optimization/threshold_refinement/ensemble_manifest.json`；可用`src.inference.ensemble.load_ensemble`加载进行单记录推理。实际检查已核对单条与批量概率、阈值、预测和指标；模型默认入口未切换。复现阈值修正：`.venv/Scripts/python.exe scripts/refine_ensemble_thresholds.py`。', '']
        uncertainty_path = ROOT / 'threshold_refinement/paired_patient_bootstrap.json'
        if uncertainty_path.exists():
            uncertainty = read_json(uncertainty_path)
            low, high = uncertainty['percentile_95_interval']
            lines += [f"固定模型/阈值下按患者做{uncertainty['replicates']}次配对bootstrap（{uncertainty['patients']}名患者，seed={uncertainty['random_seed']}）：新减旧macro-F1={uncertainty['observed_difference']:+.6f}，95%百分位区间[{low:+.6f}, {high:+.6f}]，包含0。因此本次复测的微小F1差异不足以支持稳定提升的结论；该区间不计训练/选模不确定性，也不能消除测试集复用带来的偏差。", '']
        lines += ['本轮结论：七个训练候选中，较小学习率、两种Mixup、轻度增强、两种ASL和平均/最大池化均未形成超过原标准化ResNet的稳定收益；ASL4的全量验证小幅提升未在四折维持。两个原有架构的固定概率集成是主要有效办法，四折F1均提升；测试集收益较小且存在类别取舍，不承诺每类每项指标都提高。', '',
            '两套阈值各有取舍：第一版追求独立每类F1，复测F1略高但召回下降更多；当前联合约束版在fold9满足既定Precision/Recall下限，召回率更接近旧模型。当前清单按校准约束选择，没有按复测F1高低更换版本。集成共718634个参数（ResNet450885＋Inception267749），推理需运行两个网络；未测其部署延迟。', '']
    audit_path = ROOT / 'audit.json'
    if audit_path.exists():
        audit = read_json(audit_path)
        lines += ['', '## 实现与结果核对', '', f"已核对{len(audit['checks'])}个候选留出评估；患者划分无交集，每个方案覆盖17418条唯一OOF记录；记录/标签顺序、fold9阈值、预测标签和全部指标重新计算一致。新单模型逐折训练标准化与同折原ResNet训练统计量完全一致，集成概率与对应原始预测平均值完全一致。", '']
        if 'single_record_vs_batch_max_difference' in audit:
            lines += [f"冻结模型单条推理与批量概率最大差值：{audit['single_record_vs_batch_max_difference']:.8f}，检查通过。", '']
        lines += [f"![四折F1比较]({(ROOT / 'cv_f1_comparison.png').as_posix()})", '']
    checks_path = ROOT / 'validation_checks.json'
    if checks_path.exists():
        checks = read_json(checks_path)
        lines += [f"项目测试{checks['project_tests_passed']}项通过；ASL/Mixup/平均最大池化通过CUDA AMP有限梯度检查；联合约束求解器在小问题上与完整穷举结果一致。", '']
    lines += ['', '## 复现与限制', '', '运行 `.venv/Scripts/python.exe scripts/run_further_optimization.py --stage screen`，随后 `--stage cv`、`--stage final`、`--stage audit`；最后用`scripts/refine_ensemble_thresholds.py`复现联合约束修正。独立目录含配置、checkpoint、历史、校准预测和留出预测。完整结果与选择记录均保留，未自动修改应用默认模型。', '', '概率集成需要运行两个网络，增加推理计算与存储；新方案未验证外部数据、ONNX及STM32部署。未删改原有全零/多标签数据来提高指标。', '']
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    main_report = PROJECT_ROOT / 'outputs/reports/五标签模型优化记录.md'
    marker = '\n## 第二轮优化实验\n'
    if main_report.exists():
        text = main_report.read_text(encoding='utf-8').split(marker)[0]
        text += marker + '\n继续比较ASL、Mixup、轻度增强、较小学习率、平均/最大池化与两架构概率集成。实际结果、复核及限制见[第二轮模型优化记录](../../md/模型训练与优化.md)。\n'
        if summary_path.exists():
            text += '\n四折主结果（fold9校准阈值）：\n\n| 方案 | 完成折数 | 平均AUROC | 平均Precision | 平均Recall | 平均F1 | MI平均F1 | HYP平均F1 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
            for name, item in {**summary['baseline'], **summary['models']}.items():
                m = item['policies']['f1']
                values = [f"{m['macro'][k]['mean']:.6f}" for k in KEYS] + [f"{m['per_class'][c]['f1']['mean']:.6f}" for c in ('MI', 'HYP')]
                text += f"| {name} | {item['completed_folds']} | " + ' | '.join(values) + ' |\n'
        if test_path.exists():
            result = read_json(test_path)
            text += f"\n本轮冻结方案 `{result['name']}` 的fold10复测：AUROC={result['macro_auroc']:.6f}，Precision={result['macro_precision']:.6f}，Recall={result['macro_recall']:.6f}，F1={result['macro_f1']:.6f}。fold10此前已用于第一轮评估，本轮结果不能称全新盲测；候选、集成权重及阈值在复测前已冻结。\n"
        if refined_path.exists():
            text += f"\n后续用fold9联合约束求解修正阈值搜索，沿用相同精确率/召回率下限与50/50集成权重。当前方案四折平均F1={refined_summary['macro']['macro_f1']['mean']:.6f}；fold10迭代复测AUROC={refined['macro_auroc']:.6f}、Precision={refined['macro_precision']:.6f}、Recall={refined['macro_recall']:.6f}、F1={refined['macro_f1']:.6f}。第一版结果保留，详见第二轮记录；当前模型清单为`outputs/further_optimization/threshold_refinement/ensemble_manifest.json`。\n"
        main_report.parent.mkdir(parents=True, exist_ok=True)
        main_report.write_text(text, encoding='utf-8')


def append_metrics(lines, name, m):
    values = [f'{m[k]:.6f}' for k in KEYS] + [f"{m['per_class'][c]['f1']:.6f}" for c in ('MI', 'HYP')]
    lines.append(f'| {name} | ' + ' | '.join(values) + ' |')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['screen', 'cv', 'final', 'audit', 'report'], default='screen')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    protocol = {'version': 1, 'seed': 42, 'cases': CASES, 'ensemble_weights': [0.5, 0.5], 'heldout_groups': HELDOUT_GROUPS, 'threshold_fold': 9, 'test_fold_excluded_from_selection': 10}
    write_json(ROOT / 'protocol.json', protocol)
    write_report()
    if args.stage == 'screen':
        screen()
    elif args.stage == 'cv':
        run_cv()
    elif args.stage == 'final':
        finalize()
    elif args.stage == 'audit':
        audit_and_plot()
    else:
        write_report()
    print('COMPLETE stage ' + args.stage, flush=True)


if __name__ == '__main__':
    main()
