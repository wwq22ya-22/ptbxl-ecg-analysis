"""Correct the limited threshold search, without changing ensemble weights."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.ptbxl import SUPERCLASSES, load_ecg
from src.training.data import load_labeled_splits
from src.training.metrics import constrained_pr_thresholds, calculate_multilabel_metrics
from src.inference.ensemble import load_ensemble
from scripts.run_cross_validation import predict_frame, write_json
from scripts.run_further_optimization import read_json, ROOT, CV_ROOT, KEYS, write_report

REFINED = ROOT / 'threshold_refinement'


def main():
    REFINED.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    frozen = read_json(ROOT / 'final_selection.json')
    if frozen['name'] != 'ensemble_equal':
        raise ValueError('This correction applies to the previously selected equal ensemble.')
    protocol = {'method': 'joint_PR_curve_MILP', 'architecture_weights': [0.5, 0.5], 'precision_floor': frozen['threshold_policy']['precision_floor'], 'recall_floor': frozen['threshold_policy']['recall_floor'], 'floor_source': 'same fold9 floors set before first ensemble test', 'threshold_fold': 9, 'test_is_reused': True, 'reason': 'The original one-multiplier search reported no feasible candidate on calibration; this checks the joint discrete constraints explicitly.'}
    write_json(REFINED / 'protocol.json', protocol)
    # Four-fold thresholds are calibrated against the same-fold ResNet reference.
    cv = []
    for fold in range(1, 5):
        ref = read_json(CV_ROOT / f'fold_{fold}/resnet_global/result.json')
        directory = ROOT / f'cv/fold_{fold}/ensemble_equal'
        with np.load(directory / 'calibration_predictions.npz') as saved:
            thresholds, solver = constrained_pr_thresholds(saved['targets'], saved['probabilities'], precision_floor=ref['calibration_f1']['macro_precision'], recall_floor=ref['calibration_f1']['macro_recall'])
            if thresholds is None:
                raise ValueError(f'No feasible calibrated candidate for CV fold {fold}: {solver}')
            calibration = calculate_multilabel_metrics(saved['targets'], saved['probabilities'], thresholds)
        with np.load(directory / 'heldout_predictions.npz') as saved:
            metrics = calculate_multilabel_metrics(saved['targets'], saved['probabilities'], thresholds)
            np.savez_compressed(REFINED / f'cv_fold_{fold}_predictions.npz', targets=saved['targets'], ecg_ids=saved['ecg_ids'], patient_ids=saved['patient_ids'], probabilities=saved['probabilities'], predictions=saved['probabilities'] >= thresholds)
        cv.append({'cv_fold': fold, 'thresholds': thresholds, 'solver': solver, 'calibration': calibration, 'heldout': metrics, 'reference': ref['heldout_f1']})
    summary = {'folds': cv, 'macro': {}, 'per_class': {}, 'f1_wins': sum(r['heldout']['macro_f1'] > r['reference']['macro_f1'] for r in cv)}
    for key in KEYS:
        values = [r['heldout'][key] for r in cv]
        summary['macro'][key] = {'mean': float(np.mean(values)), 'std': float(np.std(values)), 'delta_vs_resnet_global': float(np.mean([r['heldout'][key] - r['reference'][key] for r in cv]))}
    for c in SUPERCLASSES:
        values = [r['heldout']['per_class'][c]['f1'] for r in cv]
        summary['per_class'][c] = {'mean_f1': float(np.mean(values)), 'std_f1': float(np.std(values))}
    write_json(REFINED / 'cv_summary.json', summary)
    # Lock full-training thresholds solely from fold9 before a new fold10 pass.
    manifest_path = REFINED / 'ensemble_manifest.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
    else:
        with np.load(ROOT / 'screen/ensemble_equal/validation_predictions.npz') as saved:
            thresholds, solver = constrained_pr_thresholds(saved['targets'], saved['probabilities'], precision_floor=protocol['precision_floor'], recall_floor=protocol['recall_floor'])
            if thresholds is None:
                raise ValueError(f'No feasible full-training thresholds: {solver}')
            metrics = calculate_multilabel_metrics(saved['targets'], saved['probabilities'], thresholds)
        manifest = {**frozen, 'thresholds': thresholds, 'threshold_policy': {**frozen['threshold_policy'], 'feasible': True, 'method': 'joint_PR_curve_MILP', 'solver': solver}, 'validation_metrics': metrics, 'description': '同一已冻结50/50概率集成；用fold9精确联合约束阈值替换有限候选搜索，不改模型、权重和原有下限。'}
        write_json(manifest_path, manifest)
    test_path = REFINED / 'test_metrics.json'
    if not test_path.exists():
        components = manifest['components']
        checkpoints = [torch.load(c['checkpoint'], map_location='cpu', weights_only=True) for c in components]
        root = Path(checkpoints[0]['config']['dataset']['root'])
        frame = load_labeled_splits(root, 100)['test']
        scores = [predict_frame(checkpoint, frame, root, 'test_full') for checkpoint in checkpoints]
        average = np.average(np.asarray(scores), axis=0, weights=[c['weight'] for c in components])
        targets = frame[list(SUPERCLASSES)].to_numpy(dtype=np.uint8)
        metrics = calculate_multilabel_metrics(targets, average, manifest['thresholds'])
        metrics.update(name='ensemble_equal_joint_constraints', test_fold=10, threshold_source='fold9', test_role='reused_test_after_calibration_search_correction')
        np.savez_compressed(REFINED / 'test_predictions.npz', targets=targets, probabilities=average, ecg_ids=frame.ecg_id.to_numpy(), patient_ids=frame.patient_id.to_numpy(), predictions=average >= manifest['thresholds'])
        write_json(test_path, metrics)
    write_json(ROOT / 'current_best.json', {'type': 'probability_ensemble', 'manifest': str(manifest_path), 'test_metrics': str(test_path), 'previous_ensemble_version_preserved': str(ROOT / 'final_test.json')})
    # Verify that saved thresholds/predictions describe the actual inference API.
    root = Path(torch.load(manifest['components'][0]['checkpoint'], map_location='cpu', weights_only=True)['config']['dataset']['root'])
    validation = load_labeled_splits(root, 100)['validation']
    record = load_ecg(root, validation.iloc[0].filename_lr, sampling_rate_hz=100)
    prediction = load_ensemble(manifest_path).predict_record(record)
    actual = np.array([prediction.probabilities[c] for c in SUPERCLASSES])
    with np.load(ROOT / 'screen/ensemble_equal/validation_predictions.npz') as saved:
        np.testing.assert_allclose(actual, saved['probabilities'][0], rtol=1e-4, atol=1e-5)
    for c in SUPERCLASSES:
        assert prediction.thresholds[c] == manifest['thresholds'][list(SUPERCLASSES).index(c)]
    with np.load(REFINED / 'test_predictions.npz') as saved:
        np.testing.assert_array_equal(saved['predictions'], saved['probabilities'] >= manifest['thresholds'])
        recomputed = calculate_multilabel_metrics(saved['targets'], saved['probabilities'], manifest['thresholds'])
        for key, value in recomputed.items():
            assert value == read_json(test_path)[key]
    assert manifest_path.stat().st_mtime <= test_path.stat().st_mtime
    write_json(REFINED / 'audit.json', {'single_record_batch_and_thresholds_consistent': True, 'test_metrics_recomputed': True, 'manifest_frozen_before_test': True, 'all_cv_solver_results_optimal': all(r['solver']['optimal'] for r in cv)})
    write_report()
    print('REFINED CV', summary['macro'], 'wins', summary['f1_wins'], flush=True)
    print('REFINED TEST', {k: read_json(test_path)[k] for k in KEYS}, flush=True)


if __name__ == '__main__':
    main()
