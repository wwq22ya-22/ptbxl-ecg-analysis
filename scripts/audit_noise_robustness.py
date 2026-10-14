"""Independently audit saved B predictions, split isolation and frozen selection."""
from __future__ import annotations

import datetime
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_stm32 import digest
from src.data.ptbxl import SUPERCLASSES

OUT = ROOT / 'outputs/noise_robustness_b'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_predictions(path, expected, cohort):
    data = np.load(path, allow_pickle=False)
    np.testing.assert_array_equal(data['ecg_id'], cohort.ecg_id)
    np.testing.assert_array_equal(data['targets'], cohort[list(SUPERCLASSES)].to_numpy())
    scores = data['probabilities'].astype(np.float64)
    truth = data['targets'].astype(bool)
    thresholds = data['thresholds'].astype(np.float64)
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    positive = scores >= thresholds
    aggregates = {key: [] for key in ['auroc', 'precision', 'recall', 'f1']}
    for index, name in enumerate(SUPERCLASSES):
        label, prediction = truth[:, index], positive[:, index]
        tp = int((label & prediction).sum())
        fp = int((~label & prediction).sum())
        fn = int((label & ~prediction).sum())
        tn = int((~label & ~prediction).sum())
        actual = {'auroc': float(roc_auc_score(label, scores[:, index])),
                  'precision': tp / max(tp + fp, 1),
                  'recall': tp / max(tp + fn, 1),
                  'f1': 2 * tp / max(2 * tp + fp + fn, 1)}
        saved = expected['per_class'][name]
        assert saved['confusion_matrix'] == [[tn, fp], [fn, tp]]
        assert saved['support'] == int(label.sum())
        np.testing.assert_allclose(saved['threshold'], thresholds[index], rtol=0, atol=0)
        for key, value in actual.items():
            np.testing.assert_allclose(saved[key], value, rtol=0, atol=1e-12)
            aggregates[key].append(value)
    assert expected['records'] == len(cohort)
    for key, values in aggregates.items():
        np.testing.assert_allclose(expected['macro_' + key], np.mean(values), rtol=0, atol=1e-12)
    return thresholds


def main():
    summary = read(OUT / 'summary.json')
    protocol = read(OUT / 'protocol.json')
    frozen = read(OUT / 'frozen_selection.json')
    assert frozen['protocol_sha256'] == digest(OUT / 'protocol.json')
    assert frozen['selected'] == summary['selection']
    assert frozen['test_used_for_selection'] is False
    cohorts = {split: pd.read_csv(OUT / f'cohort_{split}.csv') for split in protocol['cohort']}
    for split, frame in cohorts.items():
        assert len(frame) == protocol['cohort'][split]
        assert frame.ecg_id.is_unique
        assert (frame[list(SUPERCLASSES)].sum(axis=1) > 0).all()
    for first, second in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not set(cohorts[first].patient_id) & set(cohorts[second].patient_id)
        assert not set(cohorts[first].ecg_id) & set(cohorts[second].ecg_id)
    for name, sha in protocol['protected_artifacts'].items():
        assert digest(ROOT / name) == sha, name
    count = 0
    for item in summary['models']:
        thresholds = None
        for split in ['validation', 'test']:
            report = read(OUT / f'evaluation/{split}/{item["name"]}.json')
            assert report['checkpoint_sha256'] == digest(ROOT / item['path'])
            for condition, expected in summary[split][item['name']].items():
                assert report['metrics'][condition] == expected
                current = verify_predictions(OUT / f'evaluation/{split}/{item["name"]}_{condition}_predictions.npz',
                                             expected, cohorts[split])
                if thresholds is not None:
                    np.testing.assert_array_equal(current, thresholds)
                thresholds = current
                count += 1
    for name, report in summary['quantized'].items():
        for split in ['validation', 'test']:
            for condition, expected in report[split].items():
                current = verify_predictions(OUT / f'selected_export/{name}_{split}_{condition}_predictions.npz',
                                             expected, cohorts[split])
                np.testing.assert_array_equal(current, report['thresholds'])
                count += 1
    reference = summary['validation']['A_distilled_44']['clean']['macro_f1']
    for candidate in frozen['all_candidates']:
        metrics = summary['validation'][candidate['name']]
        score = sum(weight * metrics[condition]['macro_f1']
                    for condition, weight in [('clean', .5), ('20dB', .25), ('10dB', .25)])
        np.testing.assert_allclose(score, candidate['validation_score'], rtol=0, atol=1e-15)
        assert candidate['clean_guard_passed'] == (metrics['clean']['macro_f1'] >= reference - .01)
    eligible = [c for c in frozen['all_candidates'] if c['clean_guard_passed']]
    winner = max(eligible or frozen['all_candidates'], key=lambda c: (c['validation_score'], c['clean_validation_f1']))
    assert winner == frozen['selected']
    manifest = read(OUT / 'selected_export/deployment_manifest.json')
    assert manifest['source_checkpoint_sha256'] == digest(ROOT / winner['checkpoint'])
    assert manifest['calibration']['noise_view_mix'] == {'clean': 128, '20dB': 64, '10dB': 64}
    assert manifest['calibration']['source'] == 'PTB-XL training folds 1-8 only'
    np.testing.assert_array_equal(manifest['calibration']['ecg_ids'],
                                  cohorts['train'].ecg_id.iloc[manifest['calibration']['row_indices']])
    assert len(set(manifest['calibration']['ecg_ids'])) == 256
    np.testing.assert_array_equal(np.bincount(manifest['calibration']['selected_views'], minlength=3),
                                  [128, 64, 64])
    assert summary['quantized']['B_selected_INT8']['onnx_sha256'] == digest(OUT / 'selected_export/ptbxl_cnn1d_int8.onnx')
    protected = {name.replace('\\', '/'): sha for name, sha in protocol['protected_artifacts'].items()}
    assert summary['quantized']['A_deployed_seed44_INT8']['onnx_sha256'] == protected[
        'outputs/distillation/distilled_export/ptbxl_cnn1d_int8.onnx']
    frozen_time = datetime.datetime.fromisoformat(frozen['frozen_at']).timestamp()
    assert frozen_time <= (OUT / 'noise_views/test_snr30.json').stat().st_mtime
    report = {'audit_passed': True, 'prediction_files_recomputed': count,
              'patient_isolation_verified': True, 'no_empty_five_class_targets': True,
              'metrics_and_confusion_matrices_recomputed': True, 'thresholds_fixed_across_conditions': True,
              'validation_only_selection_recomputed': True, 'selection_frozen_before_test_noise_build': True,
              'training_only_quantization_calibration': True, 'original_artifact_hashes_unchanged': True,
              'checked_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              'environment': {'python': sys.version, 'platform': platform.platform(),
                  'packages': {name: importlib.metadata.version(name) for name in
                               ['numpy', 'scipy', 'scikit-learn', 'torch', 'onnx', 'onnxruntime']}},
              'source_sha256': {name: digest(ROOT / name) for name in
                               ['src/training/noise.py', 'scripts/run_noise_robustness.py',
                                'scripts/audit_noise_robustness.py', 'tests/test_noise_robustness.py']}}
    (OUT / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
