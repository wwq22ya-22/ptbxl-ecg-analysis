"""Independently recompute the Ribeiro study's frozen choices and metrics."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.refine_student_thresholds import digest, grids, read, save
from src.data.ptbxl import SUPERCLASSES
from src.models import build_model
from src.training.data import load_labeled_splits

OUT = ROOT / 'outputs/ribeiro_teacher'


def main():
    s = read(OUT / 'summary.json')
    protocol = read(OUT / 'protocol.json')
    frozen = read(OUT / 'frozen_selection.json')
    assert s['protocol'] == protocol and frozen['test_used'] is False
    assert protocol['test_used_for_selection'] is False
    assert digest(ROOT / 'src/models/ribeiro_resnet.py') == protocol['model_source_sha256']
    base = torch.load(ROOT / 'outputs/teacher_preprocessing/inception_time1d_P0_current_seed42/checkpoint/inception_time1d_best.pt', map_location='cpu', weights_only=True)
    frames = load_labeled_splits(Path(base['config']['dataset']['root']), 100, require_diagnostic_label=False)
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not set(frames[a].patient_id) & set(frames[b].patient_id)
    assert int((frames['train'][list(SUPERCLASSES)].sum(axis=1) > 0).sum()) == 17084
    sets = 0
    for r in s['runs']:
        assert digest(Path(r['checkpoint'])) == r['checkpoint_sha256']
        cp = torch.load(r['checkpoint'], map_location='cpu', weights_only=True)
        assert cp['training_records'] == 17084 and cp['validation_records'] == 2146
        assert cp['config']['dataset']['require_diagnostic_label'] is True
        assert cp['epoch'] == r['epoch']
        for key in ['training_mean', 'training_std']:
            np.testing.assert_allclose(cp['config']['signal_processing'][key], base['config']['signal_processing'][key], rtol=0, atol=1e-12)
        source = (OUT / r['name'] if r['model'] == 'ribeiro_resnet100'
                  else ROOT / 'outputs/unfiltered_threshold_refinement' / r['name'])
        before = next(x for x in frozen['all_validation'] if x['name'] == r['name'])
        for split in ['validation', 'test']:
            suffix = 'predictions' if r['model'] == 'ribeiro_resnet100' else 'probabilities'
            file = source / f'{split}_{suffix}.npz'
            with np.load(file, allow_pickle=False) as d:
                ids, y, p = d['ecg_ids'], d['targets'], d['probabilities'].astype(np.float64)
                if suffix == 'predictions': assert str(d['checkpoint_sha256']) == r['checkpoint_sha256']
            frame = frames[split]
            np.testing.assert_array_equal(ids, frame.ecg_id)
            np.testing.assert_array_equal(y, frame[list(SUPERCLASSES)])
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            kept = y.sum(1) > 0
            assert int((~kept).sum()) == (37 if split == 'validation' else 40)
            if split == 'test' and suffix == 'predictions':
                assert file.stat().st_mtime >= (OUT / 'frozen_selection.json').stat().st_mtime
            for cohort, mask in [('filtered', kept), ('unfiltered', np.ones(len(y), dtype=bool))]:
                row = r['evaluation'][cohort]
                original = before['evaluation'][cohort]
                assert 'test' not in original
                assert row['thresholds'] == original['thresholds']
                assert row['validation'] == original['validation']
                pred = p[mask] >= np.asarray(row['thresholds'])
                expected = row[split]
                values = []
                for i, label in enumerate(SUPERCLASSES):
                    truth = y[mask, i].astype(bool)
                    guess = pred[:, i]
                    tp, fp = int((truth & guess).sum()), int((~truth & guess).sum())
                    fn, tn = int((truth & ~guess).sum()), int((~truth & ~guess).sum())
                    numbers = [roc_auc_score(truth, p[mask, i]), average_precision_score(truth, p[mask, i]),
                               tp / max(tp + fp, 1), tp / max(tp + fn, 1), 2 * tp / max(2 * tp + fp + fn, 1)]
                    c = expected['per_class'][label]
                    assert c['support'] == int(truth.sum()) and c['threshold'] == row['thresholds'][i]
                    assert c['confusion_matrix'] == [[tn, fp], [fn, tp]]
                    keys = ['auroc', 'auprc', 'precision', 'recall', 'f1']
                    np.testing.assert_allclose(numbers, [c[k] for k in keys], rtol=0, atol=1e-12)
                    values.append(numbers)
                    if split == 'validation':
                        grid = grids()['grid_0.01']
                        scores = np.array([f1_score(truth, p[mask, i] >= t, zero_division=0) for t in grid])
                        ties = np.flatnonzero(np.isclose(scores, scores.max()))
                        t = float(grid[ties[np.argmin(abs(grid[ties] - .5))]])
                        assert row['thresholds'][i] == t
                np.testing.assert_allclose(np.mean(values, 0), [expected['macro_' + k] for k in keys], rtol=0, atol=1e-12)
                if split == 'test':
                    zero = p[~kept] >= np.asarray(row['thresholds'])
                    assert row['zero_label_test']['any_positive_records'] == int(zero.any(1).sum())
                    assert row['zero_label_test']['positive_counts'] == dict(zip(SUPERCLASSES, zero.sum(0).astype(int).tolist()))
                sets += 1
    key = lambda r: (r['evaluation']['filtered']['validation']['macro_f1'], r['evaluation']['filtered']['validation']['macro_auroc'])
    new = [r for r in frozen['all_validation'] if r['model'] == 'ribeiro_resnet100']
    assert max(new, key=key)['name'] == s['selected']['name'] == frozen['selected_new_teacher']['name']
    assert max(frozen['all_validation'], key=key)['name'] == s['validation_recommendation']['name'] == frozen['validation_recommendation']['name']
    for path, sha in protocol['protected_sha256'].items(): assert digest(Path(path)) == sha
    selected = torch.load(OUT / 'selected_teacher.pt', map_location='cpu', weights_only=True)
    original = torch.load(s['selected']['checkpoint'], map_location='cpu', weights_only=True)
    assert selected['thresholds'] == s['selected']['evaluation']['filtered']['thresholds']
    for k, v in original['model_state'].items(): torch.testing.assert_close(v, selected['model_state'][k], rtol=0, atol=0)
    assert digest(OUT / 'selected_teacher.pt') == s['selected_teacher_sha256']
    assert sum(p.numel() for p in build_model('ribeiro_resnet100').parameters()) == s['parameters']
    assert sets == 36
    report = {'models': 9, 'metric_sets_verified': sets, 'patient_isolation_verified': True,
              'cohort_ids_labels_and_train_filter_verified': True, 'frozen_train_normalization_verified': True,
              'validation_thresholds_and_selection_verified': True, 'new_test_predictions_follow_freeze': True,
              'new_model_source_unchanged': True, 'original_models_ONNX_HEX_unchanged': True,
              'selected_copy_weights_and_thresholds_verified': True}
    save(OUT / 'independent_audit.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
