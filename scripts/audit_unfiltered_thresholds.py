"""Independent saved-artifact audit for the complete-cohort threshold comparison."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.refine_student_thresholds import read, save, digest
from src.data.ptbxl import SUPERCLASSES

OUT = ROOT / 'outputs/unfiltered_threshold_refinement'


def main():
    s = read(OUT / 'summary.json')
    frozen = read(OUT / 'frozen_thresholds.json')
    assert frozen['test_probabilities_read'] is False
    sets = 0
    for r in s['models']:
        f = next(x for x in frozen['models'] if x['name'] == r['name'])
        for split in ['validation', 'test']:
            frame = pd.read_csv(OUT / f'cohort_{split}.csv')
            with np.load(OUT / r['name'] / f'{split}_probabilities.npz', allow_pickle=False) as d:
                ids, y, p = d['ecg_ids'], d['targets'], d['probabilities'].astype(np.float64)
            np.testing.assert_array_equal(ids, frame.ecg_id)
            np.testing.assert_array_equal(y, frame[list(SUPERCLASSES)])
            kept = y.sum(1) > 0
            assert int((~kept).sum()) == (37 if split == 'validation' else 40)
            if split == 'test':
                assert (OUT / r['name'] / f'{split}_probabilities.npz').stat().st_mtime >= (OUT / 'frozen_thresholds.json').stat().st_mtime
            for policy, d in r['strategies'].items():
                assert d['thresholds'] == f['strategies'][policy]['thresholds']
                assert 'test' not in f['strategies'][policy]
                cutoffs = np.asarray(d['thresholds'], dtype=np.float64)
                pred = p >= cutoffs
                if split == 'test':
                    with np.load(OUT / r['name'] / f'test_decisions_{policy}.npz', allow_pickle=False) as decisions:
                        np.testing.assert_array_equal(decisions['ecg_ids'], ids)
                        np.testing.assert_array_equal(decisions['predictions'], pred)
                        np.testing.assert_array_equal(decisions['thresholds'], cutoffs)
                    zero = pred[~kept]
                    assert d['zero_label_test']['any_positive_records'] == int(zero.any(1).sum())
                for subset, mask in [('', np.ones(len(y), dtype=bool)), ('_filtered', kept)]:
                    expected = d[split + subset]
                    values = []
                    for i, label in enumerate(SUPERCLASSES):
                        precision, recall, f1, _ = precision_recall_fscore_support(y[mask, i], pred[mask, i], average='binary', zero_division=0)
                        auroc = roc_auc_score(y[mask, i], p[mask, i])
                        c = expected['per_class'][label]
                        assert c['confusion_matrix'] == confusion_matrix(y[mask, i], pred[mask, i], labels=[0, 1]).tolist()
                        assert c['support'] == int(y[mask, i].sum())
                        numbers = [auroc, precision, recall, f1]
                        np.testing.assert_allclose(numbers, [c[k] for k in ['auroc', 'precision', 'recall', 'f1']], rtol=0, atol=1e-12)
                        values.append(numbers)
                    np.testing.assert_allclose(np.mean(values, 0), [expected[k] for k in ['macro_auroc', 'macro_precision', 'macro_recall', 'macro_f1']], rtol=0, atol=1e-12)
                    sets += 1
    for path, sha in s['protocol']['protected_sha256'].items(): assert digest(Path(path)) == sha
    parity = []
    sources = {'original_seed44_int8': ROOT / 'outputs/distillation/distilled_export',
               'ensemble_kd_seed42_int8': ROOT / 'outputs/teacher_distillation/selected_export'}
    for name, folder in sources.items():
        for split in ['validation', 'test']:
            with np.load(OUT / name / f'{split}_probabilities.npz', allow_pickle=False) as d:
                ids, p = d['ecg_ids'], d['probabilities']
            path = folder / ('int8_validation_predictions.npz' if split == 'validation' else 'int8_test_predictions.npz')
            with np.load(path, allow_pickle=False) as old:
                key, score = ('ecg_id', 'probabilities_int8') if split == 'validation' else ('ecg_ids', 'probabilities')
                positions = pd.Index(ids).get_indexer(old[key])
                assert (positions >= 0).all()
                delta = float(np.abs(p[positions] - old[score]).max())
                np.testing.assert_allclose(p[positions], old[score], rtol=0, atol=1e-7)
                parity.append({'model': name, 'split': split, 'kept_records': len(positions), 'max_probability_error': delta})
    assert sets == s['audit']['metric_sets_verified']
    report = {'models': len(s['models']), 'metric_sets_verified': sets,
              'full_cohort_ids_and_targets_verified': True, 'freeze_precedes_test_prediction_files': True,
              'all_saved_decisions_verified': True, 'original_artifacts_unchanged': True,
              'filtered_INT8_score_parity': parity}
    save(OUT / 'independent_audit.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
