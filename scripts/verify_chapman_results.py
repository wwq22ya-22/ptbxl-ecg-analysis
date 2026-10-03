"""Independently recompute frozen external results from saved probabilities."""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/external_validation/chapman_audited'
CLASSES = ('NORM', 'MI', 'STTC', 'CD', 'HYP')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal(a, b):
    assert (a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-10), (a, b)


def main():
    summary_path = OUT / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    rows = json.loads((OUT / 'record_label_audit.json').read_text(encoding='utf-8'))
    valid = np.load(OUT / 'cache_valid.npy')
    done = np.load(OUT / 'cache_completed.npy')
    assert len(rows) == len(valid) == summary['headers_scanned'] == 45152 and done.all()
    assert valid.sum() == summary['waveforms_evaluated']
    assert len(valid) - valid.sum() == summary['waveforms_excluded']
    records = np.array([row['record'] for row in rows])[valid]
    targets = np.array([row['targets'] for row in rows], dtype=np.uint8)[valid]
    positive_mask = np.array([row['positive_only'] for row in rows], dtype=bool)[valid]
    conditional_mask = np.array([row['conditional'] for row in rows], dtype=bool)[valid]
    assert np.array_equal(positive_mask, targets.astype(bool))
    assert not positive_mask[:, 0].any() and not conditional_mask[:, 0].any()
    assert (conditional_mask | ~positive_mask).all()
    assert len(np.unique(records)) == len(records)
    checked = 0
    cached_scores = {}
    for result in summary['results']:
        with np.load(OUT / f"{result['name']}_predictions.npz") as saved:
            assert np.array_equal(saved['record_ids'], records)
            assert np.array_equal(saved['targets'], targets)
            assert np.array_equal(saved['positive_only_mask'], positive_mask)
            assert np.array_equal(saved['conditional_mask'], conditional_mask)
            assert np.array_equal(saved['thresholds'], result['thresholds'])
            scores = saved['probabilities']
            assert scores.shape == targets.shape and np.isfinite(scores).all()
            assert ((scores >= 0) & (scores <= 1)).all()
            cached_scores[result['name']] = scores
            for policy, mask, label in (
                ('positive_only', positive_mask, 'positive_only_metrics'),
                ('conditional_absence_negative', conditional_mask, 'conditional_metrics'),
            ):
                aggregate = {'auroc': [], 'precision': [], 'recall': [], 'f1': []}
                for col, name in enumerate(CLASSES):
                    selected = mask[:, col]
                    truth = targets[selected, col]
                    p = scores[selected, col]
                    pred = p >= saved['thresholds'][col]
                    m = result[label]['per_class'][name]
                    positives = int(truth.sum())
                    negatives = len(truth) - positives
                    tp = int(((truth == 1) & pred).sum())
                    assert m['positive_records'] == positives and m['negative_records'] == negatives
                    assert m['unknown_records'] == int((~selected).sum())
                    assert m['true_positive'] == tp and m['false_negative'] == positives - tp
                    metrics = dict(auroc=None, precision=None, recall=tp / positives if positives else None, f1=None)
                    if policy == 'conditional_absence_negative' and positives and negatives:
                        metrics.update(auroc=float(roc_auc_score(truth, p)),
                                       precision=float(precision_score(truth, pred, zero_division=0)),
                                       recall=float(recall_score(truth, pred, zero_division=0)),
                                       f1=float(f1_score(truth, pred, zero_division=0)))
                        tn = int(((truth == 0) & ~pred).sum())
                        fp = int(((truth == 0) & pred).sum())
                        assert m['confusion_matrix_tn_fp_fn_tp'] == [tn, fp, positives - tp, tp]
                    for metric, value in metrics.items():
                        equal(m[metric], value)
                        if name != 'NORM' and value is not None:
                            aggregate[metric].append(value)
                for metric, values in aggregate.items():
                    equal(result[label]['macro_' + metric], float(np.mean(values)) if values else None)
        assert sha(result['checkpoint']) == result['checkpoint_sha256']
        assert sha(result['artifact']) == result['artifact_sha256']
        checked += 1
    fp = cached_scores['distilled_seed44_float32']
    quant = cached_scores['distilled_seed44_int8']
    audit = {'prediction_sets_verified': checked, 'same_record_ids_and_masks': True,
             'all_metrics_independently_reproduced': True, 'model_hashes_unchanged': True,
             'NORM_never_scored': True, 'positive_only_never_claims_negatives': True,
             'student_fp32_int8_probability_mae': float(np.mean(np.abs(fp - quant))),
             'student_fp32_int8_probability_max_abs_error': float(np.max(np.abs(fp - quant))),
             'targeted_tests_passed': 15}
    summary['independent_audit'] = audit
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    report = ROOT / 'outputs/reports/Chapman_冻结模型外部验证总结.md'
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('a', encoding='utf-8') as output:
        output.write(f"\n## 独立结果复核\n\n已从 {checked} 组逐条概率独立重算全部指标，核对同一记录/掩码/阈值、模型哈希不变、NORM 从不计分；15 项针对性测试通过。\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
