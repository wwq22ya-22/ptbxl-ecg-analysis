"""Recompute U-Net cohort/metric/threshold/selection evidence without fold10."""
from pathlib import Path
import sys, json, hashlib, argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / 'src').exists(): ROOT = Path('D:/MyProject/college/ecg1')
sys.path.insert(0, str(ROOT))
from src.training.data import load_labeled_splits, _cache_fingerprint
from src.signal_processing.filters import FilterConfig
from src.models import build_model
from scripts.run_unet_teacher import UNetReadGuard

OUT = ROOT / 'outputs/unet_teacher_20261010'
C = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
POPS = ['filtered', 'unfiltered']
MODELS = ['unet1d', 'unet_encoder1d']

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def independently_search(y, p):
    grid = np.round(np.arange(10, 91) / 100, 2)
    grid[::5] = np.arange(.1, .901, .05)
    result = []
    for i in range(5):
        scores = np.array([f1_score(y[:, i], p[:, i] >= t, zero_division=0) for t in grid])
        ties = grid[np.isclose(scores, scores.max())]
        result.append(float(ties[np.argmin(abs(ties - .5))]))
    return result

def independently_compare(y, p, thresholds, expected):
    pred = p >= np.asarray(thresholds)
    all_values = []
    for i, c in enumerate(C):
        truth, guess = y[:, i].astype(bool), pred[:, i]
        tp, fp = int((truth & guess).sum()), int((~truth & guess).sum())
        fn, tn = int((truth & ~guess).sum()), int((~truth & ~guess).sum())
        numbers = [roc_auc_score(truth, p[:, i]), average_precision_score(truth, p[:, i]),
                   tp / max(tp + fp, 1), tp / max(tp + fn, 1), 2 * tp / max(2 * tp + fp + fn, 1)]
        keys = ['auroc', 'auprc', 'precision', 'recall', 'f1']
        v = expected['per_class'][c]
        np.testing.assert_allclose(numbers, [v[k] for k in keys], rtol=0, atol=1e-12)
        assert v['threshold'] == thresholds[i] and v['support'] == int(truth.sum())
        assert v['confusion_matrix'] == [[tn, fp], [fn, tp]]
        all_values.append(numbers)
    np.testing.assert_allclose(np.mean(all_values, axis=0), [expected['macro_' + k] for k in keys], rtol=0, atol=1e-12)

def training_moments(frame, cfg):
    rawcfg = {**cfg['signal_processing'], 'normalization': 'none'}
    rawcfg.pop('training_mean', None); rawcfg.pop('training_std', None)
    fp = _cache_fingerprint(frame, 100, FilterConfig.from_mapping(rawcfg))
    raw = np.load(ROOT / cfg['outputs']['cache_dir'] / f'train_full_{fp}.npy', mmap_mode='r')
    assert raw.shape == (len(frame), 12, 1000) and raw.dtype == np.float32
    # Separate sum/squared-sum calculation, not the training Welford function.
    total, squared, count = np.zeros(12), np.zeros(12), 0
    for start in range(0, len(raw), 113):
        x = np.asarray(raw[start:start+113], dtype=np.float64)
        assert np.isfinite(x).all()
        total += x.sum(axis=(0, 2)); squared += (x*x).sum(axis=(0, 2))
        count += x.shape[0] * x.shape[2]
    mean = total / count
    return mean, np.maximum(np.sqrt(squared / count - mean**2), 1e-8)

def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available()
    parser = argparse.ArgumentParser(); parser.add_argument('--partial', action='store_true')
    partial = parser.parse_args().partial
    splits = load_labeled_splits('D:/datasets/ptb-xl', 100)
    guard = UNetReadGuard(pd.concat(list(splits.values())))
    sys.addaudithook(guard.hook)
    frames = {'unfiltered': {k: splits[k] for k in ['train', 'validation']},
              'filtered': {k: splits[k][splits[k][C].sum(axis=1).gt(0)].copy() for k in ['train', 'validation']}}
    protocol = read(OUT / 'protocol.json')
    summary = read(OUT / 'summary.json') if not partial else {
        'protocol_sha256': sha(OUT / 'protocol.json'), 'fold10_evaluation': None,
        'firmware_generated': False, 'new_teacher_or_deployment_automatically_replaced': False,
        'runs': [read(p) for p in sorted(OUT.glob('*/*/result.json'))], 'fixed_teacher_references': []}
    assert summary['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert summary['fold10_evaluation'] is None and not summary['firmware_generated']
    assert not protocol['test_fold_evaluated'] and protocol['run_count'] == 12
    assert not summary['new_teacher_or_deployment_automatically_replaced']
    for p, digest in protocol['code_sha256'].items(): assert sha(p) == digest, p
    for p, digest in protocol['protected_artifacts'].items(): assert sha(p) == digest, p
    for pop in POPS:
        for part in ['train', 'validation']:
            frame = frames[pop][part]
            saved = pd.read_csv(OUT / f'{pop}_{part}_cohort.csv')
            np.testing.assert_array_equal(saved, frame[['ecg_id', 'patient_id', 'strat_fold', *C]])
            assert len(frame) == protocol['cohorts'][pop][part]
            assert frame.strat_fold.between(1, 8).all() if part == 'train' else frame.strat_fold.eq(9).all()
        assert not set(frames[pop]['train'].patient_id) & set(frames[pop]['validation'].patient_id)
    runs = summary['runs']; refs = summary['fixed_teacher_references']
    assert len(runs) > 0
    if not partial:
        assert len(runs) == 12 and len(refs) == 3
        assert {r['name'] for r in runs} == set(protocol['configs'])
        assert {r['seed'] for r in refs} == {42, 43, 44}
    moments, zero_label_profiles, logit_checks = {}, [], []
    sets = 0
    for r in [*runs, *refs]:
        path = Path(r['checkpoint']); cp = torch.load(path, map_location='cpu', weights_only=True)
        assert sha(path) == r['checkpoint_sha256'] and cp['model_type'] == r['model']
        pop = r['training_cohort']; train = frames[pop]['train']; cfg = cp['config']
        assert cfg['dataset']['require_diagnostic_label'] == (pop == 'filtered')
        assert cp['training_records'] == len(train) and cp['validation_records'] == len(frames[pop]['validation'])
        assert cfg['training']['random_seed'] == r['seed']
        assert cfg['signal_processing']['normalization'] == 'train_global'
        if pop not in moments: moments[pop] = training_moments(train, cfg)
        for key, expected in zip(['training_mean', 'training_std'], moments[pop]):
            np.testing.assert_allclose(cfg['signal_processing'][key], expected, rtol=0, atol=1e-12)
        labels = train[C].to_numpy(dtype=np.float32)
        positives = labels.sum(axis=0)
        weights = np.divide(len(labels)-positives, positives)
        np.testing.assert_allclose(cp['pos_weight'], weights, rtol=0, atol=1e-7)
        if r in runs:
            folder = OUT / r['name']; frozen_cfg = protocol['configs'][r['name']]
            assert read(folder / 'config.json') == frozen_cfg
            checked_cfg = json.loads(json.dumps(cfg))
            for key in ['training_mean', 'training_std']: checked_cfg['signal_processing'].pop(key)
            assert checked_cfg == frozen_cfg
            model = build_model(r['model']); model.load_state_dict(cp['model_state'], strict=True)
            assert sum(p.numel() for p in model.parameters()) == r['parameters'] == protocol['architectures'][r['model']]['parameters']
            history = read(folder / 'metrics' / f'training_history_{r["model"]}_full.json')
            assert cp['epoch'] == r['best_epoch'] == history['best_epoch']
            assert len(history['history']) == r['epochs_completed'] <= 30
            np.testing.assert_allclose(sum(e['seconds'] for e in history['history']), r['training_seconds'], rtol=0, atol=1e-9)
            score, epoch, stale = -float('inf'), 0, 0
            for e in history['history']:
                actual = e['validation_macro_auroc']
                if actual > score + 1e-4: score, epoch, stale = actual, e['epoch'], 0
                else: stale += 1
            assert epoch == cp['epoch'] and (len(history['history']) == 30 or stale == 8)
            np.testing.assert_allclose([cp['validation_macro_auroc'], history['best_validation_score']], score, rtol=0, atol=1e-12)
            calibration = read(folder / 'frozen_validation_thresholds.json')
            assert calibration['checkpoint_sha256'] == r['checkpoint_sha256'] and calibration['native_validation_cohort'] == pop
        else:
            folder = OUT / 'references' / r['name']
            assert r['weights_retrained'] is False and pop == 'filtered'
            model = build_model(r['model'], dropout=cfg['model']['dropout'])
            model.load_state_dict(cp['model_state'], strict=True)
        model.eval()
        for evaluation_pop in POPS:
            frame = frames[evaluation_pop]['validation']
            with np.load(folder / f'{evaluation_pop}_validation_predictions.npz', allow_pickle=False) as d:
                np.testing.assert_array_equal(d['ecg_ids'], frame.ecg_id)
                np.testing.assert_array_equal(d['targets'], frame[C])
                assert str(d['checkpoint_sha256']) == r['checkpoint_sha256']
                z, p, y, thresholds = d['logits'], d['probabilities'].astype(np.float64), d['targets'], d['thresholds']
                assert z.shape == p.shape == y.shape == (len(frame), 5) and np.isfinite(z).all()
                np.testing.assert_allclose(torch.from_numpy(z).sigmoid().numpy(), p, rtol=0, atol=0)
                fingerprint = _cache_fingerprint(frame, 100, FilterConfig.from_mapping(cfg['signal_processing']))
                signals = np.load(ROOT/cfg['outputs']['cache_dir']/f'validation_full_{fingerprint}.npy', mmap_mode='r')
                indices = np.asarray([0, len(frame)//2, len(frame)-1])
                with torch.inference_mode():
                    independently_loaded = model(torch.from_numpy(np.array(signals[indices], copy=True))).numpy()
                # Reproduce original GPU batch sizes and deterministic settings.
                # CPU is a cross-device diagnostic, not a substitute reference
                # for cuDNN float32-interface convolution (TF32 can be enabled).
                gpu_model = model.to('cuda')
                repeated = []
                with torch.inference_mode():
                    for index in indices:
                        begin = int(index//32*32); end = min(begin+32,len(frame))
                        output = gpu_model(torch.from_numpy(np.array(signals[begin:end],copy=True)).to('cuda')).cpu().numpy()
                        repeated.append(output[index-begin])
                repeated = np.asarray(repeated)
                np.testing.assert_allclose(repeated, z[indices], rtol=1e-5, atol=1e-6)
                model.to('cpu')
                logit_checks.append({'name': r['name'], 'validation_cohort': evaluation_pop, 'records': len(indices),
                                     'max_abs_cpu_gpu_difference': float(abs(independently_loaded-z[indices]).max()),
                                     'max_abs_original_batch_gpu_difference': float(abs(repeated-z[indices]).max())})
                np.testing.assert_array_equal(independently_search(y, p), thresholds)
                np.testing.assert_array_equal(thresholds, [r['validation'][evaluation_pop]['per_class'][c]['threshold'] for c in C])
                independently_compare(y, p, thresholds, r['validation'][evaluation_pop]); sets += 1
                if r in runs: np.testing.assert_array_equal(thresholds, calibration['thresholds'][evaluation_pop])
                if evaluation_pop == 'unfiltered':
                    outside = y.sum(axis=1) == 0; assert int(outside.sum()) == 37
                    flagged = p[outside] >= thresholds
                    zero_label_profiles.append({'name': r['name'], 'records': 37, 'any_positive_records': int(flagged.any(axis=1).sum()),
                                                'positive_counts': dict(zip(C, flagged.sum(axis=0).astype(int).tolist()))})
    for pop in ([] if partial else POPS):
        for model in MODELS:
            rows = [r for r in runs if r['training_cohort'] == pop and r['model'] == model]
            assert {r['seed'] for r in rows} == {42, 43, 44}
            key = lambda r: (r['validation'][pop]['macro_f1'], r['validation'][pop]['macro_auroc'])
            assert max(rows, key=key) == summary['selected_native_validation'][pop][model]
            for evaluation_pop in POPS:
                for k, v in summary['three_seed_statistics'][pop][model][evaluation_pop].items():
                    values = [r['validation'][evaluation_pop][k] for r in rows]
                    np.testing.assert_allclose([np.mean(values), np.std(values, ddof=1)], [v['mean'], v['sample_std']], rtol=0, atol=1e-12)
        for evaluation_pop in POPS:
            by_seed = {m: {r['seed']: r for r in runs if r['training_cohort'] == pop and r['model'] == m} for m in MODELS}
            differences = [by_seed['unet1d'][s]['validation'][evaluation_pop]['macro_f1']-by_seed['unet_encoder1d'][s]['validation'][evaluation_pop]['macro_f1'] for s in [42, 43, 44]]
            v = summary['paired_decoder_ablation'][pop][evaluation_pop]
            np.testing.assert_allclose(differences, v['unet_minus_encoder_f1_by_seed'], rtol=0, atol=1e-12)
            assert v['wins'] == sum(d > 0 for d in differences)
            np.testing.assert_allclose(np.mean(differences), v['mean_difference'], rtol=0, atol=1e-12)
    assert sets == 2*(len(runs)+len(refs)) and not guard.blocked
    if not partial:
        assert sets == 30
        training_guard = read(OUT / 'read_guard.json')
        assert training_guard['passed'] and not training_guard['blocked_reads']
    result = {'passed': True, 'partial': partial, 'fresh_runs': len(runs), 'fixed_reference_models': len(refs),
              'metric_sets_independently_recomputed': sets, 'threshold_sets_independently_researched': sets,
              'cohort_ids_labels_patient_isolation_verified': True, 'training_moments_cohorts': sorted(moments),
              'two_cohort_training_moments_recomputed': len(moments)==2,
              'train_only_positive_weights_recomputed': True, 'epoch_selection_verified': True,
              'epochs_and_native_cohort_selection_verified': not partial,
              'paired_seed_statistics_verified': not partial, 'frozen_sources_and_protected_artifacts_unchanged': True,
              'sampled_saved_logits_reinferred_from_checkpoints': logit_checks,
              'reinference_backend': {'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                                      'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                                      'cudnn_deterministic':torch.backends.cudnn.deterministic,
                                      'cpu_comparison':'diagnostic only; checkpoint binding verified using original GPU batches'},
              'fold10_evaluated': False, 'historical_test_independence_restored': False,
              'zero_label_validation_profiles': zero_label_profiles, 'blocked_reads': guard.blocked}
    (OUT / ('partial_audit.json' if partial else 'independent_audit.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ['zero_label_validation_profiles','sampled_saved_logits_reinferred_from_checkpoints']}, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
