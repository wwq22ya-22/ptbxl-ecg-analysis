"""Validation-only 0.05 versus 0.01 threshold ablation on frozen student ONNX scores."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.ptbxl import SUPERCLASSES
from src.training.metrics import calculate_multilabel_metrics, optimize_thresholds

OUT = ROOT / 'outputs/student_threshold_refinement'
KEYS = ['macro_auroc', 'macro_precision', 'macro_recall', 'macro_f1']
EXPORTS = {'original_seed44': ROOT / 'outputs/distillation/distilled_export',
           'ensemble_kd_seed42': ROOT / 'outputs/teacher_distillation/selected_export'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def grids():
    coarse = np.arange(.10, .901, .05)
    fine = np.round(np.arange(10, 91) / 100, 2)
    # Preserve the old cutoff convention exactly, including >0.5 when the
    # stored cutoff is 0.5000000000000001. INT8 may produce exact 0.5 scores.
    fine[::5] = coarse
    np.testing.assert_array_equal(fine[::5], coarse)
    return {'grid_0.05': coarse, 'grid_0.01': fine}


def tune(y, p, grid):
    assert y.shape == p.shape and y.shape[1] == 5
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    predictions = p[:, :, None] >= grid[None, None, :]
    tp = (predictions & y.astype(bool)[:, :, None]).sum(0)
    fp = (predictions & ~y.astype(bool)[:, :, None]).sum(0)
    fn = (~predictions & y.astype(bool)[:, :, None]).sum(0)
    denom = 2 * tp + fp + fn
    scores = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom != 0)
    thresholds = []
    for i in range(5):
        candidates = np.flatnonzero(np.isclose(scores[i], scores[i].max()))
        selected = candidates[np.argmin(np.abs(grid[candidates] - .5))]
        thresholds.append(float(grid[selected]) if y[:, i].sum() else .5)
        # Independent scalar sklearn verification across all candidates.
        expected = np.array([f1_score(y[:, i], p[:, i] >= t, zero_division=0) for t in grid])
        np.testing.assert_allclose(expected, scores[i], atol=1e-12, rtol=0)
    return thresholds


def check_records(data, frame, id_key):
    np.testing.assert_array_equal(data[id_key], frame.ecg_id.to_numpy())
    np.testing.assert_array_equal(data['targets'], frame[list(SUPERCLASSES)].to_numpy(dtype=np.uint8))
    assert (data['targets'].sum(1) > 0).all()


def audit_metrics(y, p, t, metrics):
    # Match the shared evaluator's float64 probability/cutoff comparisons.
    p = np.asarray(p, dtype=np.float64)
    precision, recall, f1 = [], [], []
    for i, label in enumerate(SUPERCLASSES):
        a, b = p[:, i] >= t[i], y[:, i].astype(bool)
        tp, fp = int((a & b).sum()), int((a & ~b).sum())
        fn, tn = int((~a & b).sum()), int((~a & ~b).sum())
        values = [tp / max(tp + fp, 1), tp / max(tp + fn, 1), 2 * tp / max(2 * tp + fp + fn, 1)]
        c = metrics['per_class'][label]
        assert c['confusion_matrix'] == [[tn, fp], [fn, tp]]
        for k, value in zip(['precision', 'recall', 'f1'], values):
            np.testing.assert_allclose(c[k], value, atol=1e-12, rtol=0)
        precision.append(values[0]); recall.append(values[1]); f1.append(values[2])
    for k, values in zip(['macro_precision', 'macro_recall', 'macro_f1'], [precision, recall, f1]):
        np.testing.assert_allclose(metrics[k], np.mean(values), atol=1e-12, rtol=0)


def section(path, body, marker='student-threshold-refinement'):
    start, end = f'<!-- {marker}:start -->', f'<!-- {marker}:end -->'
    text = path.read_text(encoding='utf-8')
    replacement = start + '\n' + body + '\n' + end
    if start in text:
        left, rest = text.split(start, 1)
        _, right = rest.split(end, 1)
        text = left + replacement + right
    else:
        text = text.rstrip() + '\n\n' + replacement + '\n'
    path.write_text(text, encoding='utf-8')


def report(summary):
    results = summary['models']
    lines = ['## 2026-10-08：学生阈值步长 0.05 → 0.01 消融', '',
             '模型权重、预测概率及量化方式保持固定，仅用验证折 9 的 2146 条记录，按每类 F1 选择阈值。搜索范围仍为 0.10–0.90，步长从 0.05 缩小到 0.01；平局优先接近 0.5，与原流程一致。0.01 网格严格包含原 0.05 网格，保留原浮点阈值比较语义。', '',
             '比较原 seed44 与新集成教师蒸馏 seed42 的 ONNX FP32 / INT8。每个版本按自身验证概率独立选阈值，先保存全部冻结阈值及验证选择，再读取测试预测。原阈值来自训练期间 AMP 验证；“重搜 0.05”用于区分推理版本重新校准收益与真正的步长收益。', '',
             '无需重新训练、量化或转换模型。AUROC 由概率决定，不受阈值影响；Precision、Recall、F1 随阈值改变，不能保证同时增加。测试折 10 已被多次使用，本次是探索性复核。', '',
             '### 整体指标，单位 %', '',
             '| 模型 | 阈值策略 | 验证 F1 | 测试 AUROC | Precision | Recall | F1 |',
             '|---|---|---:|---:|---:|---:|---:|']
    names = {'original_seed44': '原 seed44', 'ensemble_kd_seed42': '新蒸馏 seed42'}
    strategies = {'original': '原冻结阈值', 'grid_0.05': '重搜 0.05', 'grid_0.01': '细搜 0.01'}
    for name, r in results.items():
        label = names[r['model']] + ' ' + r['version']
        for strategy, data in r['strategies'].items():
            lines.append(f'| {label} | {strategies[strategy]} | {data["validation"]["macro_f1"]*100:.3f} | ' +
                         ' | '.join(f'{data["test"][k]*100:.3f}' for k in KEYS) + ' |')
    lines += ['', '### INT8 步长收益与选择', '']
    for name, r in results.items():
        if r['version'] != 'int8':
            continue
        a, b, c = [r['strategies'][k] for k in ['original', 'grid_0.05', 'grid_0.01']]
        lines.append(f'{names[r["model"]]}：0.01 相对重搜 0.05，验证 F1 {(c["validation"]["macro_f1"]-b["validation"]["macro_f1"])*100:+.3f} 个百分点、测试 F1 {(c["test"]["macro_f1"]-b["test"]["macro_f1"])*100:+.3f} 个百分点；相对原冻结阈值，测试 F1 {(c["test"]["macro_f1"]-a["test"]["macro_f1"])*100:+.3f} 个百分点。')
        lines.append('')
    selected = summary['frozen_validation_choice']
    r = results[selected['model']]
    chosen = r['strategies'][selected['strategy']]
    lines += [f'只按 INT8 验证宏 F1、再 AUROC、最后较少改动选择候选：**{names[r["model"]]} INT8，{strategies[selected["strategy"]]}**。该选择已在测试预测读取前冻结，不能根据测试 F1 改选。', '',
              f'该候选测试 Precision {chosen["test"]["macro_precision"]*100:.3f}%、Recall {chosen["test"]["macro_recall"]*100:.3f}%、F1 {chosen["test"]["macro_f1"]*100:.3f}%；HYP Recall {chosen["test"]["per_class"]["HYP"]["recall"]*100:.3f}%。改善幅度较小，且原 seed44 FP32 的测试 F1 在细搜后略降，说明验证最优阈值不能保证测试改善。', '',
              '### 细网格逐类阈值', '', '| 模型 | NORM | MI | STTC | CD | HYP |', '|---|---:|---:|---:|---:|---:|']
    for r in results.values():
        lines.append(f'| {names[r["model"]]} {r["version"]} | ' +
                     ' | '.join(f'{t:.2f}' for t in r['strategies']['grid_0.01']['thresholds']) + ' |')
    lines += ['', '阈值是五个独立概率的判断门槛，不改变多标签输出。已生成独立的 `outputs/student_threshold_refinement/frozen_thresholds.json` 和 `candidate_thresholds.json`，没有修改原 checkpoint、ONNX、部署 manifest、串口配置或 HEX。若后续采用，电脑端和板端阈值需同步更新；模型权重、激活和 MACC 保持不变。', '',
              '所有预测 ID / 标签与筛选后官方验证、测试清单一致，患者隔离检查通过；细网格包含粗网格，逐候选 F1 由独立 sklearn 标量计算复核；整体与逐类混淆矩阵复算一致，原模型 / 固件哈希保持不变。', '',
              '配置、网格、冻结阈值、逐条决策与全部指标保存在 `outputs/student_threshold_refinement/`。复现：`.venv/Scripts/python.exe scripts/refine_student_thresholds.py`。']
    section(ROOT / 'md/模型训练与优化.md', '\n'.join(lines))
    detailed = lines[:]
    detailed += ['', '### 每个模型、每组阈值的逐类测试指标，单位 %', '',
                 '| 模型 | 策略 | 类别 | 阳性数 | 阈值 | AUROC | Precision | Recall | F1 |',
                 '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in results.values():
        for strategy, data in r['strategies'].items():
            for label in SUPERCLASSES:
                c = data['test']['per_class'][label]
                detailed.append(f'| {names[r["model"]]} {r["version"]} | {strategies[strategy]} | {label} | {c["support"]} | {c["threshold"]:.2f} | ' +
                                ' | '.join(f'{c[k]*100:.3f}' for k in ['auroc', 'precision', 'recall', 'f1']) + ' |')
    section(ROOT / 'md/各模型整体与逐类指标总表.md', '\n'.join(detailed))
    section(ROOT / 'md/README.md', '2026-10-08 已完成原 seed44、新蒸馏 seed42 的 FP32 / INT8 阈值细搜（0.05 → 0.01）。权重与固件保留，验证选择冻结后才复算测试指标；整体 / 逐类结果见训练文档和指标总表，完整记录位于 `outputs/student_threshold_refinement/`。')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'summary.json').exists():
        report(read(OUT / 'summary.json'))
        print('Existing completed experiment reported; no threshold retuning.')
        return
    frames = {split: pd.read_csv(ROOT / f'outputs/teacher_preprocessing/cohort_{split}.csv')
              for split in ['validation', 'test']}
    assert len(frames['validation']) == 2146 and len(frames['test']) == 2158
    assert not set(frames['validation'].patient_id) & set(frames['test'].patient_id)
    grid_map = grids()
    protected = {}
    for folder in EXPORTS.values():
        manifest = read(folder / 'deployment_manifest.json')
        for path in [folder / 'ptbxl_cnn1d_float32.onnx', folder / 'ptbxl_cnn1d_int8.onnx',
                     folder / 'deployment_manifest.json', Path(manifest['source_checkpoint']),
                     folder / 'int8_validation_predictions.npz',
                     folder / 'float32_test_predictions.npz', folder / 'int8_test_predictions.npz']:
            protected[str(path)] = digest(path)
    firmware = ROOT / 'outputs/board_release/seed44_int8_stm32f407ve/ECG_Student_INT8_STM32F407VE.hex'
    protected[str(firmware)] = digest(firmware)
    protocol = {'date': '2026-10-08', 'scope': 'frozen original seed44 and ensemble KD seed42 ONNX FP32/INT8',
                'grid_range': [.1, .9], 'steps': [.05, .01], 'grid_values': {k: v.tolist() for k, v in grid_map.items()},
                'selection': 'per-class validation F1, ties closest to 0.5; no test tuning',
                'candidate_policy': 'INT8 validation macro F1 then AUROC then prefer original/coarser policy',
                'thresholds_fit_on': 'fold 9 only', 'test_previously_reused': True,
                'cohort': {'validation': 2146, 'test': 2158}, 'protected_sha256': protected}
    save(OUT / 'protocol.json', protocol)
    models = {}
    # Phase 1: use validation predictions only. All fine thresholds frozen together.
    for model, folder in EXPORTS.items():
        reference = read(folder / 'int8_validation_report.json')
        with np.load(folder / 'int8_validation_predictions.npz', allow_pickle=False) as d:
            check_records(d, frames['validation'], 'ecg_id')
            for version in ['float32', 'int8']:
                p, y = d[f'probabilities_{version}'], d['targets']
                strategies = {'original': {'thresholds': reference['thresholds']}}
                for name, grid in grid_map.items():
                    strategies[name] = {'thresholds': tune(y, p, grid)}
                np.testing.assert_array_equal(strategies['grid_0.05']['thresholds'], optimize_thresholds(y, p))
                for data in strategies.values():
                    m = calculate_multilabel_metrics(y, p, data['thresholds'])
                    audit_metrics(y, p, data['thresholds'], m)
                    data['validation'] = m
                for k in KEYS:
                    np.testing.assert_allclose(strategies['original']['validation'][k], reference[version][k], rtol=0, atol=1e-12)
                assert strategies['grid_0.01']['validation']['macro_f1'] >= strategies['grid_0.05']['validation']['macro_f1'] - 1e-12
                models[f'{model}_{version}'] = {'model': model, 'version': version, 'strategies': strategies}
    preference = {'original': 2, 'grid_0.05': 1, 'grid_0.01': 0}
    candidates = [(name, strategy, data) for name, r in models.items() if r['version'] == 'int8'
                  for strategy, data in r['strategies'].items()]
    chosen = max(candidates, key=lambda x: (x[2]['validation']['macro_f1'], x[2]['validation']['macro_auroc'], preference[x[1]]))
    selection = {'model': chosen[0], 'strategy': chosen[1], 'thresholds': chosen[2]['thresholds'],
                 'validation': chosen[2]['validation'], 'test_used': False}
    save(OUT / 'frozen_thresholds.json', {'frozen_at': datetime.now().isoformat(), 'models': models,
                                         'selected': selection, 'test_predictions_read': False})
    save(OUT / 'candidate_thresholds.json', {'superclasses': list(SUPERCLASSES), **selection,
         'onnx_sha256': protected[str(EXPORTS[models[chosen[0]]['model']] / 'ptbxl_cnn1d_int8.onnx')],
         'deployed_files_modified': False})
    # Phase 2: test scores are now evaluated, never used to choose another cutoff.
    count = 0
    for name, r in models.items():
        folder = EXPORTS[r['model']]
        with np.load(folder / f'{r["version"]}_test_predictions.npz', allow_pickle=False) as d:
            check_records(d, frames['test'], 'ecg_ids')
            y, p = d['targets'], d['probabilities']
            for strategy, data in r['strategies'].items():
                m = calculate_multilabel_metrics(y, p, data['thresholds'])
                audit_metrics(y, p, data['thresholds'], m)
                data['test'] = m
                np.savez_compressed(OUT / f'{name}_{strategy}_test_predictions.npz',
                    ecg_ids=d['ecg_ids'], targets=y, probabilities=p, thresholds=np.asarray(data['thresholds']),
                    predictions=(p >= data['thresholds']).astype(np.uint8))
                count += 2
        assert len({data['test']['macro_auroc'] for data in r['strategies'].values()}) == 1
    for path, sha in protected.items():
        assert digest(Path(path)) == sha
    summary = {'protocol': protocol, 'models': models, 'frozen_validation_choice': selection,
               'audit': {'metric_sets_verified': count, 'weights_and_firmware_unchanged': True,
                         'validation_test_ids_verified': True, 'fine_grid_contains_coarse': True,
                         'all_grid_f1_values_independently_verified': True,
                         'auroc_unchanged_across_thresholds': True, 'test_used_for_selection': False}}
    save(OUT / 'summary.json', summary)
    report(summary)
    compact = {name: {strategy: {split: {k: data[split][k] for k in KEYS} for split in ['validation', 'test']}
                      for strategy, data in r['strategies'].items()} for name, r in models.items() if r['version'] == 'int8'}
    print(json.dumps({'selected_model': selection['model'], 'strategy': selection['strategy'],
                     'thresholds': selection['thresholds'], 'INT8_metrics': compact},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
