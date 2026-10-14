"""Frozen-model 0.01 calibration on original and diagnostic-only PTB-XL cohorts."""
from __future__ import annotations
import copy
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_stm32 import get_split
from scripts.refine_student_thresholds import read, save, digest, grids, tune, audit_metrics, section
from src.data.ptbxl import SUPERCLASSES, load_ecg
from src.training.data import load_labeled_splits
from src.training.metrics import calculate_multilabel_metrics
from src.signal_processing.filters import FilterConfig, preprocess_ecg
from src.models.probability_ensemble import load_teacher

OUT = ROOT / 'outputs/unfiltered_threshold_refinement'
KEYS = ['macro_auroc', 'macro_precision', 'macro_recall', 'macro_f1']
CP_CACHE = {}


def status(message):
    print(message, flush=True)
    save(OUT / 'status.json', {'stage': message, 'updated': datetime.now().isoformat()})


def checkpoint(path):
    path = Path(path)
    if not path.is_absolute(): path = ROOT / path
    if str(path) not in CP_CACHE:
        CP_CACHE[str(path)] = torch.load(path, map_location='cpu', weights_only=True)
    return CP_CACHE[str(path)]


def descriptors():
    rows = []
    for r in read(ROOT / 'outputs/teacher_distillation/summary.json')['runs']:
        arm = {'baseline': '普通CNN', 'distilled': '原教师蒸馏CNN', 'ensemble_kd': '新教师蒸馏CNN'}[r['arm']]
        rows.append({'name': f'{r["arm"]}_seed{r["seed"]}_fp32', 'label': f'{arm} seed{r["seed"]} FP32',
                     'kind': 'torch', 'checkpoint': r['checkpoint']})
    for r in read(ROOT / 'outputs/teacher_preprocessing/summary.json')['runs']:
        rows.append({'name': r['name'], 'label': r['name'], 'kind': 'torch', 'checkpoint': r['checkpoint']})
    for r in read(ROOT / 'outputs/teacher_preprocessing/ensemble_followup/summary.json')['runs']:
        rows.append({'name': r['name'], 'label': r['name'] + ' FP32', 'kind': 'torch', 'checkpoint': r['checkpoint']})
    for name, label, folder in [
        ('original_seed44_int8', '原蒸馏 seed44 INT8', ROOT / 'outputs/distillation/distilled_export'),
        ('ensemble_kd_seed42_int8', '新蒸馏 seed42 INT8', ROOT / 'outputs/teacher_distillation/selected_export')]:
        manifest = read(folder / 'deployment_manifest.json')
        rows.append({'name': name, 'label': label, 'kind': 'onnx', 'checkpoint': manifest['source_checkpoint'],
                     'onnx': str(folder / 'ptbxl_cnn1d_int8.onnx')})
    for name, label, path in [
        ('historical_resnet_global', '历史标准化 ResNet', 'outputs/optimization/01_global_norm/checkpoint/resnet1d_best.pt'),
        ('diagnostic_resnet', '筛选后重训 ResNet seed42', 'outputs/diagnostic_only/resnet_seed42/checkpoint/resnet1d_best.pt'),
        ('historical_inception', '历史优化 Inception', 'outputs/optimization/optimized_best.pt')]:
        rows.append({'name': name, 'label': label, 'kind': 'torch', 'checkpoint': str(ROOT / path)})
    manifest = ROOT / 'outputs/further_optimization/threshold_refinement/ensemble_manifest.json'
    rows.append({'name': 'historical_equal_ensemble', 'label': '历史 ResNet+Inception 集成',
                 'kind': 'legacy_ensemble', 'manifest': str(manifest)})
    assert len({r['name'] for r in rows}) == len(rows)
    for r in rows:
        paths = [r['checkpoint']] if r['kind'] != 'legacy_ensemble' else [c['checkpoint'] for c in read(Path(r['manifest']))['components']]
        counts = [checkpoint(p)['training_records'] for p in paths]
        r['training_records'] = counts[0] if len(set(counts)) == 1 else counts
        r['training_diagnostic_only'] = all(checkpoint(p)['config']['dataset'].get('require_diagnostic_label', False) for p in paths)
    return rows


def full_inputs(cp, frame, split):
    cfg = FilterConfig.from_mapping(cp['config']['signal_processing'])
    signature = hashlib.sha256(json.dumps(cp['config']['signal_processing'], sort_keys=True).encode()).hexdigest()[:16]
    path = OUT / 'inputs' / f'{split}_{signature}.npy'
    metadata = path.with_suffix('.json')
    if path.exists() and metadata.exists():
        info = read(metadata)
        assert info['ecg_ids'] == frame.ecg_id.tolist()
        assert info['preprocessing'] == cp['config']['signal_processing']
        return np.load(path, mmap_mode='r')
    path.parent.mkdir(parents=True, exist_ok=True)
    array, source = get_split(split, cp)
    positions = pd.Index(frame.ecg_id).get_indexer(source.ecg_id)
    assert (positions >= 0).all()
    np.testing.assert_array_equal(frame.iloc[positions][list(SUPERCLASSES)], source[list(SUPERCLASSES)])
    temp = path.with_suffix('.building')
    values = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float32, shape=(len(frame), 12, 1000))
    for start in range(0, len(source), 128):
        values[positions[start:start+128]] = array[start:start+128]
    missing = np.setdiff1d(np.arange(len(frame)), positions)
    root = Path(cp['config']['dataset']['root'])
    for index in missing:
        record = load_ecg(root, str(frame.iloc[index].filename_lr), sampling_rate_hz=100)
        values[index] = preprocess_ecg(record.signal, 100, config=cfg).T
    # Independently verify retained and restored records against raw waveforms.
    checked = np.unique(np.concatenate([positions[[0, len(positions)//2, -1]], missing]))
    for index in checked:
        record = load_ecg(root, str(frame.iloc[index].filename_lr), sampling_rate_hz=100)
        expected = preprocess_ecg(record.signal, 100, config=cfg).T
        np.testing.assert_array_equal(values[index], expected)
    assert np.isfinite(values).all()
    values.flush(); del values
    temp.replace(path)
    save(metadata, {'ecg_ids': frame.ecg_id.tolist(), 'preprocessing': cp['config']['signal_processing'],
                    'source_checkpoint_training_records': cp['training_records'],
                    'copied_records': len(source), 'restored_records': len(missing),
                    'raw_records_verified': len(checked), 'normalization_refit': False})
    return np.load(path, mmap_mode='r')


def infer_torch(path, frame, split):
    cp = checkpoint(path)
    values = full_inputs(cp, frame, split)
    model, _ = load_teacher(path, device='cuda' if torch.cuda.is_available() else 'cpu')
    device = next(model.parameters()).device
    batches = []
    with torch.inference_mode():
        for start in range(0, len(values), 32):
            x = torch.from_numpy(np.array(values[start:start+32], dtype=np.float32)).to(device)
            batches.append(model(x).sigmoid().cpu().numpy())
    del model
    return np.concatenate(batches)


def scores(r, frame, split):
    path = OUT / r['name'] / f'{split}_probabilities.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as d:
            np.testing.assert_array_equal(d['ecg_ids'], frame.ecg_id)
            np.testing.assert_array_equal(d['targets'], frame[list(SUPERCLASSES)])
            return d['probabilities']
    if r['kind'] == 'torch':
        p = infer_torch(Path(r['checkpoint']), frame, split)
    elif r['kind'] == 'legacy_ensemble':
        components = read(Path(r['manifest']))['components']
        p = np.average(np.stack([infer_torch(Path(c['checkpoint']), frame, split) for c in components]),
                       axis=0, weights=[c['weight'] for c in components])
    else:
        import onnxruntime as ort
        from scripts.quantize_cubeai import probabilities
        options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
        session = ort.InferenceSession(r['onnx'], options, providers=['CPUExecutionProvider'])
        p = probabilities(session, full_inputs(checkpoint(r['checkpoint']), frame, split))
    assert p.shape == (len(frame), 5) and np.isfinite(p).all()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ecg_ids=frame.ecg_id.to_numpy(), targets=frame[list(SUPERCLASSES)].to_numpy(dtype=np.uint8), probabilities=p)
    return p


def metrics(y, p, thresholds):
    m = calculate_multilabel_metrics(y, p, thresholds)
    audit_metrics(y, p, thresholds, m)
    return m


def docs(s):
    rows = s['models']
    names = ['inception_time1d_ensemble3', 'original_seed44_int8', 'ensemble_kd_seed42_int8']
    lines = ['## 2026-10-09：未筛选五类标签数据的 0.01 阈值细搜', '',
             f'已完成 {len(rows)} 个主要模型 / 推理版本的冻结权重比较，覆盖普通 CNN、两种教师蒸馏学生（各三个种子）、预处理 / 教师架构实验、两种三教师集成、两种 INT8、历史标准化 ResNet / Inception / 集成和筛选后重训 ResNet。加噪条件和外部数据沿用各自独立实验记录。', '',
             '本次没有重新训练。各模型的训练数据是否筛选已逐项记录；只改变阈值校准和评估的样本范围，保留 checkpoint 标准化统计，未重新拟合均值或标准差。', '',
             '筛选：验证 2146、测试 2158；未筛选：验证 2183、测试 2198。加回的 37 条验证、40 条测试记录，其五类标注均为 0，按现有五类标注作为全部阴性评估，不能自动解释为正常 ECG 或无疾病。五类阳性数仍为 NORM 963、MI 550、STTC 521、CD 496、HYP 262。', '',
             '两种样本范围均独立在对应验证集按逐类 F1 细搜 0.10–0.90、步长 0.01；全部阈值先冻结，再计算测试。另保留原阈值、未筛选验证集重搜 0.05 的对照。折 10 已在历史实验中重复使用，本轮是探索性复核。', '',
             '### 重点模型：各自验证集细搜后的整体对照，单位 %', '',
             '| 模型 | 验证 / 测试是否筛选 | AUROC | Precision | Recall | F1 |', '|---|---|---:|---:|---:|---:|']
    by_name = {r['name']: r for r in rows}
    for name in names:
        r = by_name[name]
        for strategy, cohort in [('filtered_0.01', '筛选'), ('unfiltered_0.01', '未筛选')]:
            m = r['strategies'][strategy]['test_filtered' if cohort == '筛选' else 'test']
            lines.append(f'| {r["label"]} | {cohort} | ' + ' | '.join(f'{m[k]*100:.3f}' for k in KEYS) + ' |')
    chosen = s['frozen_int8_selection']
    lines += ['', f'未筛选验证集上的 INT8 候选选择：{by_name[chosen["name"]]["label"]}，仅按该验证集宏 F1、再 AUROC 比较两种 INT8 的 0.01 结果；未据测试结果切换候选。原模型、部署 manifest、串口配置和 HEX 保留。', '',
              '重点三模型在完整验证集上细搜 0.01 后，阈值与各自筛选验证集细搜 0.01 的结果相同，因此加回测试记录后的 Recall 不变，Precision / F1 下降主要来自新增记录的误报。三模型在新增 40 条记录上都至少报出一类，按当前五类全零标注，均计为 40/40 条至少一次误报。', '',
              '细网格不保证测试改善：教师集成在未筛选测试集的 F1 从原阈值的 74.856% 降到 74.754%；原 seed44 INT8 的 0.01 与重搜 0.05 接近，分别为 74.194% 和 74.200%；新 seed42 INT8 从重搜 0.05 的 74.217% 提到 74.317%。所有结果保留，未因测试分数改选阈值。', '',
              '### 重点模型：未筛选 0.01 阈值与新增记录误报', '',
              '| 模型 | NORM | MI | STTC | CD | HYP | 40 条新增记录中至少一类误报 |', '|---|---:|---:|---:|---:|---:|---:|']
    for name in names:
        r = by_name[name]; d = r['strategies']['unfiltered_0.01']
        lines.append(f'| {r["label"]} | ' + ' | '.join(f'{t:.2f}' for t in d['thresholds']) +
                     f' | {d["zero_label_test"]["any_positive_records"]}/40 |')
    lines += ['', 'Recall = TP / (TP + FN)。当阈值相同时，加回全零标签记录不会改变五类 Recall，只可能增加 FP、降低 Precision / F1，并影响 AUROC；本表 Recall 的变化来自两种验证范围选出的阈值变化。报告还保存“用未筛选阈值评估同一 2158 条保留记录”的结果，便于分离阈值与样本范围的影响。', '',
              f'复核：{s["audit"]["metric_sets_verified"]} 组指标的逐类混淆矩阵和宏平均复算一致；细搜网格各阈值由 sklearn 独立验证；患者隔离、加回记录、原始波形预处理和模型 / 固件 SHA256 检查通过。独立审计脚本另检查阈值冻结早于测试预测、全部保存决策以及两种 INT8 保留记录的概率与历史结果完全一致（最大误差 0）；详见 `independent_audit.json`。', '',
              '结果：`outputs/unfiltered_threshold_refinement/summary.json`；全部模型、训练范围与保护哈希：`protocol.json`；冻结验证阈值：`frozen_thresholds.json`；逐条预测、决策、误报：每模型子目录；未改部署的候选阈值：`candidate_thresholds.json`。', '',
              '复现：`.venv/Scripts/python.exe scripts/refine_unfiltered_thresholds.py`。完整整体和逐类对照见现有指标总表的本次章节。']
    section(ROOT / 'md/模型训练与优化.md', '\n'.join(lines), 'unfiltered-threshold-refinement')
    table = ['## 2026-10-09：筛选与未筛选的各模型 0.01 阈值对照', '',
             '所有权重冻结。表中的“训练筛选”与验证 / 测试范围是不同概念：同一模型分别在筛选验证集和完整验证集选阈值，然后评估对应测试集。筛选验证 / 测试为 2146 / 2158，未筛选为 2183 / 2198；阈值只用验证标签。单位 %。', '',
             '| 模型 / 版本 | 训练筛选 | 训练数 | 验证 / 测试范围 | 验证 F1 | 测试 AUROC | Precision | Recall | F1 |',
             '|---|---|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        for strategy, cohort in [('filtered_0.01', '筛选'), ('unfiltered_0.01', '未筛选')]:
            d = r['strategies'][strategy]
            v = d['validation_filtered' if cohort == '筛选' else 'validation']
            m = d['test_filtered' if cohort == '筛选' else 'test']
            table.append(f'| {r["label"]} | {"是" if r["training_diagnostic_only"] else "否"} | {r["training_records"]} | {cohort} | {v["macro_f1"]*100:.3f} | ' +
                         ' | '.join(f'{m[k]*100:.3f}' for k in KEYS) + ' |')
    table += ['', '### 每个模型的逐类筛选 / 未筛选对照', '',
              '每格依次为“筛选 / 未筛选”，使用各自验证集确定的 0.01 阈值。五类测试阳性数两种范围相同；指标差异同时受到阈值与新增阴性记录影响。', '',
              '| 模型 | 类别 | 阳性数 | 阈值 | AUROC | Precision | Recall | F1 | 漏检数 | 误报数 |',
              '|---|---|---:|---|---|---|---|---|---|---|']
    for r in rows:
        for c in SUPERCLASSES:
            a = r['strategies']['filtered_0.01']['test_filtered']['per_class'][c]
            b = r['strategies']['unfiltered_0.01']['test']['per_class'][c]
            table.append(f'| {r["label"]} | {c} | {a["support"]} | {a["threshold"]:.2f} / {b["threshold"]:.2f} | ' +
                         ' | '.join(f'{a[k]*100:.2f} / {b[k]*100:.2f}' for k in ['auroc', 'precision', 'recall', 'f1']) +
                         f' | {a["confusion_matrix"][1][0]} / {b["confusion_matrix"][1][0]} | {a["confusion_matrix"][0][1]} / {b["confusion_matrix"][0][1]} |')
    table += ['', '### 新增 40 条五类全零记录：未筛选 0.01 阈值的误报', '',
              '| 模型 | 至少一类报阳性（条） | NORM误报 | MI误报 | STTC误报 | CD误报 | HYP误报 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        z = r['strategies']['unfiltered_0.01']['zero_label_test']
        table.append(f'| {r["label"]} | {z["any_positive_records"]}/40 | ' +
                     ' | '.join(str(z['positive_counts'][c]) for c in SUPERCLASSES) + ' |')
    table += ['', '原始概率、原阈值 / 未筛选 0.05 / 0.01 的对照和控制相同 2158 条测试记录的结果见 `outputs/unfiltered_threshold_refinement/summary.json`。本轮没有重新训练，没有新增外部验证或替换固件。']
    section(ROOT / 'md/各模型整体与逐类指标总表.md', '\n'.join(table), 'unfiltered-threshold-refinement')
    section(ROOT / 'md/README.md', f'2026-10-09 完成 {len(rows)} 个主要模型 / 推理版本的筛选、未筛选 0.01 阈值比较。未筛选验证 / 测试为 2183 / 2198，新增全零五类标签记录的误报已单列；模型训练范围另行注明。整体、逐类列表追加到原训练文档和指标总表，结果位于 `outputs/unfiltered_threshold_refinement/`。', 'unfiltered-threshold-refinement')
    for filename in ['模型训练与优化.md', '各模型整体与逐类指标总表.md', 'README.md']:
        path = ROOT / 'md' / filename
        text = path.read_text(encoding='utf-8')
        text = text.replace('更新：2026-10-01。', '更新：2026-10-09。', 1).replace('更新：2026-10-07。', '更新：2026-10-09。', 1)
        if filename == '各模型整体与逐类指标总表.md':
            old = '模型选择依据验证集，以下测试点估计用于结果汇报，不用于重新选择模型。'
            new = '模型选择依据验证集，以下测试点估计用于结果汇报，不用于重新选择模型。首表为 2026-10-01 的历史摘要，最新 26 个模型的筛选 / 未筛选对照见本文末尾 2026-10-09 章节。'
            if new not in text: text = text.replace(old, new, 1)
        path.write_text(text, encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'summary.json').exists():
        docs(read(OUT / 'summary.json')); print('Existing completed result documented.'); return
    torch.set_num_threads(8)
    rows = descriptors()
    root = Path(checkpoint(rows[0]['checkpoint'])['config']['dataset']['root'])
    frames = load_labeled_splits(root, 100, require_diagnostic_label=False)
    assert {k: len(v) for k, v in frames.items()} == {'train': 17418, 'validation': 2183, 'test': 2198}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not set(frames[a].patient_id) & set(frames[b].patient_id)
    for split in ['validation', 'test']:
        frames[split][['ecg_id', 'patient_id', 'strat_fold', *SUPERCLASSES]].to_csv(OUT / f'cohort_{split}.csv', index=False)
    protected = {}
    for r in rows:
        paths = [Path(r['checkpoint'])] if r['kind'] != 'legacy_ensemble' else [Path(r['manifest']), *[Path(c['checkpoint']) for c in read(Path(r['manifest']))['components']]]
        if r['kind'] == 'onnx': paths.append(Path(r['onnx']))
        for path in paths: protected[str(path)] = digest(path)
    firmware = ROOT / 'outputs/board_release/seed44_int8_stm32f407ve/ECG_Student_INT8_STM32F407VE.hex'
    protected[str(firmware)] = digest(firmware)
    protocol = {'date': '2026-10-09', 'models': rows, 'model_retraining': False, 'normalization_refit': False,
                'cohort': {'train_metadata': 17418, 'validation': 2183, 'test': 2198},
                'added_zero_label_records': {'validation': 37, 'test': 40},
                'threshold_policy': 'per-class validation F1; 0.10-0.90, steps .05/.01, tie nearest .5',
                'test_used_for_selection': False, 'test_previously_reused': True, 'protected_sha256': protected}
    existing = OUT / 'protocol.json'
    if existing.exists(): assert read(existing) == protocol
    else: save(existing, protocol)
    y = frames['validation'][list(SUPERCLASSES)].to_numpy(dtype=np.uint8)
    kept = y.sum(1) > 0
    assert int((~kept).sum()) == 37
    for index, r in enumerate(rows):
        status(f'Validation {index+1}/{len(rows)}: {r["label"]}; freeze 0.01 thresholds from validation only')
        p = scores(r, frames['validation'], 'validation')
        original = read(Path(r['manifest']))['thresholds'] if r['kind'] == 'legacy_ensemble' else checkpoint(r['checkpoint'])['thresholds']
        policies = {'original': original, 'filtered_0.01': tune(y[kept], p[kept], grids()['grid_0.01']),
                    'unfiltered_0.05': tune(y, p, grids()['grid_0.05']), 'unfiltered_0.01': tune(y, p, grids()['grid_0.01'])}
        r['strategies'] = {name: {'thresholds': t, 'validation': metrics(y, p, t),
                                  'validation_filtered': metrics(y[kept], p[kept], t)} for name, t in policies.items()}
        assert r['strategies']['unfiltered_0.01']['validation']['macro_f1'] >= r['strategies']['unfiltered_0.05']['validation']['macro_f1'] - 1e-12
    candidates = [r for r in rows if r['kind'] == 'onnx']
    chosen = max(candidates, key=lambda r: (r['strategies']['unfiltered_0.01']['validation']['macro_f1'],
                                          r['strategies']['unfiltered_0.01']['validation']['macro_auroc']))
    selection = {'name': chosen['name'], 'thresholds': chosen['strategies']['unfiltered_0.01']['thresholds'],
                 'validation': chosen['strategies']['unfiltered_0.01']['validation'], 'test_used': False}
    save(OUT / 'frozen_thresholds.json', {'frozen_at': datetime.now().isoformat(), 'models': copy.deepcopy(rows),
                                         'selected_int8': selection, 'test_probabilities_read': False})
    save(OUT / 'candidate_thresholds.json', {'superclasses': list(SUPERCLASSES), **selection,
         'source_onnx_sha256': digest(Path(chosen['onnx'])), 'deployed_files_modified': False})
    # All validation thresholds are now frozen, before any test inference.
    y = frames['test'][list(SUPERCLASSES)].to_numpy(dtype=np.uint8)
    kept = y.sum(1) > 0
    assert int((~kept).sum()) == 40
    count = 0
    for index, r in enumerate(rows):
        status(f'Test {index+1}/{len(rows)}: {r["label"]}; full 2198 and matched 2158 records')
        p = scores(r, frames['test'], 'test')
        for name, d in r['strategies'].items():
            t = d['thresholds']
            d['test'] = metrics(y, p, t)
            d['test_filtered'] = metrics(y[kept], p[kept], t)
            pred = p.astype(np.float64) >= np.asarray(t, dtype=np.float64)
            zero = pred[~kept]
            d['zero_label_test'] = {'records': len(zero), 'any_positive_records': int(zero.any(1).sum()),
                 'all_negative_records': int((~zero.any(1)).sum()),
                 'positive_counts': {c: int(zero[:, i].sum()) for i, c in enumerate(SUPERCLASSES)}}
            for c in SUPERCLASSES:
                a, b = d['test']['per_class'][c], d['test_filtered']['per_class'][c]
                assert a['support'] == b['support'] and a['recall'] == b['recall']
                assert a['confusion_matrix'][0][1] - b['confusion_matrix'][0][1] == d['zero_label_test']['positive_counts'][c]
            np.savez_compressed(OUT / r['name'] / f'test_decisions_{name}.npz',
                 ecg_ids=frames['test'].ecg_id.to_numpy(), thresholds=np.asarray(t), predictions=pred.astype(np.uint8),
                 retained_diagnostic_records=kept)
            count += 4
    for path, sha in protected.items(): assert digest(Path(path)) == sha
    s = {'protocol': protocol, 'models': rows, 'frozen_int8_selection': selection,
         'audit': {'metric_sets_verified': count, 'weights_and_firmware_unchanged': True,
                   'same_threshold_recall_invariant': True, 'extra_false_positives_verified': True,
                   'patient_isolation_verified': True, 'all_thresholds_validation_only': True}}
    save(OUT / 'summary.json', s); docs(s)
    status(f'Completed {len(rows)} models: filtered/unfiltered comparisons and existing Markdown updated')
    for r in rows:
        if r['name'] in ['inception_time1d_ensemble3', 'original_seed44_int8', 'ensemble_kd_seed42_int8']:
            print(r['name'], json.dumps({name: {k: d['test'][k] for k in KEYS} for name, d in r['strategies'].items()}), flush=True)


if __name__ == '__main__': main()
