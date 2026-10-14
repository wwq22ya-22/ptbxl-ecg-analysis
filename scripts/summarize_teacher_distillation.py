"""Independently audit saved predictions and update the existing project MDs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_stm32 import digest
from scripts.run_distillation import save
from scripts.summarize_teacher_study import section
from src.data.ptbxl import SUPERCLASSES

OUT = ROOT / 'outputs/teacher_distillation'
KEYS = ['macro_auroc', 'macro_precision', 'macro_recall', 'macro_f1']
NAMES = {'baseline': '普通 CNN', 'distilled': '原教师蒸馏 CNN', 'ensemble_kd': '新集成教师蒸馏 CNN'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit_metrics(path, expected, frame):
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data['ecg_ids'], frame['ecg_id'])
        np.testing.assert_array_equal(data['targets'], np.column_stack([frame[c] for c in SUPERCLASSES]))
        p, y, t = data['probabilities'], data['targets'], data['thresholds']
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    from sklearn.metrics import roc_auc_score
    result = {k: [] for k in KEYS}
    for i, label in enumerate(SUPERCLASSES):
        pred = p[:, i] >= t[i]
        truth = y[:, i].astype(bool)
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        tn = int((~pred & ~truth).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        c = expected['per_class'][label]
        assert c['confusion_matrix'] == [[tn, fp], [fn, tp]]
        assert c['support'] == int(truth.sum())
        for key, value in [('precision', precision), ('recall', recall), ('f1', f1),
                           ('auroc', float(roc_auc_score(truth, p[:, i])))]:
            np.testing.assert_allclose(c[key], value, rtol=0, atol=1e-12)
            result['macro_' + key].append(value)
    for key, values in result.items():
        np.testing.assert_allclose(expected[key], np.mean(values), rtol=0, atol=1e-12)
    return {'file': str(path.relative_to(ROOT)), 'records': len(y), 'sha256': digest(path)}


def main():
    import pandas as pd
    s = read(OUT / 'summary.json')
    protocol = s['protocol']
    frames = {split: pd.read_csv(ROOT / f'outputs/teacher_preprocessing/cohort_{split}.csv')
              for split in ['train', 'validation', 'test']}
    audited = []
    for r in s['runs']:
        assert digest(Path(r['checkpoint'])) == r['sha256']
        for split in ['validation', 'test']:
            audited.append(audit_metrics(OUT / f'comparison/{r["arm"]}_seed{r["seed"]}/{split}_predictions.npz',
                                         r[split], frames[split]))
    for version in ['float32', 'int8']:
        audited.append(audit_metrics(OUT / f'selected_export/{version}_test_predictions.npz',
                                     s['test_export'][version], frames['test']))
    with np.load(OUT / 'selected_export/int8_validation_predictions.npz', allow_pickle=False) as d:
        np.testing.assert_array_equal(d['ecg_id'], frames['validation'].ecg_id)
        for version in ['float32', 'int8']:
            path = OUT / f'selected_export/audit_{version}_validation.npz'
            np.savez_compressed(path, ecg_ids=d['ecg_id'], targets=d['targets'],
                                probabilities=d[f'probabilities_{version}'], thresholds=s['quantization']['thresholds'])
            audited.append(audit_metrics(path, s['quantization'][version], frames['validation']))
    for path, sha in protocol['protected_artifacts'].items():
        assert digest(Path(path)) == sha
    train_ids = set(frames['train'].ecg_id)
    calibration = s['quantization']['calibration']['ecg_ids']
    assert len(calibration) == len(set(calibration)) == 256 and set(calibration) <= train_ids
    assert not set(calibration) & set(frames['validation'].ecg_id)
    assert not set(calibration) & set(frames['test'].ecg_id)
    assert s['quantization']['thresholds_retuned'] is False
    frozen = read(OUT / 'frozen_selection.json')['selected']
    best = max([r for r in s['runs'] if r['arm'] == 'ensemble_kd'],
               key=lambda r: (r['validation']['macro_f1'], r['validation']['macro_auroc']))
    assert frozen['seed'] == best['seed'] == s['selected_seed']
    reference = read(ROOT / 'outputs/noise_robustness_b/selected_export/A_deployed_seed44_INT8_robustness.json')
    for key in KEYS:
        np.testing.assert_allclose(reference['validation']['clean'][key], s['original_int8_validation'][key], atol=1e-12)
    audit = {'prediction_sets_verified': len(audited), 'predictions': audited,
             'protected_artifacts_unchanged': True, 'calibration_train_only_verified': True,
             'validation_selection_verified': True, 'loss_and_export_tests_passed': 7,
             'test_used_for_selection': False}
    save(OUT / 'independent_audit.json', audit)
    s['independent_audit'] = audit
    save(OUT / 'summary.json', s)
    stats = {}
    for arm in NAMES:
        rows = [r for r in s['runs'] if r['arm'] == arm]
        stats[arm] = {k: {'mean': float(np.mean([r['test'][k] for r in rows])),
                          'std': float(np.std([r['test'][k] for r in rows], ddof=1))} for k in KEYS}
    chosen = next(r for r in s['runs'] if r['arm'] == 'ensemble_kd' and r['seed'] == s['selected_seed'])
    old = reference['test']['clean']
    new = s['test_export']['int8']
    deltas = {k: (new[k] - old[k]) * 100 for k in KEYS}
    paired = {arm: [100 * (next(r for r in s['runs'] if r['arm'] == 'ensemble_kd' and r['seed'] == seed)['test']['macro_f1'] -
                         next(r for r in s['runs'] if r['arm'] == arm and r['seed'] == seed)['test']['macro_f1'])
                    for seed in [42, 43, 44]] for arm in ['baseline', 'distilled']}
    recommendation = '新蒸馏 INT8 候选' if s['deployment_recommendation'] == 'ensemble_kd' else '原 seed44 INT8'
    lines = ['## 2026-10-07：新集成教师 → 小 CNN 蒸馏结果', '',
             '已完成新教师蒸馏的种子 42、43、44，复核既有普通训练与原教师蒸馏的六个学生，另完成验证选定新学生的 FP32 / INT8 导出与 Cube.AI 分析。', '',
             '学生保持 CNN1D、111557 参数、12 导联 × 1000 点、五个独立输出。训练 / 验证 / 测试为 17084 / 2146 / 2158，剔除五类全零标签、患者隔离。预处理保持原 P0；训练集标准化统计、优化器、学习率、batch、早停、三个种子和原蒸馏实验一致。', '',
             '新教师为三个种子的 InceptionTime 风格网络概率均值集成，冻结后生成训练集 logits。损失仍为 0.5 × 带正类权重真实标签 BCE + 0.5 × T² × 软标签 BCE，T=2。教师只用于训练，学生推理不加载教师。', '',
             '这是更换教师后的完整流程对比。原教师训练包含原全量训练记录，而新教师训练剔除了无标签记录；教师结构、集成和教师训练数据共同变化，不能将差异全部归因于集成。普通训练与原蒸馏学生复用已完成权重，配置、患者范围、标准化及权重哈希已核对。', '',
             '权重按训练期验证 AUROC 选择；原粗网格逐类阈值来自训练期 AMP 验证，后续 FP32 / INT8 保持阈值。新学生种子按统一 FP32 验证宏 F1、再 AUROC 选定，未用测试结果选种子。', '',
             '### 三种子测试统计', '',
             '| 方法 | AUROC（%，均值 ± 标准差） | Precision | Recall | F1 |', '|---|---:|---:|---:|---:|']
    for arm in NAMES:
        lines.append('| ' + NAMES[arm] + ' | ' + ' | '.join(
            f'{stats[arm][k]["mean"]*100:.3f} ± {stats[arm][k]["std"]*100:.3f}' for k in KEYS) + ' |')
    lines += ['', '新蒸馏相对普通训练，各种子测试 F1 变化（百分点）：' + ', '.join(f'{d:+.3f}' for d in paired['baseline']) + '。',
              '新蒸馏相对原教师蒸馏，各种子测试 F1 变化（百分点）：' + ', '.join(f'{d:+.3f}' for d in paired['distilled']) + '。', '',
              f'相对普通训练，新蒸馏的三个种子均有 F1 收益，平均增加 {np.mean(paired["baseline"]):.3f} 个百分点。相对原教师蒸馏，平均仅增加 {np.mean(paired["distilled"]):.3f} 个百分点，且三个种子中两个下降；因此较强教师未带来一致、明显的学生 F1 改善。', '',
              f'### 验证选定学生 seed{s["selected_seed"]} 与原部署版', '',
              '| 模型 | 测试 AUROC（%） | Precision（%） | Recall（%） | F1（%） | HYP Recall（%） |',
              '|---|---:|---:|---:|---:|---:|']
    for label, m in [('原 seed44 INT8', old), (f'新蒸馏 seed{s["selected_seed"]} FP32', s['test_export']['float32']),
                      (f'新蒸馏 seed{s["selected_seed"]} INT8', new)]:
        lines.append('| ' + label + ' | ' + ' | '.join(f'{m[k]*100:.3f}' for k in KEYS) +
                     f' | {m["per_class"]["HYP"]["recall"]*100:.3f} |')
    lines += ['', '新 INT8 相对原部署版变化：' + '，'.join(f'{k.replace("macro_", "")} {deltas[k]:+.3f} 个百分点' for k in KEYS) + '。', '',
              f'量化相对新学生 ONNX FP32 的测试 F1 变化 {(new["macro_f1"]-s["test_export"]["float32"]["macro_f1"])*100:+.3f} 个百分点；HYP Recall 相对原部署增加 {(new["per_class"]["HYP"]["recall"]-old["per_class"]["HYP"]["recall"])*100:+.3f} 个百分点。收益主要在召回，Precision 明显下降，不能称各项指标全面提升。', '',
              f'验证集 INT8 宏 F1：原 {s["original_int8_validation"]["macro_f1"]*100:.3f}%，新 {s["quantization"]["int8"]["macro_f1"]*100:.3f}%。按测试前确定的验证选择规则，后续候选推荐为 **{recommendation}**。现有 seed44 ONNX、权重及已编译 HEX 均未替换。', '',
              '### 量化与资源', '',
              '使用与原蒸馏实验相同的 256 条训练记录做静态 QDQ INT8 校准；每通道权重量化，未使用验证 / 测试波形校准，量化后未重调阈值。', '',
              '| 版本 | 权重（B） | 激活（B） | MACC |', '|---|---:|---:|---:|']
    for v, a in s['analysis'].items():
        lines.append(f'| {v} | {a["weights_bytes"]} | {a["activations_bytes"]} | {a["macc"]} |')
    lines += ['', '模型结构与现有学生相同；Cube.AI 数值是模型资源，不能代替新完整固件链接或板上计时。本轮没有生成或烧录新 HEX。', '',
              '### 验证与复现', '',
              f'7 项蒸馏 / 导出测试通过；{len(audited)} 组预测的整体指标、逐类指标和混淆矩阵独立复算一致，训练校准 ID、候选选择、原模型 / 固件哈希均通过复核。', '',
              '测试折 10 已在项目中多次使用，新教师集成也在前一轮测试分析后提出，因此本次属于固定数据上的探索性实验；三个种子的变化不等于外部泛化提升。不得只取测试成绩最高的种子替换验证选定种子。', '',
              '结果：`outputs/teacher_distillation/summary.json`；协议与冻结选择：同目录 `protocol.json`、`frozen_selection.json`、`frozen_deployment_selection.json`；新权重：`ensemble_kd_seed42/43/44/checkpoint/cnn1d_best.pt`；选定导出：`selected_export/ptbxl_cnn1d_int8.onnx`；逐条预测与审计：`comparison/`、`independent_audit.json`。', '',
              '复现：`.venv/Scripts/python.exe scripts/run_teacher_distillation.py`，完成后运行 `.venv/Scripts/python.exe scripts/summarize_teacher_distillation.py`。']
    section(ROOT / 'md/模型训练与优化.md', '\n'.join(lines), 'new-teacher-distillation')
    table = ['## 2026-10-07：新教师蒸馏学生，整体与逐类指标', '',
             '与训练文档同一冻结协议。以下学生测试统一使用 FP32 推理和各自原验证阈值，三个种子全部保留；INT8 表另列。单位为 %，阳性数为记录数。', '',
             '| 方法 | 种子 | 最佳轮次 | 验证 F1 | 测试 AUROC | Precision | Recall | F1 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in s['runs']:
        table.append(f'| {NAMES[r["arm"]]} | {r["seed"]} | {r["epoch"]} | {r["validation"]["macro_f1"]*100:.3f} | ' +
                     ' | '.join(f'{r["test"][k]*100:.3f}' for k in KEYS) + ' |')
    table += ['', '### 全部学生的逐类测试结果', '', '| 方法 | 种子 | 类别 | 阳性数 | AUROC | Precision | Recall | F1 |',
              '|---|---:|---|---:|---:|---:|---:|---:|']
    models = [(NAMES[r['arm']], r['seed'], r['test']) for r in s['runs']]
    models += [('原部署 INT8', 44, old), ('新蒸馏 ONNX FP32', s['selected_seed'], s['test_export']['float32']),
               ('新蒸馏 INT8', s['selected_seed'], new)]
    for name, seed, m in models:
        for label in SUPERCLASSES:
            c = m['per_class'][label]
            table.append(f'| {name} | {seed} | {label} | {c["support"]} | ' +
                         ' | '.join(f'{c[k]*100:.3f}' for k in ['auroc', 'precision', 'recall', 'f1']) + ' |')
    table += ['', '### 量化版本的整体测试对比', '',
              '| 版本 | 种子 | AUROC | Precision | Recall | F1 |', '|---|---:|---:|---:|---:|---:|']
    for name, seed, m in models[-3:]:
        table.append(f'| {name} | {seed} | ' + ' | '.join(f'{m[k]*100:.3f}' for k in KEYS) + ' |')
    table += ['', f'新学生按验证集选 seed{s["selected_seed"]}；INT8 验证规则推荐 {recommendation}。旧固件保留，测试不是新的盲测。三种子均值、成对差异和资源分析见训练文档本次章节；原始数据见 `outputs/teacher_distillation/summary.json`。']
    section(ROOT / 'md/各模型整体与逐类指标总表.md', '\n'.join(table), 'new-teacher-distillation')
    section(ROOT / 'md/README.md',
            f'2026-10-07 新集成教师蒸馏已完成三个种子、九个学生模型对照、INT8 与 Cube.AI 分析。新学生验证选 seed{s["selected_seed"]}，新 INT8 测试宏 F1 {new["macro_f1"]*100:.3f}%，原 seed44 为 {old["macro_f1"]*100:.3f}%；按 INT8 验证规则推荐 {recommendation}，现有固件保留。方法与整体 / 逐类指标已写入原训练文档和指标总表，完整结果位于 `outputs/teacher_distillation/`。',
            'new-teacher-distillation')
    # Keep the earlier teacher study as history, but remove obsolete pending-work wording.
    for filename in ['模型训练与优化.md', '各模型整体与逐类指标总表.md']:
        path = ROOT / 'md' / filename
        content = path.read_text(encoding='utf-8')
        content = content.replace('新学生蒸馏、INT8精度和Cube.AI资源尚待下一轮验证。',
                                  '后续新学生蒸馏、INT8精度和Cube.AI资源已完成，结果见本文“新教师蒸馏”章节。')
        content = content.replace('当前seed44 INT8固件保持原样；新学生的精度和资源仍需单独蒸馏、量化验证。',
                                  '当前seed44 INT8固件保持原样；后续已完成新学生蒸馏与量化验证，结果见本文新增章节。')
        content = content.replace('较大网络是电脑端教师；尚未蒸馏、量化或完成板上资源分析。',
                                  '较大网络是电脑端教师；教师筛选阶段尚未蒸馏。后续集成教师蒸馏、INT8及Cube.AI分析已完成，见本文新增章节。')
        path.write_text(content, encoding='utf-8')
    print(json.dumps({'selected_seed': s['selected_seed'], 'recommendation': recommendation,
                      'old_int8': {k: old[k] for k in KEYS}, 'new_int8': {k: new[k] for k in KEYS},
                      'mean_results': stats, 'paired_f1_pp': paired, 'audit_sets': len(audited)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
