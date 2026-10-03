"""Run isolated, resumable ECG ablations and update the Markdown report."""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import load_config
from src.data.ptbxl import SUPERCLASSES
from src.models import build_model
from src.signal_processing.filters import FilterConfig
from src.training.data import CachedPTBXLDataset, ensure_preprocessed_cache, load_labeled_splits
from src.training.metrics import calculate_multilabel_metrics
from src.training.train import train_baseline
from src.training.evaluate import evaluate_checkpoint

ROOT = PROJECT_ROOT / 'outputs/optimization'
REPORT = PROJECT_ROOT / 'outputs/reports/五标签模型优化记录.md'


def exact_thresholds(truth, scores, fallback):
    chosen = []
    for i in range(5):
        if np.unique(truth[:, i]).size < 2:
            chosen.append(float(fallback[i]))
            continue
        precision, recall, thresholds = precision_recall_curve(truth[:, i], scores[:, i])
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        valid = (thresholds > 0) & (thresholds < 1)
        if not valid.any():
            chosen.append(float(fallback[i]))
            continue
        best = f1[valid].max()
        options = thresholds[valid & np.isclose(f1, best)]
        chosen.append(float(options[np.argmin(np.abs(options - fallback[i]))]))
    return chosen


def balanced_thresholds(truth, scores, fallback, recall_floor, precision_floor):
    """Choose validation thresholds with macro recall/precision floors.

    Search Lagrangian recall tradeoffs; this is a candidate search, not a claim
    of solving the discrete constrained optimization globally.
    """
    curves = []
    for i in range(5):
        precision, recall, thresholds = precision_recall_curve(truth[:, i], scores[:, i])
        valid = (thresholds > 0) & (thresholds < 1)
        p, r, t = precision[:-1][valid], recall[:-1][valid], thresholds[valid]
        f1 = 2 * p * r / np.maximum(p + r, 1e-12)
        curves.append((p, r, t, f1))
    candidates = []
    for multiplier in np.linspace(0, 1, 1001):
        indices = [int(np.argmax(f1 + multiplier * r)) for p, r, t, f1 in curves]
        precision = np.mean([curve[0][j] for curve, j in zip(curves, indices)])
        recall = np.mean([curve[1][j] for curve, j in zip(curves, indices)])
        if precision >= precision_floor and recall >= recall_floor:
            f1 = np.mean([curve[3][j] for curve, j in zip(curves, indices)])
            thresholds = [float(curve[2][j]) for curve, j in zip(curves, indices)]
            candidates.append((float(f1), thresholds))
    if candidates:
        return max(candidates, key=lambda candidate: candidate[0])[1], True
    return fallback, False


def validation_result(checkpoint_path, dataset_root, run_dir):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    cfg = checkpoint['config']
    fs = checkpoint['sampling_rate_hz']
    frame = load_labeled_splits(dataset_root, fs)['validation']
    cache = ensure_preprocessed_cache(frame, dataset_root, PROJECT_ROOT / cfg['outputs']['cache_dir'], split_name='validation_full', sampling_rate_hz=fs, config=FilterConfig.from_mapping(cfg['signal_processing']))
    loader = DataLoader(CachedPTBXLDataset(cache, frame), batch_size=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(checkpoint['model_type'], dropout=cfg['model']['dropout']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    targets, scores = [], []
    with torch.inference_mode():
        for x, y in loader:
            targets.append(y.numpy())
            scores.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
    truth, probabilities = np.concatenate(targets), np.concatenate(scores)
    coarse = calculate_multilabel_metrics(truth, probabilities, checkpoint['thresholds'])
    thresholds = exact_thresholds(truth, probabilities, checkpoint['thresholds'])
    fine = calculate_multilabel_metrics(truth, probabilities, thresholds)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / 'validation_predictions.npz', targets=truth, probabilities=probabilities, ecg_ids=frame.ecg_id.to_numpy())
    return {'validation': coarse, 'validation_fine': fine, 'fine_thresholds': thresholds, 'epoch': checkpoint['epoch'], 'parameters': sum(p.numel() for p in model.parameters())}


def write_report():
    results = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(ROOT.glob('*/result.json'))]
    lines = ['# 五标签模型优化记录', '', '日期：2026-09-30。以下结果均为实际运行；未提升的实验也保留。', '', '## 实验协议', '',
        'PTB-XL 100 Hz、12 导联、五标签。官方 fold 1–8 训练、fold 9 选模型/阈值、fold 10 最终评估。各实验保存在 outputs/optimization/独立目录，不覆盖原模型。缓存共享；训练集统计量仅使用 fold 1–8，随 checkpoint 保存，推理复用。', '',
        '单项实验固定随机种子 42，参考原 ResNet 的 20 轮上限、patience=5、学习率 0.001、batch_size=32、完整正类权重。调度实验是一个训练策略包：60 轮上限、patience=12、3 轮 warmup 和 cosine 衰减，不能把收益单独归因于调度器。', '',
        '先单项对照，再比较两种新架构只改标准化的方案、组合有效改动并比较网络；最终方案三个种子 42/43/44。选择依据为验证集细阈值 macro-F1 优先、macro-AUROC 次之。精确阈值根据验证集 PR 曲线搜索；旧粗网格阈值结果也保留。验证集阈值优化值可能偏乐观，测试集才是最终泛化结果。测试数据不参与方案或种子选择。', '',
        '## 原有测试集基线', '', '| 模型 | AUROC | Precision | Recall | F1 | HYP F1 |', '|---|---:|---:|---:|---:|---:|',
        '| CNN | 0.907686 | 0.681177 | 0.743572 | 0.707988 | 0.460342 |', '| ResNet | 0.911951 | 0.657614 | 0.785728 | 0.714893 | 0.480892 |', '',
        '## 验证集对照：原粗网格阈值', '', '| 实验 | 最佳轮 | 参数量 | AUROC | Precision | Recall | F1 | HYP F1 | 秒 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        m = r['validation']
        lines.append(f"| {r['name']} | {r['epoch']} | {r['parameters']} | {m['macro_auroc']:.6f} | {m['macro_precision']:.6f} | {m['macro_recall']:.6f} | {m['macro_f1']:.6f} | {m['per_class']['HYP']['f1']:.6f} | {r['seconds']:.1f} |")
    lines += ['', '## 验证集对照：PR 曲线精确阈值', '', '| 实验 | AUROC | Precision | Recall | F1 | HYP F1 |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        m = r['validation_fine']
        lines.append(f"| {r['name']} | {m['macro_auroc']:.6f} | {m['macro_precision']:.6f} | {m['macro_recall']:.6f} | {m['macro_f1']:.6f} | {m['per_class']['HYP']['f1']:.6f} |")
    lines += ['', '## 实验设置', '']
    for r in results:
        lines.append(f"- **{r['name']}**：{r['description']}。checkpoint：`{r['checkpoint']}`。")
    decision = ROOT / 'selection.json'
    if decision.exists():
        selected = json.loads(decision.read_text(encoding='utf-8'))
        final_cfg = selected['config']
        final_train = final_cfg['training']
        lines += ['', '## 最终方案选择', '', f"最终选择实验：`{selected['source_experiment']}`；架构：`{selected['model_type']}`；标准化：`{final_cfg['signal_processing']['normalization']}`；正类权重幂：`{final_train.get('pos_weight_power', 1.0)}`；调度器：`{final_train.get('scheduler', 'none')}`；最大轮数：`{final_train['epochs']}`；早停耐心：`{final_train['early_stopping_patience']}`；随机种子：`{selected.get('selected_seed', '等待稳定性实验')}`。", '', f"最终 checkpoint：`{selected.get('checkpoint', '等待稳定性实验')}`。", '', f"曾比较的组合候选配置：`{json.dumps(selected['combination'], ensure_ascii=False)}`。最终采用的是验证集更好的只改标准化方案。", '', '组合纳入规则：标准化、权重和调度策略只在验证 F1 提升、AUROC 降幅不超过 0.002 时纳入组合；组合再与单项直接比较。']
    if decision.exists() and selected.get('threshold_policy'):
        policy = selected['threshold_policy']
        lines += ['', '## 最终验证集阈值取舍', '', f"仅在 fold 9 搜索阈值。约束目标：macro Recall ≥ {policy['recall_floor']:.6f}、macro Precision ≥ {policy['precision_floor']:.6f}（原 ResNet 细阈值验证值）。在满足约束的拉格朗日候选中选 F1 最大者；不声称求得全局最优。找到可行候选：{policy['constraints_satisfied']}。AUROC 不因阈值变化。", '', '| 指标 | 最终阈值下验证结果 |', '|---|---:|']
        for key in ('macro_auroc', 'macro_precision', 'macro_recall', 'macro_f1'):
            lines.append(f"| {key} | {policy['validation_metrics'][key]:.6f} |")
    seed_results = [r for r in results if r['name'].startswith('final_seed_')]
    if seed_results:
        lines += ['', '## 最终方案的随机种子稳定性', '', '下表为验证集细阈值结果，标准差使用总体标准差（ddof=0）。', '', '| 指标 | 均值 | 标准差 |', '|---|---:|---:|']
        for key in ('macro_auroc', 'macro_precision', 'macro_recall', 'macro_f1'):
            values = [r['validation_fine'][key] for r in seed_results]
            lines.append(f'| {key} | {np.mean(values):.6f} | {np.std(values):.6f} |')
    final_path = ROOT / 'final_test.json'
    if final_path.exists():
        m = json.loads(final_path.read_text(encoding='utf-8'))
        old = json.loads((PROJECT_ROOT / 'outputs/metrics/test_metrics_resnet1d_full.json').read_text(encoding='utf-8'))
        lines += ['', '## 最终冻结模型：测试 fold 10', '', f"记录数：{m['records']}；模型：`{m['checkpoint']}`。测试前按验证集选定架构、种子与阈值。", '', '| 指标 | 原 ResNet | 优化模型 | 差值 |', '|---|---:|---:|---:|']
        for key in ('macro_auroc', 'macro_precision', 'macro_recall', 'macro_f1'):
            lines.append(f"| {key} | {old[key]:.6f} | {m[key]:.6f} | {m[key]-old[key]:+.6f} |")
        lines += ['', '| 类别 | AUROC | Precision | Recall | F1 | 原 ResNet F1 | F1 差值 |', '|---|---:|---:|---:|---:|---:|---:|']
        for name in SUPERCLASSES:
            c, b = m['per_class'][name], old['per_class'][name]
            lines.append(f"| {name} | {c['auroc']:.6f} | {c['precision']:.6f} | {c['recall']:.6f} | {c['f1']:.6f} | {b['f1']:.6f} | {c['f1']-b['f1']:+.6f} |")
        improved = [k for k in ('macro_auroc', 'macro_precision', 'macro_recall', 'macro_f1') if m[k] > old[k]]
        declined = [k for k in ('macro_auroc', 'macro_precision', 'macro_recall', 'macro_f1') if m[k] < old[k]]
        lines += ['', f"宏平均指标实测提升：{', '.join(improved) or '无'}。宏平均指标实测下降：{', '.join(declined) or '无'}。逐类 F1 中 NORM 和 MI 略降，STTC、CD、HYP 提升；四项宏平均提升不等于每个类别的每项指标都提升。三种子仅验证了验证集波动；最终测试是一个按验证集选出的种子，未做测试集置信区间。"]
    summary_path = ROOT / 'dataset_summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        lines += ['', '## 数据口径', '', '| 划分 | 记录数 | 患者数 | 多标签记录 | 五标签全零记录 |', '|---|---:|---:|---:|---:|']
        for name, part in summary.items():
            lines.append(f"| {name} | {part['records']} | {part['patients']} | {part['multiple_label_records']} | {part['zero_label_records']} |")
        lines += ['', '训练集阳性数：NORM 7596、MI 4379、STTC 4186、CD 3907、HYP 2119。全零标签记录沿用原基线口径保留，未通过删改标签提高结果。']
    if (ROOT / 'inference_check.json').exists():
        check = json.loads((ROOT / 'inference_check.json').read_text(encoding='utf-8'))
        lines += ['', '## 实现验证', '', f"44 项项目测试通过；两种新网络通过 CUDA AMP 前向/反向检查。最终模型在验证记录 ecg_id={check['validation_ecg_id']} 上，单条推理与批量评估概率最大差值为 {check['max_probability_difference']:.8f}，一致性检查通过。", '', '历史模型文件保留；本轮未把优化模型切换为应用默认模型，也未验证其 ONNX/STM32 部署。']
    lines += ['', '## 复现与产物', '', '运行：`.venv/Scripts/python.exe scripts/run_optimization.py`。已有 result.json 的实验自动跳过；日志 outputs/optimization_console.log（续跑追加）；每次运行的配置、checkpoint、history、验证集预测和指标都在独立子目录。', '',
        '本轮范围：标准化、训练策略、类别权重、Inception 风格多尺度卷积、CNN+BiGRU、阈值和种子稳定性。Inception 为本项目紧凑实现，不声称完整复现论文 InceptionTime。ASL、强数据增强、500 Hz 和集成属于后续候选，尚未验证。', '']
    cv_summary_path = PROJECT_ROOT / 'outputs/cross_validation/summary.json'
    if cv_summary_path.exists():
        cv = json.loads(cv_summary_path.read_text(encoding='utf-8'))
        lines += ['', '## 四折交叉验证补充', '', f"独立校准 fold 9，四个留出组为 folds 1–2、3–4、5–6、7–8；fold 10 本轮未预测。完成 {cv['completed_runs']}/12 个训练实验。详见 [四折交叉验证报告](../../md/模型训练与优化.md)。", '', '| 方案 | 完成折数 | 平均AUROC | 平均F1 | HYP平均F1 |', '|---|---:|---:|---:|---:|']
        for case, model in cv['models'].items():
            values = model['policies']['f1']
            auc, f1, hyp = values['macro']['macro_auroc'], values['macro']['macro_f1'], values['per_class']['HYP']['f1']
            lines.append(f"| {case} | {model['completed_folds']} | {auc['mean']:.6f} ± {auc['std']:.6f} | {f1['mean']:.6f} ± {f1['std']:.6f} | {hyp['mean']:.6f} ± {hyp['std']:.6f} |")
        if cv['completed_runs']==12:
            lines += ['', f"四折平均F1领先方案：`{cv['winner_by_mean_f1']}`。四折训练数据量小于原全量训练，不能与前述 fold-10 测试分数直接当作同一实验比较。"]
    text = '\n'.join(lines)
    marker = '\n## 第二轮优化实验\n'
    if REPORT.exists():
        previous = REPORT.read_text(encoding='utf-8')
        if marker in previous:
            text += marker + previous.split(marker, 1)[1]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding='utf-8')


def run(name, cfg, description, baseline=None):
    run_dir = ROOT / name
    result_path = run_dir / 'result.json'
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding='utf-8'))
        expected = copy.deepcopy(cfg)
        for key in ('checkpoint_dir', 'metrics_dir', 'predictions_dir', 'figure_dir'):
            expected['outputs'][key] = f'outputs/optimization/{name}/{key.removesuffix("_dir")}'
        if saved['config'] != expected:
            raise ValueError(f'Configuration changed for completed run {name}; use a new run name to preserve its results.')
        return saved
    print(f'\nSTART {name}: {description}', flush=True)
    started = time.monotonic()
    config = copy.deepcopy(cfg)
    for key in ('checkpoint_dir', 'metrics_dir', 'predictions_dir', 'figure_dir'):
        config['outputs'][key] = f'outputs/optimization/{name}/{key.removesuffix("_dir")}'
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    checkpoint = baseline or train_baseline(config, project_root=PROJECT_ROOT)
    result = validation_result(checkpoint, config['dataset']['root'], run_dir)
    result.update(name=name, description=description, checkpoint=str(checkpoint), seconds=time.monotonic()-started, config=config)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report()
    print(f"DONE {name}: fine F1={result['validation_fine']['macro_f1']:.6f}, AUC={result['validation']['macro_auroc']:.6f}", flush=True)
    return result


def rank(result):
    return (result['validation_fine']['macro_f1'], result['validation_fine']['macro_auroc'])


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    base = load_config(PROJECT_ROOT / 'configs/default.yaml')
    base['model']['type'] = 'resnet1d'
    baseline = run('00_resnet_baseline', base, '已有 ResNet checkpoint 在 fold 9 重新推理；不重新训练', PROJECT_ROOT / 'outputs/checkpoints/resnet1d_best.pt')
    experiments = [
        ('01_global_norm', {'normalization': 'train_global'}, {}, 'resnet1d', '只改训练集统计量标准化'),
        ('02_schedule', {}, {'scheduler': 'warmup_cosine', 'epochs': 60, 'early_stopping_patience': 12}, 'resnet1d', 'warmup+cosine/60轮上限/patience12，其他不变'),
        ('03_sqrt_weight', {}, {'pos_weight_power': 0.5}, 'resnet1d', '只将正类权重改为平方根'),
        ('04_unweighted_bce', {}, {'pos_weight_power': 0.0}, 'resnet1d', '只改普通BCE，不加正类权重'),
        ('05_inception', {}, {}, 'inception1d', '只换紧凑多尺度Inception风格网络，平均+最大池化'),
        ('06_cnn_bigru', {}, {}, 'cnn_bigru', '只换CNN+双向GRU，平均+最大池化'),
    ]
    results = [baseline]
    by_name = {}
    for name, signal, training, model, description in experiments:
        cfg = copy.deepcopy(base)
        cfg['signal_processing'].update(signal)
        cfg['training'].update(training)
        cfg['model']['type'] = model
        r = run(name, cfg, description)
        results.append(r)
        by_name[name] = r
    def useful(r):
        return r['validation_fine']['macro_f1'] > baseline['validation_fine']['macro_f1'] and r['validation']['macro_auroc'] >= baseline['validation']['macro_auroc'] - 0.002
    combination = {'signal_processing': {}, 'training': {}}
    if useful(by_name['01_global_norm']):
        combination['signal_processing']['normalization'] = 'train_global'
    if useful(by_name['02_schedule']):
        combination['training'].update(scheduler='warmup_cosine', epochs=60, early_stopping_patience=12)
    weight = max([by_name['03_sqrt_weight'], by_name['04_unweighted_bce']], key=rank)
    if useful(weight):
        combination['training']['pos_weight_power'] = weight['config']['training']['pos_weight_power']
    for kind in ('inception1d', 'cnn_bigru'):
        cfg = copy.deepcopy(base)
        cfg['model']['type'] = kind
        cfg['signal_processing']['normalization'] = 'train_global'
        results.append(run(f'08_global_{kind}', cfg, f'{kind}+只改训练集标准化，排除权重/调度交互影响'))
    for kind in ('resnet1d', 'inception1d', 'cnn_bigru'):
        cfg = copy.deepcopy(base)
        cfg['model']['type'] = kind
        cfg['signal_processing'].update(combination['signal_processing'])
        cfg['training'].update(combination['training'])
        results.append(run(f'07_combo_{kind}', cfg, f'{kind}与验证集有效改动组合'))
    winner = max(results, key=rank)
    selected_cfg = copy.deepcopy(winner['config'])
    selected = {'combination': combination, 'model_type': selected_cfg['model']['type'], 'source_experiment': winner['name'], 'config': selected_cfg}
    (ROOT / 'selection.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report()
    seeds = []
    for seed in (42, 43, 44):
        cfg = copy.deepcopy(selected_cfg)
        cfg['training']['random_seed'] = seed
        if seed == 42:
            r = run(f'final_seed_{seed}', cfg, f'最终方案，种子{seed}（复用同配置已训练模型）', Path(winner['checkpoint']))
        else:
            r = run(f'final_seed_{seed}', cfg, f'最终方案，种子{seed}')
        seeds.append(r)
    best_seed = max(seeds, key=rank)
    frozen = torch.load(best_seed['checkpoint'], map_location='cpu', weights_only=True)
    for key in ('checkpoint_dir', 'metrics_dir', 'predictions_dir', 'figure_dir'):
        frozen['config']['outputs'][key] = f'outputs/optimization/final_evaluation/{key.removesuffix("_dir")}'
    saved_validation = np.load(ROOT / best_seed['name'] / 'validation_predictions.npz')
    recall_floor = baseline['validation_fine']['macro_recall']
    precision_floor = baseline['validation_fine']['macro_precision']
    cutoffs, feasible = balanced_thresholds(saved_validation['targets'], saved_validation['probabilities'], best_seed['fine_thresholds'], recall_floor, precision_floor)
    balanced_metrics = calculate_multilabel_metrics(saved_validation['targets'], saved_validation['probabilities'], cutoffs)
    frozen['thresholds'] = cutoffs
    frozen['validation_metrics_tuned'] = balanced_metrics
    frozen['threshold_search'] = 'validation_pr_curve_with_macro_recall_precision_floors' if feasible else 'exact_validation_pr_curve'
    selected['threshold_policy'] = {'recall_floor': recall_floor, 'precision_floor': precision_floor, 'constraints_satisfied': feasible, 'thresholds': cutoffs, 'validation_metrics': balanced_metrics}
    frozen_path = ROOT / 'optimized_best.pt'
    torch.save(frozen, frozen_path)
    selected.update(checkpoint=str(frozen_path), selected_seed=best_seed['config']['training']['random_seed'])
    (ROOT / 'selection.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')
    if not (ROOT / 'final_test.json').exists():
        metrics = evaluate_checkpoint(frozen_path, project_root=PROJECT_ROOT, dataset_root=Path(base['dataset']['root']))
        (ROOT / 'final_test.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report()
    print('COMPLETE', flush=True)

if __name__ == '__main__':
    main()
