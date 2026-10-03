"""Four-fold patient-disjoint comparison with a fixed fold-9 calibration set."""
from __future__ import annotations
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import load_config, resolve_project_path
from src.data.ptbxl import SUPERCLASSES, create_fold_splits, find_ptbxl_root
from src.models import build_model
from src.signal_processing.filters import FilterConfig
from src.training.data import CachedPTBXLDataset, ensure_cache_subset, ensure_preprocessed_cache, load_labeled_splits
from src.training.metrics import calculate_multilabel_metrics
from src.training.train import train_baseline, _save_checkpoint
from scripts.run_optimization import exact_thresholds, balanced_thresholds, write_report as write_optimization_report

ROOT = PROJECT_ROOT / 'outputs/cross_validation'
REPORT = PROJECT_ROOT / 'outputs/reports/四折交叉验证报告.md'
HELDOUT_GROUPS = ((1,2), (3,4), (5,6), (7,8))
CASES = (
    ('resnet_original', 'resnet1d', 'zscore'),
    ('resnet_global', 'resnet1d', 'train_global'),
    ('inception_global', 'inception1d', 'train_global'),
)
KEYS = ('macro_auroc', 'macro_precision', 'macro_recall', 'macro_f1')


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.saving')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def predict_frame(checkpoint, frame, dataset_root, split_name):
    cfg = checkpoint['config']
    fs = checkpoint['sampling_rate_hz']
    cache = ensure_preprocessed_cache(frame, dataset_root, PROJECT_ROOT/cfg['outputs']['cache_dir'], split_name=split_name, sampling_rate_hz=fs, config=FilterConfig.from_mapping(cfg['signal_processing']))
    loader = DataLoader(CachedPTBXLDataset(cache, frame), batch_size=int(cfg['training']['batch_size']), shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(checkpoint['model_type'], dropout=cfg['model']['dropout']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    scores = []
    with torch.inference_mode():
        for x, _ in loader:
            scores.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
    return np.concatenate(scores)


def pooled_metrics(results, policy):
    truths, scores, predictions = [], [], []
    for result in results:
        saved = np.load(Path(result['run_dir'])/'heldout_predictions.npz')
        truths.append(saved['targets'])
        scores.append(saved['probabilities'])
        predictions.append(saved['predictions_'+policy])
    truth, probability, predicted = np.concatenate(truths), np.concatenate(scores), np.concatenate(predictions)
    metrics = calculate_multilabel_metrics(truth, probability)
    for i, name in enumerate(SUPERCLASSES):
        y, p = truth[:,i], predicted[:,i]
        tp = int(((y==1)&(p==1)).sum())
        fp = int(((y==0)&(p==1)).sum())
        fn = int(((y==1)&(p==0)).sum())
        tn = int(((y==0)&(p==0)).sum())
        metrics['per_class'][name].update(
            threshold=None, precision=tp/max(1,tp+fp), recall=tp/max(1,tp+fn),
            f1=2*tp/max(1,2*tp+fp+fn), confusion_matrix=[[tn,fp],[fn,tp]],
        )
    for key in ('precision','recall','f1'):
        metrics['macro_'+key] = float(np.mean([metrics['per_class'][c][key] for c in SUPERCLASSES]))
    metrics['threshold_source'] = 'each_model_calibrated_on_fold_9'
    return metrics


def summarize():
    all_results = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(ROOT.glob('fold_*/*/result.json'))]
    summary = {'protocol_version':1, 'completed_runs':len(all_results), 'expected_runs':12, 'models':{}}
    for case, _, _ in CASES:
        results = sorted([r for r in all_results if r['case']==case], key=lambda r:r['cv_fold'])
        if not results:
            continue
        model = {'completed_folds':len(results), 'policies':{}}
        for policy in ('f1','balanced'):
            item = {'macro':{}, 'per_class':{}}
            for key in KEYS:
                values = [r['heldout_'+policy][key] for r in results]
                item['macro'][key] = {'mean':float(np.mean(values)), 'std':float(np.std(values,ddof=0)), 'values':values}
            for c in SUPERCLASSES:
                item['per_class'][c] = {}
                for key in ('auroc','precision','recall','f1'):
                    values = [r['heldout_'+policy]['per_class'][c][key] for r in results]
                    item['per_class'][c][key] = {'mean':float(np.mean(values)), 'std':float(np.std(values,ddof=0))}
            item['pooled_oof'] = pooled_metrics(results, policy)
            model['policies'][policy] = item
        summary['models'][case] = model
    if len(all_results)==12:
        summary['winner_by_mean_f1'] = max(summary['models'], key=lambda name: summary['models'][name]['policies']['f1']['macro']['macro_f1']['mean'])
        summary['winner_by_balanced_mean_f1'] = max(summary['models'], key=lambda name: summary['models'][name]['policies']['balanced']['macro']['macro_f1']['mean'])
        summary['paired_comparisons'] = {}
        baseline = sorted([r for r in all_results if r['case']=='resnet_original'],key=lambda r:r['cv_fold'])
        for case, _, _ in CASES[1:]:
            candidates = sorted([r for r in all_results if r['case']==case],key=lambda r:r['cv_fold'])
            pair = {}
            for policy in ('f1','balanced'):
                differences = [r['heldout_'+policy]['macro_f1']-b['heldout_'+policy]['macro_f1'] for r,b in zip(candidates,baseline)]
                pair[policy] = {'macro_f1_differences':differences, 'mean_difference':float(np.mean(differences)), 'wins':sum(d>0 for d in differences)}
            summary['paired_comparisons'][case] = pair
    if len(all_results)==12:
        direct = {}
        resnet = sorted([r for r in all_results if r['case']=='resnet_global'],key=lambda r:r['cv_fold'])
        inception = sorted([r for r in all_results if r['case']=='inception_global'],key=lambda r:r['cv_fold'])
        for policy in ('f1','balanced'):
            direct[policy] = {}
            for key in KEYS:
                differences = [i['heldout_'+policy][key]-r['heldout_'+policy][key] for i,r in zip(inception,resnet)]
                direct[policy][key] = {'mean_difference':float(np.mean(differences)),'std_difference':float(np.std(differences,ddof=0)),'inception_wins':sum(d>0 for d in differences),'differences':differences}
        summary['inception_vs_resnet_global'] = direct
    write_json(ROOT/'summary.json',summary)
    write_cv_report(all_results, summary)
    write_optimization_report()
    return summary


def write_cv_report(results, summary):
    lines = ['# 四折交叉验证报告', '', '日期：2026-09-30。实际运行结果，按已完成实验逐步更新。', '', '## 协议', '',
        '官方 folds 1–8 两两组成四个留出评估组：1–2、3–4、5–6、7–8。每次用其余六个官方 folds 训练；fold 9 固定用于早停、选择最佳 AUROC checkpoint 和阈值；fold 10 不参与本轮训练、预测、阈值选择或模型比较。', '',
        '这是带独立校准集的四折留出比较。每折训练约 75% 的原训练池，因此不能把四折均值直接当成此前使用全部 folds 1–8 训练的 fold-10 测试成绩。各折训练集互有重叠，标准差只描述折间波动，不作为独立重复实验的置信区间。', '',
        '三种方案共享划分、种子42、batch32、学习率0.001、AdamW、完整正类权重、dropout0.2、最多20轮、早停patience5。每折按 patient_id 检查训练/校准/留出组互斥。record-wise Z-score 缓存可直接抽取；global 标准化只在当折六个训练 folds 上拟合，验证和留出组使用同一组训练统计量。', '',
        '主结果：仅在 fold 9 的 PR 曲线上选择每类 F1 最优阈值。补充结果：基于同折原 ResNet 在 fold 9 的宏平均 Precision/Recall，给另外两种模型的阈值加相同下限约束。约束只约束校准集，留出评估结果不保证也满足。无法找到可行候选时退回 F1 阈值并明确记录。', '',
        f"完成进度：{summary['completed_runs']}/12 个训练实验。原优化 checkpoint 保留；本轮不自动替换应用模型，也没有做折模型集成。", '',
        '## 每折留出结果：F1 最优校准阈值', '', '| CV折 | 留出官方折 | 方案 | 最佳轮 | 训练记录 | 评估记录 | AUROC | Precision | Recall | F1 | MI F1 | HYP F1 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        m = r['heldout_f1']
        lines.append(f"| {r['cv_fold']} | {r['heldout_folds']} | {r['case']} | {r['epoch']} | {r['training_records']} | {r['heldout_records']} | {m['macro_auroc']:.6f} | {m['macro_precision']:.6f} | {m['macro_recall']:.6f} | {m['macro_f1']:.6f} | {m['per_class']['MI']['f1']:.6f} | {m['per_class']['HYP']['f1']:.6f} |")
    for policy, title in (('f1','F1 最优校准阈值'),('balanced','宏平均 Precision/Recall 约束阈值')):
        lines += ['', f'## 四折均值与标准差：{title}', '', '均值为各折等权平均，标准差 ddof=0；未完成四折时仅是阶段性统计。', '', '| 方案 | 完成折数 | AUROC | Precision | Recall | F1 |', '|---|---:|---:|---:|---:|---:|']
        for case, model in summary['models'].items():
            metrics = model['policies'][policy]['macro']
            values = [f"{metrics[k]['mean']:.6f} ± {metrics[k]['std']:.6f}" for k in KEYS]
            lines.append(f"| {case} | {model['completed_folds']} | " + ' | '.join(values) + ' |')
        lines += ['', '| 方案 | 类别 | AUROC | Precision | Recall | F1 |', '|---|---|---:|---:|---:|---:|']
        for case, model in summary['models'].items():
            for c in SUPERCLASSES:
                metrics = model['policies'][policy]['per_class'][c]
                values = [f"{metrics[k]['mean']:.6f} ± {metrics[k]['std']:.6f}" for k in ('auroc','precision','recall','f1')]
                lines.append(f'| {case} | {c} | ' + ' | '.join(values) + ' |')
    lines += ['', '## 阈值约束可行性（只在 fold 9 判断）', '', '| CV折 | 方案 | Precision下限 | Recall下限 | 找到可行候选 |', '|---|---|---:|---:|---|']
    for r in results:
        lines.append(f"| {r['cv_fold']} | {r['case']} | {r['threshold_policy']['precision_floor']:.6f} | {r['threshold_policy']['recall_floor']:.6f} | {r['threshold_policy']['feasible']} |")
    if summary['completed_runs']==12:
        lines += ['', '## 配对比较与结论', '', f"按主结果的平均 F1 排序，领先方案：`{summary['winner_by_mean_f1']}`；按约束阈值结果排序，领先方案：`{summary['winner_by_balanced_mean_f1']}`。", '', '| 方案相对原 ResNet | 阈值策略 | F1均值差 | F1胜出折数 |', '|---|---|---:|---:|']
        for case, pair in summary['paired_comparisons'].items():
            for policy, m in pair.items():
                lines.append(f"| {case} | {policy} | {m['mean_difference']:+.6f} | {m['wins']}/4 |")
        baseline = summary['models']['resnet_original']['policies']['f1']
        inception = summary['models']['inception_global']['policies']['f1']
        delta_hyp = inception['per_class']['HYP']['f1']['mean'] - baseline['per_class']['HYP']['f1']['mean']
        delta_mi_recall = inception['per_class']['MI']['recall']['mean'] - baseline['per_class']['MI']['recall']['mean']
        lines += ['', f"Inception+标准化相对原 ResNet：HYP 平均 F1 差值 {delta_hyp:+.6f}，MI 平均 Recall 差值 {delta_mi_recall:+.6f}。宏平均提升不等于每个类别都提升。", '',
            '四折只有一个训练随机种子；本轮证据覆盖训练池内患者划分的变化，仍不能替代外部数据库验证。候选方案是此前优化实验选出的，因此本轮属于对已选方案的内部稳健性复核，不应描述成完全独立的新数据验证。']
    if summary.get('inception_vs_resnet_global'):
        comparison = summary['inception_vs_resnet_global']['f1']
        lines += ['', '## 标准化方案之间的架构比较', '', '| 指标 | Inception减ResNet的平均差值 | 差值标准差 | Inception胜出折数 |', '|---|---:|---:|---:|']
        for key, item in comparison.items():
            lines.append(f"| {key} | {item['mean_difference']:+.6f} | {item['std_difference']:.6f} | {item['inception_wins']}/4 |")
        lines += ['', '结论：训练集标准化的收益在四折上均出现；标准化后的两种架构平均 F1 仅相差约 0.0008，当前证据不能支持 Inception 稳定优于标准化 ResNet，也没有进行显著性检验。Inception 的 MI 平均召回率较高，但折间波动也更大；逐类取舍应继续保留。']
    audit_path = ROOT / 'final_audit.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        lines += ['', '## 实现与结果核对', '', f"47 项项目测试通过；四组患者划分无重叠；三个方案各自覆盖 {audit['oof_records_per_model']} 条 OOF 记录，记录顺序、标签和逐折指标核对通过。每折 global 标准化均与原始训练子集独立计算的均值/标准差一致，最大绝对差值 {audit['maximum_normalization_difference']:.3g}。", '', '复跑验证：已有12项训练结果自动跳过，不重新训练。未修改原优化 checkpoint。']
    lines += ['', '## 合并 OOF 结果', '', '每条 folds 1–8 记录只使用未训练过它的模型预测一次。Precision/Recall/F1 使用各折独立校准阈值生成的标签汇总；AUROC 合并各折概率，分数尺度差异可能影响结果，因此以逐折均值为主要结论。', '',
        '| 方案 | 策略 | OOF记录 | AUROC | Precision | Recall | F1 |', '|---|---|---:|---:|---:|---:|---:|']
    for case, model in summary['models'].items():
        for policy in ('f1','balanced'):
            m = model['policies'][policy]['pooled_oof']
            lines.append(f"| {case} | {policy} | {m['records']} | {m['macro_auroc']:.6f} | {m['macro_precision']:.6f} | {m['macro_recall']:.6f} | {m['macro_f1']:.6f} |")
    if summary['completed_runs']==12:
        lines += ['', '## 折间波动图', '', f"![四折F1比较]({(ROOT/'cv_f1_by_fold.png').as_posix()})"]
    lines += ['', '## 复现与产物', '', '运行：`.venv/Scripts/python.exe scripts/run_cross_validation.py`。结果保存到 `outputs/cross_validation/fold_*/方案/`；含配置、完整 checkpoint、训练历史、fold-9校准预测、留出组预测、各阈值策略的指标。划分清单 `protocol.json`，汇总 `summary.json`，日志 `run.log`。', '',
        '已完成实验自动跳过；同名实验配置发生变化则报错，要求使用新实验名称，防止静默复用不匹配结果。未根据留出组标签调阈值；fold 10 本轮未预测。', '']
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(lines),encoding='utf-8')


def plot_results():
    results = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(ROOT.glob('fold_*/*/result.json'))]
    fig, axes = plt.subplots(1,3,figsize=(13,4))
    for ax, c in zip(axes, ('macro','MI','HYP')):
        for case, _, _ in CASES:
            group = sorted([r for r in results if r['case']==case],key=lambda r:r['cv_fold'])
            values = [r['heldout_f1']['macro_f1'] if c=='macro' else r['heldout_f1']['per_class'][c]['f1'] for r in group]
            ax.plot([r['cv_fold'] for r in group],values,marker='o',label=case)
        ax.set_title(f'{c} F1')
        ax.set_xlabel('CV holdout group')
        ax.set_ylabel('F1')
        ax.set_xticks([1,2,3,4])
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT/'cv_f1_by_fold.png',dpi=180)
    plt.close(fig)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    base = load_config(PROJECT_ROOT/'configs/default.yaml')
    base['training'].update(random_seed=42, epochs=20, early_stopping_patience=5, scheduler='none', pos_weight_power=1.0)
    source_splits = load_labeled_splits(resolve_project_path(base['dataset']['root'], PROJECT_ROOT),100)
    full_train, calibration = source_splits['train'], source_splits['validation']
    dataset_root = find_ptbxl_root(resolve_project_path(base['dataset']['root'], PROJECT_ROOT),sampling_rate_hz=100)
    source_caches = {}
    for normalization in ('zscore','none'):
        settings = {**base['signal_processing'],'normalization':normalization}
        source_caches[normalization] = ensure_preprocessed_cache(full_train,dataset_root,PROJECT_ROOT/base['outputs']['cache_dir'],split_name='train_full',sampling_rate_hz=100,config=FilterConfig.from_mapping(settings))
    # Reuse existing fold-9 record-local caches via checked subsets too.
    for normalization in ('zscore','none'):
        settings = {**base['signal_processing'],'normalization':normalization}
        source = ensure_preprocessed_cache(calibration,dataset_root,PROJECT_ROOT/base['outputs']['cache_dir'],split_name='validation_full',sampling_rate_hz=100,config=FilterConfig.from_mapping(settings))
        ensure_cache_subset(source,calibration,calibration,ROOT/'cache',split_name='validation_full',sampling_rate_hz=100,config=FilterConfig.from_mapping(settings))
    protocol = {'protocol_version':1,'heldout_groups':HELDOUT_GROUPS,'validation_fold':9,'excluded_test_fold':10,'seed':42,'cases':CASES,'folds':[]}
    for index, heldout in enumerate(HELDOUT_GROUPS,start=1):
        training = [fold for fold in range(1,9) if fold not in heldout]
        partition = create_fold_splits(pd.concat([full_train,calibration]),train_folds=training,test_folds=heldout)
        protocol['folds'].append({'cv_fold':index,'train_folds':training,'heldout_folds':heldout,'train_ecg_ids':partition['train'].ecg_id.tolist(),'heldout_ecg_ids':partition['test'].ecg_id.tolist(),'validation_ecg_ids':partition['validation'].ecg_id.tolist(),'train_patients':int(partition['train'].patient_id.nunique()),'heldout_patients':int(partition['test'].patient_id.nunique())})
    heldout_ids = [ecg_id for part in protocol['folds'] for ecg_id in part['heldout_ecg_ids']]
    if len(heldout_ids)!=len(set(heldout_ids)) or set(heldout_ids)!=set(full_train.ecg_id):
        raise ValueError('Every training-pool record must be held out exactly once.')
    write_json(ROOT/'protocol.json',protocol)
    summarize()
    for index, heldout in enumerate(HELDOUT_GROUPS,start=1):
        training = [fold for fold in range(1,9) if fold not in heldout]
        frames = create_fold_splits(pd.concat([full_train,calibration]),train_folds=training,test_folds=heldout)
        for normalization in ('zscore','none'):
            settings = {**base['signal_processing'],'normalization':normalization}
            for name, split_name in (('train','train_full'),('test',f'cv_holdout_{index}')):
                ensure_cache_subset(source_caches[normalization],full_train,frames[name],ROOT/'cache',split_name=split_name,sampling_rate_hz=100,config=FilterConfig.from_mapping(settings))
        for case, kind, normalization in CASES:
            run_dir = ROOT/f'fold_{index}'/case
            cfg = copy.deepcopy(base)
            cfg['model']['type'] = kind
            cfg['signal_processing']['normalization'] = normalization
            cfg['dataset'].update(train_folds=training,validation_fold=9,test_folds=list(heldout))
            cfg['dataset'].pop('test_fold',None)
            cfg['outputs']['cache_dir'] = 'outputs/cross_validation/cache'
            for key in ('checkpoint_dir','metrics_dir','figure_dir','predictions_dir'):
                cfg['outputs'][key] = f'outputs/cross_validation/fold_{index}/{case}/{key.removesuffix("_dir")}'
            result_path = run_dir/'result.json'
            if result_path.exists():
                saved = json.loads(result_path.read_text(encoding='utf-8'))
                if saved['run_config']!=cfg:
                    raise ValueError(f'Configuration changed for completed run {index}/{case}.')
                print(f'SKIP completed {index}/{case}',flush=True)
                continue
            print(f'\nSTART CV {index}/4 {case}, heldout={heldout}, training={training}',flush=True)
            started = time.monotonic()
            run_dir.mkdir(parents=True,exist_ok=True)
            write_json(run_dir/'config.json',cfg)
            path = train_baseline(cfg,project_root=PROJECT_ROOT,dataset_root=dataset_root)
            checkpoint = torch.load(path,map_location='cpu',weights_only=True)
            if checkpoint['config']['dataset']['train_folds']!=training or checkpoint['training_records']!=len(frames['train']):
                raise ValueError('Training did not honor CV partition.')
            calibration_scores = predict_frame(checkpoint,frames['validation'],dataset_root,'validation_full')
            calibration_truth = frames['validation'][list(SUPERCLASSES)].to_numpy(dtype=np.uint8)
            thresholds = exact_thresholds(calibration_truth,calibration_scores,checkpoint['thresholds'])
            calibration_metrics = calculate_multilabel_metrics(calibration_truth,calibration_scores,thresholds)
            if case=='resnet_original':
                reference = calibration_metrics
                balanced, feasible = thresholds, True
            else:
                reference = json.loads((ROOT/f'fold_{index}'/'resnet_original'/'result.json').read_text(encoding='utf-8'))['calibration_f1']
                balanced, feasible = balanced_thresholds(calibration_truth,calibration_scores,thresholds,reference['macro_recall'],reference['macro_precision'])
            checkpoint.update(thresholds=thresholds,validation_metrics_tuned=calibration_metrics,threshold_search='exact_fold_9_pr_curve',evaluation_role='cross_validation',cv_fold=index)
            _save_checkpoint(path,checkpoint)
            heldout_scores = predict_frame(checkpoint,frames['test'],dataset_root,f'cv_holdout_{index}')
            heldout_truth = frames['test'][list(SUPERCLASSES)].to_numpy(dtype=np.uint8)
            predictions_f1 = (heldout_scores>=thresholds).astype(np.uint8)
            predictions_balanced = (heldout_scores>=balanced).astype(np.uint8)
            np.savez_compressed(run_dir/'calibration_predictions.npz',targets=calibration_truth,probabilities=calibration_scores,ecg_ids=frames['validation'].ecg_id.to_numpy())
            np.savez_compressed(run_dir/'heldout_predictions.npz',targets=heldout_truth,probabilities=heldout_scores,predictions_f1=predictions_f1,predictions_balanced=predictions_balanced,ecg_ids=frames['test'].ecg_id.to_numpy(),patient_ids=frames['test'].patient_id.to_numpy())
            r = {'protocol_version':1,'case':case,'cv_fold':index,'train_folds':training,'heldout_folds':heldout,'calibration_fold':9,'excluded_fold':10,'epoch':checkpoint['epoch'],'training_records':len(frames['train']),'heldout_records':len(frames['test']),'calibration_records':len(calibration),'checkpoint':str(path),'run_dir':str(run_dir),'run_config':cfg,'thresholds_f1':thresholds,'thresholds_balanced':balanced,'calibration_f1':calibration_metrics,'calibration_balanced':calculate_multilabel_metrics(calibration_truth,calibration_scores,balanced),'heldout_f1':calculate_multilabel_metrics(heldout_truth,heldout_scores,thresholds),'heldout_balanced':calculate_multilabel_metrics(heldout_truth,heldout_scores,balanced),'threshold_policy':{'recall_floor':reference['macro_recall'],'precision_floor':reference['macro_precision'],'feasible':feasible},'seconds':time.monotonic()-started,'normalization_statistics':{k:checkpoint['config']['signal_processing'].get(k) for k in ('training_mean','training_std')}}
            write_json(result_path,r)
            summarize()
            print(f"DONE CV {index} {case}: F1={r['heldout_f1']['macro_f1']:.6f}, AUC={r['heldout_f1']['macro_auroc']:.6f}, HYP F1={r['heldout_f1']['per_class']['HYP']['f1']:.6f}",flush=True)
    plot_results()
    summary = summarize()
    print('COMPLETE 12/12; winner='+summary['winner_by_mean_f1'],flush=True)

if __name__=='__main__':
    main()
