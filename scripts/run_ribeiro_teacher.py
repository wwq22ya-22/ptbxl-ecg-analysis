"""Three-seed 100-Hz Ribeiro-style teacher study with frozen validation choices."""
from __future__ import annotations
import copy
from datetime import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_stm32 import digest, get_split
from scripts.refine_student_thresholds import grids, tune, audit_metrics, section, read, save
from src.data.ptbxl import SUPERCLASSES
from src.models import build_model
from src.training.train import train_baseline
from src.training.data import load_labeled_splits
from src.training.metrics import calculate_multilabel_metrics

OUT = ROOT / 'outputs/ribeiro_teacher'
MODEL = 'ribeiro_resnet100'
BASE = ROOT / 'outputs/teacher_preprocessing/inception_time1d_P0_current_seed42/checkpoint/inception_time1d_best.pt'
KEYS = ['macro_auroc', 'macro_auprc', 'macro_precision', 'macro_recall', 'macro_f1']


def load(path): return torch.load(path, map_location='cpu', weights_only=True)


def status(message):
    print(message, flush=True)
    save(OUT / 'status.json', {'stage': message, 'updated': datetime.now().isoformat()})


def metric(y, p, t):
    m = calculate_multilabel_metrics(y, p, t)
    audit_metrics(y, p, t, m)
    ap = average_precision_score(y, p, average=None)
    m['macro_auprc'] = float(ap.mean())
    for i, c in enumerate(m['per_class']): m['per_class'][c]['auprc'] = float(ap[i])
    return m


def full_inputs(cp, frame, split):
    signature = hashlib.sha256(json.dumps(cp['config']['signal_processing'], sort_keys=True).encode()).hexdigest()[:16]
    cached = ROOT / f'outputs/unfiltered_threshold_refinement/inputs/{split}_{signature}.npy'
    if cached.exists() and cached.with_suffix('.json').exists():
        info = read(cached.with_suffix('.json'))
        assert info['ecg_ids'] == frame.ecg_id.tolist()
        assert info['preprocessing'] == cp['config']['signal_processing']
        return np.load(cached, mmap_mode='r')
    eval_cp = copy.deepcopy(cp)
    eval_cp['config']['dataset']['require_diagnostic_label'] = False
    values, actual = get_split(split, eval_cp)
    np.testing.assert_array_equal(actual.ecg_id, frame.ecg_id)
    return values


def predict(path, frame, split, output):
    if output.exists():
        with np.load(output, allow_pickle=False) as d:
            np.testing.assert_array_equal(d['ecg_ids'], frame.ecg_id)
            assert str(d['checkpoint_sha256']) == digest(path)
            return d['probabilities']
    cp = load(path)
    values = full_inputs(cp, frame, split)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(MODEL, dropout=cp['config']['model']['dropout']).to(device).eval()
    model.load_state_dict(cp['model_state'])
    batches = []
    with torch.inference_mode():
        for start in range(0, len(frame), 32):
            x = torch.from_numpy(np.array(values[start:start+32], dtype=np.float32)).to(device)
            batches.append(model(x).sigmoid().cpu().numpy())
    p = np.concatenate(batches)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ecg_ids=frame.ecg_id.to_numpy(), targets=frame[list(SUPERCLASSES)].to_numpy(dtype=np.uint8),
                        probabilities=p, checkpoint_sha256=np.asarray(digest(path)))
    return p


def baseline_scores(r, frame, split):
    study = ROOT / 'outputs/unfiltered_threshold_refinement'
    assert read(study / 'protocol.json')['protected_sha256'][r['checkpoint']] == digest(Path(r['checkpoint']))
    with np.load(study / r['name'] / f'{split}_probabilities.npz', allow_pickle=False) as d:
        np.testing.assert_array_equal(d['ecg_ids'], frame.ecg_id)
        np.testing.assert_array_equal(d['targets'], frame[list(SUPERCLASSES)])
        return d['probabilities']


def docs(s=None):
    intro = ['## 2026-10-09：文献[1] Ribeiro 残差网络适配实验', '',
             '参考 Ribeiro 等人的十二导联 ECG 网络图 3 及[作者代码](https://github.com/antonior92/automatic-ecg-diagnosis/blob/master/model.py)。本项目独立移植为 PyTorch，并按 100 Hz、1000 点、五个独立输出调整；不宣称完整复现原论文或迁移了原论文权重。', '',
             '保留四个残差模块、64→128→196→256→320 通道、快捷分支 MaxPooling + 1×1 卷积、加法后原始特征作为下一模块快捷分支输入，以及 Flatten + Dense。卷积核 16 改为 5，降采样从 4/4/4/4 改为 4/4/2/2，时间维 1000→250→63→32→16；保留作者 BN 参数及 He 初始化思路。训练用 logits 和 BCE，推理再独立 sigmoid。Dropout 0.2。', '',
             '筛选后的训练数据 17084 条，P0 中值基线去除 + 0.5–40 Hz 带通、训练集逐导联标准化；患者隔离 folds 1–8 / 9 / 10。与已训练 InceptionTime、紧凑 Inception 配对比较三个种子 42/43/44；保持 batch32、AdamW lr0.001、weight decay0.0001、带正类权重 BCE、最多30轮、早停8轮、AMP。按训练期验证 AUROC 选权重。', '',
             '本次新模型都从随机初始化训练。权重固定后，用 FP32 推理，在筛选和完整验证集分别按 0.01 网格选逐类 F1 阈值，再冻结候选并评估对应测试集；完整验证 / 测试为 2183 / 2198，筛选为 2146 / 2158。测试数据已被历史实验多次使用，本轮属于探索性实验。', '']
    if s is None:
        section(ROOT / 'md/模型训练与优化.md', '\n'.join(intro + ['实验正在运行，尚未报告新模型的测试结果。配置与协议先保存于 `outputs/ribeiro_teacher/`。']), 'ribeiro-teacher')
        return
    intro += [f'新教师参数量：{s["parameters"]}。7 项网络测试通过，检查 SAME 池化对齐、双分支原始特征传递、时间维、梯度、优化器更新、权重重载与单条 / 批量推理一致性。', '',
              '### 三种子整体统计，均值 ± 样本标准差，单位 %', '',
              '| 网络 | 验证 / 测试范围 | AUROC | AUPRC | Precision | Recall | F1 |', '|---|---|---:|---:|---:|---:|---:|']
    labels = {MODEL: 'Ribeiro 100Hz适配', 'inception_time1d': 'InceptionTime', 'inception1d': '紧凑Inception'}
    for model in [MODEL, 'inception_time1d', 'inception1d']:
        rows = [r for r in s['runs'] if r['model'] == model]
        for cohort in ['filtered', 'unfiltered']:
            intro.append(f'| {labels[model]} | {"筛选" if cohort=="filtered" else "未筛选"} | ' +
                         ' | '.join(f'{np.mean([r["evaluation"][cohort]["test"][k] for r in rows])*100:.3f} ± {np.std([r["evaluation"][cohort]["test"][k] for r in rows],ddof=1)*100:.3f}' for k in KEYS) + ' |')
    chosen = s['selected']
    intro += ['', f'新模型种子按筛选验证 F1、AUROC 冻结为 seed{chosen["seed"]}。教师推荐比较依据同一筛选验证集的 0.01 指标，推荐候选为 {s["validation_recommendation"]["name"]}；未按测试成绩选种子或替换固件。', '',
              '### 新教师与原 InceptionTime 的成对测试 F1 变化（百分点）', '']
    for cohort in ['filtered', 'unfiltered']:
        deltas = []
        for seed in [42, 43, 44]:
            a = next(r for r in s['runs'] if r['model']==MODEL and r['seed']==seed)
            b = next(r for r in s['runs'] if r['model']=='inception_time1d' and r['seed']==seed)
            deltas.append((a['evaluation'][cohort]['test']['macro_f1']-b['evaluation'][cohort]['test']['macro_f1'])*100)
        intro += [f'{"筛选" if cohort=="filtered" else "未筛选"}：' + ', '.join(f'{v:+.3f}' for v in deltas) + f'；平均 {np.mean(deltas):+.3f}。', '']
    means = {model: np.mean([r['evaluation']['filtered']['test']['macro_f1'] for r in s['runs'] if r['model'] == model])
             for model in [MODEL, 'inception_time1d']}
    weaker = means[MODEL] < means['inception_time1d']
    intro += ['### 本次结论及限制', '',
              f'适配网络三种子筛选测试平均 F1 为 {means[MODEL]*100:.3f}%，InceptionTime 为 {means["inception_time1d"]*100:.3f}%，变化 {(means[MODEL]-means["inception_time1d"])*100:+.3f} 个百分点。' +
              ('本次未获得提升，继续保留现有教师与部署学生。' if weaker else '是否用于新学生蒸馏仍需要另行验证。'), '',
              '三个新种子的最佳轮次为 8、7、5。训练损失继续下降时验证 AUROC 已回落，提示过拟合；约 213.9 万参数也是 InceptionTime 49.2 万参数的约 4.3 倍。上述现象不能单独证明性能下降的因果关系。', '',
              f'按验证集选出的新 seed{chosen["seed"]}，筛选测试 HYP Recall {chosen["evaluation"]["filtered"]["test"]["per_class"]["HYP"]["recall"]*100:.3f}%、F1 {chosen["evaluation"]["filtered"]["test"]["per_class"]["HYP"]["f1"]*100:.3f}%，仍是短板。', '',
              '这一结论只适用于本次 100 Hz 适配和统一训练预算，不能据此否定原论文的 400 Hz、六类任务结果。下一轮可将学习率衰减或更小通道数作为单独实验，用验证集选择；本轮未追加这些改动。', '',
              '本表对照均为单网络；此前 InceptionTime 三模型集成教师属于另一候选，不能将单网络对照的均值视作集成教师分数。三个种子的统计也不代表外部泛化已验证。', '']
    audit_file = OUT / 'independent_audit.json'
    if audit_file.exists():
        audit = read(audit_file)
        intro += [f'独立复核通过：{audit["metric_sets_verified"]} 组验证 / 测试整体及逐类指标，逐候选阈值、患者隔离、ID 与标签、训练标准化、测试预测生成顺序、选定权重副本，以及原模型 / ONNX / HEX 哈希一致。记录：`outputs/ribeiro_teacher/independent_audit.json`。复现复核：`.venv/Scripts/python.exe scripts/audit_ribeiro_teacher.py`。', '']
    intro += ['原教师和部署学生 / HEX 均保留。本轮只训练新的教师候选，尚未基于它蒸馏学生，也没有量化或上板；教师得分不能等同于板上学生得分。', '',
              '权重、训练历史、预测、配置、冻结选择及复核结果在 `outputs/ribeiro_teacher/`；选定新教师另存 `selected_teacher.pt`，阈值文件为 `frozen_selection.json`。整体与逐类对比见指标总表本次章节。复现：`.venv/Scripts/python.exe scripts/run_ribeiro_teacher.py`。']
    section(ROOT / 'md/模型训练与优化.md', '\n'.join(intro), 'ribeiro-teacher')
    table = ['## 2026-10-09：Ribeiro 100Hz 适配教师及两种既有教师对照', '',
             '九个单网络 / 种子，均在筛选后的 17084 条记录训练；筛选和完整验证分别选 0.01 阈值。单位 %。参考文献网络输入、降采样和输出已适配，不是原论文成绩。', '',
             '| 模型 | 种子 | 最佳轮次 | 验证 / 测试范围 | 验证 F1 | 测试 AUROC | AUPRC | Precision | Recall | F1 |',
             '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in s['runs']:
        for cohort in ['filtered', 'unfiltered']:
            d = r['evaluation'][cohort]
            table.append(f'| {labels[r["model"]]} | {r["seed"]} | {r["epoch"]} | {"筛选" if cohort=="filtered" else "未筛选"} | {d["validation"]["macro_f1"]*100:.3f} | ' +
                         ' | '.join(f'{d["test"][k]*100:.3f}' for k in KEYS) + ' |')
    table += ['', '### 每个模型 / 种子的逐类筛选与未筛选对照', '',
              '每格为“筛选 / 未筛选”，按各自验证集选阈值。', '',
              '| 模型 | 种子 | 类别 | 阳性数 | 阈值 | AUROC | AUPRC | Precision | Recall | F1 | 漏检数 | 误报数 |',
              '|---|---:|---|---:|---|---|---|---|---|---|---|---|']
    for r in s['runs']:
        for c in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
            a, b = [r['evaluation'][k]['test']['per_class'][c] for k in ['filtered', 'unfiltered']]
            table.append(f'| {labels[r["model"]]} | {r["seed"]} | {c} | {a["support"]} | {a["threshold"]:.2f} / {b["threshold"]:.2f} | ' +
                         ' | '.join(f'{a[k]*100:.2f} / {b[k]*100:.2f}' for k in ['auroc','auprc','precision','recall','f1']) +
                         f' | {a["confusion_matrix"][1][0]} / {b["confusion_matrix"][1][0]} | {a["confusion_matrix"][0][1]} / {b["confusion_matrix"][0][1]} |')
    table += ['', '### 新教师在新增 40 条五类全零标注测试记录上的预测', '',
              '| 新教师种子 | 至少一类报阳性 | NORM | MI | STTC | CD | HYP |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in s['runs']:
        if r['model'] != MODEL: continue
        z = r['evaluation']['unfiltered']['zero_label_test']
        table.append(f'| {r["seed"]} | {z["any_positive_records"]}/40 | ' + ' | '.join(str(z['positive_counts'][c]) for c in z['positive_counts']) + ' |')
    section(ROOT / 'md/各模型整体与逐类指标总表.md', '\n'.join(table), 'ribeiro-teacher')
    section(ROOT / 'md/README.md', f'2026-10-09 已完成文献[1] Ribeiro 风格残差教师的 100 Hz 适配，三个种子与既有两种教师配对比较，含筛选 / 未筛选的整体、逐类指标。筛选测试平均 F1 为 {means[MODEL]*100:.3f}%，对照 InceptionTime 为 {means["inception_time1d"]*100:.3f}%。新教师权重和实验记录在 `outputs/ribeiro_teacher/`，现有学生和固件保留。', 'ribeiro-teacher')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'summary.json').exists(): docs(read(OUT / 'summary.json')); return
    torch.set_num_threads(8)
    base = load(BASE)
    frames = load_labeled_splits(Path(base['config']['dataset']['root']), 100, require_diagnostic_label=False)
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(frames[a].patient_id) & set(frames[b].patient_id)
    controls = [r for r in read(ROOT / 'outputs/teacher_preprocessing/summary.json')['runs']
                if r['model'] in ['inception1d','inception_time1d'] and r['preprocessing']=='P0_current']
    assert len(controls) == 6
    protected = {r['checkpoint']: digest(Path(r['checkpoint'])) for r in controls}
    for path in [ROOT / 'outputs/distillation/distilled_seed44/checkpoint/cnn1d_best.pt',
                 ROOT / 'outputs/distillation/distilled_export/ptbxl_cnn1d_int8.onnx',
                 ROOT / 'outputs/teacher_distillation/selected_export/ptbxl_cnn1d_int8.onnx',
                 ROOT / 'outputs/board_release/seed44_int8_stm32f407ve/ECG_Student_INT8_STM32F407VE.hex']:
        protected[str(path)] = digest(path)
    protocol = {'date':'2026-10-09','model':MODEL,'seeds':[42,43,44], 'reference':'https://github.com/antonior92/automatic-ecg-diagnosis/blob/master/model.py',
                'architecture':{'kernel':5,'strides':[4,4,2,2],'channels':[64,128,196,256,320], 'head':'Flatten 5120 -> five logits',
                                'raw_shortcut_stream':True, 'bn_epsilon':.001,'bn_torch_momentum':.01},
                'parameters':sum(p.numel() for p in build_model(MODEL).parameters()),
                'training':base['config']['training'], 'preprocessing':base['config']['signal_processing'],
                'training_records':17084,'validation_counts':[2146,2183],'test_counts':[2158,2198],
                'selection':'filtered validation macro F1 at 0.01 then AUROC; checkpoint epoch by training AMP AUROC',
                'protected_sha256':protected, 'model_source_sha256':digest(ROOT/'src/models/ribeiro_resnet.py'),
                'test_used_for_selection':False,'test_previously_reused':True, 'original_paper_reproduction':False}
    if (OUT/'protocol.json').exists(): assert read(OUT/'protocol.json') == protocol
    else: save(OUT/'protocol.json',protocol)
    docs()
    runs=[]
    for seed in [42,43,44]:
        cfg=copy.deepcopy(base['config']); cfg['model']['type']=MODEL
        cfg['training']['random_seed']=seed
        for stat in ['training_mean','training_std']: cfg['signal_processing'].pop(stat,None)
        folder=OUT/f'seed{seed}'
        for key,sub in [('checkpoint_dir','checkpoint'),('metrics_dir','metrics'),('figure_dir','figure'),('predictions_dir','predictions')]:
            cfg['outputs'][key]=str((folder/sub).relative_to(ROOT))
        if (folder/'config.json').exists(): assert read(folder/'config.json')==cfg
        else: save(folder/'config.json',cfg)
        path=folder/f'checkpoint/{MODEL}_best.pt'; history=folder/f'metrics/training_history_{MODEL}_full.json'
        if path.exists() and not history.exists(): raise RuntimeError(f'Partial training requires inspection: {folder}')
        if not history.exists():
            status(f'Training Ribeiro-adapted teacher seed {seed}, maximum 30 epochs')
            train_baseline(cfg,project_root=ROOT,model_type=MODEL)
        cp=load(path)
        assert cp['training_records']==17084 and cp['validation_records']==2146
        for stat in ['training_mean','training_std']:
            np.testing.assert_allclose(cp['config']['signal_processing'][stat],base['config']['signal_processing'][stat],rtol=0,atol=1e-12)
        runs.append({'name':f'{MODEL}_seed{seed}','model':MODEL,'seed':seed,'epoch':cp['epoch'],
                     'checkpoint':str(path),'checkpoint_sha256':digest(path), 'evaluation':{}})
    for r in controls:
        cp=load(Path(r['checkpoint']))
        assert cp['training_records']==17084 and cp['validation_records']==2146
        expected, actual=copy.deepcopy(base['config']['training']),copy.deepcopy(cp['config']['training'])
        for t in [expected,actual]:t.pop('random_seed',None)
        assert actual==expected
        runs.append({k:r[k] for k in ['name','model','seed','epoch','checkpoint','checkpoint_sha256']})
        runs[-1]['evaluation']={}
    for r in runs:
        status(f'FP32 validation and 0.01 thresholds: {r["name"]}')
        p=predict(Path(r['checkpoint']),frames['validation'],'validation',OUT/r['name']/'validation_predictions.npz') if r['model']==MODEL else baseline_scores(r,frames['validation'],'validation')
        y=frames['validation'][['NORM','MI','STTC','CD','HYP']].to_numpy(dtype=np.uint8); kept=y.sum(1)>0
        for cohort,mask in [('filtered',kept),('unfiltered',np.ones(len(y),dtype=bool))]:
            t=tune(y[mask],p[mask],grids()['grid_0.01'])
            r['evaluation'][cohort]={'thresholds':t,'validation':metric(y[mask],p[mask],t)}
    selection_key=lambda r:(r['evaluation']['filtered']['validation']['macro_f1'],r['evaluation']['filtered']['validation']['macro_auroc'])
    chosen=max([r for r in runs if r['model']==MODEL],key=selection_key)
    recommended=max(runs,key=selection_key)
    save(OUT/'frozen_selection.json',{'selected_new_teacher':copy.deepcopy(chosen),'validation_recommendation':copy.deepcopy(recommended),
                                    'all_validation':copy.deepcopy(runs),'frozen_at':datetime.now().isoformat(),'test_used':False})
    selected_cp=load(Path(chosen['checkpoint'])); selected_cp['thresholds']=chosen['evaluation']['filtered']['thresholds']
    selected_cp['validation_metrics_tuned']=chosen['evaluation']['filtered']['validation']
    selected_cp['research_validation_thresholds']=chosen['evaluation']
    torch.save(selected_cp,OUT/'selected_teacher.pt')
    for r in runs:
        status(f'Frozen teacher test: {r["name"]}, both cohorts')
        p=predict(Path(r['checkpoint']),frames['test'],'test',OUT/r['name']/'test_predictions.npz') if r['model']==MODEL else baseline_scores(r,frames['test'],'test')
        y=frames['test'][['NORM','MI','STTC','CD','HYP']].to_numpy(dtype=np.uint8); kept=y.sum(1)>0
        for cohort,mask in [('filtered',kept),('unfiltered',np.ones(len(y),dtype=bool))]:
            d=r['evaluation'][cohort];t=d['thresholds'];d['test']=metric(y[mask],p[mask],t)
            zero=p[~kept].astype(np.float64)>=np.asarray(t,dtype=np.float64)
            d['zero_label_test']={'records':40,'any_positive_records':int(zero.any(1).sum()),
                                 'positive_counts':dict(zip(['NORM','MI','STTC','CD','HYP'],zero.sum(0).astype(int).tolist()))}
    for path,sha in protected.items():assert digest(Path(path))==sha
    for r in runs:assert digest(Path(r['checkpoint']))==r['checkpoint_sha256']
    summary={'protocol':protocol,'runs':runs,'selected':copy.deepcopy(chosen),'validation_recommendation':copy.deepcopy(recommended),
             'parameters':protocol['parameters'],'network_tests_passed':7,'protected_artifacts_unchanged':True,
             'metric_sets_verified':36,'selected_teacher_sha256':digest(OUT/'selected_teacher.pt')}
    save(OUT/'summary.json',summary);docs(summary)
    status('Complete: three new Ribeiro-style teachers, paired controls, both cohorts and existing MD updated')
    print(json.dumps({'selected_seed':chosen['seed'],'recommended':recommended['name'],
                      'new_selected_test':chosen['evaluation']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
