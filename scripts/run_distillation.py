"""Paired three-seed CNN distillation experiment, INT8 and Cube.AI analysis."""
from __future__ import annotations
import copy
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_stm32 import digest, get_split
from scripts.run_diagnostic_only import predict
from src.data.ptbxl import SUPERCLASSES
from src.models import build_model
from src.training.data import load_labeled_splits, CachedPTBXLDataset
from src.training.train import train_baseline
from src.training.evaluate import evaluate_checkpoint
from src.training.metrics import calculate_multilabel_metrics

OUT = ROOT / 'outputs/distillation'
TEACHER = ROOT / 'outputs/optimization/optimized_best.pt'
CLI = Path('D:/STM32AI/User/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.1/Utilities/windows/stedgeai.exe')
KEYS = ['macro_auroc', 'macro_precision', 'macro_recall', 'macro_f1']


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def status(message):
    print(message, flush=True)
    save(OUT/'status.json', {'stage':message, 'time':time.strftime('%Y-%m-%d %H:%M:%S')})


def command(args, log):
    with log.open('w', encoding='utf-8') as f:
        result = subprocess.run([str(a) for a in args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Command failed; inspect {log}')


def teacher_logits(selected):
    path = OUT/'teacher_training_logits.npz'
    checkpoint = torch.load(TEACHER, map_location='cpu', weights_only=True)
    assert checkpoint['superclasses']==list(SUPERCLASSES)
    assert checkpoint['sampling_rate_hz']==100
    sha = digest(TEACHER)
    if path.exists():
        data=np.load(path)
        np.testing.assert_array_equal(data['ecg_ids'], selected.ecg_id)
        if str(data['teacher_sha256']) != sha:
            raise ValueError('Cached teacher changed.')
        return path
    array, frame = get_split('train', checkpoint)
    positions = pd.Index(frame.ecg_id).get_indexer(selected.ecg_id)
    assert (positions>=0).all()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build_model(checkpoint['model_type'], dropout=checkpoint['config']['model']['dropout']).to(device).eval()
    model.load_state_dict(checkpoint['model_state'])
    logits=[]
    with torch.inference_mode():
        for start in range(0,len(positions),32):
            x=np.ascontiguousarray(array[positions[start:start+32]])
            logits.append(model(torch.from_numpy(x).to(device)).cpu().numpy())
    np.savez_compressed(path, ecg_ids=selected.ecg_id.to_numpy(), logits=np.concatenate(logits),
                        teacher_sha256=np.asarray(sha))
    return path


def analysis(export, version):
    folder=export/f'{version}_analysis'; folder.mkdir(exist_ok=True)
    model=export/f'ptbxl_cnn1d_{"int8" if version=="int8" else "float32"}.onnx'
    name=f'ecg_cnn_{version}'
    command([CLI, 'analyze', '--model', model, '--name', name, '--target', 'stm32f4', '--compression', 'none',
             '--optimization', 'ram', '--output', folder,
             '--workspace', export/f'{version}_workspace',
             '--inputs-ch-position', 'chfirst', '--output-data-type', 'float32', '--c-api', 'legacy'],
            export/f'cubeai_{version}.log')
    info=json.loads((folder/f'{name}_c_info.json').read_text(encoding='utf-8'))
    text=(folder/f'{name}_analyze_report.txt').read_text(encoding='utf-8')
    macc=int(re.search(r'^macc\s+:\s+([\d,]+)', text, re.M).group(1).replace(',',''))
    return {'weights_bytes':info['memory_footprint']['weights'],
            'activations_bytes':info['memory_footprint']['activations'], 'macc':macc}


def write_report(summary):
    lines=['# 小 CNN 多标签蒸馏与 INT8 实验', '', '日期：2026-10-01', '',
           '## 实验设置', '',
           '十二导联、100 Hz、10 秒，五个独立诊断输出。剔除五类标签全零记录后，训练 17084、验证 2146、测试 2158；官方 folds 1–8 / 9 / 10，患者隔离。', '',
           '普通训练与蒸馏使用相同 CNN、种子 42/43/44、batch 32、AdamW、学习率 0.001、weight decay 0.0001、最多 30 轮、早停 patience 8，验证 AUROC 选权重。标准化仅拟合筛选后的训练集。', '',
           f"学生参数量 {summary['student_parameters']}，教师参数量 {summary['teacher_parameters']}。学生采用 32/64/128/128 通道卷积、时间降采样和全局平均池化。", '',
           '教师是既有优化 Inception，冻结权重和原标准化统计；教师曾使用原训练集（含无五类标签记录），学生只使用筛选后的训练集。对每条训练记录保存五个原始 logits，按 ECG ID 严格对齐。', '',
           '蒸馏损失：0.5 × 带正类权重的真实标签 BCE + 0.5 × T² × BCEWithLogits(student_logits/T, sigmoid(teacher_logits/T))，T=2。各类概率独立，不使用 softmax。未加入数据增强以避免缓存教师与变化后的波形不一致。', '',
           '每组依据验证宏 F1、再以 AUROC 打破平局，冻结量化候选；三个种子的成对结果全部报告，没有根据测试结果选种子或调参。多种子衡量训练随机性，不代替患者交叉验证或外部验证。', '',
           '## 每个模型的整体指标', '', '| 模型 | 种子 | 最佳轮次 | 验证 F1 | 测试 AUROC | Precision | Recall | F1 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['runs']:
        m=r['test']; lines.append(f"| {r['arm']} | {r['seed']} | {r['epoch']} | {r['validation']['macro_f1']:.6f} | "+' | '.join(f'{m[k]:.6f}' for k in KEYS)+' |')
    lines+=['', '## 跨种子统计', '', '| 方法 | 测试宏 AUROC，均值 ± 样本标准差 | 测试宏 F1，均值 ± 样本标准差 |', '|---|---:|---:|']
    for arm in ['baseline','distilled']:
        rows=[r for r in summary['runs'] if r['arm']==arm]
        vals={k:np.asarray([r['test'][k] for r in rows]) for k in ['macro_auroc','macro_f1']}
        lines.append(f'| {arm} | '+ ' | '.join(f'{vals[k].mean():.6f} ± {vals[k].std(ddof=1):.6f}' for k in vals)+' |')
    delta=[next(r['test']['macro_f1'] for r in summary['runs'] if r['arm']=='distilled' and r['seed']==seed)-next(r['test']['macro_f1'] for r in summary['runs'] if r['arm']=='baseline' and r['seed']==seed) for seed in [42,43,44]]
    lines+=['',f'成对测试 F1 变化（蒸馏−普通，百分点）：{[round(d*100,3) for d in delta]}；平均 {np.mean(delta)*100:+.3f} 个百分点。三个种子不足以证明统计显著或外部泛化改善。', '',
            '## 量化候选与 Cube.AI', '', '| 方法 | 验证选定种子 | 版本 | 验证 AUROC | 验证 F1 | 测试 AUROC | 测试 F1 | 权重 B | 激活 B | MACC |', '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, dep in summary['deployment'].items():
        for version in ['float32','int8']:
            v=dep['quantization'][version]; t=dep['test'][version]; a=dep['analysis'][version]
            lines.append(f"| {arm} | {dep['seed']} | {version} | {v['macro_auroc']:.6f} | {v['macro_f1']:.6f} | {t['macro_auroc']:.6f} | {t['macro_f1']:.6f} | {a['weights_bytes']} | {a['activations_bytes']} | {a['macc']} |")
    recommendation=max(summary['deployment'],key=lambda arm:(summary['deployment'][arm]['quantization']['int8']['macro_f1'],summary['deployment'][arm]['quantization']['int8']['macro_auroc']))
    chosen=summary['deployment'][recommendation]
    lines+=['',f"按测试前约定的 INT8 验证宏 F1（AUROC 打破平局）推荐后续部署候选：{recommendation}，种子 {chosen['seed']}。该选择不使用测试指标。baseline 表示普通 CNN，distilled 表示蒸馏 CNN。", '',
            f"相对既有 ResNet INT8，推荐候选权重减少 {(1-chosen['analysis']['int8']['weights_bytes']/453012)*100:.2f}%，MACC 减少 {(1-chosen['analysis']['int8']['macc']/73639711)*100:.2f}%；这只是模型分析，不能直接推算整机速度或固件节省。"]
    lines+=['', 'INT8 使用相同的 256 条筛选后训练记录校准，静态 QDQ、每通道对称 INT8 权重、每张量非对称 INT8 激活；量化后阈值保持各自 FP32 验证阈值，没有重新校准。', '',
            '训练期间验证使用 AMP；导出和量化表采用 ONNX Runtime FP32 / INT8 推理，阈值附近的少量判断可能不同，因此两个表的 FP32 验证 F1 可能略有差别。', '',
            'Cube.AI 是模型资源分析，权重不等于完整固件 Flash，激活不等于总 RAM。尚未替换串口工程、链接完整学生固件或实板测时。既有 ResNet INT8 权重 453012 B、激活 44032 B、MACC 73639711；其完整固件 Flash 506496 B、RAM 62504 B。', '',
            '## 每个模型的逐类测试指标', '', '| 模型 | 种子 | 类别 | 阳性数 | AUROC | Precision | Recall | F1 |', '|---|---:|---|---:|---:|---:|---:|---:|']
    rows=[(r['arm'],r['seed'],r['test']) for r in summary['runs']]
    rows += [(arm+' INT8',dep['seed'],dep['test']['int8']) for arm,dep in summary['deployment'].items()]
    for name,seed,m in rows:
        for label in SUPERCLASSES:
            c=m['per_class'][label]
            lines.append(f"| {name} | {seed} | {label} | {c['support']} | "+' | '.join(f'{c[k]:.6f}' for k in ['auroc','precision','recall','f1'])+' |')
    reference=json.loads((ROOT/'outputs/diagnostic_only/summary.json').read_text(encoding='utf-8'))
    lines+=['', '## 同一测试范围的既有模型参考', '', '| 模型 | 测试 AUROC | Precision | Recall | F1 |', '|---|---:|---:|---:|---:|']
    for r in reference['comparisons']:
        lines.append('| '+r['name']+' | '+' | '.join(f'{r["test"][k]:.6f}' for k in KEYS)+' |')
    lines+=['', '这些参考模型使用相同的 2158 条保留测试记录，但结构、训练数据范围与训练预算不同，不能作为纯蒸馏消融对照。纯蒸馏效果应看同种子、同结构的成对学生实验。', '',
            '## 文件与边界', '',
            '所有配置、权重、训练曲线、逐条预测、量化日志与 Cube.AI 原报告位于 `outputs/distillation/`；`frozen_selection.json` 记录测试前的候选选择，`summary.json` 保存全部结果。', '',
            '原始数据和旧模型保留。本轮测试与此前多次使用的 fold 10 重叠，不能当作新盲测；没有外部数据或实板新采集信号证据。泛化能力尚未充分证明。', '',
            '复现：`.venv/Scripts/python.exe scripts/run_distillation.py`。论文方法依据：[PTB-XL 多标签压缩研究](https://www.informatica.vu.lt/journal/INFORMATICA/article/1401/read)；部署依据：[ST 量化支持](https://stedgeai-dc.st.com/assets/embedded-docs/quantization.html)。']
    if 'independent_audit' in summary:
        audit=summary['independent_audit']
        lines+=['', '## 独立复核', '',
                f"{audit['tests_passed']} 项测试通过；{audit['prediction_sets_verified']} 组保存的预测复算整体指标一致，测试记录一致。标准化仅拟合筛选后的训练集；两个量化模型使用相同的训练校准记录，候选选择依据验证指标。详见 `outputs/distillation/independent_audit.json`。"]
    (ROOT / 'outputs/reports').mkdir(parents=True, exist_ok=True)
    (ROOT/'outputs/reports/小CNN蒸馏与INT8实验总结.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'summary.json').exists():
        write_report(json.loads((OUT/'summary.json').read_text(encoding='utf-8'))); return
    torch.set_num_threads(8)
    cp=torch.load(ROOT/'outputs/diagnostic_only/resnet_seed42/checkpoint/resnet1d_best.pt',map_location='cpu',weights_only=True)
    cfg=copy.deepcopy(cp['config'])
    cfg['signal_processing'].pop('training_mean',None)
    cfg['signal_processing'].pop('training_std',None)
    cfg['model']['type']='cnn1d'
    cfg['training'].update(epochs=30, early_stopping_patience=8)
    full=load_labeled_splits(Path(cfg['dataset']['root']),100)
    selected=load_labeled_splits(Path(cfg['dataset']['root']),100,require_diagnostic_label=True)
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(selected[a].patient_id)&set(selected[b].patient_id)
    status('Preparing frozen Inception teacher logits on 17084 training records only')
    logits=teacher_logits(selected['train'])
    teacher_sha=digest(TEACHER)
    save(OUT/'protocol.json',{'seeds':[42,43,44],'epochs':30,'patience':8,'alpha':0.5,'temperature':2,
         'teacher':str(TEACHER),'teacher_sha256':teacher_sha,'selection':'validation macro F1 then AUROC',
         'deployment_selection':'INT8 validation macro F1 then AUROC; never test metrics',
         'cohort':{k:len(f) for k,f in selected.items()},'test_used_for_selection':False})
    save(OUT/'deployment_selection_policy.json',{'criterion':'INT8 validation macro F1, AUROC tie-break; test scores excluded',
         'written_before_test_evaluation':True})
    runs=[]
    for seed in [42,43,44]:
        for arm in ['baseline','distilled']:
            trial=copy.deepcopy(cfg); trial['training']['random_seed']=seed
            trial['training'].pop('distillation',None)
            if arm=='distilled':
                trial['training']['distillation']={'alpha':0.5,'temperature':2,'logits_path':str(logits),
                                                  'teacher_checkpoint':str(TEACHER),'teacher_sha256':teacher_sha}
            folder=OUT/f'{arm}_seed{seed}'
            for key,sub in [('checkpoint_dir','checkpoint'),('metrics_dir','metrics'),('figure_dir','figure'),('predictions_dir','predictions')]:
                trial['outputs'][key]=str((folder/sub).relative_to(ROOT))
            save(folder/'config.json',trial)
            checkpoint=folder/'checkpoint/cnn1d_best.pt'
            history=folder/'metrics/training_history_cnn1d_full.json'
            if checkpoint.exists() and not history.exists():
                raise RuntimeError(f'Partial training exists: {folder}; inspect before resuming.')
            if not history.exists():
                status(f'Training {arm}, seed {seed}, fixed 30-epoch maximum')
                train_baseline(trial,project_root=ROOT,model_type='cnn1d')
            trained=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert trained['training_records']==17084 and trained['validation_records']==2146
            for stat in ['training_mean','training_std']:
                np.testing.assert_allclose(trained['config']['signal_processing'][stat],
                                           cp['config']['signal_processing'][stat], rtol=0, atol=1e-12)
            runs.append({'arm':arm,'seed':seed,'checkpoint':str(checkpoint),'epoch':trained['epoch'],
                         'validation':trained['validation_metrics_tuned'],'sha256':digest(checkpoint)})
    candidates={arm:max([r for r in runs if r['arm']==arm],key=lambda r:(r['validation']['macro_f1'],r['validation']['macro_auroc'])) for arm in ['baseline','distilled']}
    frozen={arm:{k:r[k] for k in ['seed','checkpoint','sha256','validation']} for arm,r in candidates.items()}
    save(OUT/'frozen_selection.json',frozen)
    status('All six training runs complete; candidates frozen using validation only; starting INT8 exports')
    deploy={}
    for arm,r in candidates.items():
        export=OUT/f'{arm}_export';export.mkdir(exist_ok=True)
        if not (export/'deployment_manifest.json').exists():
            command([sys.executable,ROOT/'scripts/prepare_stm32.py','--checkpoint',r['checkpoint'],'--output-dir',export],export/'export.log')
        if not (export/'int8_validation_report.json').exists():
            status(f'Quantizing {arm} seed {r["seed"]}; comparing all 2146 validation records')
            command([sys.executable,ROOT/'scripts/quantize_cubeai.py','--export-dir',export],export/'quantization.log')
        quant=json.loads((export/'int8_validation_report.json').read_text(encoding='utf-8'))
        status(f'Cube.AI FP32 and INT8 analysis: {arm}')
        analyses={v:analysis(export,v) for v in ['float32','int8']}
        deploy[arm]={'seed':r['seed'],'export_dir':str(export),'quantization':quant,'analysis':analyses}
    status('Fixed configurations and quantization complete; reporting all six models on the same filtered test records')
    for r in runs:
        r['test']=evaluate_checkpoint(r['checkpoint'],project_root=ROOT)
    import onnxruntime as ort
    from scripts.quantize_cubeai import probabilities
    for arm,r in candidates.items():
        export=Path(deploy[arm]['export_dir'])
        model_cp=torch.load(r['checkpoint'],map_location='cpu',weights_only=True)
        array,frame=get_split('test',model_cp)
        truth=frame[list(SUPERCLASSES)].to_numpy(dtype=np.uint8)
        options=ort.SessionOptions(); options.intra_op_num_threads=4; options.inter_op_num_threads=1
        tests={}
        for v in ['float32','int8']:
            session=ort.InferenceSession(str(export/f'ptbxl_cnn1d_{v}.onnx'),options,providers=['CPUExecutionProvider'])
            scores=probabilities(session,array)
            tests[v]=calculate_multilabel_metrics(truth,scores,model_cp['thresholds'])
            np.savez_compressed(export/f'{v}_test_predictions.npz',ecg_ids=frame.ecg_id.to_numpy(),targets=truth,probabilities=scores)
        deploy[arm]['test']=tests
    assert digest(TEACHER)==teacher_sha
    for r in runs:
        assert digest(Path(r['checkpoint']))==r['sha256']
    recommendation=max(deploy,key=lambda arm:(deploy[arm]['quantization']['int8']['macro_f1'],
                                             deploy[arm]['quantization']['int8']['macro_auroc']))
    summary={'runs':runs,'deployment':deploy,'teacher_unchanged':True,'test_used_for_selection':False,
             'deployment_recommendation':recommendation, 'training_normalization_verified':True,
             'student_parameters':sum(p.numel() for p in build_model('cnn1d').parameters()),
             'teacher_parameters':sum(p.numel() for p in build_model('inception1d').parameters())}
    save(OUT/'summary.json',summary);write_report(summary)
    status('Completed: six models trained, paired three-seed comparison, INT8, Cube.AI and Markdown saved')


if __name__=='__main__': main()
