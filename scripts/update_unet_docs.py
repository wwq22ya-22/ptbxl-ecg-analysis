"""Insert a reproducible U-Net study into the existing five-document flow."""
from pathlib import Path
import argparse, json, re, zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT/'src').exists(): ROOT = Path('D:/MyProject/college/ecg1')
OUT = ROOT/'outputs/unet_teacher_20261010'
C = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
NAMES = {'filtered':'筛选后', 'unfiltered':'未筛选', 'unet1d':'1D U-Net', 'unet_encoder1d':'相同编码器对照', 'inception_time1d':'InceptionTime旧权重'}

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def pct(v): return f'{100*v:.3f}'
def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(str(x) for x in row)+' |' for row in rows]])+'\n'

def block(text, name, content, before):
    start, end = f'<!-- unet-{name}-20261010:start -->', f'<!-- unet-{name}-20261010:end -->'
    replacement = start+'\n'+content.strip()+'\n'+end+'\n\n'
    if start in text:
        return re.sub(re.escape(start)+r'.*?'+re.escape(end)+r'\n*', lambda _: replacement, text, flags=re.S)
    assert before in text, before
    return text.replace(before, replacement+before, 1)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=['running','complete'], default='complete')
    stage = parser.parse_args().stage
    protocol = read(OUT/'protocol.json')
    summary = audit = None
    if stage == 'complete':
        summary, audit = read(OUT/'summary.json'), read(OUT/'independent_audit.json')
        assert audit['passed'] and audit['metric_sets_independently_recomputed']==30
        assert summary['fold10_evaluation'] is None
    files = [ROOT/'README.md', *[ROOT/'md'/n for n in ['README.md','模型训练与优化.md','各模型整体与逐类指标总表.md','STM32部署与串口验证.md','Chapman_冻结模型外部验证总结.md']]]
    backup = OUT/'markdown_before_unet.zip'
    if not backup.exists():
        with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as z:
            for path in files: z.write(path, str(path.relative_to(ROOT)))
    texts = {p.name if p.parent.name=='md' else 'root':p.read_text(encoding='utf-8') for p in files}
    status = '进行中；12次训练全部完成后统一复核，暂不据部分种子更新推荐。' if stage=='running' else '已完成12次新训练及30组整体/逐类指标、30组阈值独立复核；仅折9评估。'
    setup = '''<a id="unet-study-20261010"></a>

### E15：1D U-Net分类与编码器对照（2026-10-10）

问题：在记录级五类多标签任务中，保留时间位置的U-Net解码器是否有帮助；筛选与未筛选训练人群下结果是否一致。原始U-Net面向分割，本次是1D分类适配，没有逐点分割真值，也不是原论文完整复现。

结构使用核5、12/24/48/96/192通道，编码长度1000/500/250/125/62；线性插值对齐125/250/500/1000，1×1投影后拼接跳跃特征。末端全局平均/最大池化、dropout0.2和五类logits。U-Net参数579053；相同编码器、无解码器对照参数372053。两者分类头通道及总参数量不同，未做参数量匹配，不能把差异完全归因于解码器。

评估的FP32指张量/接口精度。本环境cuDNN允许TF32、CUDA矩阵乘TF32关闭，没有另做禁用TF32的整队列对照。输出绑定审计按原GPU批次与确定性设置重新推理；CPU不同批量的差值仅作跨平台诊断，不当作原GPU结果的严格复现。

| 固定项 | 方案 |
|---|---|
| 训练人群 | 筛选后17084条与未筛选17418条，各自从头训练 |
| 验证人群 | 每个模型均在筛选后2146条、未筛选2183条折9记录上评估 |
| 候选数 | 两结构×两训练人群×42/43/44三配对种子，共12次；训练前全部冻结 |
| 训练设置 | AdamW，lr0.001，weight decay0.0001，batch32，30轮上限，早停8，AMP；无蒸馏/新增增强/学习率调度 |
| 损失与标准化 | 加权BCE和train_global，分别仅从各组训练集计算 |
| checkpoint选择 | 原训练人群对应折9宏AUROC改善超过1e-4 |
| 比较与种子选择 | FP32折9推理，0.10–0.90步长0.01逐类校准；每训练人群/结构按对应验证人群宏F1、再AUROC选种子 |
| 口径边界 | 两验证人群分别校准阈值，不能跨分母排名；无折10推理/计分 |
| 固定参照 | 三个旧P0 InceptionTime单网络权重，重算同样两套折9；它们仅在筛选后人群训练，未筛选训练比较不完全匹配 |

未筛选训练中的334条全零标签记录保留五个0作为BCE目标，不改成NORM；这增加了五类阴性目标，但数据库未列出五类诊断不等于临床证实全部阴性。未筛选验证中的37条同样保留，属于该人群定义的限制。

''' + '状态：'+status+'\n\n' + '''结果和阈值见[U-Net指标](各模型整体与逐类指标总表.md#metric-unet)，复现入口为`scripts/run_unet_teacher.py`和`scripts/audit_unet_teacher.py`。训练前[协议](../outputs/unet_teacher_20261010/protocol.json)冻结配置、代码哈希与保护产物。读取拦截针对Python层文件访问，不把它表述成对所有原生库访问的完全证明。

架构依据：[U-Net原论文](https://arxiv.org/abs/1505.04597)。本轮不替换既有教师、学生、ONNX或HEX，不进行蒸馏、量化、Chapman或实板验证。历史测试独立性不会因本轮不读折10而恢复。
'''
    conclusion = ''
    weights = []
    for pop in ['filtered','unfiltered']:
        frame = pd.read_csv(OUT/f'{pop}_train_cohort.csv')
        positive = frame[C].to_numpy(dtype=np.float32).sum(axis=0)
        weights.append([NAMES[pop],len(frame),*[f'{w:.6f}' for w in (len(frame)-positive)/positive]])
    setup += '\n两组训练正类权重（阳性数相同、训练分母不同）：\n\n'+table(['训练人群','记录数',*C],weights)+'\n'
    if summary:
        clauses = []
        for pop in ['filtered','unfiltered']:
            full = summary['three_seed_statistics'][pop]['unet1d'][pop]['macro_f1']
            enc = summary['three_seed_statistics'][pop]['unet_encoder1d'][pop]['macro_f1']
            paired = summary['paired_decoder_ablation'][pop][pop]
            clauses.append(f'{NAMES[pop]}训练/验证：U-Net三种子F1均值{pct(full["mean"])}%，样本标准差{pct(full["sample_std"])}个百分点；编码器对照均值{pct(enc["mean"])}%，平均差{100*paired["mean_difference"]:+.3f}个百分点，U-Net胜出{paired["wins"]}/3种子。')
        refmean = np.mean([r['validation']['filtered']['macro_f1'] for r in summary['fixed_teacher_references']])
        unetmean = summary['three_seed_statistics']['filtered']['unet1d']['filtered']['macro_f1']['mean']
        clauses.append(f'筛选后同训练人群的旧InceptionTime三种子折9均值{pct(refmean)}%，U-Net差{100*(unetmean-refmean):+.3f}个百分点。')
        conclusion = '\n\n'.join(clauses)+'\n\n这些是用来选择模型和阈值的验证成绩，不能据此宣布泛化提升；三种子不足以证明统计显著性。本轮按冻结方案保留全部结果，既有部署候选保持原状态。'
        setup += '\n### 本轮结论\n\n'+conclusion+'\n'
    training = texts['模型训练与优化.md']
    if '| unet1d |' not in training:
        training = training.replace('| ribeiro_resnet100 | 2139109 | 开题文献网络的100 Hz适配 | 保留四残差模块/双分支/Flatten头；核5、降采样4/4/2/2，未提升 |',
            '| ribeiro_resnet100 | 2139109 | 开题文献网络的100 Hz适配 | 保留四残差模块/双分支/Flatten头；核5、降采样4/4/2/2，未提升 |\n| unet1d | 579053 | 1D编码/解码与跳跃特征分类 | [E15固定实验](#unet-study-20261010)，两训练人群、三种子，仅折9 |\n| unet_encoder1d | 372053 | 同编码器、去掉解码器对照 | 全局平均/最大池化；非参数量匹配对照 |')
    e15 = '| E15 1D U-Net分类 | 两训练人群、同编码器对照、三种子 | '+('进行中，固定12次训练；无折10评估' if not summary else '12次训练完成，双人群折9与独立复核；不替换部署')+' | `outputs/unet_teacher_20261010` |'
    if '| E15 1D U-Net分类 |' in training: training = re.sub(r'^\| E15 1D U-Net分类 \|.*$', lambda _:e15, training, flags=re.M)
    else: training = training.replace('\n\n第一轮15个候选', '\n'+e15+'\n\n第一轮15个候选',1)
    training = re.sub(r'(\| E14[^\n]*\n)\n+(\| E15)', r'\1\2', training)
    training = block(training,'method',setup,'<a id="step-10"></a>')
    texts['模型训练与优化.md'] = training
    metrics = '''<a id="metric-unet"></a>

## 新增E15：1D U-Net，筛选/未筛选双训练人群（仅fold9）

''' + status+'\n\n'+'''每个模型在两套验证人群分别校准自己的0.01步长阈值；表中P/R/F1为宏平均百分数，AUROC/PR-AUC为0–1。原生人群指训练和验证采用相同纳入规则；只在各原生人群内选种子，交叉人群列不参与该选择。详细方法见[训练流程E15](模型训练与优化.md#unet-study-20261010)。
'''
    if summary:
        metrics += '\n### 三种子统计（F1为均值±样本标准差，单位百分点）\n\n'
        stats = []
        for pop in ['filtered','unfiltered']:
            for model in ['unet1d','unet_encoder1d']:
                for evaluation in ['filtered','unfiltered']:
                    m = summary['three_seed_statistics'][pop][model][evaluation]
                    stats.append([NAMES[pop],NAMES[model],NAMES[evaluation],pct(m['macro_precision']['mean']),pct(m['macro_recall']['mean']),f'{pct(m["macro_f1"]["mean"])} ± {pct(m["macro_f1"]["sample_std"])}',f'{m["macro_auroc"]["mean"]:.6f}'])
        metrics += table(['训练人群','结构','验证人群','平均P %','平均R %','F1 %','平均AUROC'],stats)
        metrics += '\n### 各结构原生人群所选种子\n\n'
        rows = []
        for pop, selection in summary['selected_native_validation'].items():
            for model, r in selection.items():
                m = r['validation'][pop]
                rows.append([NAMES[pop],NAMES[model],r['seed'],r['best_epoch'],r['epochs_completed'],pct(m['macro_precision']),pct(m['macro_recall']),pct(m['macro_f1']),pct(m['per_class']['HYP']['recall'])])
        metrics += table(['训练/验证人群','结构','种子','最佳轮','完成轮数','P %','R %','F1 %','HYP R %'],rows)
        metrics += '\n### 全部模型的双人群整体指标\n\n'
        all_rows = [*summary['runs'],*summary['fixed_teacher_references']]
        for evaluation in ['filtered','unfiltered']:
            metrics += f'\n#### {NAMES[evaluation]}验证（'+('2146' if evaluation=='filtered' else '2183')+'条）\n\n'
            rows = []
            for r in all_rows:
                m = r['validation'][evaluation]
                rows.append([r['name'],NAMES[r['training_cohort']],f'{m["macro_auroc"]:.6f}',f'{m["macro_auprc"]:.6f}',pct(m['macro_precision']),pct(m['macro_recall']),pct(m['macro_f1']),pct(m['per_class']['HYP']['recall'])])
            metrics += table(['版本','训练人群','AUROC','PR-AUC','P %','R %','F1 %','HYP R %'],rows)
        metrics += '\n### 全部逐类指标及该验证人群的阈值\n\n'
        for evaluation in ['filtered','unfiltered']:
            metrics += f'\n#### {NAMES[evaluation]}验证逐类\n\n'
            rows = []
            for r in all_rows:
                for c, m in r['validation'][evaluation]['per_class'].items():
                    rows.append([r['name'],c,m['support'],f'{m["threshold"]:.2f}',f'{m["auroc"]:.6f}',f'{m["auprc"]:.6f}',pct(m['precision']),pct(m['recall']),pct(m['f1'])])
            metrics += table(['版本','类别','阳性数','阈值','AUROC','PR-AUC','P %','R %','F1 %'],rows)
        metrics += '\n### 37条全零标签验证记录的预测（未筛选阈值）\n\n'
        metrics += '这是未筛选折9人群内五个目标均为0的记录，不补正常标签；“任一阳性”按该任务目标可记为范围外预测，但不等于临床诊断错误的独立证据。\n\n'
        profiles = audit['zero_label_validation_profiles']
        metrics += table(['版本','任一阳性记录/37',*C],[[r['name'],r['any_positive_records'],*[r['positive_counts'][c] for c in C]] for r in profiles])
        metrics += '\n### 结论、复核与产物\n\n'+conclusion+'\n\n'
        if (OUT/'paired_validation_f1.png').exists(): metrics += '![原生人群三种子配对验证F1](../outputs/unet_teacher_20261010/paired_validation_f1.png)\n\n'
        metrics += '独立复算30组整体/逐类、30组阈值，检查两组训练统计/正类权重、四个原生分组选种子、配对差值和受保护产物哈希；37项网络/训练策略/读取边界测试通过。预测、权重、完整历史及细阈值按模型分目录保存。checkpoint保留训练结束时原粗阈值，细阈值在每次运行的`frozen_validation_thresholds.json`中，不混用。\n\n'
        checks = audit['sampled_saved_logits_reinferred_from_checkpoints']
        metrics += f'输出抽查覆盖30套预测，每套抽查3条并复原其GPU批次。GPU重算最大logits差{max(r["max_abs_original_batch_gpu_difference"] for r in checks):.8g}；CPU不同批量诊断最大差{max(r["max_abs_cpu_gpu_difference"] for r in checks):.8g}。初始CPU严格容差检查曾失败；后续通过的是原GPU批次绑定检查，不把CPU差异抹去或解释成指标公式错误。精度设置与逐模型差值保存在独立审计中。\n\n'
        metrics += '[总结果](../outputs/unet_teacher_20261010/summary.json)、[独立审计](../outputs/unet_teacher_20261010/independent_audit.json)、[读取记录](../outputs/unet_teacher_20261010/read_guard.json)。旧InceptionTime参照权重未重训；对未筛选训练组的比较同时改变训练人群。无新折10/外部结果、量化或固件，不继承旧部署资源。\n'
    metrics += '\n协议和复现：[冻结协议](../outputs/unet_teacher_20261010/protocol.json)，`scripts/run_unet_teacher.py`，`scripts/audit_unet_teacher.py`。同轮复现会核对哈希并复用已完整完成的训练记录，不覆盖原模型。\n'
    total = texts['各模型整体与逐类指标总表.md']
    total = total.replace('更新：2026-10-10。仅调整组织与状态说明，原指标数值不改写。','更新：2026-10-10。历史数值保持原记录，新增E15的折9双人群U-Net实验。')
    if '- [新增E15：1D U-Net' not in total:
        total = total.replace('- [3. 早期优化', '- [新增E15：1D U-Net双人群折9实验](#metric-unet)\n- [3. 早期优化',1)
    texts['各模型整体与逐类指标总表.md'] = block(total,'metrics',metrics,'<a id="metric-early-val"></a>')
    overview = '### 新增网络研究：1D U-Net\n\n'+status+'\n\n'
    if summary: overview += conclusion+'\n\n'
    overview += '本轮训练分别使用筛选后17084条与未筛选17418条，对应两套折9验证；两结构×三种子，数据与选择口径见[训练流程](模型训练与优化.md#unet-study-20261010)、[指标总表](各模型整体与逐类指标总表.md#metric-unet)。没有新增折10、外部或板端结果，当前seed43部署候选与旧HEX保持原状态。'
    texts['README.md'] = block(texts['README.md'],'overview',overview,'<a id="overview-files"></a>')
    texts['README.md'] = texts['README.md'].replace('本次整理未重训、未新增折10评估、未改原模型/固件/Word、未推送。', '前一阶段的文档整理本身未重训、未新增折10评估、未改原模型/固件/Word、未推送；后续E15的12次训练另按本轮协议执行，仍不新增折10评估。')
    texts['root'] = block(texts['root'],'root','1D U-Net双训练人群实验：'+status+'详见[方法与结果](md/模型训练与优化.md#unet-study-20261010)。既有部署版本保持原状态。','## 项目与实验文档')
    deploy = '本轮1D U-Net研究：'+status+'模型579053参数，编码器对照372053参数。仅有训练及折9评估范围；未执行ONNX/INT8导出、Cube.AI、C代码、完整固件或真实板端验证，不能继承旧学生的Flash/SRAM/MACC。见[训练文档](模型训练与优化.md#unet-study-20261010)。'
    texts['STM32部署与串口验证.md'] = block(texts['STM32部署与串口验证.md'],'deployment',deploy,'<a id="deployment-chain"></a>')
    texts['Chapman_冻结模型外部验证总结.md'] = block(texts['Chapman_冻结模型外部验证总结.md'],'external','新增1D U-Net和编码器对照只在PTB-XL折9评估，尚无Chapman结果；筛选/未筛选验证分数不作为外部证据。现有六个冻结版本的结果保持原记录。','<a id="external-input"></a>')
    for path in files:
        key = path.name if path.parent.name=='md' else 'root'
        path.write_text(texts[key],encoding='utf-8')
    print('Updated five core MDs and root README; stage='+stage)

if __name__ == '__main__': main()
