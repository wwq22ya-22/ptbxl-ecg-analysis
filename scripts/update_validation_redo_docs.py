"""Maintain the five existing core MD documents for validation-only correction."""
from pathlib import Path
import argparse,json,re
ROOT=Path(__file__).resolve().parents[1]
if not (ROOT/'md').exists():ROOT=Path('D:/MyProject/college/ecg1')
OUT=ROOT/'outputs/validation_only_redo_20261010'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def replace_block(text,marker,body,top=False):
    start=f'<!-- {marker}:start -->';end=f'<!-- {marker}:end -->'
    if start in text:
        left,tail=text.split(start,1);_,right=tail.split(end,1)
        return left+start+'\n'+body+'\n'+end+right
    block=start+'\n'+body+'\n'+end+'\n'
    if top:
        title,tail=text.split('\n',1)
        return title+'\n\n'+block+'\n'+tail.lstrip('\n')
    return text.rstrip()+'\n\n'+block
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['running','complete'],required=True);args=parser.parse_args()
    complete=args.stage=='complete'
    intro='**评价流程纠正（2026-10-10）：** 以下已有fold10成绩全部保留为历史探索性结果，不能作为新的独立盲测。三种子集成教师及其派生新seed42蒸馏模型撤回“经纠正流程推荐候选”的地位，原文件保留。'
    intro+='纠正流程已完成；最新候选仅按fold9选择，详见本文纠正记录。' if complete else '纠正工作正在执行，当前没有已完成的新推荐候选；结果未完成前不写入新成绩。'
    intro+='原seed44 HEX继续作为既有编译与回放演示产物，不因本次纠正自动替换，也不因此获得新的泛化或实板验证结论。'
    common=['## 受测试反馈影响流程的纠正（2026-10-10）','',
            '### 责任与范围','',
            '复审没有发现训练代码把fold10样本用于梯度更新，也没有发现按fold10指标直接排序选择种子或阈值的代码。已确认的问题是研发反馈：联合约束阈值搜索修正在第一版测试复测之后提出；三种子教师集成在查看父实验测试结果后提出，后续蒸馏继承了该追加教师。因此撤回这些流程作为完全独立测试证据的解释，不能把所有正确的训练集隔离检查误写成“整个研发过程从未受测试影响”。','',
            '### 本轮预先固定的重做方法','',
            '- 使用原官方folds1–8训练、fold9验证；五个独立sigmoid、100 Hz、12导联、10秒不变。教师/新学生沿用筛选后验证2146条；历史联合阈值分支沿用未筛选验证2183条，二者不能直接作同人群成绩比较。',
            '- 联合约束阈值只读取fold9保存概率及第一次集成测试之前已记录的Precision/Recall下限；重新求解，并重做原训练池四组留出验证及模型采用条件。没有再次评估fold10。',
            '- 教师回到三种子追加集成之前的9个单网络候选，重新在fold9进行FP32推理，按原预处理、架构、种子选择顺序复算。教师训练权重沿用原训练集所得权重，没有重训教师；取消本轮使用追加集成教师。',
            '- 普通CNN和单教师蒸馏CNN分别使用42/43/44三个成对种子，从随机初始化重新训练，共6次；不载入历史学生权重。沿用30轮上限、早停8、学习率0.001、batch32、加权BCE、AdamW和原数据预处理；KD alpha=0.5、T=2。',
            '- 最佳epoch按fold9 AUROC选择；最终学生在全部6个候选中按fold9 FP32宏F1、再AUROC选择。每个候选五类阈值只按固定0.10–0.90、步长0.01网格在fold9选择。',
            '- 新学生独立导出FP32/INT8；校准只使用256条训练记录。旧seed44 INT8与新INT8均使用自身fold9细阈值，按验证宏F1、再AUROC推荐。保留原ONNX、参数、阈值及HEX。',
            '- 新入口拒绝fold10波形、历史test结果文件和混有历史测试成绩的summary读取；不调用旧脚本的测试阶段。原始元数据仍用于确定官方划分；文件拦截属于Python读取边界，并不声称审计所有第三方C库内部文件访问。','',
            '### 对历史结论的处理','',
            '早期优化、患者隔离内部验证、单教师首轮蒸馏、噪声研究、Ribeiro适配网络及26版本指标复核的原文件和实测数字均保留。本轮没有发现其代码直接按fold10选参数的证据，不为“洗掉历史”而重复所有训练；有关分支仍有共享验证集、测试反复使用及探索性设计的边界。追加集成和派生seed42不能再作为本次纠正推荐，旧表中的“当前”“推荐”“最优”只描述当时历史状态。','',
            '重新训练或重新选择不能使已经查看过的fold10恢复独立。当前这轮是在已知历史基础上约束新的选择过程，不是预注册研究，不是新的盲测，也不是独立泛化确认。重新随机划分同一批已用于开发的数据不能解决这个问题；如需要确认性结论，必须使用此前未用于开发且标签兼容的独立数据。Chapman现有标签不足以完成五类确认，新候选也没有Chapman结果。','',
            '配置与冻结选择见 `outputs/validation_only_redo_20261010/protocol.json`；入口 `scripts/run_validation_only_redo.py`，默认执行完整纠正流程，不提供fold10评估阶段。文档更新入口 `scripts/update_validation_redo_docs.py`。']
    overall=[];detail=[];deployment=[]
    if complete:
        s=read(OUT/'summary.json');a=read(OUT/'audit.json');ind=read(OUT/'independent_audit.json')
        assert ind['passed']
        assert a['passed'] and s['protected_artifacts_unchanged'] and s['fold10_evaluation'] is None
        teacher=s['teacher'];student=s['selected_student'];dep=s['deployment']['selected'];joint=s['joint_thresholds']
        common+=['','### 实际完成结果','',
                 f'9个原单网络候选重新推理及选择完成，仍选 `{teacher["name"]}`；验证宏F1 {teacher["validation"]["macro_f1"]*100:.3f}%，验证AUROC {teacher["validation"]["macro_auroc"]:.6f}。教师权重复用，学生6次从随机初始化重训均已完成。',
                 f'联合约束重算：原训练池内部CV模型选择 `{joint["cv_model_choice"]}`；全量fold9宏F1 {joint["validation"]["macro_f1"]*100:.3f}%，内部四折宏F1均值 {joint["cv_mean_macro_f1"]*100:.3f}%。这是内部验证，未给出新fold10成绩。',
                 f'新训练学生按fold9选 `{student["name"]}`，最佳epoch {student["epoch"]}；FP32验证宏F1 {student["validation"]["macro_f1"]*100:.3f}%。INT8部署比较按fold9选择 `{dep["name"]}`，验证宏F1 {dep["validation"]["macro_f1"]*100:.3f}%。',
                 f'旧模型、原ONNX和HEX哈希保持一致；没有读取历史fold10预测/指标或fold10波形，没有生成新固件或实板验证。独立复算{ind["metric_sets_independently_recomputed"]}组整体与逐类指标、重新搜索8组学生/INT8细阈值，全部通过；训练统计、教师与学生选择、INT8选择及校准ID也独立复核通过。新入口及既有蒸馏/训练策略相关检查30项通过；新入口本身13项。']
        for arm in ['supervised','single_teacher_kd']:
            rows=[r for r in s['students'] if r['arm']==arm]
            mean=sum(r['validation']['macro_f1'] for r in rows)/3
            common.append(f'{arm}三个种子验证宏F1平均 {mean*100:.3f}%（各自已调验证阈值；不是独立测试成绩）。')
        ordinary={r['seed']:r for r in s['students'] if r['arm']=='supervised'}
        distilled={r['seed']:r for r in s['students'] if r['arm']=='single_teacher_kd'}
        deltas=[(distilled[seed]['validation']['macro_f1']-ordinary[seed]['validation']['macro_f1'])*100 for seed in [42,43,44]]
        common.append('蒸馏减普通CNN的成对验证宏F1差值（42/43/44，百分点）：'+', '.join(f'{v:+.3f}' for v in deltas)+'。不同种子并非都改善，不据此宣称各项指标全面提升。')
        common.append('联合阈值可加载清单为 `outputs/validation_only_redo_20261010/joint_thresholds/ensemble_manifest.json`，原组件和原集成清单保留。')
        common.append('完整复现依次执行：`.venv/Scripts/python.exe scripts/run_validation_only_redo.py`、`.venv/Scripts/python.exe scripts/write_validation_redo_manifest.py`、`.venv/Scripts/python.exe scripts/audit_validation_only_redo.py`、`.venv/Scripts/python.exe scripts/update_validation_redo_docs.py --stage complete`。已有完成训练会检查配置后跳过；部分训练不静默续跑。本轮校准张量也与清单对应的256条真实训练输入逐元素一致。')
        overall=['## 纠正流程：仅验证集指标（2026-10-10）','',
                 '以下是fold9指标，不能与历史fold10指标当作同一测试集成绩比较。普通/蒸馏6个学生均为本轮重训。教师为原单网络候选验证选择复算；不使用追加集成。','',
                 '| 版本 | 最佳epoch | AUROC | PR-AUC | Precision % | Recall % | F1 % |','|---|---:|---:|---:|---:|---:|---:|']
        for r in s['students']:
            m=r['validation'];overall.append(f'| {r["name"]} | {r["epoch"]} | {m["macro_auroc"]:.6f} | {m["macro_auprc"]:.6f} | {m["macro_precision"]*100:.3f} | {m["macro_recall"]*100:.3f} | {m["macro_f1"]*100:.3f} |')
        overall+=['','| INT8验证比较 | AUROC | Precision % | Recall % | F1 % |','|---|---:|---:|---:|---:|']
        for r in s['deployment']['candidates']:
            m=r['validation'];overall.append(f'| {r["name"]} | {m["macro_auroc"]:.6f} | {m["macro_precision"]*100:.3f} | {m["macro_recall"]*100:.3f} | {m["macro_f1"]*100:.3f} |')
        overview=[{'name':teacher['name'],'validation':teacher['validation']},{'name':'ensemble_equal_joint_thresholds','validation':joint['validation']}]
        overall+=['','### 教师选择与历史联合阈值重算','',
                  '教师/学生及INT8使用筛选后fold9的2146条；历史联合阈值重算沿用未筛选2183条，各分支按自身原队列报告。9个教师候选及四组内部留出结果完整保存在本轮JSON；内部留出不等于独立测试。','',
                  '| 版本 | 验证记录数 | AUROC | PR-AUC | Precision % | Recall % | F1 % |','|---|---:|---:|---:|---:|---:|---:|']
        for r in overview:
            m=r['validation'];overall.append(f'| {r["name"]} | {m["records"]} | {m["macro_auroc"]:.6f} | {m["macro_auprc"]:.6f} | {m["macro_precision"]*100:.3f} | {m["macro_recall"]*100:.3f} | {m["macro_f1"]*100:.3f} |')
        detail=['','### 新学生、INT8、所选教师与联合阈值的逐类验证指标','',
                '| 版本 | 类别 | 阳性数 | 阈值 | AUROC | PR-AUC | Precision % | Recall % | F1 % |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in s['students']+s['deployment']['candidates']+overview:
            for c,m in r['validation']['per_class'].items():
                detail.append(f'| {r["name"]} | {c} | {m["support"]} | {m["threshold"]:.2f} | {m["auroc"]:.6f} | {m["auprc"]:.6f} | {m["precision"]*100:.3f} | {m["recall"]*100:.3f} | {m["f1"]*100:.3f} |')
        common+=['','完整整体与逐类验证指标见[指标总表](各模型整体与逐类指标总表.md#metric-current)当前纠正章节；全部新验证概率、checkpoint、训练历史、冻结阈值、ONNX及审计位于本轮独立目录。']
        deployment=['## 纠正后候选的部署状态（2026-10-10）','',
                    f'仅按fold9 INT8比较推荐 `{dep["name"]}`；对应模型文件 `{dep["onnx"]}`，SHA-256 `{dep["onnx_sha256"]}`；候选阈值 `{dep["thresholds"]}`。这是验证选择，尚无本轮独立测试结果。',
                    f'本轮重新训练并导出的新学生为 `{student["name"]}`，产物位于 `outputs/validation_only_redo_20261010/selected_export/`，必须使用本目录自己的标准化、量化参数和候选阈值。',
                    '该候选只有FP32/INT8 ONNX及验证核对；未进行新的Cube.AI资源分析、C代码生成、完整固件编译、烧录或实板串口计时。不能沿用旧模型的板端结果作为新模型证据。原seed44固件、原部署阈值和所有可回退文件未修改；芯片VG/VE配置差异仍须在生成新工程前处理。']
    else:common+=['','执行状态：重做入口及防泄露检查30项通过；协议冻结、教师复算与内部验证重做正在进行。后续6次新训练和INT8验证未完成，不预填成绩。']
    names=['README.md','模型训练与优化.md','各模型整体与逐类指标总表.md','STM32部署与串口验证.md','Chapman_冻结模型外部验证总结.md']
    for name in names:
        path=ROOT/'md'/name;text=path.read_text(encoding='utf-8')
        text=replace_block(text,'validation-redo-notice-20261010',intro,top=True)
        if name=='模型训练与优化.md':
            text=text.replace('验证集阈值优化值可能偏乐观，测试集才是最终泛化结果。测试数据不参与方案或种子选择。','验证集阈值优化值可能偏乐观；fold10已多次使用，只能报告历史探索性结果。当轮代码按验证指标选择，但整个研发历史存在测试反馈，不能称为全程未受测试影响。')
            body='\n'.join(common)
        elif name=='各模型整体与逐类指标总表.md':body='\n'.join(overall+detail) if complete else '## 纠正后的指标口径（2026-10-10）\n\n重做正在执行。已有fold10表格均为历史探索性结果；新表将单独报告fold9验证指标，尚未完成前不预填。'
        elif name=='STM32部署与串口验证.md':body='\n'.join(deployment) if complete else '## 评价纠正期间的部署状态（2026-10-10）\n\n原seed44 HEX及部署参数保留。新集成教师派生seed42不作为本次纠正推荐；纠正候选尚未完成，没有新固件或实板结果。'
        elif name=='Chapman_冻结模型外部验证总结.md':body='## 纠正流程与外部验证边界（2026-10-10）\n\n现有Chapman结果保留为其对应的原冻结模型证据。本次纠正候选没有执行Chapman推理，不能继承旧模型外部成绩。Chapman NORM真值不足，不是完整五类独立验证；本轮也未使用Chapman选模或调阈值。'
        else:
            body='## 当前纠正流程入口（2026-10-10）\n\n'
            if complete:
                body+=f'纠正已完成：6个学生重新训练，按fold9推荐 `{dep["name"]}`，验证宏F1 {dep["validation"]["macro_f1"]*100:.3f}%。自身阈值约为 `[0.68,0.45,0.61,0.57,0.78]`，必须使用新导出目录的标准化与量化参数。新候选尚无独立测试、Chapman结果或完整固件。\n\n'
                body+='详细方法见[训练与优化](模型训练与优化.md)末尾，整体和逐类结果见[指标总表](各模型整体与逐类指标总表.md)末尾。\n\n'+'\n'.join(common[-8:])
            else:body+='正在重做受测试反馈影响的两项流程，细节见[训练与优化](模型训练与优化.md)末尾；独立目录 `outputs/validation_only_redo_20261010/`。原模型和固件保留。'
        # Keep the reorganized chapter structure when refreshing this block.
        if any(a in text for a in ['id="step-01"','id="deployment-status"','id="external-role"']):
            body=re.sub(r'^(#{2,}) ',r'#\1 ',body,flags=re.M)
        elif 'id="overview-route"' in text:
            body=re.sub(r'^## ', '### ',body,flags=re.M)
        elif 'id="metric-current"' in text:
            body=body.replace('## 纠正流程：仅验证集指标（2026-10-10）','## 2. 当前纠正结果：仅fold9（2026-10-10）',1)
        text=replace_block(text,'validation-redo-results-20261010',body)
        text=text.replace('更新：2026-10-09。','更新：2026-10-10。',1)
        text=text.replace('新教师蒸馏 seed42 是尚未打包上板的候选。','原追加集成教师派生seed42改为历史探索性候选；当前纠正状态见本文件首条说明和末尾记录。')
        text=text.replace('新学生按验证集选 seed42；INT8 验证规则推荐 新蒸馏 INT8 候选。','历史该轮按验证集选seed42；该追加集成派生候选已撤回本次纠正推荐，仅保留历史记录。')
        path.write_text(text,encoding='utf-8')
    rootread=ROOT/'README.md';text=rootread.read_text(encoding='utf-8')
    text=replace_block(text,'validation-redo-notice-20261010',intro+'最新流程与证据见[核心文档入口](md/README.md)。',top=True)
    rootread.write_text(text,encoding='utf-8')
    assert len(list((ROOT/'md').glob('*.md')))==5
    print('Updated five existing core MDs and root README:',args.stage)
if __name__=='__main__':main()
