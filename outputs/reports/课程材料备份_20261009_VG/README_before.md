# 项目运行与文档入口

更新：2026-10-08。四个旧对话的完整历史已合并为下方接续总览，并与现有实验、发布包和报告材料核对。当前已编译固件仍为 seed44 INT8；新教师蒸馏 seed42 是尚未打包上板的候选。实板型号有历史冲突，尚待只读识别；真实烧录、串口闭环和板上计时仍未验证。

<!-- project-handoff:start -->
## 四个对话接续总览（2026-10-08）

本节汇总四个旧对话的可读取完整历史，并与当前代码、实验 JSON、发布清单和报告文件核对。分支共有历史已合并；历史建议、后来取消的工作和当前已完成成果分别标记。后续新对话先读本节，再按需要读取其余四份核心文档。

### 1. 项目目标与必须保留的约定

课题为“基于 STM32 的十二导联心电信号五类多标签识别系统设计与实现”。完整链路为公开 ECG 数据读取、预处理与信号分析、模型训练和评价、网页/API 展示、蒸馏与 INT8 量化、电脑发送波形、STM32 返回分类结果。

- 用户已明确：这是公开数据库波形回放的课程展示，不要求人体采集。无需扩展人体电极、模拟前端或采集硬件。
- 输入：十二导联、100 Hz、10 秒，模型形状 `[1,12,1000]`。导联顺序为 I、II、III、aVR、aVL、aVF、V1、V2、V3、V4、V5、V6。
- 输出顺序固定为 `NORM, MI, STTC, CD, HYP`。五类独立 sigmoid/阈值判断，允许多标签；不能改为五选一 argmax。
- PTB-XL 官方 folds 1–8 训练、9 验证、10 测试，患者隔离。标准化和量化校准只用训练集；选模型和阈值只用验证集。
- 用户要求记录每次工作的完成清单、整体与逐类指标、实验配置、无效尝试和验证边界。
- 用户此前要求核心 MD 只保留五份。本次总览并入本文件，后续实验仍追加现有训练/指标/部署文档，自动报告放 `outputs/reports/`。
- 用户个人分工为模型训练、知识蒸馏与量化。报告中的日期、讨论经历、个人信息应按真实经历补齐。
- 历史上已明确取消 LUDB 方向，相关数据、代码和报告已清理；不把旧对话中的 LUDB 待办恢复到当前任务。

### 2. 实际目录与环境

| 项目 | 当前位置或状态 |
|---|---|
| 实际项目根目录 | `D:/MyProject/college/ecg1`，本轮确认存在 |
| Codex 中旧项目绑定目录 | `D:/MyProject/college/大三上/ecg1`，本轮确认不存在；保存的 ecg 项目仍指向这里 |
| PTB-XL 数据 | `D:/datasets/ptb-xl`，本轮确认存在；本机 `configs/local.yaml` 指向该目录 |
| Chapman 原始数据 | `outputs/external_validation/chapman_data/`，已确认有 WFDBRecords 和诊断字典 |
| Chapman 评估结果 | `outputs/external_validation/chapman_audited/` |
| 独立 Python 环境 | 项目 `.venv/`；沿用现有环境，避免覆盖系统或其他课程环境 |
| 当前核心文档 | 本文件、指标总表、训练与优化、STM32 部署、Chapman 外部验证 |
| 个人课程材料 | 项目 `报告材料/` |
| Git 远程 | `https://github.com/wwq22ya-22/ptbxl-ecg-analysis.git`，本轮核对本地 remote 配置 |

旧对话中的搬迁前路径和已合并/删除的 MD 文件名不再作为入口。项目目前使用完整 100 Hz 子集；500 Hz 扩展不属于当前展示必需项。历史记录称本机 Python 3.14.3、PyTorch 2.12.1+cu126；本轮未重新检测版本，不据此重装环境。

仓库在后续对话中曾按用户要求改为公开，早期“私有仓库”表述已过时。本轮没有查询远程当前可见性、提交状态，也没有推送；10 月 7 日新实验不能仅凭早期“已上传”就认定已同步。

### 3. 已完成工作与可继续使用的成果

| 工作 | 目前结论 | 证据入口 |
|---|---|---|
| 数据读取与标签映射 | PTB-XL WFDB、十二导联、官方患者划分和缺失数据提示已实现 | `src/data/`、`scripts/prepare_data.py` |
| 信号分析 | 基线处理、带通、R 峰、RR/心率、SDNN/RMSSD、波形图已实现 | `src/signal_processing/`、`src/visualization/` |
| 网页/API | 十二导联、信号分析、分类、模型关注区域、性能展示和 API 已实现 | `app/streamlit_app.py`、`api/main.py` |
| 基线与训练优化 | CNN、ResNet、紧凑 Inception、CNN+BiGRU；标准化、权重、调度、Mixup、ASL 等已比较 | [训练与优化](模型训练与优化.md) |
| 四折内部复核 | 原训练池 folds 1–8 分四组，fold9 校准；训练集统计量标准化的收益得到复核 | `outputs/cross_validation/`、`outputs/further_optimization/` |
| 五类全零记录筛选 | 保留任一五类阳性记录，筛选后的训练/验证/测试为 17084/2146/2158 | `outputs/diagnostic_only/` |
| 原教师蒸馏 | 普通 CNN 与蒸馏 CNN 三个成对种子已完成，平均测试 F1 0.737798 → 0.749464 | `outputs/distillation/summary.json` |
| 原 seed44 INT8 固件 | Cube.AI 转换、完整编译、HEX、源码 ZIP、电脑 C 实现和协议模拟核对已完成 | [部署文档](STM32部署与串口验证.md) |
| Chapman 冻结外部验证 | 六个版本、45053 条可读取 ECG；保守标签映射，未调外部阈值 | [外部验证](Chapman_冻结模型外部验证总结.md) |
| B 真实 ECG 加噪 | 白噪声增强、教师/学生训练、蒸馏、三个种子、INT8 已完成；强噪声改善但干净表现下降 | `outputs/noise_robustness_b/` |
| 更强教师与预处理消融 | 9 次训练和集成复核已完成；本轮保留原预处理，SE-ResNet18 未带来收益 | `outputs/teacher_preprocessing/` |
| 新教师蒸馏 | 三种子学生、九个学生对照、FP32/INT8 导出及 Cube.AI 分析已完成，验证选 seed42 | `outputs/teacher_distillation/` |

这里的“信号分析已实现”表示课程功能和既有检查通过，不表示 R 峰检测完成临床准确性验证。网页/API 默认模型仍是原正式检查点，尚未因学生训练而自动替换。

#### 最新模型对比：保留“现有固件”和“新候选”的区别

以下三行来自同一批筛选后的 2158 条 PTB-XL 测试记录，指标均为五类宏平均：

| 版本 | AUROC | Precision | Recall | F1 | HYP Recall |
|---|---:|---:|---:|---:|---:|
| 原 seed44 INT8，现有固件 | 92.428% | 74.243% | 75.959% | 74.944% | 59.160% |
| 新集成教师蒸馏 seed42 ONNX FP32 | 92.851% | 72.181% | 78.631% | 75.228% | 65.267% |
| 新集成教师蒸馏 seed42 INT8，候选 | 92.864% | 72.155% | 78.508% | 75.166% | 65.267% |

新 INT8 相对原版本：AUROC +0.436、Recall +2.549、F1 +0.222、HYP Recall +6.107 个百分点，Precision -2.088 个百分点。新蒸馏与原蒸馏的三种子平均 F1 仅差约 +0.075 个百分点，两个种子下降。因此用户提出的“各项指标都提升”尚未实现，不能把实验完成写成目标全部达成。

INT8 验证宏 F1 为原版本 0.754595、新版本 0.760370；按冻结验证规则推荐新候选。新候选尚未生成完整新 HEX、未烧录，现有 seed44 发布包保持原样。

- 原学生检查点：`outputs/distillation/distilled_seed44/checkpoint/cnn1d_best.pt`。
- 原 INT8：`outputs/distillation/distilled_export/ptbxl_cnn1d_int8.onnx`。
- 新学生检查点：`outputs/teacher_distillation/ensemble_kd_seed42/checkpoint/cnn1d_best.pt`。
- 新 INT8：`outputs/teacher_distillation/selected_export/ptbxl_cnn1d_int8.onnx`。
- 新候选依据：`outputs/teacher_distillation/frozen_selection.json`、`frozen_deployment_selection.json`、`summary.json`。
- 网页原 CNN 为 `outputs/checkpoints/baseline_best.pt`，ResNet 为 `outputs/checkpoints/resnet1d_best.pt`；不要凭模型名称猜文件名。

新旧学生结构相同，Cube.AI 模型资源为 INT8 权重 112276 B（109.64 KiB）、激活 24000 B（23.44 KiB）。新版本完整固件的大小和耗时仍需要单独编译/上板确认，不能照搬旧包的链接结果。

#### 已知无效尝试与待研究方案

B 加噪候选 INT8：干净 F1 0.741773，低于原版本 0.749441；10 dB F1 0.737271，高于原版本 0.732486。它保留为抗噪实验候选，未替换展示固件，且本轮只覆盖高斯白噪声。

MATLAB/ECGSYN 合成心电、C 组“少量合成 ECG”、500 Hz 扩展、自监督预训练、人工特征/导联注意力、更细的选模和蒸馏策略仍是讨论或可选研究方向，不能写成已经完成。合成波形的五类标签和十二导联关系尚未验证。旧对话“接下来训练新教师/蒸馏”的部分待办已由 10 月 7 日实验完成，不需要重复训练。

### 4. 上板现状与必须核对的问题

当前发布包为 [seed44 烧录 HEX](../outputs/board_release/seed44_int8_stm32f407ve/ECG_Student_INT8_STM32F407VE.hex) 与 [完整源码 ZIP](../outputs/board_release/seed44_int8_stm32f407ve/ECG_Student_INT8_源码工程.zip)。

| 项目 | 已确认的工程/电脑验证结果 |
|---|---|
| 编译目标 | STM32F407VE；这是现有工程目标 |
| 完整固件 Flash / 主 SRAM | 158876 B / 43672 B，即 155.15 / 42.65 KiB |
| 栈预留 | 8192 B，已包含在上述 SRAM；早期 35.65 KiB 是修正栈之前的旧结果 |
| 编译结果 | 0 错误、0 警告，既有发布清单有记录 |
| 128 条电脑 C 实现与 INT8 ONNX 核对 | 记录中输出完全一致；协议模拟通过，不属于板上实测 |
| 当前时钟 | 工程按外部 8 MHz、主频 168 MHz 配置 |
| 当前串口 | USART1，PA9 TX / PA10 RX，115200、8N1、3.3 V TTL、共地 |
| 物理开发板验证 | 发布清单 `physical_board_test=false`；本轮未见约定的实板结果文件 |

用户早先确认过 F407VE，旧三分类工程/日志曾选择 F407VG，较晚查到的 9 月 30 日日志含实际 JTAG ID `0x06419041`，旧对话提示器件系列可能与 F407 配置不同。**实板完整型号尚需复核**：连接后读取器件 ID、Flash 容量并对照芯片丝印，再确认晶振和工程目标。软件选择项不能代替实物识别；现有部署文档的“型号已确认”只反映当时用户提供的型号。

学生固件专用参数为阈值 `[0.55,0.60,0.55,0.75,0.65]`、输入 scale `0.2008364498615265`、zero point 0。新 seed42 的阈值不同，必须读取它自己的清单，不能套用旧值。电脑完成对应检查点的预处理，芯片执行 INT8 推理。

使用 `start_ecg_student_viz.cmd` 连接现有学生固件；`start_ecg_stm32_viz.cmd` 是旧部署入口，两者清单必须匹配。Cube.AI 曾受中文用户路径影响，专用启动器为 `start_stm32cubemx_ascii.cmd`，使用纯英文工作路径；无需因旧错误记录再重复重装。

### 5. 外部验证结果的使用边界

当前外部验证依据为 Chapman，不再使用已取消的 LUDB 报告。原 seed44 INT8 在明确阳性上的召回率为：

| MI | STTC | CD | HYP | 四类平均 |
|---:|---:|---:|---:|---:|
| 93.66% | 77.31% | 67.35% | 78.91% | 79.31% |

NORM 没有可靠真值，不计分；未提及诊断保留为未知。四类 F1 0.4955 和 AUROC 0.8970 是采用“未标出类别为阴性”假设的附加结果，不能当作已确认的五类外部指标。新 seed42 尚无本次核实过的 Chapman 评估成果。

PTB-XL 测试折已经在多轮实验中重复使用，更强教师集成也有探索性追加，不称为新的盲测。报告区分原始 2198 条口径、筛选后 2158 条口径、四折内部均值和外部四类口径，不能把这些数字拼成同条件对比。

### 6. 课程材料与剩余交付

已核对以下三份文件存在：

- [个人实习记录第 3–10 周](../报告材料/个人实习记录_第3至10周.docx)：已有初稿，个人信息和实际讨论记录仍需本人核对。
- [第 4–5 周修订开题报告](../报告材料/《科技实习与创新》开题报告_十二导联心电五类识别_第4至5周修订版.docx)：作为开题阶段的使用版本，把后续蒸馏/固件成果改为计划。
- [原开题报告](../报告材料/《科技实习与创新》开题报告_十二导联心电五类识别.docx)：保留原稿，部分内容基于较晚成果，不适合作为第 4–5 周当时已经完成工作的证明。

课程要求来自旧对话对 `课程安排与介绍.pptx` 的阅读：第 11 周交小组中期报告和个人第 3–10 周记录；第 16 周交小组最终报告和完整第 3–16 周记录；第 17–18 周进行 7 分钟答辩。课程安排写开题第 3–4 周，用户说明自己的开题形成于第 4–5 周，需按实际提交时间填写。此轮未重新解读 PPT 或修改 Word。

当前 `报告材料/` 未见小组中期报告、最终报告、个人第 11–16 周记录、答辩 PPT 成稿；它们列为待交付/待确认，不能仅凭已有技术文档视为完成。参考文献真实性、周次与实际成果先后关系继续按用户要求核对。

### 7. 后续任务顺序与完成标准

| 顺序 | 待办 | 完成标准/依赖 |
|---|---|---|
| 1 | 修正后续对话的项目目录 | 绑定实际 `D:/MyProject/college/ecg1`，能在正确目录读写与运行 |
| 2 | 识别实板型号、Flash、晶振与连接 | 实板供电并连接调试器；只读识别结果与丝印一致后匹配工程 |
| 3 | 建立现有 seed44 的实板串口闭环 | 电脑发十二导联波形，真实芯片返回五类概率/标签，记录模型哈希和样本 ID |
| 4 | 板上输出一致性、耗时与稳定性 | 同样本逐条核对；分别记录传输/推理耗时、连续发送、超时和断线恢复 |
| 5 | 如推进新候选，将 seed42 单独打包 | 使用新清单、阈值和标准化参数，生成/编译独立固件并验证，保留 seed44 回退包 |
| 6 | 若继续追求精确率与召回同时提升 | 先基于已有逐类结果明确评价/选模规则，再做验证集控制实验；不重复现成实验，不承诺各项必涨 |
| 7 | 完成课程书面材料与演示 | 按实际周次补记录；完成中期/结题报告、7 分钟答辩 PPT 和操作脚本 |
| 8 | 同步最终交付并更新核心文档 | 保存实际板上证据、冻结最终包；确认远程状态后再按用户已授权范围同步 |

本次整理到第 1 项的问题说明和全部接续依据，没有烧录、改模型、改环境、推送仓库或新增训练。实板相关工作需要用户连接硬件；其余材料与实验准备可在正确目录继续。

### 8. 本轮核对范围与历史来源

本轮读取四个对话全部可返回页面：合计 166 个回合，按回合 ID 去除分支重复后 110 个。三个旧对话可见压缩记录，材料对话没有可见压缩记录；这是已发生的历史事件，不是它们正在同时压缩。

| 旧对话标题（工具返回原文） | 对话 ID | 主要承接内容 |
|---|---|---|
| 现在我这个五分类你想想怎么去优化，把各项指标都提升一下 | `01a0f166-47dd-7551-92c5-b6a772ef742e` | 优化、蒸馏、外部验证、固件、课程任务与最新实验 |
| 现在我有一个想法 能否用matlab生成标准的五类心电信号来训练模型 | `01a100d0-d213-74e3-ad9d-4055bfe6b7e3` | 后续已转为个人周记录、开题/阶段报告专用对话 |
| 仔细阅读这项任务 然后微调流程重新给我一版 在当前目录下新建md文档写入 项目任务：基于 PTB-XL 的十二… (2) | `01a0f624-d4be-7d81-953c-2e124f87daa2` | 系统实现、数据/模型基线、网页/API、Cube.AI 准备 |
| 仔细阅读这项任务 然后微调流程重新给我一版 在当前目录下新建md文档写入 项目任务：基于 PTB-XL 的十二导联 E… | `01a0cec2-ea22-7cf3-a32b-721d35b06e3b` | 与分支共有的实现历史、部署研究与项目定位 |

当前模型数字核对自新蒸馏 JSON；旧固件空间和未实板状态核对自发布清单；HEX 与源码 ZIP 的 SHA-256 本轮重算一致；关键权重、ONNX 和课程文档路径已检查。已有实验记录包含 82 项项目测试、最新 7 项蒸馏/导出检查和 22 组预测复核；本次只整合文档，未重新运行训练、这些测试或所有预测审计。

后续读取优先级：本总览 → [指标总表](各模型整体与逐类指标总表.md) → [训练与优化](模型训练与优化.md) → [部署与串口验证](STM32部署与串口验证.md) → [Chapman 外部验证](Chapman_冻结模型外部验证总结.md)。同一结论有冲突时，以日期更晚且有配置/清单/JSON 支撑的记录为准。

### 9. 新对话可直接使用的接续说明

> 继续 D:/MyProject/college/ecg1 项目。先读 md/README.md 中“四个对话接续总览（2026-10-08）”，再按需要读取其余四份核心文档。项目是公开数据库十二导联 ECG 的五类多标签课程展示，不需要人体采集。训练、优化、三种子蒸馏、INT8 和原 seed44 完整固件已完成；新教师蒸馏 seed42 已导出，但未替换固件。先核对实际目录和实板型号，再完成真实串口闭环、一致性、耗时/稳定性与课程交付。模型性能全面提升仍未达成，不能把精确率下降说成所有指标提升。LUDB 已取消；不重复已完成实验，不新增核心 MD。每轮将完成清单、结果、限制和下一步写回现有文档。涉及个人经历与硬件实测的内容按真实证据填写。

<!-- project-handoff:end -->

## 保留的五份文档

| 文档 | 内容 |
|---|---|
| [本文件](README.md) | 环境、运行、项目结构和小组协作 |
| [各模型整体与逐类指标总表](各模型整体与逐类指标总表.md) | PTB-XL 历史模型、筛选模型、蒸馏、量化与交叉验证指标 |
| [模型训练与优化](模型训练与优化.md) | 数据剔除、患者划分、训练方法、蒸馏和复现入口 |
| [STM32 部署与串口验证](STM32部署与串口验证.md) | 当前学生固件、Cube.AI 资源、接线、协议和验证边界 |
| [Chapman 外部验证](Chapman_冻结模型外部验证总结.md) | 标签映射、四类检出率、条件指标和量化差异 |

## 数据与模型

模型输入为十二导联、100 Hz、10 秒，形状 `[1,12,1000]`。输出依次为 NORM、MI、STTC、CD、HYP，逐类 sigmoid 与独立阈值；一条 ECG 可以对应多个类别。

PTB-XL 1.0.3 位于本机 `D:/datasets/ptb-xl`，通过不提交的 `configs/local.yaml` 配置。官方 folds 1–8 训练、9 验证、10 测试，患者隔离。原始训练/验证/测试为 17418/2183/2198；剔除五类全零记录后为 17084/2146/2158。筛选规则和指标口径见训练文档，不把无诊断标签解释为正常。

当前部署候选为 `outputs/distillation/distilled_seed44/checkpoint/cnn1d_best.pt` 及 `outputs/distillation/distilled_export/ptbxl_cnn1d_int8.onnx`。旧网页/API 默认 CNN 与 ResNet 仍使用 `outputs/checkpoints/` 的正式检查点；没有因生成学生固件而自动切换网页默认模型。

2026-10-03 已完成 B 实验：真实 ECG 加高斯白噪声、教师重训练、三个种子的普通 / 蒸馏学生、INT8 导出与 Cube.AI 资源分析。加噪普通训练有平均收益，本轮蒸馏尚未带来额外收益；B 蒸馏 INT8 在 10 dB 噪声下改善、干净测试 F1 下降，保留为实验候选。当前部署继续使用原 seed44。方法和结论见训练文档，所有模型的整体 / 逐类指标见指标总表；完整记录位于 `outputs/noise_robustness_b/`，112 份预测已独立复核。

## 安装与启动

在项目根目录运行。已有 `.venv` 时直接使用其中的 Python；小组成员在自己的克隆目录执行：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt -r requirements-train.txt -r requirements-app.txt -r requirements-api.txt -r requirements-stm32.txt
.venv/Scripts/python.exe scripts/download_ptbxl.py --workers 8
.venv/Scripts/python.exe scripts/prepare_data.py
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
```

网页地址为 `http://localhost:8501/`。GPU 构建须与本机 Python、驱动匹配；本机环境为 Python 3.14.3、PyTorch 2.12.1+cu126。不要直接复制 `.venv` 到其他电脑。

```powershell
# 单条推理
.venv/Scripts/python.exe scripts/predict_record.py --ecg-id 9
# API，接口文档为 localhost:8000/docs
.venv/Scripts/python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
# 项目测试
.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp outputs/diagnostic_only/pytest_run
```

串口学生界面双击根目录 `start_ecg_student_viz.cmd`；接线、HEX 和真实/模拟模式见部署文档。模拟流程不代表板上运行或板上耗时。

## 项目结构与清理

| 位置 | 用途 |
|---|---|
| `src/` | 数据、信号处理、模型、训练和推理 |
| `scripts/` | 下载、训练、评估、导出及报告入口 |
| `app/`、`api/` | Streamlit 与 FastAPI |
| `firmware/` | 串口协议及 MCU 接入源码 |
| `tests/`、`configs/` | 测试、默认及本机配置 |
| `outputs/` | 检查点、指标、逐条预测、工程及生成报告 |
| `md/` | 五份核心文档 |

原始数据、缓存与训练产物不随普通代码发布自动下载。`outputs/cache` 可以重建；模型、部署清单、配置、JSON 指标及预测是复现依据，清理时保留。详细自动生成报告在 `outputs/reports/`，核心 `md` 不因重跑实验自动增加文件。

## 小组协作与归属

公开仓库为 [wwq22ya-22/ptbxl-ecg-analysis](https://github.com/wwq22ya-22/ptbxl-ecg-analysis)，可直接浏览或克隆；需要向原仓库推送的小组成员，由所有者在 Settings → Collaborators 邀请。各成员以自己的克隆目录和本机数据路径为准，建议用分支和 Pull Request 协作。

仓库提供选定模型、学生串口演示样本、STM32F407VE 源码工程及烧录包；完整原始数据库和中间缓存不上传。演示样本的归属、处理方式及 CC BY 4.0 许可随包保存，主 README 给出直接下载链接。

本项目参考 [AndrewLucenko/ptb-xl-ecg-classification](https://github.com/AndrewLucenko/ptb-xl-ecg-classification) 的数据划分与工程组织思路，未复制其模型权重或成绩；参考项目的 MIT 许可不自动成为本项目的代码许可。数据按 [PTB-XL 官方页](https://physionet.org/content/ptb-xl/1.0.3/) 和 [Chapman 官方页](https://physionet.org/content/ecg-arrhythmia/1.0.0/) 的要求引用。

当前数据上的指标不等于临床适用性证明。短时 HRV、R 峰和模型响应图属于课程实验功能。最新回归测试为 82 项通过，实板结果仍待验证。

<!-- teacher-study-entry:start -->
2026-10-07 完成更强教师与预处理消融，共9次训练，包含入选教师与紧凑Inception的三个种子配对复核。预处理按验证集规则保留原方案；教师候选为 `inception_time1d_P0_current_seed42`，权重另存 `outputs/teacher_preprocessing/selected_teacher.pt`。当前部署仍为原seed44 INT8。整体/逐类结果已追加到现有训练文档和指标总表，配置、预测及独立审计保存在 `outputs/teacher_preprocessing/`。
<!-- teacher-study-entry:end -->

<!-- teacher-ensemble-followup:start -->
教师探索性追加实验：三种子概率集成的验证选择为 `inception_time1d_ensemble3`；权重和训练logits位于 `outputs/teacher_preprocessing/ensemble_followup/`。该步骤在首轮测试分析后提出，属于固定测试集上的探索性复核；当前板上候选继续使用原seed44 INT8。结果见现有训练文档和指标总表。
<!-- teacher-ensemble-followup:end -->

<!-- new-teacher-distillation:start -->
2026-10-07 新集成教师蒸馏已完成三个种子、九个学生模型对照、INT8 与 Cube.AI 分析。新学生验证选 seed42，新 INT8 测试宏 F1 75.166%，原 seed44 为 74.944%；按 INT8 验证规则推荐 新蒸馏 INT8 候选，现有固件保留。方法与整体 / 逐类指标已写入原训练文档和指标总表，完整结果位于 `outputs/teacher_distillation/`。
<!-- new-teacher-distillation:end -->

<!-- student-threshold-refinement:start -->
2026-10-08 已完成原 seed44、新蒸馏 seed42 的 FP32 / INT8 阈值细搜（0.05 → 0.01）。权重与固件保留，验证选择冻结后才复算测试指标；整体 / 逐类结果见训练文档和指标总表，完整记录位于 `outputs/student_threshold_refinement/`。
<!-- student-threshold-refinement:end -->

<!-- unfiltered-threshold-refinement:start -->
2026-10-09 完成 26 个主要模型 / 推理版本的筛选、未筛选 0.01 阈值比较。未筛选验证 / 测试为 2183 / 2198，新增全零五类标签记录的误报已单列；模型训练范围另行注明。整体、逐类列表追加到原训练文档和指标总表，结果位于 `outputs/unfiltered_threshold_refinement/`。
<!-- unfiltered-threshold-refinement:end -->
