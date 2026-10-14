# 十二导联ECG五类多标签识别与STM32回放

**小组成员先看：[阅读指南与目录说明](md/小组成员阅读指南.md)**。按用途查找文档、模型、固件和实板记录。

更新：2026-10-14。课程项目使用PTB-XL公开波形，不要求人体采集。现有两类STM32流程：A由电脑预处理、芯片推理，另提供独立二进制INT8快速版；B接收原始mV波形，由芯片完成预处理及推理。完整实验过程按“数据集选择 → 数据分析 → 标签与划分 → 预处理 → 网络与训练 → 验证选择 → 蒸馏量化 → 部署及课程交付”整理。

<!-- validation-redo-notice-20261010:start -->
**评价状态（2026-10-10）：** 当前验证候选为单教师蒸馏CNN seed43 INT8，折9宏F1为75.893%；现有完整HEX仍为原seed44及原部署阈值。折10在历史研发中已反复使用，且部分追加方案受测试反馈影响，所有已有折10成绩按历史探索结果解释。重做没有恢复折10的独立性；新候选没有本轮独立测试或完整固件。
<!-- validation-redo-notice-20261010:end -->

<!-- unet-root-20261010:start -->
1D U-Net双训练人群实验：已完成12次新训练及30组整体/逐类指标、30组阈值独立复核；仅折9评估。详见[方法与结果](md/模型训练与优化.md#unet-study-20261010)。既有部署版本保持原状态。
<!-- unet-root-20261010:end -->

## 项目与实验文档

| 文档 | 内容 |
|---|---|
| [项目总览](md/README.md) | 流程、状态、目录、启动、课程交付 |
| [STM32AI全流程](md/STM32AI心电识别与部署全流程.md) | 按结构组织的数据、训练、转换、工程与上板验收 |
| [GitHub发布范围](md/GitHub同步与实板验证.md) | 已共享产物、个人材料排除和实板证据 |
| [模型训练与优化](md/模型训练与优化.md) | 从数据集选择开始的完整实验过程 |
| [指标总表](md/各模型整体与逐类指标总表.md) | 当前验证结果与历史全部整体/逐类表 |
| [STM32部署](md/STM32部署与串口验证.md) | 模型版本、资源、串口及真实板端验证 |
| [Chapman外部验证](md/Chapman_冻结模型外部验证总结.md) | 标签兼容性、明确阳性召回和限制 |

## 数据与模型状态

- 实际项目`D:/MyProject/college/ecg1`，数据`D:/datasets/ptb-xl`，版本v1.0.3。
- 输入`[B,12,1000]`，100 Hz、10秒；输出NORM/MI/STTC/CD/HYP，五个独立sigmoid，允许多标签。
- 官方folds1–8训练、9验证、10历史测试；原始17418/2183/2198，当前筛选后17084/2146/2158。
- 当前仅按折9选出的单教师蒸馏seed43 INT8验证宏F1为75.893%；原seed44同口径75.683%。这是验证成绩，不是新测试成绩。
- 原seed44已按原sine/三分类配置编译两个VG工程，旧VE包保留；模型及原部署阈值相同。2026-10-11通过COM8确认A仅推理固件完成3条不同折10记录及1次重复的真实串口推理，返回1012–1013ms，与电脑C参考最大概率差约3.006e-08。B板端预处理、全量回放和长时间稳定性仍待验证。
- 新seed43仅有FP32/INT8导出与折9核对，不能继承旧固件资源或Chapman成绩。当前部署仍为 seed44 INT8；后续候选与实板模型分别记录。

## 软件启动

命令在实际项目根目录执行，数据路径用`configs/local.yaml`配置。本机已有`.venv`可直接复用。

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt -r requirements-train.txt -r requirements-app.txt -r requirements-api.txt -r requirements-stm32.txt
.venv/Scripts/python.exe scripts/prepare_data.py --dataset-root D:/datasets/ptb-xl
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
.venv/Scripts/python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

网页/API默认仍使用原正式模型，独立于新候选和旧学生固件；API文档为`http://localhost:8000/docs`。缺少数据时再下载，见[data目录说明](data/README.md)。

串口接线：PA9/TX→串口RX、PA10/RX→串口TX、GND共地，115200/8N1、3.3 V TTL。烧录和参数核对以[部署流程](md/STM32部署与串口验证.md#deployment-run)为准，不把模拟输出当成实板成绩。

## 串口验证工程

两个工程均使用原seed44 INT8及原部署阈值，按原参考工程配置STM32F407VG、8 MHz HSE、168 MHz主频、J-Link。在VS Code选择项目`.venv`后可直接运行对应Python文件。

| 项目 | A：仅推理 | B：板端预处理＋推理 |
|---|---|---|
| 发送数据 | 电脑预处理后的浮点信号 | 原始mV浮点波形，12导联×1000点 |
| 板端流程 | INT8量化→模型→五类判断 | 双中值去基线→0.5–40 Hz前后向带通→固定训练标准化→量化→模型→五类判断 |
| Keil工程 | [A工程](outputs/distillation/stm32_student_project_vg/MDK_Project/ECG_Student_INT8_VG.uvprojx) | [B工程](outputs/distillation/stm32_student_raw_vg_project/MDK_Project/ECG_Student_RAW_INT8_VG.uvprojx) |
| 固件 | [A HEX](outputs/board_release/seed44_int8_stm32f407vg/ECG_Student_INT8_STM32F407VG.hex) | [B HEX](outputs/board_release/seed44_int8_stm32f407vg_preprocess/ECG_Student_RAW_INT8_STM32F407VG.hex) |
| VS Code入口 | [ecg_stm32_viz.py](scripts/ecg_stm32_viz.py) | [ecg_stm32_raw_viz.py](scripts/ecg_stm32_raw_viz.py) |
| 双击启动 | [start_ecg_student_viz.cmd](start_ecg_student_viz.cmd) | [start_ecg_board_preprocess_viz.cmd](start_ecg_board_preprocess_viz.cmd) |
| 完整Flash / 主SRAM | 158876 B / 43672 B | 162420 B / 108368 B |

两种程序通过协议版本与预处理位置区分，错误配对会拒绝连接。B另核对预处理哈希。新工程编译0错误/0警告；27项检查通过，128条真实验证ECG的主机C预处理、INT8输入及分类与原流程一致。完整RAM已包含波形缓冲、工作区和8 KiB栈，主SRAM余量22704 B。

B先接收完整10秒片段再预处理，尚未改成连续逐点滤波；ARM实板数值、栈峰值、预处理和推理耗时待连接板子验证。详见[两个工程的部署记录](md/STM32部署与串口验证.md)。

2026-10-11：两个串口界面新增“波形来源”下拉框，默认折9的128条联调记录，可切换为筛选后的完整折10共2158条。折10使用原seed44固定预处理和阈值，电脑参考来自Cube.AI host C-model；切换后支持上一条/下一条/跳转，发送时禁止切换。折数、来源、样本及模型哈希写入每条验证记录，无需重新烧录HEX。

命令行可添加`--fold 10`，例如双击启动器对应的命令：`start_ecg_student_viz.cmd --fold 10`或`start_ecg_board_preprocess_viz.cmd --fold 10`。样本和核对记录在`outputs/serial_samples/seed44_fold10/`；详见[部署文档](md/STM32部署与串口验证.md)。本项目历史折10曾反复用于研发，这次回放不称为新的独立盲测。

A新增[二进制INT8快速版](md/STM32部署与串口验证.md#serial-fast-a)：先烧录[Fast HEX](outputs/board_release/seed44_int8_stm32f407vg_fast/ECG_Student_INT8_Fast_STM32F407VG.hex)，再在VS Code运行[scripts/ecg_stm32_fast_viz.py](scripts/ecg_stm32_fast_viz.py)，或双击[start_ecg_student_fast_viz.cmd](start_ecg_student_fast_viz.cmd)。[Fast Keil工程](outputs/distillation/stm32_student_fast_vg_project/MDK_Project/ECG_Student_INT8_Fast_VG.uvprojx)已编译0错误/0警告，Flash153040 B、RAM40664 B。仍为原seed44及原阈值，115200/8N1、折9/折10；每条传12000字节INT8负载，共12包。全部2286条量化输入与原C规则一致，44条C协议/C模型与GUI核对通过。预计整次约2–3秒，**快速固件尚未烧录实测**；旧文本A和B保留。

## 可回退固件与最新验证候选

- [原seed44 HEX](outputs/board_release/seed44_int8_stm32f407ve/ECG_Student_INT8_STM32F407VE.hex)。
- [原源码ZIP](outputs/board_release/seed44_int8_stm32f407ve/ECG_Student_INT8_源码工程.zip)。
- [原发布清单](outputs/board_release/seed44_int8_stm32f407ve/release_manifest.json)。
- [新seed43 INT8](outputs/validation_only_redo_20261010/selected_export/ptbxl_cnn1d_int8.onnx)。
- [新候选阈值](outputs/validation_only_redo_20261010/selected_export/candidate_thresholds.json)。
- [最新纠正审计](outputs/validation_only_redo_20261010/independent_audit.json)。

旧固件完整Flash158876 B、主SRAM43672 B（含8192 B栈），是编译链接数据；A仅推理已于2026-10-11完成小批实板回放，[原始串口记录](outputs/serial_samples/seed44_fold10/physical_inference_check_20261011_103033.json)另存，不将其视为VE包或B工程均已验证。模型激活/权重不能再重复叠加到完整固件资源上。

## 源码与证据

`src/`包括数据/滤波/网络/训练/推理，`scripts/`包括实验和工具，`app/`、`api/`为展示，`firmware/ecg_serial/`为协议与适配，`tests/`为相关检查，`outputs/`保留原始实验配置、权重、预测、导出及审计。

完整数据/缓存和部分本机实验产物不会随普通克隆下载；本机存在文件不证明远程已发布。五份核心MD按完整流程维护，整理前原文保存在`outputs/documentation_reorganization_20261010/original_markdown.zip`。

## 数据来源与许可

PTB-XL 来自 [PhysioNet v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)，数据采用 CC BY 4.0。默认联调样本来自筛选后的官方验证折9，可选择完整筛选后的折10；原始mV信号与预处理数据分别保存，记录ID保存在各样本包。随包附 [数据许可](outputs/distillation/distilled_export/LICENSE_PTBXL.txt) 与 [归属及处理说明](outputs/distillation/distilled_export/DATA_ATTRIBUTION.txt)，折10包同样保留许可及归属说明。

引用：Wagner et al. (2022), *PTB-XL, a large publicly available electrocardiography dataset*, v1.0.3, [doi:10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45)；原始论文 Wagner et al. (2020), *Scientific Data*, [doi:10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6)。PhysioNet 平台引用见官方数据页。

ST 的网络代码、运行库与第三方外设/CMSIS 文件保留各自版权与许可；Cube.AI 许可见 [LICENSE.txt](outputs/distillation/stm32_student_project/User/ai/LICENSE.txt)。参考项目 [AndrewLucenko/ptb-xl-ecg-classification](https://github.com/AndrewLucenko/ptb-xl-ecg-classification) 的划分和工程组织思路，未使用其模型权重或成绩。本项目的课程分类结果不构成临床适用性证明。
