# 十二导联 ECG 五类多标签识别与 STM32 部署

基于 PTB-XL 的课程展示项目，完成数据处理、神经网络训练、知识蒸馏、静态 INT8 量化，以及 STM32F407VE 串口波形分类程序。展示时由电脑发送数据库中的 ECG 波形，芯片返回五类概率和分类结果。

**当前状态（2026-10-03）：** seed44 蒸馏 CNN INT8 已由 STM32Cube.AI 转换并完整编译，HEX 与源码工程已提供；实际烧录、板上串口收发和推理耗时尚待验证。展示使用公开数据库，无需人体采集。

## 输入与类别

- 十二导联，100 Hz，10 秒，输入形状 `[1,12,1000]`。
- 导联顺序：`I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`。
- 输出顺序：`NORM, MI, STTC, CD, HYP`，每类独立 sigmoid 和独立阈值，一条记录可以同时属于多个类别。

| 类别 | 含义 |
|---|---|
| NORM | 正常心电 |
| MI | 心肌梗死 |
| STTC | ST/T 改变 |
| CD | 传导异常 |
| HYP | 心肌肥厚 |

输出对应 PTB-XL 的诊断大类，节律标注不会自动归入五类。学生训练剔除五类标签全部为零的记录，保留任一类阳性的记录。

## 当前模型结果

使用官方患者隔离划分：folds 1–8 训练、fold 9 验证、fold 10 测试。筛选后分别为 **17084 / 2146 / 2158** 条记录。标准化、量化校准使用训练集，阈值与部署候选根据验证集确定。

| 部署模型 | 测试 Macro AUROC | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| 蒸馏 CNN seed44 INT8 | 0.924276 | 0.742429 | 0.759592 | 0.749441 |

已记录的测试宏 F1 最高学生是 seed43 FP32（0.751618）；部署采用验证集选出的 seed44 INT8。上述指标来自筛选后的 PTB-XL 测试集；该测试折已在多轮实验中使用，不能当作新的盲测。Chapman 的标签映射与外部评估单独记录。

完整对比和逐类指标见 [指标总表](md/各模型整体与逐类指标总表.md)，蒸馏汇总见 [summary.json](outputs/distillation/summary.json)。

## 快速开始

以下命令在 Windows PowerShell 的项目根目录执行。

```powershell
git clone https://github.com/wwq22ya-22/ptbxl-ecg-analysis.git
cd ptbxl-ecg-analysis
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt -r requirements-train.txt -r requirements-app.txt -r requirements-api.txt -r requirements-stm32.txt
```

仓库提供正式 CNN / ResNet 检查点、优化教师、筛选后 ResNet、seed43 / seed44 蒸馏学生、seed44 ONNX 和串口演示样本。训练和完整数据评估需要另行下载 PTB-XL；串口演示可以直接使用随包的 128 条验证折样本。

### STM32 波形分类展示

1. 下载 [烧录 HEX](outputs/board_release/seed44_int8_stm32f407ve/ECG_Student_INT8_STM32F407VE.hex)，通过 STM32CubeProgrammer 与 ST-LINK/SWD 下载到 **STM32F407VE**。
2. 核对板载晶振：当前固件使用 **8 MHz 外部晶振、168 MHz 主频**。晶振不同时修改时钟配置并重新编译。
3. 连接 USB 转串口：板子 `PA9/TX → RX`、`PA10/RX → TX`、`GND → GND`，采用 3.3 V TTL、115200 / 8N1 / 无流控。
4. 双击 [start_ecg_student_viz.cmd](start_ecg_student_viz.cmd)，选择实际串口并连接，发送波形，查看芯片分类，保存验证记录。

界面显示所选的一条导联，发送完整十二导联。电脑完成训练时相同的预处理，芯片执行 INT8 神经网络推理。连接时核对固件模型哈希、输入规格、类别和阈值。

| 完整固件资源 | 使用 | STM32F407VE 预算 |
|---|---:|---:|
| Flash | 158876 B（155.15 KiB） | 512 KiB |
| 主 SRAM | 43672 B（42.65 KiB） | 128 KiB |
| 栈预留 | 8192 B，已计入 SRAM | — |

编译为 0 错误、0 警告。Cube.AI 电脑 C 实现与 INT8 ONNX 在 128 条样本上的输出完全一致；这些结果不代表已经在芯片上验证。

- [完整 Keil 源码工程 ZIP](outputs/board_release/seed44_int8_stm32f407ve/ECG_Student_INT8_源码工程.zip)
- [仓库中的 Keil 工程](outputs/distillation/stm32_student_project/MDK_Project/ECG_Student_INT8.uvprojx)
- [烧录包使用说明](outputs/board_release/seed44_int8_stm32f407ve/使用说明.txt)
- [资源与文件校验清单](outputs/board_release/seed44_int8_stm32f407ve/release_manifest.json)

重新编译需要 Keil 与 ARM Compiler 5.06 update 4；源码包包含本工程使用的 Cube.AI Cortex-M4 运行库。重新转换需要安装 STM32Cube.AI，工具和参考工程路径须按本机位置设置。详细协议、输入量化参数与验证步骤见 [部署文档](md/STM32部署与串口验证.md)。

### 数据、网页与 API

```powershell
# 下载 PTB-XL 100 Hz 波形，默认位置 data/ptb-xl
.venv/Scripts/python.exe scripts/download_ptbxl.py --workers 8
.venv/Scripts/python.exe scripts/prepare_data.py

# 网页展示，默认使用原正式检查点
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py

# API，接口文档 http://localhost:8000/docs
.venv/Scripts/python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# 项目测试
.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
```

数据路径可通过 `configs/local.yaml` 覆盖，该文件不上传。网页/API 的默认模型与学生固件分别配置；生成学生固件不会自动替换网页模型。完整说明见 [运行文档](md/README.md) 与 [训练文档](md/模型训练与优化.md)。

## 代码与文档

| 目录 | 内容 |
|---|---|
| `src/` | 数据处理、网络、训练、蒸馏和推理 |
| `scripts/` | 数据下载、实验、量化、工程打包及串口界面 |
| `firmware/ecg_serial/` | 串口协议和 STM32 适配源码 |
| `app/`、`api/` | 网页与 API |
| `tests/` | 数据、训练、导出与串口协议测试 |
| `outputs/distillation/` | 学生模型、ONNX、实验汇总和 STM32 源码 |
| `outputs/board_release/` | 当前可烧录包 |
| `md/` | 五份核心说明文档 |

- [项目运行与协作](md/README.md)
- [各模型整体与逐类指标](md/各模型整体与逐类指标总表.md)
- [模型训练与优化](md/模型训练与优化.md)
- [STM32 部署与串口验证](md/STM32部署与串口验证.md)
- [Chapman 外部验证](md/Chapman_冻结模型外部验证总结.md)

仓库保留选定权重和主要实验汇总，完整训练缓存、全部实验中间产物、个人报告与课程资料留在本机。历史文档中部分中间产物路径供实验复现定位，未全部随仓库发布。

## 数据来源与许可

PTB-XL 来自 [PhysioNet v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)，数据采用 CC BY 4.0。演示样本来自筛选后的官方验证折 9；原始 mV 信号与预处理数据分别保存，记录 ID 在部署清单中列出。随包附 [数据许可](outputs/distillation/distilled_export/LICENSE_PTBXL.txt) 与 [归属及处理说明](outputs/distillation/distilled_export/DATA_ATTRIBUTION.txt)。

引用：Wagner et al. (2022), *PTB-XL, a large publicly available electrocardiography dataset*, v1.0.3, [doi:10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45)；原始论文 Wagner et al. (2020), *Scientific Data*, [doi:10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6)。PhysioNet 平台引用见官方数据页。

ST 的网络代码、运行库与第三方外设/CMSIS 文件保留各自版权与许可；Cube.AI 许可见 [LICENSE.txt](outputs/distillation/stm32_student_project/User/ai/LICENSE.txt)。参考项目 [AndrewLucenko/ptb-xl-ecg-classification](https://github.com/AndrewLucenko/ptb-xl-ecg-classification) 的划分和工程组织思路，未使用其模型权重或成绩。本项目的课程分类结果不构成临床适用性证明。
