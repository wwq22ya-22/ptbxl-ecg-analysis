# STM32部署与串口验证：版本、资源和板端证据

<!-- vscode-direct-serial-20261010:start -->
VS Code直接运行（2026-10-10）：选择项目`.venv`解释器后，A仅推理打开`D:/MyProject/college/ecg1/scripts/ecg_stm32_viz.py`；B板端预处理＋推理打开`D:/MyProject/college/ecg1/scripts/ecg_stm32_raw_viz.py`，点击右上角“运行Python文件”。两个入口各自默认加载配套原seed44清单；选择COM及115200并连接。A启动器为`start_ecg_student_viz.cmd`，B启动器为`start_ecg_board_preprocess_viz.cmd`；旧`start_ecg_stm32_viz.cmd`显式指定ResNet清单。配套工程、HEX、完整资源和数值复核见[两种独立工程](#board-preprocess-two-projects)。
<!-- vscode-direct-serial-20261010:end -->

更新：2026-10-11。各串口入口均支持折9联调或折10历史测试。仅切换折数无需重新烧录；新增[A快速版](#serial-fast-a)采用二进制INT8传输，必须先烧录配套快速HEX。原A文本版与B保留；快速版已完成编译和电脑核对，尚未烧录实测。

<!-- validation-redo-notice-20261010:start -->
**评价状态（2026-10-10）：** 当前验证候选为单教师蒸馏CNN seed43 INT8，折9宏F1为75.893%；现有完整HEX仍为原seed44及原部署阈值。折10在历史研发中已反复使用，且部分追加方案受测试反馈影响，所有已有折10成绩按历史探索结果解释。重做没有恢复折10的独立性；新候选没有本轮独立测试或完整固件。
<!-- validation-redo-notice-20261010:end -->

- [1. 芯片、模型与证据状态](#deployment-status)
- [2. 回放链路、输入与接线](#deployment-chain)
- [3. 已编译seed44学生：资源、核对与文件](#deployment-student)
- [4. 串口协议与结果含义](#deployment-serial)
- [5. 学生包运行与真实板端验证顺序](#deployment-run)
- [6. 防错缺口及已完成验证的边界](#deployment-issues)
- [7. 当前seed43候选的导出状态](#deployment-new)
- [8. HAL接入参考：旧ResNet示例](#deployment-hal)
- [9. 旧ResNet工程、参数与Cube.AI历史对照](#deployment-old)

- [10. 高采样率候选的资源与边界](#highrate-study-20261010)
- [11. 仅推理与板端预处理＋推理：工程、程序及资源对比](#board-preprocess-two-projects)
- [A快速版：二进制INT8、使用顺序及验证证据](#serial-fast-a)

<a id="deployment-status"></a>

## 1. 芯片、模型与证据状态

实板最新记录是**STM32F407VG**；旧三分类参考工程Device为STM32F407VGTx。2026-10-10已另生成并重新编译**VG/1 MiB Flash**五类seed44工程；原**VE/512 KiB Flash**包保留作历史回退。原工程的芯片配置可以查证，但不代替实物、时钟和接线核对；不修改旧包的chip字段来冒充VG重新编译。

| 版本 | 模型/阈值 | 已完成 | 未完成 |
|---|---|---|---|
| 原seed44学生INT8 | 原部署阈值`[0.55,0.60,0.55,0.75,0.65]` | Cube.AI、主机C核对；A/B独立VG工程及发布包；2026-10-11 A通过COM8完成3条不同折10记录及重复回放，返回1012–1013ms，最大概率差约3.006e-08 | B板端预处理、全量回放、长期稳定性及板载时钟/栈峰值核对 |
| A快速版seed44 INT8 | 与原seed44一致 | 独立VG编译/HEX、完整折9/折10量化逐值核对、44条C协议/C模型核对、GUI验证 | 快速HEX烧录、真实UART提速与稳定性 |
| 当前单教师KD seed43 INT8 | 自身折9细阈值约`[0.68,0.45,0.61,0.57,0.78]` | 新训练、FP32/INT8 ONNX、折9核对 | 新Cube.AI资源/C代码/完整固件、实板、独立测试、外部验证 |
| 追加集成派生seed42 | 历史研究细阈值约`[0.49,0.67,0.55,0.67,0.68]` | 历史导出和资源分析 | 已撤回纠正推荐；无完整新固件或已核实外部结果 |
| 旧ResNet INT8 | 自己的清单与阈值 | 历史对照工程及HEX | 未作为当前学生；不能混用参数或启动器 |

公开数据回放不要求人体采集。2026-10-11新增A仅推理的小批真实串口证据，详见文末实板核对；B尚无本轮实板证据。折10历史成绩按探索性解释，不随回放恢复为独立测试。


<!-- unet-deployment-20261010:start -->
本轮1D U-Net研究：已完成12次新训练及30组整体/逐类指标、30组阈值独立复核；仅折9评估。模型579053参数，编码器对照372053参数。仅有训练及折9评估范围；未执行ONNX/INT8导出、Cube.AI、C代码、完整固件或真实板端验证，不能继承旧学生的Flash/SRAM/MACC。见[训练文档](模型训练与优化.md#unet-study-20261010)。
<!-- unet-deployment-20261010:end -->

<a id="deployment-chain"></a>

## 2. 回放链路、输入与接线

A有两个独立版本：原文本版由电脑预处理后发送12×1000个FP32文本数，芯片逐块量化后推理；快速版由电脑完成相同INT8量化，直接发送12000字节，由芯片推理。B由电脑发送原始mV波形，芯片接收完整片段后执行双中值去基线、前后向带通、固定训练标准化、量化及推理。两者均返回五个sigmoid概率与独立阈值分类；界面只显示所选的一条导联，实际发送全部十二导联。

输入顺序为I、II、III、aVR、aVL、aVF、V1、V2、V3、V4、V5、V6，100 Hz、10秒。标准化使用原模型训练统计，不做记录内min-max/Z-score。A零相位预处理在电脑完成，B在STM32完成；两者均为整段处理，尚未实现连续逐点采集算法。

| 开发板 | USB转串口 |
|---|---|
| PA9 / USART1_TX | RX |
| PA10 / USART1_RX | TX |
| GND | GND |

串口为115200、8N1、无流控、3.3 V TTL、共地。旧工程配置8 MHz外部晶振、168 MHz主频，实际板载晶振须核对，不能由芯片型号推断。确认VG目标、Flash、下载器、HSE/PLL与接线后再进行板端验证。


<a id="deployment-student"></a>

## 3. 已编译seed44学生：资源、核对与文件

本节保留2026-10-03实际编译及主机核对记录，工程目标VE，参数只适用于原seed44。以下PTB-XL测试数字对应原部署阈值，是历史探索成绩，不是新候选或实板成绩。

日期：2026-10-03

### 结论

已用 STM32Cube.AI 重新将 seed44 INT8 ONNX 转换为网络 C 代码和权重，再与串口程序完整编译和链接通过：Flash 158876 B（155.15 KiB）、RAM 43672 B（42.65 KiB）。按 STM32F407VE 的 512 KiB Flash、128 KiB 主 SRAM 限制，空间足够。

本轮没有烧录实板或实测板上耗时。已整理独立 HEX 和包含全部库文件的源码 ZIP，可用于烧录和后续联调；实际接线、板载晶振和板上运行仍待核对。

学生检查点确认使用筛选后的训练 / 验证记录 17084 / 2146，`require_diagnostic_label=True`、随机种子 44。检查点、INT8 模型与串口清单的 SHA-256 已逐项一致核对。

冻结阈值的蒸馏 INT8 模型在 PTB-XL 保留的 2158 条测试记录上，五类宏 AUROC 0.924276、宏 F1 0.749441。

### 完整固件资源

| 项目 | 蒸馏 CNN INT8 | 既有 ResNet INT8 |
|---|---:|---:|
| 完整固件 Flash | 158876 B | 506496 B |
| 完整固件 RAM | 43672 B | 62504 B |
| 512 KiB Flash 下余量 | 365412 B（356.85 KiB） | 17792 B |
| 128 KiB SRAM 下余量 | 87400 B（85.35 KiB） | 68568 B |
| 模型权重 | 112276 B | 453012 B |
| 模型激活内存 | 24000 B | 44032 B |
| 模型 MACC | 14698207 | 73639711 |

Flash 采用链接 map 的 Total ROM Size，包含压缩后的初始化数据；RAM 采用 Total RW Size。模型激活包含输入和输出，不能与完整固件 RAM 再相加。编译器 ARMCC 5.06 update 4，0 错误、0 警告。

本轮修正了参考工程的栈配置：原为 1024 B，链接器报告已知最大调用链至少 1504 B，来自串口 `strtof` 浮点解析，且包含无法完整静态估计的回调路径。学生工程现预留 8192 B 栈，RAM 比旧版本增加 7168 B，Flash 不变。工程生成脚本也同步设置该栈容量，避免重新打包时退回旧配置；运行时栈峰值仍需板上验证。

### 转换与协议验证

2026-10-03 重新执行 Cube.AI `generate` 和 `validate --mode host`，电脑 C 实现验证 128 条波形，logits RMSE 0.000000000、最大误差 0.000000000，冻结阈值的五类判断一致率 100%；这批输出也与学生串口样本缓存的参考概率完全一致。该检查证明抽样转换一致性，不能代替全部测试集或实板精度验证。

重新生成的七个网络 C/H 文件与原版本的算子和权重一致，仅生成日期及工具生成的内部图签名变化。先备份原代码，再安装新代码并完整重新编译。

128 条波形通过实际 C 协议解析器的分块传输、CRC 和回复模拟，分类与电脑 C 参考的逐标签一致率 100.00%。模拟推理不是 STM32 板上运行。

原始波形重新预处理与缓存已核对。独立学生工程使用自己的模型哈希、INT8 输入 scale / zero point、训练标准化统计和五个阈值，旧 ResNet 部署包未替换。

本轮串口协议回归测试 12 项全部通过。源码 ZIP 完整性与 Keil 项目引用的源文件、运行库均已检查，编译产物未包含在源码 ZIP 中。

### PTB-XL 逐类测试指标

| 类别 | AUROC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| NORM | 0.946161 | 0.823970 | 0.913811 | 0.866568 |
| MI | 0.919610 | 0.711340 | 0.752727 | 0.731449 |
| STTC | 0.935457 | 0.730104 | 0.809981 | 0.767971 |
| CD | 0.919408 | 0.806236 | 0.729839 | 0.766138 |
| HYP | 0.900746 | 0.640496 | 0.591603 | 0.615079 |

### 独立复核

检查点与导出文件哈希已核对。HEX 每条校验和与全部 Flash 地址范围均核对通过，没有越过 512 KiB 边界。

### 文件与运行

- [学生 Keil 工程](../outputs/distillation/stm32_student_project/MDK_Project/ECG_Student_INT8.uvprojx)。
- [烧录 HEX](../outputs/board_release/seed44_int8_stm32f407ve/ECG_Student_INT8_STM32F407VE.hex)。
- [完整源码工程 ZIP](../outputs/board_release/seed44_int8_stm32f407ve/ECG_Student_INT8_源码工程.zip)：包含启动代码、外设库、Cube.AI 运行库、模型权重与串口代码。
- [烧录与接线说明](../outputs/board_release/seed44_int8_stm32f407ve/使用说明.txt)。
- [本轮资源、校验与文件哈希](../outputs/board_release/seed44_int8_stm32f407ve/release_manifest.json)。
- [完整空间结果](../outputs/distillation/distilled_export/student_firmware_summary.json)。
- [PTB-XL 蒸馏与量化结果](../outputs/distillation/summary.json)。
- [学生串口界面启动器](../start_ecg_student_viz.cmd)。

用 STM32CubeProgrammer 经 ST-LINK / SWD 下载本学生 HEX，或在 Keil 中选择实际下载器后下载；HEX 已包含从 `0x08000000` 起的 Flash 地址。工程采用 **8 MHz 外部晶振、168 MHz 主频**，来自参考工程的实际 `HSE_VALUE` 与 PLL 配置；Keil 界面 `CLOCK(12000000)` 字段不参与固件时钟初始化。若板载晶振不同，须修改 `HSE_VALUE` 和 `PLL_M` 后重新编译，不能仅凭 F407VE 型号认定晶振相同。

烧录后，用学生启动器选择实际串口。USART1：PA9 TX / PA10 RX，115200，3.3 V TTL，和 USB 转串口共地。最新实板记录为VG；该工程为VE配置，实际接线与兼容性尚未验证。模型仍要求十二导联输入；界面显示单条导联，发送全部十二导联。

学生固件输入量化为 `q = clip(round_to_nearest_even(x / 0.2008364498615265), -128, 127)`，五类阈值依次为 `[0.55, 0.60, 0.55, 0.75, 0.65]`；这些参数由学生清单和头文件固定，不要使用下文旧 ResNet 的参数。

实板命令行示例（将 COM7 换成实际端口）：

```powershell
.\start_ecg_student_viz.cmd --headless --port COM7 --sample 0 --result-file outputs/board_release/seed44_int8_stm32f407ve/physical_serial_result.json
```

旧seed44工程的历史复核入口（存在旧输出复用风险；不能用于新seed43固件）：

```powershell
.venv/Scripts/python.exe scripts/verify_student_firmware.py
```

本轮强制转换、全量编译与 HEX 检查的原始证据在 `outputs/distillation/distilled_export/recheck_20261003/`，包括 `generate.log`、`host_validate.log`、`source_comparison.json` 和 `release_audit.json`。原脚本在生成代码和验证结果存在时会复用，不代表每次都强制重新转换。

分类训练与部署候选选择使用 PTB-XL 训练和验证集。其他来源的外部测试见独立报告，不影响本固件空间结论。


<a id="deployment-serial"></a>

## 4. 串口协议与结果含义

| 项目 | 原参考程序 | 本次程序 |
|---|---|---|
| 输入 | 单导联、360 点 | 12 导联、每导联 1000 点，100 Hz、10 秒 |
| 分类 | ECG / Sine / Noise 三选一 | NORM / MI / STTC / CD / HYP 五个独立判断 |
| 数据 | 模拟 ADC 数值，按记录做 min-max | A发送预处理信号；B发送原始mV，由芯片预处理；均按模型参数转INT8 |
| 发送 | 一条长 CSV | 每块最多 64 个浮点数，收到确认才发下一块 |
| 对齐 | 当前界面的样本索引 | 请求序号 + 数据位置 + 完整记录 CRC32 |
| 回复 | 三个概率和一个 argmax 类别 | 五个sigmoid概率、五位分类掩码、推理时间；B另返回预处理时间 |
| 核对 | 默认相信芯片模型 | 模型SHA-256、形状、类别、阈值及协议；B另核对预处理哈希 |

当前任务为五标签分类，多类可以同时为阳性。概率之和不要求等于 1，不能使用 softmax 或只返回最大概率类别。没有标签达到阈值时显示“无标签达到阈值”，不会自动改成 NORM。

### 传输命令与CRC

每条命令以 CRLF 结束。每块收到确认才发送下一块，默认等待确认 5 秒，推理结果等待上限 60 秒；后者是软件超时设置，不是实测推理时长。

| 命令 | 意义 | 回复 |
|---|---|---|
| `$P` | 查询模型、阈值和输入协议 | JSON `hello` |
| `$B,seq,12000` | 开始发送一条记录 | JSON `begin`，带 seq |
| `$D,seq,offset,count,x1,...` | 发送最多 64 个浮点数 | JSON `ack`，带 seq 和下一位置 next |
| `$R,seq,crc32` | 检查完整输入；B先预处理，再执行推理 | JSON `result`，带五类概率、mask、inference_ms；B另含preprocessing_ms |

CRC32 与 Python `zlib.crc32` 一致，计算对象是按导联展开的 12000 个 little-endian float32 的原始字节。ASCII 数值使用 9 位有效数字，确保 float32 往返。芯片检查序号、位置、数量、有限数值及完整记录 CRC；错误时回复 JSON `error`，终止本次记录，下次 `$B` 可重新开始。

分类 mask 的 bit 0 到 bit 4 依次代表 NORM、MI、STTC、CD、HYP。例如 `mask=10` 表示 MI 和 CD 同时达到阈值。回复必须与本次 seq 一致，程序会忽略旧序号回复。


<a id="deployment-run"></a>

## 5. 学生包运行与真实板端验证顺序

A入口为项目根`start_ecg_student_viz.cmd`，B入口为`start_ecg_board_preprocess_viz.cmd`，各自配套原seed44独立VG固件；新seed43尚未接入。旧ResNet入口`start_ecg_stm32_viz.cmd`仅供历史对照，不混用部署清单。

1. 使用本轮新VG工程及HEX；硬件配置沿用原sine/三分类工程，连接对应下载器和USART1。
2. 冻结一个完整包：checkpoint、ONNX、标准化、量化参数、阈值、C代码、工程、HEX及哈希一致；保留旧包回退。
3. 按该包完成真实烧录并连接串口，确认返回真实芯片版本信息。模拟器能连通不登记为实板完成。
4. 发送固定真实波形，逐条保存样本ID及主机/板端概率、标签差异；分别检查数值误差与阈值附近标签变化。
5. 分开统计传输、B预处理和推理耗时，记录测量方法/样本量、连续回放、超时、断线恢复和栈稳定性。

接线和工程目标核对后，可从项目根运行：

```powershell
.\start_ecg_student_viz.cmd
.\start_ecg_student_viz.cmd --headless --port COM7 --sample 0 --result-file outputs/board_release/seed44_int8_stm32f407vg/physical_serial_result.json
.\start_ecg_board_preprocess_viz.cmd
.\start_ecg_board_preprocess_viz.cmd --headless --port COM7 --sample 0 --result-file outputs/board_release/seed44_int8_stm32f407vg_preprocess/physical_serial_result.json
```

COM7替换为实际端口；界面显示“无标签达到阈值”时不自动改为NORM。五类全零标注并不等于正常/起搏器，原始标注和模型预测分别展示。A已完成2026-10-11文末列出的少量回放和返回耗时核对；全量实板指标、长期运行、断线/超时恢复及B工程仍待验证。

结果证据至少包含时间戳、运行模式、样本ID、模型/部署/HEX哈希、导联和输入参数、五类概率/阈值/标签、传输与推理时间。当前单条命令行结果版本字段不足，正式实测前仍需完善。


<a id="deployment-issues"></a>

## 6. 防错缺口及已完成验证的边界

原A握手核对模型哈希、输入形状、类别、阈值及host协议，仍未显式核对input scale、zero point及预处理指纹。新B另核对board/raw_mv协议和冻结C预处理哈希；两种输入混用已在双向拒绝测试中验证。旧A已复现同模型ID下错误scale仍通过；当前发行包参数相符，不能将防错缺口写成已发生错配。两个协议均未直接返回量化scale/zero point，仍须以配套冻结文件及哈希复核为准。

`package_serial_demo.py`已支持显式VG/VE目标，默认VE保留兼容；新A/B生成器显式选择VG。`verify_student_firmware.py`仍固定旧seed44路径，并会按文件存在复用生成/host验证产物。后续换模型须绑定源哈希、工具版本、配置和生成文件，不能直接把该脚本当作新seed43固件入口。下面验证表保留旧ResNet历史记录，新A/B检查见[两种工程记录](#board-preprocess-two-projects)。

| 验证 | 结果 |
|---|---|
| 原始 WFDB 波形重新预处理，对齐发送缓存 | 128 条通过 |
| 界面初始化、导联绘图 | 通过 |
| 真正的 C 协议解析器 + ONNX 推理模拟 | 128 条完整记录，确认和 CRC 全通过 |
| 模拟与 Cube.AI 电脑 C 模型五类判定 | 100% 一致，最大概率差约 0.001887 |
| 新增数据对齐和串口协议测试 | 13 项通过 |
| 项目完整测试目录 | 71 项通过 |
| 标准外设库固件编译与 512 KiB 链接限制 | 0 错误、0 警告，生成 HEX |
| 真实开发板串口与推理耗时 | 未验证 |

证据文件：`outputs/cubeai/serial_simulation_validation.json`、`outputs/cubeai/keil_build.log`、`outputs/cubeai_tests.log` 和 `outputs/cubeai/resnet/host_validation_summary.json`。

原始参考程序本身的三类权重和输入长度不能直接运行本程序；需要本次生成的五类固件。当前没有已确认连接的开发板，因此没有把枚举出的蓝牙串口当作硬件测试结果，也没有将模拟模式的 `inference_ms=0` 作为芯片测速。

### 已存最新复审证据

以下表格保留早期复审结果，涉及旧VE固件。2026-10-10已按原参考配置另建并完整编译A/B两个VG工程，原VE工程及发布清单保持不变。最新配套入口与资源见[两种工程记录](#board-preprocess-two-projects)。

| 项目 | 当前事实 |
|---|---|
| 原 seed44 发布包 | Flash 158876 B，主 SRAM 43672 B，含 8192 B 栈；发布 HEX/ZIP 哈希复核相符 |
| 当前固件阈值 | [0.55,0.60,0.55,0.75,0.65]，原部署阈值；对应旧 F1=74.944% |
| 原 seed44 0.01 研究阈值 | [0.55,0.55,0.53,0.72,0.63]，F1=75.097%；未同步现有 HEX |
| 新 seed42 0.01 研究阈值 | [0.49,0.67,0.55,0.67,0.68]，F1=75.256%；无完整新固件 |
| 当前输入量化 | scale=0.2008364498615265，zero point=0，串口清单与发布清单一致 |
| 真实烧录/串口/耗时/稳定性 | 尚无已核实完成证据，`physical_board_test=false` |

早期复审时`package_serial_demo.py`硬编码VE；现已支持显式VG/VE，新生成器明确选择VG。`verify_student_firmware.py`固定旧模型路径、按文件存在复用旧生成/验证产物，不能直接作为新seed42/43的完整固件入口。新B发布包已记录模型、预处理配置/源码、验证及编译产物哈希，但未接入研究候选seed43。

旧A握手未返回/核对input scale、zero point和预处理指纹，早期复审模拟中保留相同模型哈希和阈值、将scale改为两倍仍通过；这是防错缺口，不代表已发生错配。新B加入预处理指纹及协议区分，直接回传量化参数仍未实现。实测保存模型/部署合同/固件哈希、样本ID、时间戳、运行模式以及传输、预处理和推理时间，单条命令行结果的完整版本字段仍待完善。

实板顺序：核对丝印、Flash、晶振和接线 → 独立 VG 工程配置/编译 → 同模型真实回放对照 → 计时、连续回放、超时/断线恢复和栈稳定性。只有完成对应测试才更新实板验证状态；不把主机 C 验证或模拟结果登记成实板完成。全过程问题与证据见 [训练与优化](模型训练与优化.md) 本次复审。


<a id="deployment-new"></a>

## 7. 当前seed43候选的导出状态

<!-- validation-redo-results-20261010:start -->
### 纠正后候选的部署状态（2026-10-10）

仅按fold9 INT8比较推荐 `single_teacher_kd_seed43_int8`；对应模型文件 `D:\MyProject\college\ecg1\outputs\validation_only_redo_20261010\selected_export\ptbxl_cnn1d_int8.onnx`，SHA-256 `f244cf3769d4d3171913d85195ef711f899a78c0a64cdfef078ac2bd94b58179`；候选阈值 `[0.68, 0.45000000000000007, 0.61, 0.57, 0.78]`。这是验证选择，尚无本轮独立测试结果。
本轮重新训练并导出的新学生为 `single_teacher_kd_seed43`，产物位于 `outputs/validation_only_redo_20261010/selected_export/`，必须使用本目录自己的标准化、量化参数和候选阈值。
该候选只有FP32/INT8 ONNX及验证核对；未进行新的Cube.AI资源分析、C代码生成、完整固件编译、烧录或实板串口计时。不能沿用旧模型的板端结果作为新模型证据。原seed44固件、原部署阈值和所有可回退文件未修改；芯片VG/VE配置差异仍须在生成新工程前处理。
<!-- validation-redo-results-20261010:end -->


<a id="deployment-hal"></a>

## 8. HAL接入参考：旧ResNet示例

以下路径及适配示例来自旧ResNet，不是当前seed44或新seed43的自动替换入口。完整编译证据属于标准外设库版本，HAL完整工程尚未编译/实板验证。

工作区提供以下文件：

- [ecg_protocol.c](../firmware/ecg_serial/ecg_protocol.c) / [ecg_protocol.h](../firmware/ecg_serial/ecg_protocol.h)：分块接收、INT8 输入转换、CRC32、推理回复。
- [ecg_uart.c](../firmware/ecg_serial/ecg_uart.c) / [ecg_uart.h](../firmware/ecg_serial/ecg_uart.h)：STM32F4 HAL 适配。
- [ecg_deployment_config.h](../firmware/ecg_serial/ecg_deployment_config.h)：该旧ResNet路径的模型哈希、输入量化参数和五类阈值。
- [main_spl.c](../firmware/ecg_serial/main_spl.c)：参考标准外设库工程的完整主程序。
- `outputs/cubeai/resnet/generated/`：Cube.AI 网络源文件和权重。

HAL 工程中加入网络代码、协议和 HAL 适配文件，以及同版本 Cortex-M4 Cube.AI 库和头文件。USART1 初始化后调用：

```c
/* main.c 的 USER CODE 区域 */
#include "ecg_uart.h"

/* MX_USART1_UART_Init() 之后 */
ECG_UART_Init(&huart1);

/* while (1) 内 */
ECG_UART_Poll();
```

HAL 版本使用主循环轮询接收，不应同时在同一串口启动另一个接收服务。推理也在主循环执行，不在串口中断中运行。HAL 版本尚未在完整 HAL 工程中编译或实板验证；已完整编译的是标准外设库版本。


<a id="deployment-old"></a>

## 9. 旧ResNet工程、参数与Cube.AI历史对照

本章全部为旧ResNet历史产物。其量化scale/zero point/阈值与学生包不同，不作为学生启动步骤；日期后的“当前”仅指原轮次。

### 工程与编译

可直接用 Keil 打开 [ECG_ResNet_INT8.uvprojx](../outputs/cubeai/stm32_spl_project/MDK_Project/ECG_ResNet_INT8.uvprojx)。已生成 [ECG_ResNet_INT8.hex](../outputs/cubeai/stm32_spl_project/MDK_Project/Objects/ECG_ResNet_INT8.hex)。

工程沿用参考程序的启动代码、时钟初始化、延时和 USART1 驱动，替换模型与主程序，并将串口中断共享状态改为 `volatile`。使用 ARM Compiler 5.06 update 4，Cube.AI 运行库 `NetworkRuntime1020_CM4_Keil.lib`。

本次编译明确按 STM32F407VE 的 512 KiB Flash 限制配置；参考工程原为 VG。烧录前核对实板型号和 Keil 下载器设置。代码和模型已生成并通过编译，本次未进行下载或烧录。

参考工程 USART1 接线：

| 开发板 | USB 转串口 |
|---|---|
| PA9 / USART1_TX | RX |
| PA10 / USART1_RX | TX |
| GND | GND |

使用与开发板兼容的 3.3 V TTL 电平，参数为 `115200 / 8N1 / 无流控`；开发板正常供电。界面的波特率必须与固件一致，标准外设库版本在 `main_spl.c` 的 `uart_init(115200)` 中设置。

完整固件链接结果为 Flash 506496 B（494.63 KiB）、RAM 62504 B（61.04 KiB），0 错误、0 警告。512 KiB Flash 剩余约 17.38 KiB；新增模块后需重新检查资源占用。RAM 中包含 44032 B 模型激活区，串口接收缓冲和程序状态已计入完整固件。

标准外设库工程的 SysTick 中断每毫秒更新计时，用于接收超时和推理时间。新增功能不能再用原 `delay_ms/delay_us` 改写同一个 SysTick，应改用独立定时器或兼容的计时方式。

### 输入及内存处理

电脑端按检查点的配置进行基线处理、0.5–40 Hz 四阶零相位带通和训练集全局标准化，保留原始幅值的统计含义。标准化均值和标准差来自训练集，并保存于部署清单。不能再按每条波形做 min-max 或记录内 z-score。

发送布局为逐导联展开：先发送 I 导联的 1000 点，再发送 II 导联，直到 V6。导联顺序为 `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`。

串口传输为预处理后的 FP32 数值的 ASCII 表示；芯片收到一个小块就直接量化写入网络输入区，不另外保留一整条 48000 字节浮点输入。

旧ResNet量化公式（不能用于seed44或seed43）：

```text
q = clip(round_to_nearest_even(x / 0.1990550309419632) + 5, -128, 127)
```

Cube.AI 生成版本必须使用 `--inputs-ch-position chfirst --output-data-type float32 --c-api legacy`。输入为 INT8，输出为 FP32 logits。芯片对每个 logits 单独做 sigmoid，再按 `[0.60, 0.65, 0.55, 0.75, 0.70]` 判定。

本节旧ResNet程序验证“电脑预处理 + 芯片神经网络推理”的链路。新seed44 B工程已实现板端整段预处理，见[两种工程记录](#board-preprocess-two-projects)；课程范围仍是公开数据回放，没有接入采样ADC或12导联模拟前端。

### 旧Cube.AI对照分析

以下是旧模型的工具分析与工程路径，保留作资源比较；不代表当前学生资源。

### 分析环境与芯片依据

- 安装包目录：X-CUBE-AI 10.2.1；实际命令行引擎自报 `ST Edge AI Core v2.2.0-20266 2adc00962`、`STM32CubeAI 10.2.0-RC1`。
- 使用本机 `stedgeai` 工具，属于 Cube.AI 的分析、生成和验证工具；分析目标 `stm32f4`，内存优化 `ram`，权重压缩 `none`。
- 当前 `D:/MyProject/MyProject.ioc` 设置为 STM32F407VET6：512 KiB Flash，128 KiB 主 SRAM + 64 KiB CCM。
- 用户提供的串口参考工程设置为 STM32F407VG：1 MiB Flash。新工程按更小的 VE 预算配置；2026-10-03 用户已确认实板型号为 STM32F407VE。
- Cube.AI CLI 的普通分析不会自动按某个具体芯片的容量拒绝模型，因此另外进行了 512 KiB Flash、128 KiB 主 SRAM 的 Keil 链接检查。

芯片规格参考：[ST STM32F407VE](https://www.st.com/en/microcontrollers-microprocessors/stm32f407ve.html)。工具说明参考：[ST 命令行文档](https://stedgeai-dc.st.com/assets/embedded-docs/2.0.0/stm32_command_line_interface.html)。

### Cube.AI 资源结果

| 模型 | 权重 / B | 激活内存 / B | MACC | 部署判断 |
|---|---:|---:|---:|---|
| 优化 Inception FP32 | 1,071,668 | 624,096 | 80,000,021 | 权重和 RAM 均超限 |
| 优化 Inception INT8 | 272,564 | 288,000 | 80,504,405 | RAM 超限 |
| 标准化 ResNet INT8（部署版） | 453,012 | 44,032 | 73,639,711 | 完整固件已链接通过 |

以上激活内存已包含模型输入、输出，不能再重复加一次输入缓冲。权重占用不等于固件总 Flash；总 Flash 还包括推理库、网络代码、串口程序及初始化数据。

部署版显式指定 `--inputs-ch-position chfirst --output-data-type float32 --c-api legacy`。输入为 INT8 `[12,1000]`，输出为五个 FP32 logits；输出转换使 MACC 比纯 INT8 输出版多 10。

### 量化精度：同一验证集整体对比

以下均为 **官方验证折 9，共 2183 条记录**。校准仅使用训练折 1–8 的 256 条记录；没有用验证标签选择量化方法，没有重新调整阈值，没有读取测试折 10。
使用 ONNX Runtime 1.30.0 的静态 QDQ 量化，权重有符号对称、按通道，激活有符号非对称、按张量，MinMax 校准。

| 模型 | Macro AUROC | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Inception FP32 | 0.929047 | 0.711349 | 0.807543 | 0.754900 |
| Inception INT8 | 0.928731 | 0.699917 | 0.811278 | 0.748792 |
| ResNet FP32 | 0.927554 | 0.720955 | 0.786018 | 0.749293 |
| ResNet INT8（选用） | 0.927532 | 0.719735 | 0.782779 | 0.746859 |

ResNet 宏 F1 降低 **0.243 个百分点**，五类二值判定一致率 **99.432%**。该一致率是对 2183×5 个判断计算，不是对真实标签的准确率。

阈值保持为 `[0.60, 0.65, 0.55, 0.75, 0.70]`，来自该 ResNet 检查点的验证阈值。这组阈值与电脑端集成或优化版 Inception 的阈值不同，不能混用。

### 选用模型逐类指标

| 类别 | 版本 | 阳性数 | AUROC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| NORM | FLOAT32 | 955 | 0.947026 | 0.839286 | 0.885864 | 0.861946 |
| NORM | INT8 | 955 | 0.946989 | 0.843531 | 0.880628 | 0.861680 |
| MI | FLOAT32 | 540 | 0.922062 | 0.791667 | 0.703704 | 0.745098 |
| MI | INT8 | 540 | 0.921901 | 0.790254 | 0.690741 | 0.737154 |
| STTC | FLOAT32 | 528 | 0.932886 | 0.707593 | 0.829545 | 0.763731 |
| STTC | INT8 | 528 | 0.932968 | 0.702060 | 0.839015 | 0.764452 |
| CD | FLOAT32 | 495 | 0.927877 | 0.721818 | 0.802020 | 0.759809 |
| CD | INT8 | 495 | 0.928042 | 0.717902 | 0.802020 | 0.757634 |
| HYP | FLOAT32 | 268 | 0.907919 | 0.544413 | 0.708955 | 0.615883 |
| HYP | INT8 | 268 | 0.907761 | 0.544928 | 0.701493 | 0.613377 |

HYP 仍是最弱类别，INT8 F1 为 0.613377；MI 召回率为 0.690741，后续仍需改进。此次工作验证的是已有模型的芯片转换和资源可用性，不代表解决了全部类别短板。

### 转换与程序验证

- PyTorch → FP32 ONNX：ResNet 在 128 条验证记录上最大 logits 误差 0.000003815，通过容差检查。
- INT8 ONNX → Cube.AI 电脑端 C 实现：128 条记录，logits RMSE 0.004305452、最大 logits 差 0.077018261，最大概率差 0.001886985，五类判定一致率 100%。
- 最大 logits 差约为一个输出量化步长，属于已观察到的转换差异，不能称为浮点数完全一致。
- 128 条记录从原始 WFDB 波形重新预处理，与发送数据缓存一致；串口 C 协议的分块确认和 CRC 全部通过模拟验证。
- Keil ARM Compiler 5.06 update 4：0 错误、0 警告，生成 HEX；工程链接区域明确限制 Flash 为 0x80000。
- 项目测试：71 项通过，其中新增导出抽样和串口协议测试覆盖数据对齐、错误 CRC、缺包、错序、无效浮点数、旧响应和多标签返回。

### 文件与复现

- 选用模型：[ptbxl_resnet1d_int8.onnx](../outputs/cubeai/resnet/ptbxl_resnet1d_int8.onnx)。
- 源检查点：`outputs/optimization/01_global_norm/checkpoint/resnet1d_best.pt`。
- 生成代码与原始报告：[generated](../outputs/cubeai/resnet/generated/ecg_network_generate_report.txt)。
- 完整资源及精度明细：[deployment_summary.json](../outputs/cubeai/deployment_summary.json)。
- 可编译串口工程：[ECG_ResNet_INT8.uvprojx](../outputs/cubeai/stm32_spl_project/MDK_Project/ECG_ResNet_INT8.uvprojx)。
- HEX：[ECG_ResNet_INT8.hex](../outputs/cubeai/stm32_spl_project/MDK_Project/Objects/ECG_ResNet_INT8.hex)。
- 串口使用与接入：[串口波形分类验证程序总结](STM32部署与串口验证.md)。

INT8 ONNX SHA-256：`73e6a0a740e6f5af059ed933add9513e6403264418e1c55d9b4c2cedf5a9a1a7`。

在项目根目录导出和量化：

```powershell
.venv/Scripts/python.exe scripts/prepare_stm32.py --checkpoint outputs/optimization/01_global_norm/checkpoint/resnet1d_best.pt --output-dir outputs/cubeai/resnet
.venv/Scripts/python.exe scripts/quantize_cubeai.py --export-dir outputs/cubeai/resnet
```

Cube.AI 的完整实际命令保存在各原始报告的 `Parameters` 字段。先用 `validate --mode host` 生成验证输出，再用 `generate` 生成代码，随后运行 `package_serial_demo.py` 准备串口样本和工程。更换模型后需要重新生成所有对应文件和固件。

### 尚未完成的实板验证

该轮文档曾记录实板为STM32F407VE；最新更正为VG，旧工程仍为VE，尚未进行烧录、真实串口波形收发和芯片耗时测量。旧轮次仅枚举到蓝牙串口，不能据此认定已连接开发板。后续接上 USB 转串口及开发板，按本文件当前学生固件说明运行程序即可验证；电脑端的模拟时间不能当作 STM32 推理时间。

此次 ResNet INT8 只报告验证折 9 的量化对比和已有交叉验证结论，没有新增测试折 10 结果。历史模型的完整整体和逐类测试结果见 [各模型整体与逐类指标总表](各模型整体与逐类指标总表.md)。

### 旧界面运行记录

项目目录：`D:/MyProject/college/ecg1`。本次依赖已安装在项目虚拟环境。

旧 ResNet 包可双击项目根目录 [start_ecg_stm32_viz.cmd](../start_ecg_stm32_viz.cmd)，打开界面。也可以运行：

```powershell
.venv/Scripts/python.exe scripts/ecg_stm32_viz.py
```

界面操作：

1. 选择“真实串口”、开发板对应的 COM 端口和 `115200` 波特率，点击“连接”。
2. 程序先检查芯片固件是否使用同一个模型和阈值；不一致会停止发送。
3. 选择样本和显示导联，点击“发送波形并分类”。切换显示导联不会改变发送的 12 导联数据。
4. 传输完成后，查看每类真实标签、电脑 C 模型参考概率、芯片概率、阈值及芯片判断。
5. 点击“保存验证记录”，结果保存在 `outputs/cubeai/serial_validation_log.json`，包含模型哈希、记录编号、真实标签、概率、参考值、分类结果、工作模式和耗时。

无开发板时，可在界面选择“模拟验证”，或运行：

```powershell
.venv/Scripts/python.exe scripts/ecg_stm32_viz.py --simulate
.venv/Scripts/python.exe scripts/ecg_stm32_viz.py --headless --simulate --sample 0 --result-file outputs/cubeai/serial_simulation_result.json
```

模拟模式运行实际编译的 C 协议解析器，推理由 ONNX Runtime 完成，用于检查收发流程。模拟结果和模拟耗时均不代表实板结果。第一次使用模拟模式需要本机 GCC 编译协议库；真实串口模式不需要 GCC。

命令行实板验证示例，`COM7` 要替换为开发板实际端口：

```powershell
.venv/Scripts/python.exe scripts/ecg_stm32_viz.py --headless --port COM7 --baudrate 115200 --sample 0 --result-file outputs/cubeai/physical_serial_result.json
```

默认数据为官方验证折 9 的 128 条真实 ECG。数据文件是 `outputs/cubeai/resnet/serial_samples.npz`，包含原始 mV 波形、预处理输入、真实五标签和电脑 C 实现参考概率。它不是原三分类工程的 `X_test.npy`，也没有使用官方测试折 10。

#### 为什么有“未标注这五类诊断标签”的记录

五类标签只取原始 SCP 标注中的诊断大类，节律或其他标注不会自动归入这五类。例如样本索引 5、ECG 记录 190 的原始标注为 `PACE: 100`，标注表将其列为起搏器相关节律，诊断大类为空，因此五列为 `[0,0,0,0,0]`。这不表示该记录没有任何原始标注，也不表示正常 NORM。

旧 ResNet 验证包保留了五类映射后全 0 的记录：训练集 334 / 17418、验证集 37 / 2183、测试集 40 / 2198；当前 128 条串口验证波形中有 2 条。界面对此显示“未标注这五类诊断标签（不代表正常）”，逐类真实标签栏显示“未标注”。

已有训练和指标计算将这些记录按五类全 0 目标处理。如果后续改为仅使用至少有一个五类诊断标签的记录，需要重新训练、评估并注明新的样本数量，不能直接沿用已有指标。真实标签缺少五类诊断标注与模型“没有类别达到阈值”是两件不同的事。

### 当轮联调计划

开发板型号已确认 VE，检查板载晶振与下载器设置后烧录当前学生 HEX，连接 USART1。先在学生界面确认握手成功，再逐条发送验证波形，保存真实模式日志，核对芯片与电脑参考的五类概率差，并记录芯片推理时间。如果握手报模型不一致，应重新生成相同模型的部署清单和固件；如果接收超时，先检查 COM 端口、TX/RX、GND 和波特率。

旧 ResNet Flash 余量较小；当前学生固件余量约 356.85 KiB，加入功能后仍应重新编译检查。不应直接把模型时间长度或导联数改短后继续沿用旧性能数字。

<!-- highrate-study-20261010:start -->
<a id="highrate-study-20261010"></a>

## 10. 高采样率候选：资源、量化与串口边界

STM32F407VG：1 MiB Flash、128 KiB主SRAM+64 KiB CCM；旧VE工程Flash为512 KiB。CCM独立寻址，不能把192 KiB当成单一主SRAM区，DMA亦不能直接访问CCM。
以下权重、激活、MACC来自本轮实际Cube.AI分析；完整固件估算暂借旧学生额外Flash 46600 B、额外主SRAM 19672 B，未做新固件链接，不代表板端验证。模型激活中的输入/输出不能再次重复相加。
输入12导联×10秒，250/500 Hz的INT8输入分别30000/60000 B，FP32分别120000/240000 B。现有协议电脑预处理后分块发送FP32并在板端量化；板上不需要保留整片FP32缓存。

|网络|Hz|训练队列|版本|权重B|激活B|MACC|估算固件FlashB|估算主SRAM B|主SRAM可容纳|
|---|---:|---|---|---:|---:|---:|---:|---:|---|
|cnn1d|250|filtered|float32|444820|240000|36912957|491420|259672|False|
|cnn1d|250|filtered|int8|112276|60000|36773063|158876|79672|True|
|cnn1d|500|filtered|float32|444820|480000|73866133|491420|499672|False|
|cnn1d|500|filtered|int8|112276|120000|73586207|158876|139672|False|
|cnn1d|250|unfiltered|float32|444820|240000|36912957|491420|259672|False|
|cnn1d|250|unfiltered|int8|112276|60000|36773063|158876|79672|True|
|cnn1d|500|unfiltered|float32|444820|480000|73866133|491420|499672|False|
|cnn1d|500|unfiltered|int8|112276|120000|73586207|158876|139672|False|
|inception1d|250|filtered|float32|1071668|1560096|199952957|1118268|1579768|False|
|inception1d|250|filtered|int8|272564|720000|201212967|319164|739672|False|
|inception1d|500|filtered|float32|1071668|3120096|399992021|1118268|3139768|False|
|inception1d|500|filtered|int8|272564|1440000|402512415|319164|1459672|False|
|inception_time1d|250|filtered|float32|1968788|4796160|1236586861|2015388|4815832|False|
|inception_time1d|250|filtered|int8|498708|2155488|1242347383|545308|2175160|False|

115200/8N1只计数据的FP32发送下限：100/250/500 Hz为4.17/10.42/20.83秒，实际ASCII浮点协议、ACK与CRC更慢。改为二进制INT8传输后的下限为1.04/2.60/5.21秒，尚未实现该协议。MACC不能直接换算真实推理时延。
当前HEX、阈值、UART固定样本数仍是100 Hz；新模型要改协议输入长度、量化参数、模型C代码、工程目标与完整链接，再做真实烧录/一致性/耗时/栈峰值验证。
<!-- highrate-study-supplement:start -->

#### F407的250/500 Hz内存结论与输入协议

250 Hz CNN INT8激活60000 B，借旧额外开销估算主SRAM为79672 B，主SRAM余量51400 B；500 Hz激活120000 B，估算139672 B，比128 KiB多8600 B。500 Hz激活自身能放进主SRAM，但旧栈/串口等一起放主SRAM会超限；不能简单相加192 KiB当作一块连续内存。

旧栈8192 B与CPU专用`line[2048]`缓冲合计10240 B，理论上移到CCM可覆盖8600 B差额，仅留约1640 B额外主SRAM余量，仍很紧。当前串口使用接收中断缓冲与发送轮询，未用DMA；若后续改DMA，DMA缓冲必须留主SRAM。此处仅为布局可行性分析，尚未修改散装/链接配置或重新编译500 Hz固件。不能承诺完整工程/栈峰值够用。

两种Inception网络按各自实际Cube.AI激活与权重判断；模型需要连续大激活区时不能靠64 KiB CCM直接扩展主SRAM。教师优先留电脑端，高采样率上板优先INT8轻量CNN。FP32 CNN两种频率仅激活便超过全部192 KiB，不能直接用片内RAM运行。

现有电脑发送端`src/inference/serial_link.py`强制`[12,1000]`，固件`ECG_SAMPLE_COUNT=12000`且hello返回`[12,1000]`，双方必须同步修改，再更新量化参数、模型哈希、阈值、输入长度和CRC协议验证。原100 Hz模型/HEX未替换。

芯片资料：[STM32F407VG官方页](https://www.st.com/en/microcontrollers-microprocessors/stm32f407vg.html)、[F407参考手册RM0090](https://www.st.com/resource/en/reference_manual/dm00031020.pdf)。采样源：[PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)；抗混叠：[SciPy resample_poly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)。

#### 同结构100/250/500 Hz资源增长

100 Hz列来自已有Cube.AI报告（CNN为旧seed44学生，Inception为历史同结构）；它们用于架构资源对照，与本轮100 Hz成绩参考检查点不是同一份权重。量化设置/输出类型略有差别，因此MACC比值按实际工具报告解释。

|网络|Hz|INT8权重B|激活B|相对100 Hz激活倍数|MACC|相对100 Hz MACC倍数|
|---|---:|---:|---:|---:|---:|---:|
|cnn1d|100|112276|24000|1.00|14698207|1.00|
|cnn1d|250|112276|60000|2.50|36773063|2.50|
|cnn1d|500|112276|120000|5.00|73586207|5.01|
|inception1d|100|272564|288000|1.00|80504405|1.00|
|inception1d|250|272564|720000|2.50|201212967|2.50|
|inception1d|500|272564|1440000|5.00|402512415|5.00|

本轮资源证据支持：保持当前网络与10秒输入长度时，500 Hz相比100 Hz使激活约增至5倍、计算约增至5倍，权重大小基本不随输入长度增加。500 Hz不适合作为现有F407工程的默认输入方案；这不是“任何500 Hz网络都无法部署”的普遍证明。紧凑Inception即使100 Hz也已超过片内RAM，不能把它的部署失败全部归因于500 Hz。

#### 量化成绩与资源证据分开解释

指标总表中的新模型成绩均为FP32推理。资源实测包括CNN两训练队列×两采样率的4组，以及筛选教师的3组；教师未筛选版本只完成训练与FP32评价。同结构教师的资源按已测筛选图作架构筛查，未对其未筛选版本另做量化精度验证。CNN使用128条对应训练队列样本校准，并在完整折9复核INT8成绩，复用原FP32阈值而不重搜。

|训练队列|Hz|原FP32宏F1/%|INT8宏F1/%|下降/百分点|
|---|---:|---:|---:|---:|
|filtered|250|75.430|75.015|0.416|
|filtered|500|75.210|75.007|0.203|
|unfiltered|250|74.960|74.764|0.196|
|unfiltered|500|74.874|74.608|0.266|

Inception教师的INT8图仅用32条训练样本作资源筛查，尚无完整折9量化成绩。8条校准训练样本上的可执行性核对如下，数值误差不代表验证集F1。

|模型/Hz|最大概率差|平均概率差|完整INT8验证F1|
|---|---:|---:|---|
|filtered_inception1d_250hz|0.091154|0.013363|未测|
|filtered_inception1d_500hz|0.138810|0.025911|未测|
|filtered_inception_time1d_250hz|0.665142|0.098398|未测|

250 Hz InceptionTime资源版INT8图最大概率差达到0.665，数值误差较大，不作为可用部署候选；其资源报告只描述这份实际导出图，不能借用FP32成绩声称INT8精度已保留。教师若后续研究部署，须重新处理量化校准/算子与完整验证。CNN的完整INT8验证证据不受此限制。

- 当前ASCII格式按8条真实波形估算，filtered/250 Hz仅数据包线速时间中位数33.26秒（不含ACK、接收解析、推理）；这不是实测延迟。

- 当前ASCII格式按8条真实波形估算，filtered/500 Hz仅数据包线速时间中位数66.60秒（不含ACK、接收解析、推理）；这不是实测延迟。

- 当前ASCII格式按8条真实波形估算，unfiltered/250 Hz仅数据包线速时间中位数33.27秒（不含ACK、接收解析、推理）；这不是实测延迟。

- 当前ASCII格式按8条真实波形估算，unfiltered/500 Hz仅数据包线速时间中位数66.62秒（不含ACK、接收解析、推理）；这不是实测延迟。

<!-- highrate-study-supplement:end -->
<!-- highrate-study-20261010:end -->

<!-- seed44-vg-reference-20261010:start -->
## 2026-10-10：沿用原 sine / ECG 三分类配置的五类 VG 工程

按用户要求，将原三分类工程的 VG 芯片、Flash、RAM、时钟及 J-Link 配置用于原 seed44 INT8 五类串口工程。已与 sine 工程交叉核对 Device 与时钟源码；原参考工程和旧 VE 回退包均未修改。

| 配置 | 本轮实际工程 |
|---|---|
| 芯片 / Flash | STM32F407VGTx / 1 MiB，STM32F4xx_1024 下载算法 |
| 固件时钟 | HSE 8 MHz，PLL M=8/N=336/P=2/Q=7，主频168 MHz；时钟源码原样复制 |
| 串口 | USART1，PA9 TX / PA10 RX，115200、8N1、无流控 |
| 调试下载 | 沿用 Project.uvoptx 的 Segger/J-Link 与注册参数 |
| Keil CLOCK 字段 | 保留参考工程12000000；它不改变固件实际HSE配置 |
| 工具 / 栈 | 本机 ARMCC 5.06 update4；8 KiB栈，满足五类协议与运行库需要 |

原参考工程写 ARMCC update7，本机沿用已有 update4；不是逐字复制不可用编译器。五类工程继续使用原 seed44 权重、预处理、量化参数和原部署阈值，不是当前研究候选 seed43。

本轮执行全量重新编译，**0错误、0警告**；Flash 158876 B（155.15 KiB），主SRAM 43672 B（42.65 KiB，含栈）。15项芯片配置/串口协议测试通过；HEX校验和、VG地址范围和向量表验证通过，源码ZIP完整性通过。

- [新VG Keil工程](../outputs/distillation/stm32_student_project_vg/MDK_Project/ECG_Student_INT8_VG.uvprojx)
- [新VG HEX](../outputs/board_release/seed44_int8_stm32f407vg/ECG_Student_INT8_STM32F407VG.hex)
- [完整VG源码ZIP](../outputs/board_release/seed44_int8_stm32f407vg/ECG_Student_INT8_VG_源码工程.zip)
- [本轮发布清单及保护哈希](../outputs/board_release/seed44_int8_stm32f407vg/release_manifest.json)

使用项目根 `start_ecg_student_viz.cmd`，它与本包仍是同一个原seed44模型/阈值。本轮只生成和编译，没有自动下载到开发板，也没有板上推理计时。

生成入口：`.venv/Scripts/python.exe scripts/build_seed44_vg.py`，已有包时拒绝覆盖；`package_serial_demo.py` 新增 `--chip STM32F407VG`，原VE入口保留兼容，后续应显式指定芯片。
<!-- seed44-vg-reference-20261010:end -->

<!-- board-preprocessing-two-projects-20261010:start -->
<a id="board-preprocess-two-projects"></a>

## 2026-10-10：两种独立VG工程——仅推理与板端预处理＋推理

按用户要求保留原仅推理VG工程，并新增从原始mV波形开始的完整板端处理工程。两个工程使用同一原seed44 INT8模型及原部署阈值，均沿用原三分类的VG、8 MHz HSE、168 MHz主频、USART1 PA9/PA10和J-Link配置；新seed43研究候选未接入本轮固件。

| 项目 | A：仅推理 | B：板端预处理＋推理 |
|---|---|---|
| 上位机发送 | 预处理后的浮点波形 | 原始mV浮点波形 |
| STM32执行 | 量化→模型→分类 | 双中值去基线→前后向带通→固定训练标准化→量化→模型→分类 |
| 输入形状 | 十二导联×1000点，100 Hz | 相同；先接收完整10秒片段 |
| 握手 | protocol=1，preprocessing=host | protocol=2，preprocessing=board，raw_mv，预处理哈希 |
| MDK工程 | `outputs/distillation/stm32_student_project_vg/MDK_Project/ECG_Student_INT8_VG.uvprojx` | `outputs/distillation/stm32_student_raw_vg_project/MDK_Project/ECG_Student_RAW_INT8_VG.uvprojx` |
| 直接运行Python | `scripts/ecg_stm32_viz.py` | `scripts/ecg_stm32_raw_viz.py` |
| 双击启动 | `start_ecg_student_viz.cmd` | `start_ecg_board_preprocess_viz.cmd` |
| 清单 | `outputs/distillation/distilled_export/serial_manifest.json` | `outputs/distillation/board_preprocess_export/serial_manifest.json` |
| 发布包 | `outputs/board_release/seed44_int8_stm32f407vg/` | `outputs/board_release/seed44_int8_stm32f407vg_preprocess/` |
| 完整Flash | 158876 B / 155.15 KiB | 162420 B / 158.61 KiB |
| 完整主SRAM，含8 KiB栈 | 43672 B / 42.65 KiB | 108368 B / 105.83 KiB |

### 本轮预处理实现及一致性

C实现的双中值窗口为21/61点，边界使用reflect；带通为4阶0.5–40 Hz Butterworth、四组SOS、前后向计算、odd延拓27点。滤波系数和初始状态冻结自SciPy；使用原训练集逐导联均值/标准差，没有在回放数据上重新拟合。保留Python各阶段的float32舍入，SOS与标准化中间计算用double。

原始波形缓冲48000 B；复用的单导联预处理工作区16680 B。两者均已计入链接map的完整RAM，不能重复相加。B工程仍是10秒整段处理，尚未改为连续逐点流式算法。M4的double运算性能需要板上计时。

27项预处理/原新串口/芯片配置测试通过。真实验证折9的128条ECG用实际C源码在电脑复核：相对Python原始float32输入的最大误差0.0，相对原训练流程缓存最大误差0.0；INT8差异值均为0，串口接收量化输入差异0。分类与原电脑C参考一致率100.00%，最大概率差5.96046448e-08。

这批检查是GCC主机数值及协议模拟，不是ARMCC实板数值、模型新测试集成绩或板上耗时。完整新固件已重新编译0错误/0警告，HEX校验和、VG地址范围和向量表通过。真实板端一致性、预处理耗时、推理耗时及栈峰值仍待连接开发板验证。

### 运行与防止混用

A烧录仅推理HEX，运行`ecg_stm32_viz.py`；B烧录RAW预处理HEX，运行`ecg_stm32_raw_viz.py`。各自界面标题及发送说明明确显示工程类型。二者均可在VS Code已选项目`.venv`后直接点击运行，或双击各自CMD。选择真实串口、实际COM和115200；CPU时钟及J-Link设置沿用原工程。

两种固件即使模型哈希相同也会因协议/预处理位置不同拒绝错误上位机。B另核对预处理参数及实现哈希。CRC和完整样本数通过后才执行预处理；不完整、非有限输入和错误CRC不会触发推理。返回结果分别记录`preprocessing_ms`和`inference_ms`，模拟模式不登记芯片时间。

- [B工程](../outputs/distillation/stm32_student_raw_vg_project/MDK_Project/ECG_Student_RAW_INT8_VG.uvprojx)
- [B HEX](../outputs/board_release/seed44_int8_stm32f407vg_preprocess/ECG_Student_RAW_INT8_STM32F407VG.hex)
- [B完整源码ZIP](../outputs/board_release/seed44_int8_stm32f407vg_preprocess/ECG_Student_RAW_INT8_VG_源码工程.zip)
- [B发布清单](../outputs/board_release/seed44_int8_stm32f407vg_preprocess/release_manifest.json)
- [C预处理与协议验证记录](../outputs/distillation/board_preprocess_export/preprocessing_validation.json)

源码：`firmware/ecg_preprocess/`及带`ECG_BOARD_PREPROCESSING`条件分支的`firmware/ecg_serial/`。冻结配置：`scripts/prepare_board_preprocess.py`；主机复核：`scripts/verify_board_preprocessing.py`；独立B工程编译：`scripts/build_seed44_board_preprocess.py`。发布目录存在时构建脚本拒绝覆盖。原参考工程、原A/VE工程及发布包保持哈希一致。
<!-- board-preprocessing-two-projects-20261010:end -->

<!-- board-preprocessing-independent-audit-20261010:start -->
发布后的独立复核已通过：源码ZIP内47个工程源码/库引用均存在且与编译工程一致，J-Link与时钟配置沿用原参考；C预处理、协议、部署头与冻结版本一致；HEX与map一致；原A、VE包及参考工程保护哈希均未变化。两个Python默认入口均通过界面初始化/绘图和单条模拟收发，验证日志按各自清单目录保存，并记录协议及预处理位置。见[独立复核记录](../outputs/board_release/seed44_int8_stm32f407vg_preprocess/independent_audit.json)及[上位机检查](../outputs/distillation/board_preprocess_export/application_checks.json)。复核入口：`scripts/audit_board_preprocess_release.py`。
<!-- board-preprocessing-independent-audit-20261010:end -->

<!-- serial-fold10-20261011:start -->
## 2026-10-11：两个串口程序新增折10波形选择

A仅推理与B板端预处理＋推理均新增“波形来源”下拉框，可选择“折9（联调）”与“折10（历史测试）”。默认仍为原折9的128条；折10为按原seed44纳入规则剔除五类全零后的完整2158条，不随机抽取或重新筛选。切换回到第0条，可用上一条/下一条/跳转查看任意记录；发送期间禁止切换。A发送预处理后的信号，B发送原始mV；两者模型、训练统计量、量化参数、阈值及固件均未修改，无需重新烧录HEX。

命令行增加`--fold 10`，例如：

```powershell
.\start_ecg_student_viz.cmd --fold 10
.\start_ecg_board_preprocess_viz.cmd --fold 10
.\start_ecg_board_preprocess_viz.cmd --fold 10 --headless --port COM7 --sample 0 --result-file outputs/serial_samples/seed44_fold10/physical_result.json
```

COM7替换为实际串口；不添加`--simulate`就是实际串口模式。模型握手仍使用原配套清单，折10样本另校验模型、预处理、量化参数、阈值和样本SHA-256；不符合当前模型则拒绝加载。折10电脑参考概率由本轮STM32Cube.AI host C验证重新生成，不是FP32教师或研究seed43预测。记录逐条保存折数、来源、样本/模型哈希、协议/预处理位置、阈值、真实标签、电脑参考、芯片输出和运行模式；折9/折10混合回放也可区分。

样本包及证据位于`outputs/serial_samples/seed44_fold10/`：[样本清单](../outputs/serial_samples/seed44_fold10/sample_manifest.json)、[预处理及host C验证](../outputs/serial_samples/seed44_fold10/preparation_validation.json)、[应用核对](../outputs/serial_samples/seed44_fold10/application_validation.json)。生成入口`scripts/prepare_serial_fold10.py`，已有完整样本包时校验后正常退出并提示串口入口，不覆盖文件；校验失败则报错。复核入口`scripts/verify_serial_fold10.py`。

本轮35项测试通过，完整2158条的Python/C预处理及INT8输入核对通过，并验证两个实际C串口解析器的抽样模拟及界面折9→折10→折9切换、收发和日志来源。这些是电脑核对，不代表本轮已完成折10实板验证。历史折10曾反复用于研发，不能将此次回放称为新的独立盲测。

折9同样经过样本准备，原128条已缓存；折10此次首次完成完整2158条打包，后续启动直接读取，无需每次执行prepare。这里的prepare整理波形、标签和电脑参考，不训练模型；B仍发送原始波形，由板端完成预处理。

复核发现Cube.AI CLI自动量化在两个点与固件舍入相差1：样本索引1203的[6,460]和1440的[0,778]。保留原CLI输入/输出作为证据，并用同一ST生成的host C模型及原固件权重，按实际固件INT8输入重算这两条参考。该调用对原CLI输入的输出逐值核对一致；其余参考沿用完整host验证。未修改模型、量化尺度、舍入规则或阈值。首次host验证遇到工作目录更新错误；换用独立短目录后成功，原失败日志及成功重试日志均保留。

保护检查另发现原Keil工作目录有49个工程设置或构建文件与旧审计哈希不同；用户明确授权保留当前文件并记录差异。原始检查失败及逐文件哈希保存于`artifact_guard_audit.json`，精确授权列表保存于`artifact_guard_approval.json`；只允许列表中已确认的当前哈希，未来变化仍拒绝通过。模型、源码、发布HEX和折9样本没有哈希差异，未覆盖或还原任何现有工程。
<!-- serial-fold10-20261011:end -->

<!-- serial-timing-difference-display-20261011:start -->
### 推理时间和概率差的含义（2026-10-11）

现有A/B标准库固件用1毫秒SysTick计数，记录`ai_ecg_network_run`调用前后的差值，并由串口返回`inference_ms`；上位机直接显示该字段，没有将时间写成常数。固定12导联×1000点输入及固定网络的计算量基本相同，小于1毫秒的时间变化可能无法从整数结果中看出，因此连续记录显示相同毫秒数可以是正常现象。截图中的1013ms是单次芯片推理的返回值，16.12s是上位机测得的整次收发耗时，包含数据传输、协议等待和推理。

“与电脑C模型最大概率差”是五个类别中最大的`abs(芯片概率－电脑参考概率)`，不是分类概率本身，也不代表分类准确率。原界面保留六位小数，约`1.5e-08`的差异会显示为`0.000000`；现改为科学计数法，例如`1.502e-08`。验证JSON一直保留未按六位小数显示舍入的差值。显示为`0.000e+00`表示这次比较在已返回、已保存的数值精度下相同，不能据此判断真实标签是否预测正确。

本次显示修改不更改模型、固件计时或HEX。最初微小误差示例来自已有模拟核对；随后执行的真实COM8回放另列如下，保留原始返回，不混合两类证据。
<!-- serial-timing-difference-display-20261011:end -->

<a id="physical-inference-confirmation-20261011"></a>

<!-- physical-inference-confirmation-20261011:start -->
### 2026-10-11：确认芯片实际完成推理

使用物理串口COM8、115200/8N1，直接发送折10样本索引0、1、2157、0，未使用模拟器，也未修改或烧录固件。两次握手均返回原seed44模型哈希`ef31602cf10e211c13b1dd99597fd4750d8d2c7e1032dbc9e1a756d9cdf3f610`、协议1、`ready=true`、`preprocessing=host`。因此当前板上是A仅推理，不能将本次结果记为B板端预处理已完成。

| 回放 | ECG记录 | 芯片预测 | 芯片返回推理耗时 | 与电脑C最大概率差 |
|---|---:|---|---:|---:|
| 第1次 | 9 | NORM | 1013ms | 1.502e-08 |
| 第2次 | 38 | NORM、CD | 1013ms | 1.884e-09 |
| 第3次 | 21826 | MI | 1013ms | 3.006e-08 |
| 重复第1条 | 9 | NORM | 1012ms | 1.502e-08 |

四次响应对应四个新序号；不同输入的完整五类概率不同，重复输入的概率逐值相同。整次收发耗时16.50–16.97秒，包括发送12000个浮点值、分块确认、推理和返回。固件仅在`ai_ecg_network_run`返回成功处理1个batch后发出`result`，失败会发出`ai_run`错误；配合真实串口原始返回和不同输入/重复输入核对，可以确认本次芯片完成了推理。

三次1013ms和一次1012ms符合固定网络及1ms计时分辨率下的表现；这里记录的是固件返回的毫秒数，未独立校准板载时钟。“概率差0.000000”是旧显示精度舍入，本次实测差值并非全为零。两端接近证明本次样本的部署数值一致性，不等于分类准确率100%，也不等于已验证完整折10或长期稳定性。

原始握手、四次返回字符串、序号、完整概率、真实标签和电脑参考保存于[实板核对记录](../outputs/serial_samples/seed44_fold10/physical_inference_check_20261011_103033.json)。复核入口`scripts/verify_physical_inference.py --port COM8`会自动识别A/B协议并保存新的时间戳记录；运行前关闭其他程序对同一串口的占用。本次完成后串口已释放，A界面可重新连接。
<!-- physical-inference-confirmation-20261011:end -->

<!-- physical-timing-audit-20261011:start -->
### 2026-10-11：独立核对为什么经常返回1013ms

用户提出同一毫秒数是否可疑后，新增电脑端独立计时：先完整传送波形，然后只测量发送`$R`推理命令至收到`result`的时间，排除前面的波形传输。测试仅针对COM8上原seed44 A固件，未修改或烧录固件；两条真实折10波形、全零计时输入和重复波形均收到真实返回。

| 输入 | 芯片返回推理时间 | 电脑测得最终命令往返时间 |
|---|---:|---:|
| ECG 9 | 1013ms | 1043.8554ms |
| ECG 21826 | 1013ms | 1045.8717ms |
| 全零预处理输入，仅用于计时 | 1013ms | 1046.1396ms |
| 重复ECG 9 | 1013ms | 1042.8906ms |

四次往返时间比返回的推理时间多29.89–33.14ms，包含命令/结果串口传输、USB接收、概率计算和格式化等开销；按115200/8N1和实际收发字节数计算的串口线速时间约14.41–14.76ms，其余开销没有单独分解。电脑计时支持计算实际占用约1秒，没有出现芯片返回1013ms而命令立即完成的情况。本方法不是微秒级校准，也未读取芯片DWT周期计数。

**为何经常是相同整数：** 该模型输入固定为12×1000，每条执行相同的卷积、池化和全连接图；Cube.AI生成报告给出每条14,698,207 MACC。改变输入数值会改变结果，但不会改变各层遍历规模；全零计时输入也没有减少网络规模。`infer`中没有写入常数1013或调用定时等待，而是取`ai_ecg_network_run`前后的SysTick计数差。计数单位为1ms；相近的实际耗时落入同一整数时就都显示1013，早先真实回放也出现过1012。固定运算规模、稳定执行环境和整数计时共同解释这一现象。

全零输入只是计时探针，不是真实ECG样本，不纳入分类性能统计。原始返回和独立计时保存于[计时核对记录](../outputs/serial_samples/seed44_fold10/physical_timing_audit_20261011_103850.json)。复核命令：`.venv\Scripts\python.exe scripts/verify_physical_inference.py --port COM8 --timing-audit`；A界面须先断开串口。此次完成后COM8已释放。
<!-- physical-timing-audit-20261011:end -->

<!-- serial-transfer-breakdown-20261011:start -->
### 2026-10-11：为什么整次收发约16秒

现有A协议将12000个float32格式化为`.9g`十进制文本，每64个数一块，共188块，每块均等待芯片ACK后才发送下一块。115200/8N1每字节占10位，理想吞吐率只有11520字节/秒。根据上述真实回放的保存输入和序号重建命令长度、按固件格式计算ACK长度，并使用实际最终返回字符串计数，ECG 9共发送155183字节、接收8467字节，累计163650字节。这里是字节量重建和线速计算，不是抓取UART线上的物理字节流。

该记录的耗时组成：理想线速传输14.2057秒＋芯片返回推理1.013秒＋剩余分块等待、USB及程序处理开销约1.5428秒＝实测16.7615秒。ECG 21826同口径为14.2604＋1.013＋1.4405＝16.7139秒。额外开销未逐项独立测量。全零计时输入的推理同为1013ms，整次仅7.0539秒，其文本更短；这是传输开销随文本内容变化的直接对照，不是分类性能测试。

因此目前整次收发的主要瓶颈是文本传输。针对A可研究二进制float32或直接二进制INT8传输：12000点float32负载48000字节，当前波特率下纯负载理想时间4.1667秒；INT8负载12000字节，理想时间1.0417秒。完整收发仍需加推理、帧头、CRC和等待开销；二进制INT8还需确保电脑量化与原固件输入逐值一致。该耗时拆解阶段只分析旧协议；随后已生成独立A快速版，见[A快速版](#serial-fast-a)，其新协议尚未实板测量。

详见[收发耗时拆解](../outputs/serial_samples/seed44_fold10/serial_transfer_breakdown_20261011.json)。
<!-- serial-transfer-breakdown-20261011:end -->

<a id="serial-fast-a"></a>

## A快速版：二进制INT8传输

原A文本版已完成小批真实板端推理，整次收发约16.7秒；本次另建A快速工程，直接传送相同的12000个INT8输入字节。电脑仍完成原预处理，再按原固件规则量化：float32除以0.2008364498615265、最近偶数舍入、零点0、饱和至[-128,127]。模型仍为原蒸馏seed44，五类阈值仍为[0.55,0.60,0.55,0.75,0.65]。本次没有重训练模型，没有修改B。

| 项目 | A原文本版 | A快速版 |
|---|---|---|
| 协议 | 1，十进制float32文本 | 3，二进制INT8 |
| 每条数据包 | 188包，每包64点 | 12包，每包最多1024字节 |
| 每条收发字节 | ECG 9约163650 B | 请求12288 B，结果长度随概率文本略变，总计约12700 B |
| 115200/8N1理想线速时间 | ECG 9约14.2057秒 | 已核对样本约1.10秒，不含推理及等待 |
| 整次耗时 | 原A实测约16.7秒 | 预计约2–3秒，尚未烧录实测 |
| 完整Flash / 主SRAM | 158876 B / 43672 B | 153040 B / 40664 B，含8 KiB栈 |
| 验证状态 | 小批真实COM8回放 | 0编译错误、0编译警告；电脑数值和协议验证通过 |

不要把模拟运行的几毫秒当作芯片速度。INT8数值一致只证明传输没有改变模型输入，不代表模型分类准确率100%。推理图没有变化，仍有约14.7M MACC，约1秒的旧板端耗时可作为估算基准，快速固件计时尚待实测。

### 使用顺序

1. 在Keil打开[快速版工程](../outputs/distillation/stm32_student_fast_vg_project/MDK_Project/ECG_Student_INT8_Fast_VG.uvprojx)，或下载[快速版HEX](../outputs/board_release/seed44_int8_stm32f407vg_fast/ECG_Student_INT8_Fast_STM32F407VG.hex)。目标仍为STM32F407VG，沿用原sine/三分类的8 MHz HSE、168 MHz、J-Link及USART1配置。
2. 烧录快速HEX并复位后，在VS Code运行[ecg_stm32_fast_viz.py](../scripts/ecg_stm32_fast_viz.py)，或双击[start_ecg_student_fast_viz.cmd](../start_ecg_student_fast_viz.cmd)。选择实际COM端口及115200，支持折9/折10。界面标题为“A仅推理·二进制INT8”，额外显示波形上传时间，记录也包含收发字节数、包数和最终推理命令往返时间。
3. 快速版必须配套快速固件；旧文本固件无法仅靠更换电脑程序提速。恢复原A时重新烧录原A HEX，并运行原start_ecg_student_viz.cmd。B入口与B工程保留原样。
4. 需要复核真实速度时，先在GUI断开串口，再运行下面的只读验证命令。它不会烧录或模拟，会发3条不同记录及1次重复，保存真实返回与分阶段计时。

```powershell
.venv/Scripts/python.exe scripts/verify_physical_fast_serial.py --port COM8 --fold 10
```

实际COM端口以本机为准。新版尚未烧录，本轮没有执行上述物理验证命令。电脑端模拟核对入口如下，其耗时不代表芯片：

```powershell
.venv/Scripts/python.exe scripts/ecg_stm32_fast_viz.py --simulate --headless --fold 10
```

### 传输校验及已完成验证

协议3使用16字节小端帧头：ECG3魔数4字节、版本/类型各1字节、负载长度2字节、请求序号4字节、输入偏移4字节；尾部4字节为CRC32。HELLO/BEGIN/DATA/RUN类型为1/2/3/4，回应分别为0x81–0x84，错误为0xFF。DATA负载是按导联顺序排列的INT8二进制；RUN另检查完整12000字节的CRC32，完整收齐且校验通过才推理。每包等待芯片确认下一偏移，CRC错误、漏包、重复包、错序号、超长帧或接收超时终止本次，不自动跳过。HELLO核对模型哈希、输入、标签、阈值和量化参数。

UART接收改为2048字节环形缓冲，保留所有0–255字节，不再把CR/LF当作波形结束；ISR只收字节，主循环解析和推理。单包最大1044字节，分包确认限制正常发送积压；UART错误或缓冲溢出会报错，5秒没有新字节则放弃未完成的帧/记录。没有改成实时逐点采集。

已完成：42项新旧串口及折数回归测试；全部128条折9和2158条折10，共27432000个输入值，与原C量化规则逐值比较，INT8差异0。44条记录通过实际新C协议＋同一Cube.AI host C模型，最大概率差5.9764e-08；难舍入的折10索引1203和1440也通过原文本C协议直接对照。GUI折9→折10→折9、两折发送和日志来源核对通过；快速Python入口另验证折10索引1203。模型、权重、原发布HEX、旧A/B工程及冻结折10样本的保护哈希核对通过。

pytest有一条旧缓存目录写入警告，不影响42项通过结果；Keil编译为0警告。HEX校验和、VG地址范围、向量表和源码ZIP完整性检查通过。板上UART环形缓冲接收稳定性、快速固件数值与实测总耗时仍待烧录后验证。

- [完整快速源码ZIP](../outputs/board_release/seed44_int8_stm32f407vg_fast/ECG_Student_INT8_Fast_VG_源码工程.zip)
- [发布清单、资源和保护哈希](../outputs/board_release/seed44_int8_stm32f407vg_fast/release_manifest.json)
- [逐值核对及44条C模型结果](../outputs/distillation/fast_serial_export/verification/verification.json)
- [电脑GUI验证日志](../outputs/distillation/fast_serial_export/verification/A_fast_GUI_log.json)
- 源码：firmware/ecg_serial_fast/；生成入口：scripts/build_seed44_fast_vg.py。现有发布包拒绝覆盖，未发布工程编译失败时才使用--resume。

