# STM32 部署与串口验证

更新：2026-10-03。用户已确认开发板芯片为 STM32F407VE。当前推荐的是蒸馏 CNN seed44 INT8 独立工程；旧 ResNet 包保留用于对照。两个包各有自己的模型、标准化参数、阈值和校验清单，启动器必须匹配固件。

## 当前学生固件与资源

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

烧录后，用学生启动器选择实际串口。USART1：PA9 TX / PA10 RX，115200，3.3 V TTL，和 USB 转串口共地。芯片型号已确认，实际接线尚未验证。模型仍要求十二导联输入；界面显示单条导联，发送全部十二导联。

学生固件输入量化为 `q = clip(round_to_nearest_even(x / 0.2008364498615265), -128, 127)`，五类阈值依次为 `[0.55, 0.60, 0.55, 0.75, 0.65]`；这些参数由学生清单和头文件固定，不要使用下文旧 ResNet 的参数。

实板命令行示例（将 COM7 换成实际端口）：

```powershell
.\start_ecg_student_viz.cmd --headless --port COM7 --sample 0 --result-file outputs/board_release/seed44_int8_stm32f407ve/physical_serial_result.json
```

命令行复核：

```powershell
.venv/Scripts/python.exe scripts/verify_student_firmware.py
```

本轮强制转换、全量编译与 HEX 检查的原始证据在 `outputs/distillation/distilled_export/recheck_20261003/`，包括 `generate.log`、`host_validate.log`、`source_comparison.json` 和 `release_audit.json`。原脚本在生成代码和验证结果存在时会复用，不代表每次都强制重新转换。

分类训练与部署候选选择使用 PTB-XL 训练和验证集。其他来源的外部测试见独立报告，不影响本固件空间结论。


## 串口使用与接入

学生包使用 [start_ecg_student_viz.cmd](../start_ecg_student_viz.cmd)。以下通用协议与接线对两个包适用；涉及 `outputs/cubeai/resnet`、旧启动器和旧 HEX 的示例属于旧 ResNet 对照包。学生路径、HEX 和资源以本文件前半部分为准。

### 与参考程序的区别

| 项目 | 原参考程序 | 本次程序 |
|---|---|---|
| 输入 | 单导联、360 点 | 12 导联、每导联 1000 点，100 Hz、10 秒 |
| 分类 | ECG / Sine / Noise 三选一 | NORM / MI / STTC / CD / HYP 五个独立判断 |
| 数据 | 模拟 ADC 数值，按记录做 min-max | 电脑按训练配置预处理，芯片按模型量化参数转 INT8 |
| 发送 | 一条长 CSV | 每块最多 64 个浮点数，收到确认才发下一块 |
| 对齐 | 当前界面的样本索引 | 请求序号 + 数据位置 + 完整记录 CRC32 |
| 回复 | 三个概率和一个 argmax 类别 | 五个 sigmoid 概率、五位分类掩码、芯片推理时间 |
| 核对 | 默认相信芯片模型 | 连接时核对模型 SHA-256、输入形状、类别顺序和阈值 |

当前任务为五标签分类，多类可以同时为阳性。概率之和不要求等于 1，不能使用 softmax 或只返回最大概率类别。没有标签达到阈值时显示“无标签达到阈值”，不会自动改成 NORM。

### 如何运行

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

### 芯片端工程和接线

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

### Cube HAL 工程接入

工作区提供以下文件：

- [ecg_protocol.c](../firmware/ecg_serial/ecg_protocol.c) / [ecg_protocol.h](../firmware/ecg_serial/ecg_protocol.h)：分块接收、INT8 输入转换、CRC32、推理回复。
- [ecg_uart.c](../firmware/ecg_serial/ecg_uart.c) / [ecg_uart.h](../firmware/ecg_serial/ecg_uart.h)：STM32F4 HAL 适配。
- [ecg_deployment_config.h](../firmware/ecg_serial/ecg_deployment_config.h)：当前模型哈希、输入量化参数和五类阈值。
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

### 输入预处理与内存

电脑端按检查点的配置进行基线处理、0.5–40 Hz 四阶零相位带通和训练集全局标准化，保留原始幅值的统计含义。标准化均值和标准差来自训练集，并保存于部署清单。不能再按每条波形做 min-max 或记录内 z-score。

发送布局为逐导联展开：先发送 I 导联的 1000 点，再发送 II 导联，直到 V6。导联顺序为 `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`。

串口传输为预处理后的 FP32 数值的 ASCII 表示；芯片收到一个小块就直接量化写入网络输入区，不另外保留一整条 48000 字节浮点输入。

当前量化公式：

```text
q = clip(round_to_nearest_even(x / 0.1990550309419632) + 5, -128, 127)
```

Cube.AI 生成版本必须使用 `--inputs-ch-position chfirst --output-data-type float32 --c-api legacy`。输入为 INT8，输出为 FP32 logits。芯片对每个 logits 单独做 sigmoid，再按 `[0.60, 0.65, 0.55, 0.75, 0.70]` 判定。

该程序验证“电脑预处理 + 芯片神经网络推理”的链路；尚未实现采样 ADC、12 导联模拟前端或把预处理搬到芯片上。

### 串口协议

每条命令以 CRLF 结束。每块收到确认才发送下一块，默认等待确认 5 秒，推理结果等待上限 60 秒；后者是软件超时设置，不是实测推理时长。

| 命令 | 意义 | 回复 |
|---|---|---|
| `$P` | 查询模型、阈值和输入协议 | JSON `hello` |
| `$B,seq,12000` | 开始发送一条记录 | JSON `begin`，带 seq |
| `$D,seq,offset,count,x1,...` | 发送最多 64 个浮点数 | JSON `ack`，带 seq 和下一位置 next |
| `$R,seq,crc32` | 检查完整输入并执行推理 | JSON `result`，带五类概率、mask、inference_ms |

CRC32 与 Python `zlib.crc32` 一致，计算对象是按导联展开的 12000 个 little-endian float32 的原始字节。ASCII 数值使用 9 位有效数字，确保 float32 往返。芯片检查序号、位置、数量、有限数值及完整记录 CRC；错误时回复 JSON `error`，终止本次记录，下次 `$B` 可重新开始。

分类 mask 的 bit 0 到 bit 4 依次代表 NORM、MI、STTC、CD、HYP。例如 `mask=10` 表示 MI 和 CD 同时达到阈值。回复必须与本次 seq 一致，程序会忽略旧序号回复。

### 已完成的验证与边界

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

### 后续实板联调

开发板型号已确认 VE，检查板载晶振与下载器设置后烧录当前学生 HEX，连接 USART1。先在学生界面确认握手成功，再逐条发送验证波形，保存真实模式日志，核对芯片与电脑参考的五类概率差，并记录芯片推理时间。如果握手报模型不一致，应重新生成相同模型的部署清单和固件；如果接收超时，先检查 COM 端口、TX/RX、GND 和波特率。

旧 ResNet Flash 余量较小；当前学生固件余量约 356.85 KiB，加入功能后仍应重新编译检查。不应直接把模型时间长度或导联数改短后继续沿用旧性能数字。


## 历史 Cube.AI 对照分析

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

实板型号现已确认 STM32F407VE，尚未进行烧录、真实串口波形收发和芯片耗时测量。旧轮次仅枚举到蓝牙串口，不能据此认定已连接开发板。后续接上 USB 转串口及开发板，按本文件当前学生固件说明运行程序即可验证；电脑端的模拟时间不能当作 STM32 推理时间。

此次 ResNet INT8 只报告验证折 9 的量化对比和已有交叉验证结论，没有新增测试折 10 结果。历史模型的完整整体和逐类测试结果见 [各模型整体与逐类指标总表](各模型整体与逐类指标总表.md)。
