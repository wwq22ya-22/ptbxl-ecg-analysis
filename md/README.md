# 项目运行与文档入口

更新：2026-10-03。项目完成 PTB-XL 十二导联五类多标签训练、网页/API 展示、模型蒸馏、INT8 导出、STM32 固件编译及串口验证程序；Chapman 外部测试已完成。STM32F407VE 型号已确认，学生 HEX 和完整源码工程已整理；课程展示尚待实板烧录、串口收发和板上计时验证。

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

当前数据上的指标不等于临床适用性证明。短时 HRV、R 峰和模型响应图属于课程实验功能。最新回归测试为 78 项通过，实板结果仍待验证。
