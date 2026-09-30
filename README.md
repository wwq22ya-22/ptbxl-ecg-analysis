# PTB-XL 十二导联 ECG 智能分析与可视化系统

面向小组课程实践的 ECG 分析项目：读取真实十二导联心电信号，进行预处理、R 峰/心率/RR/短时 HRV 分析，使用 CNN 1D 或 ResNet1D 完成五类多标签推理，并通过 Streamlit 和 FastAPI 展示结果。另提供 STM32F407 的 ONNX 导出与 CubeAI 前期准备。

## 小组成员从这里开始

- [项目总结：数据、模型大小、正式测试结果](md/项目总结.md)
- [完整运行与开发说明](md/README.md)
- [LUDB 外部验证报告](md/LUDB_外部验证报告.md)
- [STM32 部署准备](md/README_STM32_PREP.md)
- [项目交付说明](md/项目交付说明.md)

模型输出 NORM（正常）、MI（心肌梗死）、STTC（ST/T 改变）、CD（传导异常）、HYP（肥厚）五个独立分数。一条记录可对应多个类别。项目用于教学与算法实验，性能结果不构成临床诊断依据。

## 已保存的模型结果

PTB-XL 官方测试 fold 10 共 2,198 条记录；模型与判定阈值由训练集、验证集确定。

| 模型 | Macro AUROC | Macro F1 | 检查点 |
| --- | ---: | ---: | --- |
| CNN 1D | 0.9077 | 0.7080 | `outputs/checkpoints/baseline_best.pt` |
| ResNet1D | 0.9120 | 0.7149 | `outputs/checkpoints/resnet1d_best.pt` |

LUDB 的 200 条外部 ECG 已完成冻结模型验证；四个可映射类别的 CNN 平均 AUROC 为 0.7939、平均 F1 为 0.4857。NORM 未评估，MI 阳性只有 8 条，外部 HYP 召回率较低。四类外部平均值与 PTB-XL 五类平均值口径不同，详见外部验证报告。

## 本机运行

仓库包含代码、正式检查点、指标、训练图表与 ONNX。原始 ECG 数据需要按 [数据说明](data/README.md)另行下载；首次克隆使用 `configs/default.yaml` 的 `data/ptb-xl` 路径。可用不提交的 `configs/local.yaml` 覆盖本机数据位置。

本项目原环境验证于 Windows、Python 3.14.3。请使用本机 PyTorch 支持的 Python 版本，并在项目根目录创建环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-train.txt -r requirements-app.txt -r requirements-api.txt
.\.venv\Scripts\python.exe scripts\download_ptbxl.py --workers 8
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

打开 <http://localhost:8501/>。已有正式模型可直接用于推理，无需先重新训练。GPU 安装方式见完整运行说明。

启动 API 或运行测试：

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp outputs\pytest-run
```

## 目录

| 目录 | 内容 |
| --- | --- |
| `src/` | 数据读取、信号处理、模型、训练、推理与解释 |
| `app/` / `api/` | Streamlit 页面 / FastAPI 服务 |
| `scripts/` | 数据下载、训练、评估、单条推理与 STM32 导出入口 |
| `tests/` | 自动测试 |
| `configs/` | 默认配置 |
| `outputs/` | 正式模型、指标、图表、外部验证与 STM32 导出 |
| `md/` | 项目总结、阶段记录与验证文档 |

## 数据与参考

- [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)：数据许可 CC BY 4.0，使用时按官方要求引用。
- [LUDB 1.0.1](https://physionet.org/content/ludb/1.0.1/)：数据许可 Open Data Commons Attribution License v1.0。
- [参考项目](https://github.com/AndrewLucenko/ptb-xl-ecg-classification)：架构参考与实际实现说明见完整运行文档。

历史报告中的 `D:\...` 是原开发机器位置；组员以克隆后的项目目录和本机数据配置为准。
