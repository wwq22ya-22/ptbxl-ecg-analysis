# STM32Cube AI 导入前准备

以正式训练的 PTB-XL CNN1D 检查点为部署候选。ONNX 只包含神经网络；信号预处理保留在图外。

## 输入输出约定

- 输入 ecg：float32，固定形状 [1, 12, 1000]，布局 [batch, lead, time]。
- 导联顺序：I、II、III、aVR、aVL、aVF、V1、V2、V3、V4、V5、V6。
- 每条记录 100 Hz、10 秒；输入必须是训练流程预处理后的信号。
- 固件端要复现 deployment_manifest.json 中的基线处理、0.5–40 Hz 四阶零相位带通和逐导联 z-score；这些操作没有并入 ONNX。
- 输出 logits：[1, 5]，依次是 NORM、MI、STTC、CD、HYP。对每个分数单独做 sigmoid，再使用清单内的逐类阈值。

## 文件

- outputs/stm32/ptbxl_cnn1d_float32.onnx：导入 STM32Cube AI。
- outputs/stm32/calibration_train.npy 和 calibration_train.npz：训练 folds 1–8 随机抽取的校准信号；NPZ 数据键为 ecg。
- outputs/stm32/validation_reference.npz：fold 9 输入、标签、记录号及 PyTorch 参考分数。
- outputs/stm32/onnx_parity.json：PyTorch 对 ONNX Runtime CPU 的误差报告。
- outputs/stm32/cnn1d_weights_only.pt：移除优化器状态的权重备份。
- outputs/stm32/deployment_manifest.json：模型契约、预处理、阈值、算子、样本来源和 SHA-256。

本次校准只使用 folds 1–8，参考样本只使用 fold 9；未读取官方测试 fold 10。INT8 量化后若要调整阈值，使用 fold 9；fold 10 留作最终独立评估。

## 重生成

项目根目录运行：

    .\.venv\Scripts\python.exe -m pip install -r requirements-stm32.txt
    .\.venv\Scripts\python.exe scripts\prepare_stm32.py

确认 onnx_parity.json 为 passed 后，再在 STM32Cube AI 中选择 STM32F407VGTx 并导入 ptbxl_cnn1d_float32.onnx。算子兼容性、INT8 精度、Flash/RAM/MACC 和推理时延须在 AI 工具中确认；导出成功不表示固件已经生成或烧录。
