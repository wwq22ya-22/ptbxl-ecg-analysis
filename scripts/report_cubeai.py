"""Summarize actual Cube.AI reports, quantization results and firmware build."""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/cubeai"


def load(path): return json.loads(path.read_text(encoding="utf-8"))


def main():
    inception = load(OUT / "inception/int8_validation_report.json")
    resnet = load(OUT / "resnet/int8_validation_report.json")
    manifest = load(OUT / "resnet/serial_manifest.json")
    parity = load(OUT / "resnet/host_validation_summary.json")
    simulation = load(OUT / "serial_simulation_validation.json")
    analysis = []
    for title, path in [
        ("优化 Inception FP32", OUT / "inception/float_analysis/ecg_inception_c_info.json"),
        ("优化 Inception INT8", OUT / "inception/int8_analysis/ecg_inception_int8_c_info.json"),
        ("标准化 ResNet INT8（部署版）", OUT / "resnet/generated/ecg_network_c_info.json"),
    ]:
        info = load(path)
        report = path.with_name(path.name.replace("_c_info.json", "_generate_report.txt" if "generated" in path.parts else "_analyze_report.txt"))
        text = report.read_text(encoding="utf-8")
        macc = int(re.search(r"^macc\s+:\s+([\d,]+)", text, re.M).group(1).replace(",", ""))
        analysis.append({"model": title, "weights_bytes": info["memory_footprint"]["weights"],
                         "activations_bytes": info["memory_footprint"]["activations"], "macc": macc})
    mapping = (OUT / "stm32_spl_project/MDK_Project/Listings/ECG_ResNet_INT8.map").read_text(errors="replace")
    build = (OUT / "keil_build.log").read_text(encoding="utf-8", errors="replace")
    checks = re.search(r"(\d+) Error\(s\),\s*(\d+) Warning\(s\)", build)
    if not checks or int(checks.group(1)):
        raise RuntimeError("A successful Keil build is required before reporting deployment.")
    flash = int(re.search(r"Total ROM Size[^\n]*?\s(\d+)\s+\(", mapping).group(1))
    ram = int(re.search(r"Total RW\s+Size[^\n]*?\s(\d+)\s+\(", mapping).group(1))
    firmware = {"flash_bytes": flash, "ram_bytes": ram, "flash_limit_bytes": 512 * 1024,
                "main_sram_limit_bytes": 128 * 1024, "flash_margin_bytes": 512 * 1024 - flash,
                "physical_board_validation": False, "build_errors": int(checks.group(1)),
                "build_warnings": int(checks.group(2))}
    summary = {"selection": manifest, "analysis": analysis, "firmware": firmware,
               "quantization": {"inception": inception, "resnet": resnet},
               "host_c_validation": parity, "serial_simulation": simulation}
    (OUT / "deployment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Cube.AI 优化模型分析总结", "", "更新日期：2026-10-01", "",
        "## 结论", "",
        "本次部署选择 **训练集全局标准化 ResNet1D 的 INT8 版本**。",
        "已有交叉验证中，标准化 ResNet 的宏 F1 为 0.748078，标准化 Inception 为 0.747293，二者差距很小；"
        "部署选择进一步依据 Cube.AI 实测内存和完整固件的编译结果。双模型集成仍是电脑端的较优方案。", "",
        "优化版 Inception 已完成 FP32 和 INT8 分析，但 INT8 仍需 288000 字节运行内存，超过 F407 的容量。"
        "ResNet INT8 的运行内存为 44032 字节，并已生成、编译串口分类固件。", "",
        f"完整固件使用 **{flash} 字节 Flash（{flash/1024:.2f} KiB）**、"
        f"**{ram} 字节 RAM（{ram/1024:.2f} KiB）**。在 512 KiB Flash 限制下链接成功，"
        f"Flash 余量约 **{(512*1024-flash)/1024:.2f} KiB**。新增界面、文件系统或其他模块前需重新检查链接结果。", "",
        "当前状态：Cube.AI 分析完成、模型代码生成完成、电脑端 C 实现验证完成、串口固件编译完成。"
        "尚未烧录和进行实板串口测试，所以没有实测芯片推理耗时。", "",
        "## 分析环境与芯片依据", "",
        "- 安装包目录：X-CUBE-AI 10.2.1；实际命令行引擎自报 `ST Edge AI Core v2.2.0-20266 2adc00962`、`STM32CubeAI 10.2.0-RC1`。",
        "- 使用本机 `stedgeai` 工具，属于 Cube.AI 的分析、生成和验证工具；分析目标 `stm32f4`，内存优化 `ram`，权重压缩 `none`。",
        "- 当前 `D:/MyProject/MyProject.ioc` 设置为 STM32F407VET6：512 KiB Flash，128 KiB 主 SRAM + 64 KiB CCM。",
        "- 用户提供的串口参考工程设置为 STM32F407VG：1 MiB Flash。本次新工程按更小的 VE 预算配置；实板型号还需核对。",
        "- Cube.AI CLI 的普通分析不会自动按某个具体芯片的容量拒绝模型，因此另外进行了 512 KiB Flash、128 KiB 主 SRAM 的 Keil 链接检查。", "",
        "芯片规格参考：[ST STM32F407VE](https://www.st.com/en/microcontrollers-microprocessors/stm32f407ve.html)。"
        "工具说明参考：[ST 命令行文档](https://stedgeai-dc.st.com/assets/embedded-docs/2.0.0/stm32_command_line_interface.html)。", "",
        "## Cube.AI 资源结果", "",
        "| 模型 | 权重 / B | 激活内存 / B | MACC | 部署判断 |",
        "|---|---:|---:|---:|---|",
    ]
    for i, row in enumerate(analysis):
        verdict = "权重和 RAM 均超限" if i == 0 else "RAM 超限" if i == 1 else "完整固件已链接通过"
        lines.append(f"| {row['model']} | {row['weights_bytes']:,} | {row['activations_bytes']:,} | {row['macc']:,} | {verdict} |")
    lines += ["", "以上激活内存已包含模型输入、输出，不能再重复加一次输入缓冲。权重占用不等于固件总 Flash；"
              "总 Flash 还包括推理库、网络代码、串口程序及初始化数据。", "",
              "部署版显式指定 `--inputs-ch-position chfirst --output-data-type float32 --c-api legacy`。"
              "输入为 INT8 `[12,1000]`，输出为五个 FP32 logits；输出转换使 MACC 比纯 INT8 输出版多 10。", "",
              "## 量化精度：同一验证集整体对比", "",
              "以下均为 **官方验证折 9，共 2183 条记录**。校准仅使用训练折 1–8 的 256 条记录；"
              "没有用验证标签选择量化方法，没有重新调整阈值，没有读取测试折 10。",
              "使用 ONNX Runtime 1.30.0 的静态 QDQ 量化，权重有符号对称、按通道，激活有符号非对称、按张量，MinMax 校准。", "",
              "| 模型 | Macro AUROC | Macro Precision | Macro Recall | Macro F1 |", "|---|---:|---:|---:|---:|"]
    for title, result in [("Inception FP32", inception["float32"]), ("Inception INT8", inception["int8"]),
                          ("ResNet FP32", resnet["float32"]), ("ResNet INT8（选用）", resnet["int8"])]:
        lines.append("| " + title + " | " + " | ".join(f"{result[key]:.6f}" for key in
                     ["macro_auroc", "macro_precision", "macro_recall", "macro_f1"]) + " |")
    lines += ["", f"ResNet 宏 F1 降低 **{(resnet['float32']['macro_f1']-resnet['int8']['macro_f1'])*100:.3f} 个百分点**，"
              f"五类二值判定一致率 **{resnet['label_agreement']*100:.3f}%**。该一致率是对 2183×5 个判断计算，"
              "不是对真实标签的准确率。", "",
              "阈值保持为 `[0.60, 0.65, 0.55, 0.75, 0.70]`，来自该 ResNet 检查点的验证阈值。"
              "这组阈值与电脑端集成或优化版 Inception 的阈值不同，不能混用。", "",
              "## 选用模型逐类指标", "",
              "| 类别 | 版本 | 阳性数 | AUROC | Precision | Recall | F1 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for name in manifest["labels"]:
        for version in ["float32", "int8"]:
            m = resnet[version]["per_class"][name]
            lines.append(f"| {name} | {version.upper()} | {m['support']} | " +
                         " | ".join(f"{m[key]:.6f}" for key in ["auroc", "precision", "recall", "f1"]) + " |")
    lines += ["", "HYP 仍是最弱类别，INT8 F1 为 0.613377；MI 召回率为 0.690741，后续仍需改进。"
              "此次工作验证的是已有模型的芯片转换和资源可用性，不代表解决了全部类别短板。", "",
              "## 转换与程序验证", "",
              "- PyTorch → FP32 ONNX：ResNet 在 128 条验证记录上最大 logits 误差 0.000003815，通过容差检查。",
              f"- INT8 ONNX → Cube.AI 电脑端 C 实现：128 条记录，logits RMSE {parity['rmse_logits']:.9f}、"
              f"最大 logits 差 {parity['max_absolute_logit_error']:.9f}，最大概率差 {parity['max_probability_error']:.9f}，五类判定一致率 100%。",
              "- 最大 logits 差约为一个输出量化步长，属于已观察到的转换差异，不能称为浮点数完全一致。",
              "- 128 条记录从原始 WFDB 波形重新预处理，与发送数据缓存一致；串口 C 协议的分块确认和 CRC 全部通过模拟验证。",
              "- Keil ARM Compiler 5.06 update 4：0 错误、0 警告，生成 HEX；工程链接区域明确限制 Flash 为 0x80000。",
              "- 项目测试：71 项通过，其中新增导出抽样和串口协议测试覆盖数据对齐、错误 CRC、缺包、错序、无效浮点数、旧响应和多标签返回。", "",
              "## 文件与复现", "",
              "- 选用模型：[ptbxl_resnet1d_int8.onnx](../cubeai/resnet/ptbxl_resnet1d_int8.onnx)。",
              "- 源检查点：`outputs/optimization/01_global_norm/checkpoint/resnet1d_best.pt`。",
              "- 生成代码与原始报告：[generated](../cubeai/resnet/generated/ecg_network_generate_report.txt)。",
              "- 完整资源及精度明细：[deployment_summary.json](../cubeai/deployment_summary.json)。",
              "- 可编译串口工程：[ECG_ResNet_INT8.uvprojx](../cubeai/stm32_spl_project/MDK_Project/ECG_ResNet_INT8.uvprojx)。",
              "- HEX：[ECG_ResNet_INT8.hex](../cubeai/stm32_spl_project/MDK_Project/Objects/ECG_ResNet_INT8.hex)。",
              "- 串口使用与接入：[串口波形分类验证程序总结](../../md/STM32部署与串口验证.md)。", "",
              f"INT8 ONNX SHA-256：`{manifest['model_sha256']}`。", "",
              "在项目根目录导出和量化：", "", "```powershell",
              ".venv/Scripts/python.exe scripts/prepare_stm32.py --checkpoint outputs/optimization/01_global_norm/checkpoint/resnet1d_best.pt --output-dir outputs/cubeai/resnet",
              ".venv/Scripts/python.exe scripts/quantize_cubeai.py --export-dir outputs/cubeai/resnet", "```", "",
              "Cube.AI 的完整实际命令保存在各原始报告的 `Parameters` 字段。先用 `validate --mode host` 生成验证输出，"
              "再用 `generate` 生成代码，随后运行 `package_serial_demo.py` 准备串口样本和工程。更换模型后需要重新生成所有对应文件和固件。", "",
              "## 尚未完成的实板验证", "",
              "尚未确认实板的 VE/VG 型号，也没有进行烧录、真实串口波形收发和芯片耗时测量。"
              "当前机器枚举到的串口描述为蓝牙串口，不能据此认定已连接开发板。"
              "后续接上 USB 转串口及开发板，按另一份文档运行程序即可验证；电脑端的模拟时间不能当作 STM32 推理时间。", "",
              "此次 ResNet INT8 只报告验证折 9 的量化对比和已有交叉验证结论，没有新增测试折 10 结果。"
              "历史模型的完整整体和逐类测试结果见 [各模型整体与逐类指标总表](../../md/各模型整体与逐类指标总表.md)。", ""]
    (ROOT / 'outputs/reports').mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/reports/CubeAI_优化模型分析总结.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved analysis summary; firmware Flash {flash} B, RAM {ram} B.")


if __name__ == "__main__": main()
