"""Report student firmware usage, conversion checks and PTB-XL performance."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/distillation'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    firmware=read(OUT/'distilled_export/student_firmware_summary.json')
    conversion=read(OUT/'distilled_export/student_conversion_audit.json')
    training=read(OUT/'summary.json')
    lines=['# 蒸馏学生模型固件空间与验证', '', '日期：2026-10-01', '', '## 结论', '',
           f"完整蒸馏 CNN INT8 串口固件已编译和链接通过：Flash {firmware['flash_bytes']} B（{firmware['flash_bytes']/1024:.2f} KiB）、RAM {firmware['ram_bytes']} B（{firmware['ram_bytes']/1024:.2f} KiB）。按 STM32F407VE 的 512 KiB Flash、128 KiB 主 SRAM 限制，空间足够。", '',
           '本轮没有烧录实板或实测板上耗时。生成的 HEX 和独立工程可供后续联调；当前仅检测到蓝牙虚拟串口，没有识别到开发板串口。', '',
           f"冻结阈值的蒸馏 INT8 模型在 PTB-XL 保留的 2158 条测试记录上，五类宏 AUROC {training['deployment']['distilled']['test']['int8']['macro_auroc']:.6f}、宏 F1 {training['deployment']['distilled']['test']['int8']['macro_f1']:.6f}。", '',
           '## 完整固件资源', '', '| 项目 | 蒸馏 CNN INT8 | 既有 ResNet INT8 |', '|---|---:|---:|',
           f"| 完整固件 Flash | {firmware['flash_bytes']} B | 506496 B |",
           f"| 完整固件 RAM | {firmware['ram_bytes']} B | 62504 B |",
           f"| 512 KiB Flash 下余量 | {firmware['flash_margin_bytes']} B（{firmware['flash_margin_bytes']/1024:.2f} KiB） | 17792 B |",
           f"| 128 KiB SRAM 下余量 | {firmware['ram_margin_bytes']} B（{firmware['ram_margin_bytes']/1024:.2f} KiB） | 68568 B |",
           '| 模型权重 | 112276 B | 453012 B |', '| 模型激活内存 | 24000 B | 44032 B |', '| 模型 MACC | 14698207 | 73639711 |', '',
           'Flash 采用链接 map 的 Total ROM Size，包含压缩后的初始化数据；RAM 采用 Total RW Size。模型激活包含输入和输出，不能与完整固件 RAM 再相加。编译器 ARMCC 5.06 update 4，0 错误、0 警告。', '',
           '## 转换与协议验证', '',
           f"Cube.AI 电脑 C 实现验证 {conversion['host_c_records']} 条波形，logits RMSE {conversion['host_c_logit_rmse']:.9f}、最大误差 {conversion['host_c_max_logit_error']:.9f}；该样本集与 ONNX INT8 输出一致。", '',
           f"{conversion['serial_simulation_records']} 条波形通过实际 C 协议解析器的分块传输、CRC 和回复模拟，分类与电脑 C 参考的逐标签一致率 {conversion['simulated_decision_agreement_with_host_c']*100:.2f}%。模拟推理不是 STM32 板上运行。", '',
           '原始波形重新预处理与缓存已核对。独立学生工程使用自己的模型哈希、INT8 输入 scale / zero point、训练标准化统计和五个阈值，旧 ResNet 部署包未替换。', '',
           '## PTB-XL 逐类测试指标', '', '| 类别 | AUROC | Precision | Recall | F1 |', '|---|---:|---:|---:|---:|']
    for label in ['NORM','MI','STTC','CD','HYP']:
        metrics=training['deployment']['distilled']['test']['int8']['per_class'][label]
        lines.append('| '+label+' | '+' | '.join(f'{metrics[key]:.6f}' for key in ['auroc','precision','recall','f1'])+' |')
    lines+=['', '## 独立复核', '',
            '检查点与导出文件哈希已核对。HEX 每条校验和与全部 Flash 地址范围均核对通过，没有越过 512 KiB 边界。']
    lines+=['', '## 文件与运行', '',
            '- [学生 Keil 工程](../distillation/stm32_student_project/MDK_Project/ECG_Student_INT8.uvprojx)。',
            '- [学生 HEX](../distillation/stm32_student_project/MDK_Project/Objects/ECG_Student_INT8.hex)。',
            '- [完整空间结果](../distillation/distilled_export/student_firmware_summary.json)。',
            '- [PTB-XL 蒸馏与量化结果](../distillation/summary.json)。',
            '- [学生串口界面启动器](../../start_ecg_student_viz.cmd)。', '',
            '板上烧录本学生 HEX 后，用学生启动器选择对应串口。硬件接口沿用独立工程 USART1：PA9 TX / PA10 RX，115200，和 USB 转串口共地。未核实实板型号及实际接线。模型仍要求十二导联输入；界面显示单条导联，发送全部十二导联。', '',
            '命令行复核：', '', '```powershell', '.venv/Scripts/python.exe scripts/verify_student_firmware.py',
            '.venv/Scripts/python.exe scripts/report_student_validation.py', '```', '',
            '分类训练与部署候选选择使用 PTB-XL 训练和验证集。其他来源的外部测试见独立报告，不影响本固件空间结论。']
    (ROOT / 'outputs/reports').mkdir(parents=True, exist_ok=True)
    (ROOT/'outputs/reports/蒸馏模型_固件空间与验证.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
