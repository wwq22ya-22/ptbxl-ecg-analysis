"""A快速版：上位机预处理、INT8二进制传输、芯片推理。"""
from pathlib import Path
from ecg_stm32_viz import main

if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1] / 'outputs/distillation/fast_serial_export/serial_manifest.json')
