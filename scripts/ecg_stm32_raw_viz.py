"""Direct VS Code entry: send raw ECG to board preprocessing + inference firmware."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ecg_stm32_viz import main
if __name__ == '__main__':
    main(ROOT / 'outputs/distillation/board_preprocess_export/serial_manifest.json')
