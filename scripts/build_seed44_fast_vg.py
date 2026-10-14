"""Build a separate A binary-INT8 firmware; never rebuild the released A/B projects."""
import argparse
import json
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_seed44_vg import REFERENCE, SDK, EXPORT, audit_hex, snapshot
from scripts.package_serial_demo import write_spl_project
from scripts.prepare_stm32 import digest

PROJECT = ROOT / 'outputs/distillation/stm32_student_fast_vg_project'
FAST_EXPORT = ROOT / 'outputs/distillation/fast_serial_export'
RELEASE = ROOT / 'outputs/board_release/seed44_int8_stm32f407vg_fast'
NAME = 'ECG_Student_INT8_Fast_VG'
SOURCE = ROOT / 'firmware/ecg_serial_fast'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='仅重新编译未发布的快速工程。')
    args = parser.parse_args()
    if (RELEASE / 'release_manifest.json').exists():
        raise FileExistsError('快速发布包已冻结，请检查后另建版本，不覆盖。')
    if PROJECT.exists() and not args.resume:
        raise FileExistsError('快速工程已存在，可用--resume继续未发布的编译。')
    protected_dirs = [EXPORT, ROOT/'outputs/distillation/stm32_student_project_vg',
                      ROOT/'outputs/distillation/stm32_student_raw_vg_project',
                      ROOT/'outputs/distillation/board_preprocess_export',
                      ROOT/'outputs/board_release/seed44_int8_stm32f407vg',
                      ROOT/'outputs/board_release/seed44_int8_stm32f407vg_preprocess',
                      ROOT/'outputs/serial_samples/seed44_fold10', REFERENCE]
    protected = {}
    for folder in protected_dirs: protected.update(snapshot(folder))
    target = PROJECT / f'MDK_Project/{NAME}.uvprojx'
    if not PROJECT.exists():
        target = write_spl_project(REFERENCE, PROJECT, EXPORT, SDK, NAME, chip='STM32F407VG')
    user = PROJECT/'User'
    for name in ['ecg_binary.c', 'ecg_binary.h', 'uart_ring.inc']:
        shutil.copy2(SOURCE/name, user/name)
    shutil.copy2(SOURCE/'main_spl.c', user/'main.c')
    # Preserve the reference UART configuration and printf hooks; replace text ISR only.
    uart = (REFERENCE/'User/usart.c').read_bytes()
    marker = uart.find(b'u8 USART_RX_BUF[')
    assert marker >= 0
    (user/'usart.c').write_bytes(uart[:marker] + b'\n#include "uart_ring.inc"\n')
    tree = ET.parse(target)
    for file in tree.findall('.//File'):
        if file.findtext('FileName') == 'ecg_protocol.c':
            file.find('FileName').text = 'ecg_binary.c'
            file.find('FilePath').text = '..\\User\\ecg_binary.c'
    tree.write(target, encoding='utf-8', xml_declaration=True)
    for node in tree.findall('.//FilePath'):
        assert (target.parent/node.text.replace('\\','/')).resolve().is_file(), node.text
    for source in (EXPORT/'generated').iterdir():
        if source.suffix in ('.c','.h'):
            assert digest(source) == digest(user/'ai'/source.name)
    FAST_EXPORT.mkdir(parents=True, exist_ok=True)
    m = json.loads((EXPORT/'serial_manifest.json').read_text(encoding='utf-8'))
    assert m['model_sha256'] == digest(EXPORT/m['model_file'])
    for name in [m['model_file'], m['sample_file'], 'ecg_deployment_config.h']:
        shutil.copy2(EXPORT/name, FAST_EXPORT/name)
    for source in EXPORT.glob('*.txt'): shutil.copy2(source, FAST_EXPORT/source.name)
    m.update(protocol=3, preprocessing_location='host', input_kind='int8', chunk_bytes=1024,
             quantization='float32_divide_rint_ties_even_saturate', baudrate=115200,
             firmware_variant='A_fast_binary_int8', transport='binary_int8',
             original_manifest_sha256=digest(EXPORT/'serial_manifest.json'))
    (FAST_EXPORT/'serial_manifest.json').write_text(json.dumps(m, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    audit_file = target.parent/'protected_before.json'
    if audit_file.exists():
        original = json.loads(audit_file.read_text(encoding='utf-8'))
        for path, sha in original.items(): assert digest(Path(path)) == sha, path
        protected = original
    else: audit_file.write_text(json.dumps(protected, ensure_ascii=False, indent=2), encoding='utf-8')
    log = target.parent/'build_fast.log'
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW; startup.wShowWindow = subprocess.SW_HIDE
    print(f'Building {target}', flush=True)
    with (target.parent/'keil_process.log').open('w') as stream:
        result = subprocess.run(['C:/Keil_v5/UV4/UV4.exe','-r',str(target),'-j0','-o',str(log)],
                                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, startupinfo=startup,
                                creationflags=subprocess.CREATE_NO_WINDOW, timeout=300)
    text = log.read_text(errors='replace')
    counts = re.search(r'(\d+) Error\(s\),\s*(\d+) Warning\(s\)', text)
    assert result.returncode == 0 and counts and counts.groups() == ('0','0'), text[-5000:]
    mapping = (target.parent/f'Listings/{NAME}.map').read_text(errors='replace')
    flash = int(re.search(r'Total ROM Size[^\n]*?\s(\d+)\s+\(', mapping).group(1))
    ram = int(re.search(r'Total RW\s+Size[^\n]*?\s(\d+)\s+\(', mapping).group(1))
    assert flash <= 1024*1024 and ram <= 128*1024
    hex_file = target.parent/f'Objects/{NAME}.hex'
    hex_validation = audit_hex(hex_file)
    for path, sha in protected.items(): assert digest(Path(path)) == sha, path
    RELEASE.mkdir(parents=True, exist_ok=True)
    published = RELEASE/'ECG_Student_INT8_Fast_STM32F407VG.hex'
    shutil.copy2(hex_file, published)
    archive = RELEASE/'ECG_Student_INT8_Fast_VG_源码工程.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in PROJECT.rglob('*'):
            if p.is_file() and not ({'Objects','Listings'} & set(p.relative_to(PROJECT).parts)) and p.suffix not in {'.log','.bak','.uvguix'} and p.name != 'protected_before.json':
                z.write(p, Path(PROJECT.name)/p.relative_to(PROJECT))
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None
    report = {'built_at':datetime.now().isoformat(), 'chip':'STM32F407VG', 'model':'original distilled seed44 INT8',
              'model_sha256':m['model_sha256'], 'thresholds':m['thresholds'], 'protocol':3,
              'uart':{'instance':'USART1','TX':'PA9','RX':'PA10','baud':115200,'format':'8N1'},
              'clock':{'HSE_hz':8000000,'core_hz':168000000}, 'physical_board_test':False,
              'firmware':{'flash_bytes':flash,'ram_bytes':ram,'stack_bytes':8192,'project':str(target),
                          'build_log':str(log),'build_errors':0,'build_warnings':0},
              'hex_validation':hex_validation, 'original_A_B_and_fold10_unchanged':True,
              'protected_sha256':protected,
              'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in SOURCE.iterdir() if p.is_file()},
              'files':{published.name:digest(published),archive.name:digest(archive)}}
    (RELEASE/'release_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (RELEASE/'使用说明.txt').write_text('A快速版：原seed44 INT8模型，协议3二进制传输。\n'
        'STM32F407VG，HSE8MHz，168MHz；USART1 PA9 TX/PA10 RX，115200 8N1，J-Link沿用原工程。\n'
        '先用Keil烧录本目录Fast HEX，再运行根目录start_ecg_student_fast_viz.cmd。\n'
        'VS Code运行scripts/ecg_stm32_fast_viz.py，可选择折9或折10。\n'
        '上位机完成原预处理和相同的INT8量化；芯片仍只推理。旧A/B工程保留。\n'
        '只完成编译及电脑协议验证，尚未烧录快速固件或实测提速。\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['firmware','hex_validation','physical_board_test']},indent=2),flush=True)

if __name__ == '__main__': main()
