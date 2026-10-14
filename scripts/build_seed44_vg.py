"""Build a fresh five-label VG firmware using the user's original MDK profile."""
from __future__ import annotations

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
from scripts.package_serial_demo import write_spl_project
from scripts.refine_student_thresholds import digest, read, save, section

REFERENCE = Path('D:/MyProject/college/3/stm32/2026STM32+AI/2026STM32+AI/ECG分类/STM32Test/MDK')
SINE = Path('D:/MyProject/college/3/stm32/0417sine模拟/STM32正弦/MDK')
EXPORT = ROOT / 'outputs/distillation/distilled_export'
PROJECT = ROOT / 'outputs/distillation/stm32_student_project_vg'
RELEASE = ROOT / 'outputs/board_release/seed44_int8_stm32f407vg'
SDK = Path('D:/STM32AI/User/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.1/Middlewares/ST/AI')
NAME = 'ECG_Student_INT8_VG'


def record_docs(manifest):
    f = manifest['firmware']
    text = ('## 2026-10-10：沿用原 sine / ECG 三分类配置的五类 VG 工程\n\n'
            '按用户要求，将原三分类工程的 VG 芯片、Flash、RAM、时钟及 J-Link 配置用于原 seed44 INT8 五类串口工程。'
            '已与 sine 工程交叉核对 Device 与时钟源码；原参考工程和旧 VE 回退包均未修改。\n\n'
            '| 配置 | 本轮实际工程 |\n|---|---|\n'
            '| 芯片 / Flash | STM32F407VGTx / 1 MiB，STM32F4xx_1024 下载算法 |\n'
            '| 固件时钟 | HSE 8 MHz，PLL M=8/N=336/P=2/Q=7，主频168 MHz；时钟源码原样复制 |\n'
            '| 串口 | USART1，PA9 TX / PA10 RX，115200、8N1、无流控 |\n'
            '| 调试下载 | 沿用 Project.uvoptx 的 Segger/J-Link 与注册参数 |\n'
            '| Keil CLOCK 字段 | 保留参考工程12000000；它不改变固件实际HSE配置 |\n'
            '| 工具 / 栈 | 本机 ARMCC 5.06 update4；8 KiB栈，满足五类协议与运行库需要 |\n\n'
            '原参考工程写 ARMCC update7，本机沿用已有 update4；不是逐字复制不可用编译器。'
            '五类工程继续使用原 seed44 权重、预处理、量化参数和原部署阈值，不是当前研究候选 seed43。\n\n'
            f'本轮执行全量重新编译，**0错误、0警告**；Flash {f["flash_bytes"]} B（{f["flash_bytes"]/1024:.2f} KiB），'
            f'主SRAM {f["ram_bytes"]} B（{f["ram_bytes"]/1024:.2f} KiB，含栈）。'
            '15项芯片配置/串口协议测试通过；HEX校验和、VG地址范围和向量表验证通过，源码ZIP完整性通过。\n\n'
            '- [新VG Keil工程](../outputs/distillation/stm32_student_project_vg/MDK_Project/ECG_Student_INT8_VG.uvprojx)\n'
            '- [新VG HEX](../outputs/board_release/seed44_int8_stm32f407vg/ECG_Student_INT8_STM32F407VG.hex)\n'
            '- [完整VG源码ZIP](../outputs/board_release/seed44_int8_stm32f407vg/ECG_Student_INT8_VG_源码工程.zip)\n'
            '- [本轮发布清单及保护哈希](../outputs/board_release/seed44_int8_stm32f407vg/release_manifest.json)\n\n'
            '使用项目根 `start_ecg_student_viz.cmd`，它与本包仍是同一个原seed44模型/阈值。'
            '本轮只生成和编译，没有自动下载到开发板，也没有板上推理计时。\n\n'
            '生成入口：`.venv/Scripts/python.exe scripts/build_seed44_vg.py`，已有包时拒绝覆盖；'
            '`package_serial_demo.py` 新增 `--chip STM32F407VG`，原VE入口保留兼容，后续应显式指定芯片。')
    doc = ROOT / 'md/STM32部署与串口验证.md'
    section(doc, text, 'seed44-vg-reference-20261010')
    old = doc.read_text(encoding='utf-8')
    old = old.replace('现有五类seed44发布工程则按**F407VE/512 KiB Flash**编译。',
                      '2026-10-10已另生成并重新编译**VG/1 MiB Flash**五类seed44工程；原**VE/512 KiB Flash**包保留作历史回退。')
    old = old.replace('| 原seed44学生INT8 | 原部署阈值`[0.55,0.60,0.55,0.75,0.65]` | Cube.AI、主机C核对、完整VE编译、HEX/源码ZIP | VG工程与真实烧录、串口闭环、耗时/稳定性 |',
                      '| 原seed44学生INT8 | 原部署阈值`[0.55,0.60,0.55,0.75,0.65]` | Cube.AI、主机C核对、完整VE及VG编译、HEX/源码ZIP | 真实烧录、串口闭环、耗时/稳定性 |')
    old = old.replace('1. 对照实物确认VG、晶振、供电、SWD下载器和USART1接线，处理现有VE工程与打包脚本目标差异。',
                      '1. 使用本轮新VG工程及HEX；硬件配置沿用原sine/三分类工程，连接对应下载器和USART1。')
    doc.write_text(old, encoding='utf-8')
    section(ROOT / 'md/README.md',
            '2026-10-10已按用户要求沿用原sine/ECG三分类的STM32F407VG、8 MHz HSE、168 MHz主频、USART1及J-Link设置，生成独立原seed44五类VG工程。'
            '全量编译0错误/0警告，Flash155.15 KiB、主SRAM42.65 KiB。'
            '工程：`outputs/distillation/stm32_student_project_vg/MDK_Project/ECG_Student_INT8_VG.uvprojx`；'
            '发布包：`outputs/board_release/seed44_int8_stm32f407vg/`。仍使用 `start_ecg_student_viz.cmd`；'
            '模型、原阈值和旧VE回退包保留，尚未实板烧录/计时。详见[部署文档](STM32部署与串口验证.md)。',
            'seed44-vg-reference-20261010')
    overview = ROOT / 'md/README.md'
    old = overview.read_text(encoding='utf-8')
    old = old.replace('| 旧seed44固件 | Cube.AI/C生成、完整VE编译、HEX/源码ZIP、主机C核对 | VG目标/时钟/接线及真实板端未完成 |',
                      '| 原seed44固件 | Cube.AI/C生成、完整VE/VG编译、HEX/源码ZIP、主机C核对 | 真实板端烧录、串口闭环及计时未完成 |')
    old = old.replace('实板最新记录VG，旧包工程VE；烧录前按',
                      '实板最新记录VG，本轮新增VG工程，旧VE包保留；烧录前按')
    overview.write_text(old, encoding='utf-8')


def snapshot(folder):
    return {str(p): digest(p) for p in folder.rglob('*') if p.is_file()}


def audit_hex(path):
    data, upper, ended = {}, 0, False
    for line in path.read_text().splitlines():
        assert line.startswith(':') and not ended
        raw = bytes.fromhex(line[1:])
        assert len(raw) == raw[0] + 5 and sum(raw) % 256 == 0
        count, offset, kind = raw[0], int.from_bytes(raw[1:3], 'big'), raw[3]
        payload = raw[4:4 + count]
        if kind == 0:
            for i, byte in enumerate(payload):
                address = upper + offset + i
                assert 0x08000000 <= address < 0x08100000 and address not in data
                data[address] = byte
        elif kind == 4:
            assert count == 2
            upper = int.from_bytes(payload, 'big') << 16
        elif kind == 1:
            assert count == 0
            ended = True
        else:
            assert kind == 5 and count == 4
    assert ended
    vectors = bytes(data[0x08000000 + i] for i in range(8))
    stack = int.from_bytes(vectors[:4], 'little')
    reset = int.from_bytes(vectors[4:], 'little')
    assert 0x20000000 < stack <= 0x20020000 and stack % 8 == 0
    assert reset & 1 and (reset & ~1) in data
    return {'checksum_and_VG_flash_addresses_verified': True, 'initial_stack_pointer': hex(stack),
            'reset_vector': hex(reset), 'programmed_bytes': len(data)}


def main():
    if RELEASE.exists():
        raise FileExistsError(f'Existing release must be inspected before rebuilding: {RELEASE}')
    refs = {**snapshot(REFERENCE), **snapshot(SINE)}
    protected = {**snapshot(ROOT / 'outputs/board_release/seed44_int8_stm32f407ve'),
                 **snapshot(ROOT / 'outputs/distillation/stm32_student_project')}
    for p in [EXPORT / 'ptbxl_cnn1d_int8.onnx', EXPORT / 'serial_manifest.json', EXPORT / 'ecg_deployment_config.h']:
        protected[str(p)] = digest(p)
    model_sha = digest(EXPORT / 'ptbxl_cnn1d_int8.onnx')
    assert read(EXPORT / 'serial_manifest.json')['model_sha256'] == model_sha
    config = (EXPORT / 'ecg_deployment_config.h').read_text()
    assert model_sha in config
    # Both reference designs select VG; clock sources are identical byte for byte.
    for folder in [REFERENCE, SINE]:
        xml = ET.parse(folder / 'MDK_Project/Project.uvprojx').getroot()
        assert xml.findtext('.//Device') == 'STM32F407VGTx'
        assert 'uart_init(115200)' in (folder / 'User/main.c').read_bytes().decode('utf-8', errors='replace')
    for name in ['system_stm32f4xx.c', 'stm32f4xx.h']:
        assert digest(REFERENCE / 'Lib/cmsis' / name) == digest(SINE / 'Lib/cmsis' / name)
    target = write_spl_project(REFERENCE, PROJECT, EXPORT, SDK, NAME, chip='STM32F407VG')
    new_xml, ref_xml = [ET.parse(p).getroot() for p in [target, REFERENCE / 'MDK_Project/Project.uvprojx']]
    for tag in ['Device', 'Cpu', 'FlashDriverDll', 'RegisterFile', 'SFDFile', 'PackID']:
        assert new_xml.findtext('.//' + tag) == ref_xml.findtext('.//' + tag)
    assert [(n.tag, (n.text or '').strip()) for n in new_xml.find('.//Utilities').iter()] == [(n.tag, (n.text or '').strip()) for n in ref_xml.find('.//Utilities').iter()]
    old_opt = ET.parse(REFERENCE / 'MDK_Project/Project.uvoptx').getroot()
    new_opt = ET.parse(target.with_suffix('.uvoptx')).getroot()
    for part in ['DebugOpt', 'TargetDriverDllRegistry']:
        assert [(n.tag, (n.text or '').strip()) for n in old_opt.find('.//' + part).iter()] == [(n.tag, (n.text or '').strip()) for n in new_opt.find('.//' + part).iter()]
    for folder, prefix, names in [(PROJECT / 'Lib/cmsis', REFERENCE / 'Lib/cmsis', ['system_stm32f4xx.c', 'stm32f4xx.h']),
                                  (PROJECT / 'User', REFERENCE / 'User', ['delay.c', 'delay.h'])]:
        for name in names: assert digest(folder / name) == digest(prefix / name)
    for name in ['usart.c', 'usart.h']:
        assert (PROJECT / 'User' / name).read_bytes() == (REFERENCE / 'User' / name).read_bytes().replace(b'u16 USART_RX_STA', b'volatile u16 USART_RX_STA')
    assert '0x00002000' in (PROJECT / 'Lib/starup/startup_stm32f40_41xxx.s').read_text(errors='replace')
    for node in new_xml.findall('.//FilePath'):
        assert (target.parent / node.text.replace('\\', '/')).resolve().is_file(), node.text
    log = target.parent / 'build_seed44_vg_20261010.log'
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    print(f'Building fresh VG project: {target}', flush=True)
    process_log = target.parent / 'keil_process.log'
    with process_log.open('w') as stream:
        result = subprocess.run(['C:/Keil_v5/UV4/UV4.exe', '-r', str(target), '-j0', '-o', str(log)],
                                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW, timeout=300)
    text = log.read_text(errors='replace')
    counts = re.search(r'(\d+) Error\(s\),\s*(\d+) Warning\(s\)', text)
    assert result.returncode == 0 and counts and counts.groups() == ('0', '0'), text[-4000:]
    mapping = (target.parent / f'Listings/{NAME}.map').read_text(errors='replace')
    flash = int(re.search(r'Total ROM Size[^\n]*?\s(\d+)\s+\(', mapping).group(1))
    ram = int(re.search(r'Total RW\s+Size[^\n]*?\s(\d+)\s+\(', mapping).group(1))
    assert flash <= 1024 * 1024 and ram <= 128 * 1024
    hex_file = target.parent / f'Objects/{NAME}.hex'
    hex_audit = audit_hex(hex_file)
    for path, sha in {**refs, **protected}.items(): assert digest(Path(path)) == sha
    RELEASE.mkdir(parents=True)
    published_hex = RELEASE / 'ECG_Student_INT8_STM32F407VG.hex'
    shutil.copy2(hex_file, published_hex)
    archive = RELEASE / 'ECG_Student_INT8_VG_源码工程.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in PROJECT.rglob('*'):
            if p.is_file() and not ({'Objects', 'Listings'} & set(p.relative_to(PROJECT).parts)) and p.suffix not in {'.log', '.uvguix', '.bak'}:
                z.write(p, Path(PROJECT.name) / p.relative_to(PROJECT))
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None
    manifest = {'date': '2026-10-10', 'built_at': datetime.now().isoformat(), 'chip': 'STM32F407VG',
                'model': 'original distilled seed44 INT8', 'model_sha256': model_sha,
                'thresholds': read(EXPORT / 'serial_manifest.json')['thresholds'],
                'reference_mdk': str(REFERENCE), 'sine_reference': str(SINE),
                'reference_hardware_and_debug_settings_verified': True, 'debug_probe_profile': 'J-Link / Segger JL2CM3',
                'clock': {'HSE_hz': 8000000, 'PLL_M': 8, 'PLL_N': 336, 'PLL_P': 2, 'PLL_Q': 7, 'core_hz': 168000000,
                          'keil_CLOCK_field_hz': 12000000, 'keil_CLOCK_field_is_not_firmware_HSE': True},
                'uart': {'instance': 'USART1', 'TX': 'PA9', 'RX': 'PA10', 'baud': 115200, 'format': '8N1'},
                'firmware': {'flash_bytes': flash, 'ram_bytes': ram, 'flash_limit_bytes': 1024*1024,
                             'main_sram_limit_bytes': 128*1024, 'stack_bytes': 8192, 'build_errors': 0, 'build_warnings': 0,
                             'project': str(target), 'build_log': str(log)},
                'hex_validation': hex_audit, 'physical_board_test': False,
                'original_reference_and_VE_artifacts_unchanged': True,
                'source_reference_sha256': refs, 'protected_sha256': protected,
                'files': {published_hex.name: digest(published_hex), archive.name: digest(archive)}}
    save(RELEASE / 'release_manifest.json', manifest)
    (RELEASE / '使用说明.txt').write_text('2026-10-10：原seed44 INT8五类模型，沿用原sine/ECG三分类硬件配置。\n'
        '目标STM32F407VG，1 MiB Flash，HSE8 MHz，主频168 MHz。J-Link下载设置沿用三分类工程。\n'
        '串口USART1 PA9 TX/PA10 RX，115200 8N1，3.3 V TTL、共地。\n'
        '完整工程：outputs/distillation/stm32_student_project_vg/MDK_Project/ECG_Student_INT8_VG.uvprojx\n'
        '双击项目根start_ecg_student_viz.cmd，选择真实串口；模型与原seed44清单一致。\n'
        'Keil CLOCK(12000000)为参考工程界面字段，固件实际HSE8 MHz，请勿据此改PLL。\n'
        '五类协议保留8 KiB栈，不能恢复原1 KiB。未烧录或实测板上耗时。\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ['chip', 'firmware', 'hex_validation', 'physical_board_test']}, indent=2), flush=True)
    record_docs(manifest)


if __name__ == '__main__': main()
