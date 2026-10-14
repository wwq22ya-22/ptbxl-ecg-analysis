"""Independently check the released raw-input MDK project and old artifact hashes."""
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_seed44_vg import REFERENCE, audit_hex
from scripts.build_seed44_board_preprocess import PROJECT, RELEASE, NAME
from scripts.prepare_board_preprocess import OUT, OLD
from scripts.refine_student_thresholds import read, save, digest, section

def nodes(element):
    return [(n.tag, (n.text or '').strip()) for n in element.iter()]

def main():
    manifest = read(RELEASE / 'release_manifest.json')
    serial = read(OUT / 'serial_manifest.json')
    protocol = read(OUT / 'build_protocol.json')
    assert manifest['protected_sha256'] == protocol['protected_sha256']
    for path, expected in manifest['protected_sha256'].items():
        assert digest(Path(path)) == expected, path
    for name, expected in manifest['files'].items():
        assert digest(RELEASE / name) == expected, name
    assert serial['model_sha256'] == digest(OUT / serial['model_file']) == digest(OLD / serial['model_file'])
    assert serial['preprocessing_sha256'] == manifest['preprocessing_sha256'] == protocol['preprocessing_sha256']
    assert serial['protocol'] == manifest['protocol'] == 2 and serial['preprocessing_location'] == 'board'
    assert serial['thresholds'] == read(OLD / 'serial_manifest.json')['thresholds']
    assert serial['preprocessing_implementation']['source_sha256'] == digest(ROOT / 'firmware/ecg_preprocess/ecg_preprocess.c')
    for name in ['ecg_preprocess.c', 'ecg_preprocess.h']:
        assert digest(PROJECT / 'User' / name) == digest(ROOT / 'firmware/ecg_preprocess' / name)
    for name in ['ecg_protocol.c', 'ecg_protocol.h', 'ecg_tick.h']:
        assert digest(PROJECT / 'User' / name) == digest(ROOT / 'firmware/ecg_serial' / name)
    assert digest(PROJECT / 'User/main.c') == digest(ROOT / 'firmware/ecg_serial/main_spl.c')
    assert digest(PROJECT / 'User/ecg_preprocess_config.h') == digest(OUT / 'ecg_preprocess_config.h')
    assert serial['preprocessing_sha256'] in (PROJECT / 'User/ecg_preprocess_config.h').read_text()
    assert digest(PROJECT / 'User/ecg_deployment_config.h') == digest(OLD / 'ecg_deployment_config.h')
    a_project = ROOT / 'outputs/distillation/stm32_student_project_vg'
    ai_files = list((a_project / 'User/ai').rglob('*'))
    for source in ai_files:
        if source.is_file():
            assert digest(source) == digest(PROJECT / source.relative_to(a_project))
    target = PROJECT / f'MDK_Project/{NAME}.uvprojx'
    current, reference = [ET.parse(p).getroot() for p in [target, REFERENCE / 'MDK_Project/Project.uvprojx']]
    for tag in ['Device', 'Cpu', 'FlashDriverDll', 'RegisterFile', 'SFDFile', 'PackID']:
        assert current.findtext('.//' + tag) == reference.findtext('.//' + tag)
    assert current.findtext('.//Device') == 'STM32F407VGTx'
    assert 'ECG_BOARD_PREPROCESSING=1' in current.findtext('.//Cads/VariousControls/Define')
    assert nodes(current.find('.//Utilities')) == nodes(reference.find('.//Utilities'))
    old_opt, new_opt = [ET.parse(p).getroot() for p in [REFERENCE / 'MDK_Project/Project.uvoptx', target.with_suffix('.uvoptx')]]
    for part in ['DebugOpt', 'TargetDriverDllRegistry']:
        assert nodes(old_opt.find('.//' + part)) == nodes(new_opt.find('.//' + part))
    for name in ['system_stm32f4xx.c', 'stm32f4xx.h']:
        assert digest(PROJECT / 'Lib/cmsis' / name) == digest(REFERENCE / 'Lib/cmsis' / name)
    mapping = (target.parent / f'Listings/{NAME}.map').read_text(errors='replace')
    flash = int(re.search(r'Total ROM Size[^\n]*?\s(\d+)\s+\(', mapping).group(1))
    ram = int(re.search(r'Total RW\s+Size[^\n]*?\s(\d+)\s+\(', mapping).group(1))
    assert (flash, ram) == (manifest['firmware']['flash_bytes'], manifest['firmware']['ram_bytes'])
    hex_path = RELEASE / 'ECG_Student_RAW_INT8_STM32F407VG.hex'
    assert digest(hex_path) == digest(target.parent / f'Objects/{NAME}.hex')
    assert audit_hex(hex_path) == manifest['hex_validation']
    references = current.findall('.//FilePath')
    with zipfile.ZipFile(RELEASE / 'ECG_Student_RAW_INT8_VG_源码工程.zip') as archive:
        assert archive.testzip() is None
        for node in references:
            source = (target.parent / node.text.replace('\\', '/')).resolve()
            assert archive.read(PROJECT.name + '/' + source.relative_to(PROJECT).as_posix()) == source.read_bytes()
        for source in [target, target.with_suffix('.uvoptx'), PROJECT / 'User/ecg_preprocess_config.h']:
            assert archive.read(PROJECT.name + '/' + source.relative_to(PROJECT).as_posix()) == source.read_bytes()
    application = read(OUT / 'application_checks.json')
    assert len(application['GUI']) == 2 and all(r['GUI_initialized_and_waveform_rendered'] for r in application['GUI'])
    old_result = read(OUT / 'inference_only_entry_simulation.json')
    new_result = read(OUT / 'raw_entry_simulation.json')
    assert old_result['mode'] == new_result['mode'] == 'simulation'
    assert old_result['probabilities'] == new_result['probabilities'] and old_result['mask'] == new_result['mask']
    assert 'preprocessing_ms' not in old_result and new_result['preprocessing_ms'] == 0
    result = {'project_source_and_library_references_verified_in_zip': len(references),
              'JLink_and_clock_settings_identical_to_original_reference': True,
              'generated_AI_files_and_thresholds_identical_to_original_seed44': True,
              'preprocessing_and_protocol_sources_match_verified_C': True,
              'original_inference_only_VG_VE_and_reference_artifacts_unchanged': True,
              'HEX_checksum_vectors_addresses_and_compiled_map_verified': True,
              'both_GUI_and_default_headless_simulation_entries_verified': True,
              'model_sha256': serial['model_sha256'], 'preprocessing_sha256': serial['preprocessing_sha256'],
              'flash_bytes': flash, 'ram_bytes': ram, 'physical_board_test': False}
    save(RELEASE / 'independent_audit.json', result)
    section(ROOT / 'md/STM32部署与串口验证.md',
            f'发布后的独立复核已通过：源码ZIP内{len(references)}个工程源码/库引用均存在且与编译工程一致，J-Link与时钟配置沿用原参考；C预处理、协议、部署头与冻结版本一致；HEX与map一致；原A、VE包及参考工程保护哈希均未变化。两个Python默认入口均通过界面初始化/绘图和单条模拟收发，验证日志按各自清单目录保存，并记录协议及预处理位置。见[独立复核记录](../outputs/board_release/seed44_int8_stm32f407vg_preprocess/independent_audit.json)及[上位机检查](../outputs/distillation/board_preprocess_export/application_checks.json)。复核入口：`scripts/audit_board_preprocess_release.py`。',
            'board-preprocessing-independent-audit-20261010')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
