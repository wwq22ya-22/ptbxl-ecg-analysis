"""Build an independent raw-input VG project, preserving inference-only releases."""
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
from scripts.build_seed44_vg import REFERENCE, SDK, snapshot, audit_hex
from scripts.package_serial_demo import write_spl_project
from scripts.prepare_board_preprocess import OUT, OLD
from scripts.refine_student_thresholds import read, save, digest, section

PROJECT = ROOT / 'outputs/distillation/stm32_student_raw_vg_project'
RELEASE = ROOT / 'outputs/board_release/seed44_int8_stm32f407vg_preprocess'
NAME = 'ECG_Student_RAW_INT8_VG'

def record_docs(m):
    f = m['firmware']; a = m['preprocessing_validation']
    text = ['<a id="board-preprocess-two-projects"></a>', '', '## 2026-10-10：两种独立VG工程——仅推理与板端预处理＋推理', '',
            '按用户要求保留原仅推理VG工程，并新增从原始mV波形开始的完整板端处理工程。两个工程使用同一原seed44 INT8模型及原部署阈值，均沿用原三分类的VG、8 MHz HSE、168 MHz主频、USART1 PA9/PA10和J-Link配置；新seed43研究候选未接入本轮固件。', '',
            '| 项目 | A：仅推理 | B：板端预处理＋推理 |', '|---|---|---|',
            '| 上位机发送 | 预处理后的浮点波形 | 原始mV浮点波形 |',
            '| STM32执行 | 量化→模型→分类 | 双中值去基线→前后向带通→固定训练标准化→量化→模型→分类 |',
            '| 输入形状 | 十二导联×1000点，100 Hz | 相同；先接收完整10秒片段 |',
            '| 握手 | protocol=1，preprocessing=host | protocol=2，preprocessing=board，raw_mv，预处理哈希 |',
            '| MDK工程 | `outputs/distillation/stm32_student_project_vg/MDK_Project/ECG_Student_INT8_VG.uvprojx` | `outputs/distillation/stm32_student_raw_vg_project/MDK_Project/ECG_Student_RAW_INT8_VG.uvprojx` |',
            '| 直接运行Python | `scripts/ecg_stm32_viz.py` | `scripts/ecg_stm32_raw_viz.py` |',
            '| 双击启动 | `start_ecg_student_viz.cmd` | `start_ecg_board_preprocess_viz.cmd` |',
            '| 清单 | `outputs/distillation/distilled_export/serial_manifest.json` | `outputs/distillation/board_preprocess_export/serial_manifest.json` |',
            '| 发布包 | `outputs/board_release/seed44_int8_stm32f407vg/` | `outputs/board_release/seed44_int8_stm32f407vg_preprocess/` |',
            f'| 完整Flash | 158876 B / 155.15 KiB | {f["flash_bytes"]} B / {f["flash_bytes"]/1024:.2f} KiB |',
            f'| 完整主SRAM，含8 KiB栈 | 43672 B / 42.65 KiB | {f["ram_bytes"]} B / {f["ram_bytes"]/1024:.2f} KiB |', '',
            '### 本轮预处理实现及一致性', '',
            'C实现的双中值窗口为21/61点，边界使用reflect；带通为4阶0.5–40 Hz Butterworth、四组SOS、前后向计算、odd延拓27点。滤波系数和初始状态冻结自SciPy；使用原训练集逐导联均值/标准差，没有在回放数据上重新拟合。保留Python各阶段的float32舍入，SOS与标准化中间计算用double。', '',
            f'原始波形缓冲48000 B；复用的单导联预处理工作区{a["workspace_bytes"]} B。两者均已计入链接map的完整RAM，不能重复相加。B工程仍是10秒整段处理，尚未改为连续逐点流式算法。M4的double运算性能需要板上计时。', '',
            f'27项预处理/原新串口/芯片配置测试通过。真实验证折9的{a["records"]}条ECG用实际C源码在电脑复核：相对Python原始float32输入的最大误差{a["host_C_vs_python_raw32_max_absolute_error"]}，相对原训练流程缓存最大误差{a["host_C_vs_original_training_cache_max_absolute_error"]}；INT8差异值均为{a["int8_different_values_vs_original_cache"]}，串口接收量化输入差异{a["serial_parser_int8_mismatches"]}。分类与原电脑C参考一致率{a["classification_agreement_with_original_host_C"]*100:.2f}%，最大概率差{a["max_probability_difference_from_original_host_C"]:.9g}。', '',
            '这批检查是GCC主机数值及协议模拟，不是ARMCC实板数值、模型新测试集成绩或板上耗时。完整新固件已重新编译0错误/0警告，HEX校验和、VG地址范围和向量表通过。真实板端一致性、预处理耗时、推理耗时及栈峰值仍待连接开发板验证。', '',
            '### 运行与防止混用', '',
            'A烧录仅推理HEX，运行`ecg_stm32_viz.py`；B烧录RAW预处理HEX，运行`ecg_stm32_raw_viz.py`。各自界面标题及发送说明明确显示工程类型。二者均可在VS Code已选项目`.venv`后直接点击运行，或双击各自CMD。选择真实串口、实际COM和115200；CPU时钟及J-Link设置沿用原工程。', '',
            '两种固件即使模型哈希相同也会因协议/预处理位置不同拒绝错误上位机。B另核对预处理参数及实现哈希。CRC和完整样本数通过后才执行预处理；不完整、非有限输入和错误CRC不会触发推理。返回结果分别记录`preprocessing_ms`和`inference_ms`，模拟模式不登记芯片时间。', '',
            '- [B工程](../outputs/distillation/stm32_student_raw_vg_project/MDK_Project/ECG_Student_RAW_INT8_VG.uvprojx)',
            '- [B HEX](../outputs/board_release/seed44_int8_stm32f407vg_preprocess/ECG_Student_RAW_INT8_STM32F407VG.hex)',
            '- [B完整源码ZIP](../outputs/board_release/seed44_int8_stm32f407vg_preprocess/ECG_Student_RAW_INT8_VG_源码工程.zip)',
            '- [B发布清单](../outputs/board_release/seed44_int8_stm32f407vg_preprocess/release_manifest.json)',
            '- [C预处理与协议验证记录](../outputs/distillation/board_preprocess_export/preprocessing_validation.json)', '',
            '源码：`firmware/ecg_preprocess/`及带`ECG_BOARD_PREPROCESSING`条件分支的`firmware/ecg_serial/`。冻结配置：`scripts/prepare_board_preprocess.py`；主机复核：`scripts/verify_board_preprocessing.py`；独立B工程编译：`scripts/build_seed44_board_preprocess.py`。发布目录存在时构建脚本拒绝覆盖。原参考工程、原A/VE工程及发布包保持哈希一致。']
    section(ROOT / 'md/STM32部署与串口验证.md', '\n'.join(text), 'board-preprocessing-two-projects-20261010')
    section(ROOT / 'md/README.md', '2026-10-10补齐原seed44的完整板端预处理，保留两个独立VG工程：A仅推理使用`ecg_stm32_viz.py` / `start_ecg_student_viz.cmd`；B发送原始波形、板端预处理＋推理使用`ecg_stm32_raw_viz.py` / `start_ecg_board_preprocess_viz.cmd`。B工程在`outputs/distillation/stm32_student_raw_vg_project/`，发布包在`outputs/board_release/seed44_int8_stm32f407vg_preprocess/`。两个包协议不同，禁止混用。均已编译；C预处理与128条真实回放在主机核对通过，尚未实板验证。详见[部署文档](STM32部署与串口验证.md)。', 'board-preprocessing-two-projects-20261010')

def main():
    if RELEASE.exists(): raise FileExistsError(f'Inspect existing release before rebuilding: {RELEASE}')
    m = read(OUT / 'serial_manifest.json'); report = read(OUT / 'preprocessing_validation.json')
    assert report['model_sha256'] == m['model_sha256'] and report['preprocessing_sha256'] == m['preprocessing_sha256']
    assert report['int8_different_values_vs_python_raw32'] == 0 and report['classification_agreement_with_original_host_C'] == 1.
    assert digest(ROOT / 'firmware/ecg_preprocess/ecg_preprocess.c') == m['preprocessing_implementation']['source_sha256']
    protected = {**snapshot(ROOT / 'outputs/distillation/stm32_student_project_vg'),
                 **snapshot(ROOT / 'outputs/board_release/seed44_int8_stm32f407vg'),
                 **snapshot(ROOT / 'outputs/board_release/seed44_int8_stm32f407ve'),
                 **snapshot(REFERENCE), **snapshot(OLD)}
    protocol_file = OUT / 'build_protocol.json'
    if protocol_file.exists(): assert read(protocol_file)['protected_sha256'] == protected
    save(protocol_file, {'protected_sha256': protected, 'preprocessing_sha256': m['preprocessing_sha256'], 'model_sha256': m['model_sha256']})
    target = PROJECT / f'MDK_Project/{NAME}.uvprojx'
    if not PROJECT.exists():
        target = write_spl_project(REFERENCE, PROJECT, OLD, SDK, NAME, chip='STM32F407VG')
        for name in ['ecg_preprocess.c', 'ecg_preprocess.h']:
            shutil.copy2(ROOT / 'firmware/ecg_preprocess' / name, PROJECT / 'User' / name)
        shutil.copy2(OUT / 'ecg_preprocess_config.h', PROJECT / 'User/ecg_preprocess_config.h')
        xml = ET.parse(target); root = xml.getroot()
        root.find('.//Cads/VariousControls/Define').text += ',ECG_BOARD_PREPROCESSING=1'
        group = ET.SubElement(root.find('.//Groups'), 'Group')
        ET.SubElement(group, 'GroupName').text = 'ECG preprocessing'
        files = ET.SubElement(group, 'Files'); item = ET.SubElement(files, 'File')
        for tag, value in [('FileName','ecg_preprocess.c'),('FileType','1'),('FilePath','..\\User\\ecg_preprocess.c')]:
            ET.SubElement(item, tag).text = value
        ET.indent(xml); xml.write(target, encoding='UTF-8', xml_declaration=True)
    else:
        for name in ['ecg_preprocess.c', 'ecg_preprocess.h']:
            assert digest(PROJECT / 'User' / name) == digest(ROOT / 'firmware/ecg_preprocess' / name)
        assert digest(PROJECT / 'User/ecg_preprocess_config.h') == digest(OUT / 'ecg_preprocess_config.h')
    root = ET.parse(target).getroot()
    assert 'ECG_BOARD_PREPROCESSING=1' in root.findtext('.//Cads/VariousControls/Define')
    assert root.findtext('.//Device') == 'STM32F407VGTx'
    for node in root.findall('.//FilePath'): assert (target.parent / node.text.replace('\\','/')).resolve().is_file()
    startup = subprocess.STARTUPINFO(); startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW; startup.wShowWindow = subprocess.SW_HIDE
    log = target.parent / 'build_raw_vg_20261010.log'
    print(f'Full rebuild: {target}', flush=True)
    with (target.parent / 'keil_process.log').open('w') as stream:
        result = subprocess.run(['C:/Keil_v5/UV4/UV4.exe','-r',str(target),'-j0','-o',str(log)],cwd=ROOT,
                                stdout=stream,stderr=subprocess.STDOUT,startupinfo=startup,
                                creationflags=subprocess.CREATE_NO_WINDOW,timeout=300)
    text = log.read_text(errors='replace'); counts = re.search(r'(\d+) Error\(s\),\s*(\d+) Warning\(s\)',text)
    assert result.returncode == 0 and counts and counts.groups() == ('0','0'), text[-5000:]
    mapping = (target.parent / f'Listings/{NAME}.map').read_text(errors='replace')
    flash = int(re.search(r'Total ROM Size[^\n]*?\s(\d+)\s+\(',mapping).group(1))
    ram = int(re.search(r'Total RW\s+Size[^\n]*?\s(\d+)\s+\(',mapping).group(1))
    assert flash <= 1024*1024 and ram <= 128*1024
    hex_file = target.parent / f'Objects/{NAME}.hex'; hex_result = audit_hex(hex_file)
    for path, sha in protected.items(): assert digest(Path(path)) == sha
    RELEASE.mkdir(parents=True)
    published_hex = RELEASE / 'ECG_Student_RAW_INT8_STM32F407VG.hex'; shutil.copy2(hex_file,published_hex)
    archive = RELEASE / 'ECG_Student_RAW_INT8_VG_源码工程.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in PROJECT.rglob('*'):
            if p.is_file() and not ({'Objects','Listings'} & set(p.relative_to(PROJECT).parts)) and p.suffix not in {'.log','.uvguix','.bak'}:
                z.write(p,Path(PROJECT.name)/p.relative_to(PROJECT))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for node in root.findall('.//FilePath'):
            p = (target.parent / node.text.replace('\\','/')).resolve()
            assert z.read(PROJECT.name+'/'+p.relative_to(PROJECT).as_posix()) == p.read_bytes()
    release = {'date':'2026-10-10','built_at':datetime.now().isoformat(),'chip':'STM32F407VG',
               'pipeline':'raw_mV -> board baseline + zero-phase bandpass + train_global -> INT8 -> five-label inference',
               'protocol':2,'model':'original distilled seed44 INT8','model_sha256':m['model_sha256'],
               'preprocessing_sha256':m['preprocessing_sha256'],'thresholds':m['thresholds'],
               'serial_manifest':str(OUT/'serial_manifest.json'),'preprocessing_validation':report,
               'tests_passed':27,'firmware':{'project':str(target),'build_log':str(log),
                  'flash_bytes':flash,'ram_bytes':ram,'flash_limit_bytes':1024*1024,'main_sram_limit_bytes':128*1024,
                  'main_sram_margin_bytes':128*1024-ram,'stack_bytes':8192,'build_errors':0,'build_warnings':0},
               'clock':{'HSE_hz':8000000,'core_hz':168000000},'UART':'USART1 PA9/PA10 115200 8N1',
               'hex_validation':hex_result,'original_inference_only_and_reference_unchanged':True,
               'protected_sha256':protected,'physical_board_test':False,
               'files':{published_hex.name:digest(published_hex),archive.name:digest(archive)}}
    save(RELEASE/'release_manifest.json',release)
    (RELEASE/'使用说明.txt').write_text('B工程：板端预处理＋推理，原seed44 INT8，STM32F407VG。\n'
        '先接收完整12导联×1000点原始mV波形，板端执行双中值去基线、0.5–40 Hz前后向带通、训练统计标准化、量化和模型。\n'
        '烧录ECG_Student_RAW_INT8_STM32F407VG.hex；在VS Code直接运行scripts/ecg_stm32_raw_viz.py或双击start_ecg_board_preprocess_viz.cmd。\n'
        '旧A仅推理工程仍用ecg_stm32_viz.py/start_ecg_student_viz.cmd，二者协议不同，握手拒绝混用。\n'
        '沿用原三分类VG、8 MHz HSE、168 MHz主频、J-Link设置，USART1 PA9 TX/PA10 RX，115200 8N1，3.3 V TTL共地。\n'
        '本轮已编译和主机核对，尚未烧录实板或测量板上预处理/推理耗时。\n',encoding='utf-8')
    record_docs(release)
    print(json.dumps({'firmware':release['firmware'],'hex_validation':hex_result,'physical_board_test':False},indent=2),flush=True)
if __name__ == '__main__': main()
