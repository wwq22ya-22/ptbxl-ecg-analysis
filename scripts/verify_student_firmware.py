"""Build the distilled student's independent serial firmware and verify host C."""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_stm32 import digest

OUT=ROOT/'outputs/distillation/distilled_export'
PROJECT=ROOT/'outputs/distillation/stm32_student_project'
CLI=Path('D:/STM32AI/User/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.1/Utilities/windows/stedgeai.exe')
SDK=CLI.parents[2]/'Middlewares/ST/AI'
REFERENCE=Path('D:/MyProject/college/3/stm32/2026STM32+AI/2026STM32+AI/ECG分类/STM32Test/MDK')


def run(args, log):
    print('Running:',Path(args[0]).name,flush=True)
    with log.open('w',encoding='utf-8') as f:
        result=subprocess.run([str(x) for x in args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Failed: {log}')


def main():
    source=np.load(OUT/'validation_reference.npz')
    np.save(OUT/'validation_input.npy',source['ecg'])
    model=OUT/'ptbxl_cnn1d_int8.onnx'
    base=[CLI,'--model',model,'--target','stm32f4','--name','ecg_network','--c-api','legacy',
          '--compression','none','--optimization','ram','--inputs-ch-position','chfirst','--output-data-type','float32','--quiet']
    if not (OUT/'generated/ecg_network.c').exists():
        run([base[0],'generate',*base[1:],'--workspace',OUT/'generated_workspace','--output',OUT/'generated'],OUT/'student_generate.log')
    if not (OUT/'host_validation/ecg_network_val_io.npz').exists():
        run([base[0],'validate',*base[1:],'--mode','host','--valinput',OUT/'validation_input.npy',
             '--workspace',OUT/'host_workspace','--output',OUT/'host_validation'],OUT/'student_host_validate.log')
    io=np.load(OUT/'host_validation/ecg_network_val_io.npz')
    print('Host validation arrays:',{k:io[k].shape for k in io.files},flush=True)
    if not PROJECT.exists():
        run([sys.executable,ROOT/'scripts/package_serial_demo.py','--export-dir',OUT,'--reference-mdk',REFERENCE,
             '--sdk',SDK,'--project-dir',PROJECT,'--project-name','ECG_Student_INT8'],OUT/'student_package.log')
    build=PROJECT/'MDK_Project/student_build.log'
    run(['C:/Keil_v5/UV4/UV4.exe','-b',PROJECT/'MDK_Project/ECG_Student_INT8.uvprojx','-j0','-o',build],OUT/'student_keil_process.log')
    text=build.read_text(errors='replace')
    errors=re.search(r'(\d+) Error\(s\),\s*(\d+) Warning\(s\)',text)
    assert errors and int(errors.group(1))==0, text
    mapping=(PROJECT/'MDK_Project/Listings/ECG_Student_INT8.map').read_text(errors='replace')
    flash=int(re.search(r'Total ROM Size[^\n]*?\s(\d+)\s+\(',mapping).group(1))
    ram=int(re.search(r'Total RW\s+Size[^\n]*?\s(\d+)\s+\(',mapping).group(1))
    assert flash<=512*1024 and ram<=128*1024
    summary={'flash_bytes':flash,'ram_bytes':ram,'flash_limit_bytes':512*1024,'main_sram_limit_bytes':128*1024,
             'flash_margin_bytes':512*1024-flash,'ram_margin_bytes':128*1024-ram,
             'build_errors':int(errors.group(1)),'build_warnings':int(errors.group(2)),
             'model_sha256':digest(model),'physical_board_test':False,
             'project':str(PROJECT/'MDK_Project/ECG_Student_INT8.uvprojx')}
    (OUT/'student_firmware_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(summary,flush=True)


if __name__=='__main__': main()
