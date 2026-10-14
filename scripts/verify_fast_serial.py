"""Numerically verify A-fast against original C quantization and Cube.AI host C."""
import ctypes as C
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_stm32 import digest
from scripts.verify_serial_fold10 import check_gui
from src.inference.serial_link import load_serial_manifest, STM32Link
from src.inference.serial_samples import read_samples
from src.inference.serial_simulator import FirmwareSimulator
from src.inference.binary_serial_link import BinarySTM32Link, quantize_input
from src.inference.binary_serial_simulator import BinaryFirmwareSimulator
from src.inference.cubeai_host_reference import CubeAIHostReference

def main():
    directory = ROOT/'outputs/distillation/fast_serial_export'
    output = directory/'verification'
    output.mkdir(exist_ok=True)
    manifest = load_serial_manifest(directory/'serial_manifest.json')
    source = output/'original_quantization.c'
    # Include the unchanged original parser and apply exactly its scalar quantization expression.
    source.write_text('#include "ecg_protocol.c"\n'
        '__declspec(dllexport) void original_quantize(const float *x, int8_t *out, unsigned n, float scale, int zero) {\n'
        ' unsigned i; for(i=0;i<n;++i) { float q = nearbyintf(x[i]/scale)+(float)zero;\n'
        ' q=fminf(127.0f,fmaxf(-128.0f,q)); out[i]=(int8_t)q; } }\n',encoding='ascii')
    library = output/'original_quantization.dll'
    subprocess.run(['D:/MyProject/MinGW64/bin/gcc.exe','-shared','-std=c99','-O2','-ffp-contract=off',
                    '-I',str(ROOT/'firmware/ecg_serial'),str(source),'-o',str(library),'-lm'],
                   capture_output=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    lib = C.CDLL(str(library))
    lib.original_quantize.argtypes = [C.POINTER(C.c_float),C.POINTER(C.c_int8),C.c_uint,C.c_float,C.c_int]
    lib.original_quantize.restype = None
    ref = CubeAIHostReference(ROOT/'outputs/f10host_retry/inspector_ecg_network/workspace',output)
    report = {'verified_at':datetime.now().isoformat(),'physical_board_test':False,
              'model_sha256':manifest['model_sha256'],'protocol':3,
              'host_C_dll_sha256':digest(ref.dll),'folds':[]}
    fast = BinaryFirmwareSimulator(manifest,lambda q:ref.logits(q.reshape(12,1000)))
    link = BinarySTM32Link(fast,manifest); link.handshake()
    max_difference = 0.
    for fold in [9,10]:
        samples = read_samples(manifest,directory,fold)
        q_differences = 0
        for values in samples['ecg']:
            exact = np.empty((12,1000),np.int8)
            values = np.ascontiguousarray(values,np.float32)
            lib.original_quantize(values.ctypes.data_as(C.POINTER(C.c_float)),exact.ctypes.data_as(C.POINTER(C.c_int8)),
                                  12000,manifest['input_scale'],manifest['input_zero_point'])
            q_differences += int(np.count_nonzero(quantize_input(values,manifest) != exact))
        assert q_differences == 0
        indices = sorted(set([0,1,len(samples['ecg_id'])-1]+list(np.linspace(0,len(samples['ecg_id'])-1,20,dtype=int))+
                             ([1203,1440] if fold == 10 else [])))
        results = []
        # Exercise the old full text parser on difficult boundary cases too.
        old_m = dict(manifest,protocol=1)
        old = FirmwareSimulator(old_m,lambda q:ref.logits(q.reshape(12,1000)))
        old_link = STM32Link(old,old_m); old_link.handshake()
        for index in indices:
            result = link.predict(samples['ecg'][index])
            delta = float(np.max(abs(np.asarray(result['probabilities'])-samples['reference_probabilities'][index])))
            assert delta < 1e-6, (fold,index,delta)
            max_difference = max(max_difference,delta)
            if index in [0,1,1203,1440,len(samples['ecg_id'])-1]:
                original = old_link.predict(samples['ecg'][index])
                np.testing.assert_array_equal(np.ctypeslib.as_array(fast.input),np.ctypeslib.as_array(old.input))
                np.testing.assert_array_equal(result['probabilities'],original['probabilities'])
            results.append({'index':int(index),'ecg_id':int(samples['ecg_id'][index]),
                            'probabilities':result['probabilities'],'max_probability_difference':delta,
                            'tx_bytes':result['tx_bytes'],'rx_bytes':result['rx_bytes'],
                            'ideal_wire_seconds_115200':(result['tx_bytes']+result['rx_bytes'])*10/115200})
        report['folds'].append({'fold':fold,'quantization_records_checked':len(samples['ecg_id']),
                               'quantization_values_checked':len(samples['ecg_id'])*12000,
                               'different_INT8_values':q_differences,'C_protocol_and_C_model_cases':len(results),
                               'cases':results})
        print(f'Fold {fold}: {len(samples["ecg_id"])} exact quantization records; {len(results)} C roundtrips passed.',flush=True)
    report['max_probability_difference_from_frozen_host_C'] = max_difference
    report['GUI'] = check_gui(manifest,directory,'A_fast',output)
    release = json.loads((ROOT/'outputs/board_release/seed44_int8_stm32f407vg_fast/release_manifest.json').read_text(encoding='utf-8'))
    for path,sha in release['protected_sha256'].items(): assert digest(Path(path)) == sha, path
    report['original_A_B_and_fold10_unchanged'] = True
    report['completed'] = True
    (output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'folds'},ensure_ascii=False,indent=2),flush=True)

if __name__ == '__main__': main()
