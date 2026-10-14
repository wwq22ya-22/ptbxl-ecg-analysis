"""Read-only physical A-fast timing/consistency check; does not flash the board."""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import serial
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_stm32 import digest
from src.inference.serial_link import load_serial_manifest
from src.inference.serial_samples import read_samples, result_metadata
from src.inference.binary_serial_link import BinarySTM32Link

class CaptureSerial:
    def __init__(self, port):
        self.port = port
        self.responses = []
    def write(self, data): return self.port.write(data)
    def flush(self): self.port.flush()
    def read(self, count):
        data = self.port.read(count)
        if data: self.responses.append({'received_at':datetime.now().isoformat(),'hex':data.hex()})
        return data

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',required=True)
    parser.add_argument('--fold',type=int,choices=[9,10],default=10)
    args = parser.parse_args()
    directory = ROOT/'outputs/distillation/fast_serial_export'
    manifest = load_serial_manifest(directory/'serial_manifest.json')
    assert digest(directory/manifest['model_file']) == manifest['model_sha256']
    samples = read_samples(manifest,directory,args.fold)
    output = directory/'physical_validation'
    output.mkdir(exist_ok=True)
    path = output/f'physical_fast_{datetime.now():%Y%m%d_%H%M%S}.json'
    report = {'physical_board_test':True,'mode':'physical_serial','completed':False,
              'simulation_used':False,'firmware_modified_or_flashed':False,
              'port':args.port,'baudrate':115200,'protocol':3,'model_sha256':manifest['model_sha256'],
              'results':[]}
    capture = None
    try:
        with serial.Serial(args.port,115200,timeout=.2,write_timeout=3) as port:
            port.reset_input_buffer(); capture = CaptureSerial(port)
            link = BinarySTM32Link(capture,manifest)
            report['hello'] = link.handshake()
            link.sequence = int(time.time()) & 0x3fffffff
            for index in [0,1,len(samples['ecg_id'])-1,0]:
                start = time.monotonic()
                result = link.predict(samples['ecg'][index])
                result.update(result_metadata(manifest,samples,index),mode='physical_serial',
                              total_seconds=time.monotonic()-start,
                              max_probability_difference_from_host_c=float(np.max(abs(
                                  np.asarray(result['probabilities'])-samples['reference_probabilities'][index]))))
                report['results'].append(result)
                print(json.dumps({k:result[k] for k in ['ecg_id','inference_ms','upload_seconds',
                    'total_seconds','positive_labels','max_probability_difference_from_host_c']},ensure_ascii=False),flush=True)
            probabilities = np.asarray([r['probabilities'] for r in report['results']])
            report['repeated_input_max_probability_difference'] = float(np.max(abs(probabilities[0]-probabilities[-1])))
            report['max_probability_difference_from_host_c'] = max(r['max_probability_difference_from_host_c'] for r in report['results'])
            report['distinct_response_sequences'] = len({r['seq'] for r in report['results']}) == 4
            report['distinct_inputs_produce_different_probabilities'] = len(np.unique(probabilities[:3],axis=0)) == 3
            report['inference_confirmed'] = (report['distinct_response_sequences']
                and report['distinct_inputs_produce_different_probabilities']
                and report['repeated_input_max_probability_difference'] < 1e-7
                and all(r['inference_ms'] > 0 for r in report['results'])
                and report['max_probability_difference_from_host_c'] < 1e-5)
            assert report['inference_confirmed']
            report['completed'] = True
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        report['raw_rx_chunks'] = capture.responses if capture else []
        report['finished_at'] = datetime.now().isoformat()
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(f'Physical record: {path}',flush=True)

if __name__ == '__main__': main()
