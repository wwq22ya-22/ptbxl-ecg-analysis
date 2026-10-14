"""Capture physical STM32 responses to distinct ECGs and a repeated input."""
import argparse
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import serial

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ecg_stm32_viz import transmission_sample
from scripts.prepare_stm32 import digest
from src.inference.serial_link import STM32Link, load_serial_manifest
from src.inference.serial_samples import FOLD10_DIRECTORY, read_samples, result_metadata


class CaptureSerial:
    def __init__(self, transport):
        self.transport = transport
        self.responses = []

    def write(self, value):
        return self.transport.write(value)

    def flush(self):
        self.transport.flush()

    def readline(self):
        raw = self.transport.readline()
        try:
            value = json.loads(raw.decode('ascii'))
            if isinstance(value, dict) and value.get('type') in {'hello', 'result', 'error'}:
                self.responses.append({'received_at': datetime.now(timezone.utc).isoformat(),
                                       'raw_ascii': raw.decode('ascii'), 'parsed': value})
        except (ValueError, UnicodeDecodeError):
            pass
        return raw


class TimedLink(STM32Link):
    """Measure only the final run request, excluding waveform transmission."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request_timings = []

    def _request(self, command, *args, **kwargs):
        start = time.monotonic_ns()
        reply = super()._request(command, *args, **kwargs)
        elapsed_ms = (time.monotonic_ns() - start) / 1e6
        if command.startswith(('$P', '$R')):
            raw = self.transport.responses[-1]['raw_ascii']
            request_bytes = len(command.encode('ascii')) + 2
            response_bytes = len(raw.encode('ascii'))
            self.request_timings.append({'command': command, 'seq': reply.get('seq'),
                                         'host_request_response_ms': elapsed_ms,
                                         'request_bytes': request_bytes, 'response_bytes': response_bytes,
                                         'wire_time_at_configured_baud_ms':
                                             (request_bytes + response_bytes) * 10 / self.transport.transport.baudrate * 1000})
        return reply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--baudrate', type=int, default=115200)
    parser.add_argument('--timing-audit', action='store_true',
                        help='Audit A firmware timing using real ECGs, an all-zero timing probe and a repeat.')
    args = parser.parse_args()
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    report_file = FOLD10_DIRECTORY / f"physical_{'timing_audit' if args.timing_audit else 'inference_check'}_{timestamp}.json"
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'physical_board_test': True,
              'mode': 'physical_serial', 'port': args.port, 'baudrate': args.baudrate,
              'simulation_used': False, 'firmware_modified_or_flashed': False, 'results': [],
              'completed': False}
    capture = None
    link = None
    try:
        with serial.Serial(args.port, args.baudrate, timeout=.2, write_timeout=3) as port:
            port.reset_input_buffer()
            capture = CaptureSerial(port)
            probe = STM32Link(capture, {})
            hello = probe._request('$P', 'hello')
            folder = {1: 'distilled_export', 2: 'board_preprocess_export'}.get(hello.get('protocol'))
            if folder is None:
                raise ValueError(f'Unsupported physical firmware protocol: {hello}')
            if args.timing_audit and hello.get('protocol') != 1:
                raise ValueError('Timing audit is limited to A inference-only firmware.')
            directory = ROOT / 'outputs/distillation' / folder
            manifest = load_serial_manifest(directory / 'serial_manifest.json')
            if digest(directory / manifest['model_file']) != manifest['model_sha256']:
                raise ValueError('Host model hash mismatch.')
            link = TimedLink(capture, manifest)
            report['hello'] = link.handshake()
            report['model_sha256'] = manifest['model_sha256']
            report['protocol'] = manifest['protocol']
            report['preprocessing_location'] = manifest.get('preprocessing_location', 'host')
            link.sequence = int(time.time()) & 0x3FFFFFFF
            samples = read_samples(manifest, directory, 10)
            plan = [0, len(samples['ecg_id']) - 1, None, 0] if args.timing_audit else [0, 1, len(samples['ecg_id']) - 1, 0]
            for index in plan:
                start = time.monotonic()
                signal = np.zeros((12, 1000), dtype=np.float32) if index is None else transmission_sample(manifest, samples, index)
                result = link.predict(signal)
                if index is None:
                    result.update(ecg_id=None, sample_index=None, sample_source='All-zero prepared input; timing probe only',
                                  model_sha256=manifest['model_sha256'], truth=None)
                else:
                    result.update(result_metadata(manifest, samples, index))
                result.update(mode='physical_serial', total_seconds=time.monotonic() - start,
                              max_probability_difference_from_host_c=None if index is None else float(np.max(abs(
                                  np.asarray(result['probabilities']) - samples['reference_probabilities'][index]))))
                timing = link.request_timings[-1]
                assert timing['seq'] == result['seq'] and timing['command'].startswith('$R')
                result.update(run_request_response_ms=timing['host_request_response_ms'],
                              run_request_wire_ms=timing['wire_time_at_configured_baud_ms'],
                              run_request_overhead_beyond_reported_inference_ms=timing['host_request_response_ms'] - result['inference_ms'])
                report['results'].append(result)
                print(json.dumps({key: result[key] for key in ['ecg_id', 'seq', 'inference_ms',
                                       'positive_labels', 'probabilities', 'total_seconds',
                                       'run_request_response_ms', 'run_request_wire_ms',
                                       'max_probability_difference_from_host_c']}, ensure_ascii=False), flush=True)
            probabilities = np.asarray([r['probabilities'] for r in report['results']])
            report['distinct_inputs_produce_different_probabilities'] = len(np.unique(probabilities[:3], axis=0)) == 3
            report['repeated_input_max_probability_difference'] = float(np.max(abs(probabilities[0] - probabilities[-1])))
            report['distinct_response_sequences'] = len({r['seq'] for r in report['results']}) == 4
            report['all_inference_times_positive'] = all(r['inference_ms'] > 0 for r in report['results'])
            report['max_probability_difference_from_host_c'] = max(r['max_probability_difference_from_host_c'] for r in report['results']
                                                                  if r['max_probability_difference_from_host_c'] is not None)
            report['inference_confirmed'] = (report['distinct_inputs_produce_different_probabilities']
                                            and report['distinct_response_sequences']
                                            and report['all_inference_times_positive']
                                            and report['repeated_input_max_probability_difference'] <= 1e-7)
            report['completed'] = True
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        report['raw_responses'] = capture.responses if capture else []
        report['request_timings'] = link.request_timings if link else []
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'Physical verification record: {report_file}', flush=True)


if __name__ == '__main__':
    main()
