"""Execute the actual binary firmware C parser, with injected inference."""
import ctypes as C
import os
import subprocess
from pathlib import Path
import numpy as np
from src.inference.serial_simulator import ROOT, RUN

SEND_BINARY = C.CFUNCTYPE(None, C.POINTER(C.c_uint8), C.c_size_t, C.c_void_p)
class BinaryProtocol(C.Structure):
    _fields_ = [('input', C.POINTER(C.c_int8)), ('scale', C.c_float), ('zero_point', C.c_int),
                ('thresholds', C.POINTER(C.c_float)), ('model_sha256', C.c_char_p),
                ('send', SEND_BINARY), ('run', RUN), ('context', C.c_void_p),
                ('sequence', C.c_uint32), ('received', C.c_uint32), ('crc', C.c_uint32),
                ('used', C.c_uint32), ('needed', C.c_uint32), ('active', C.c_int),
                ('frame', C.c_uint8 * 1044)]

def load_binary_library():
    source = ROOT / 'firmware/ecg_serial_fast/ecg_binary.c'
    target = ROOT / ('outputs/cubeai/serial_protocol/ecg_binary' + ('.dll' if os.name == 'nt' else '.so'))
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_mtime < max(source.stat().st_mtime, source.with_suffix('.h').stat().st_mtime):
        import shutil
        compiler = shutil.which('gcc') or 'D:/MyProject/MinGW64/bin/gcc.exe'
        flags = ['-shared', '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror']
        if os.name != 'nt': flags.append('-fPIC')
        subprocess.run([compiler, *flags, str(source), '-o', str(target), '-lm'], check=True,
                       capture_output=True, creationflags=0x08000000 if os.name == 'nt' else 0)
    lib = C.CDLL(str(target))
    lib.ecg_binary_byte.argtypes = [C.POINTER(BinaryProtocol), C.c_uint8]
    lib.ecg_binary_byte.restype = None
    lib.ecg_binary_abort.argtypes = [C.POINTER(BinaryProtocol), C.c_char_p]
    lib.ecg_binary_abort.restype = None
    return lib

class BinaryFirmwareSimulator:
    def __init__(self, manifest, infer):
        self.library = load_binary_library()
        self.responses = bytearray()
        self.input = (C.c_int8 * 12000)()
        self.thresholds = (C.c_float * 5)(*manifest['thresholds'])
        self.model_id = manifest['model_sha256'].encode('ascii')
        self.is_open = True
        self.callback_error = None
        self.inference_calls = 0
        def send(data, length, _): self.responses.extend(C.string_at(data, length))
        def run(output, elapsed, _):
            try:
                self.inference_calls += 1
                logits = np.asarray(infer(np.ctypeslib.as_array(self.input).copy()), np.float32).reshape(5)
                for i, value in enumerate(logits): output[i] = value
                elapsed[0] = 0  # Host simulation supplies no MCU timing.
                return 1
            except Exception as error:
                self.callback_error = error
                return 0
        self.send, self.run = SEND_BINARY(send), RUN(run)
        self.protocol = BinaryProtocol(self.input, manifest['input_scale'], manifest['input_zero_point'],
                                       self.thresholds, self.model_id, self.send, self.run, None)
    def write(self, data):
        if not self.is_open: raise OSError('Transport closed')
        for byte in data: self.library.ecg_binary_byte(C.byref(self.protocol), byte)
        return len(data)
    def read(self, count):
        data = bytes(self.responses[:count]); del self.responses[:count]; return data
    def flush(self): pass
    def close(self): self.is_open = False
