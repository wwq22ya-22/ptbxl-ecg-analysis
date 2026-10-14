"""Load the exact board C implementation on the host for numerical verification."""
import ctypes as C
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'outputs/distillation/board_preprocess_export'
SOURCES = ROOT / 'firmware/ecg_preprocess'

def load_library(*, protocol=False):
    sources = [SOURCES / 'ecg_preprocess.c']
    dependencies = sources + [SOURCES / 'ecg_preprocess.h', CONFIG / 'ecg_preprocess_config.h']
    if protocol:
        sources += [ROOT / 'firmware/ecg_serial/ecg_protocol.c']
        dependencies += sources[1:] + [ROOT / 'firmware/ecg_serial/ecg_protocol.h']
    fingerprint = hashlib.sha256(b''.join(p.read_bytes() for p in dependencies)).hexdigest()[:20]
    directory = CONFIG / 'host_simulation'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (('raw_protocol_' if protocol else 'preprocess_') + fingerprint + ('.dll' if os.name == 'nt' else '.so'))
    if not target.exists():
        compiler = shutil.which('gcc') or (r'D:\MyProject\MinGW64\bin\gcc.exe' if os.name == 'nt' else None)
        if not compiler or not Path(compiler).is_file(): raise RuntimeError('Host C checks require GCC; physical serial mode does not.')
        flags = ['-shared', '-std=c99', '-O2', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror']
        if os.name != 'nt': flags.append('-fPIC')
        if protocol: flags.append('-DECG_BOARD_PREPROCESSING=1')
        result = subprocess.run([compiler, *flags, '-I', str(CONFIG), '-I', str(SOURCES),
                                 *map(str, sources), '-o', str(target), '-lm'], capture_output=True,
                                creationflags=0x08000000 if os.name == 'nt' else 0)
        if result.returncode: raise RuntimeError(result.stderr.decode(errors='replace'))
    lib = C.CDLL(str(target))
    pointer = C.POINTER(C.c_float)
    lib.ecg_preprocess_lead.argtypes = [pointer, C.c_uint, pointer]
    lib.ecg_preprocess_lead.restype = C.c_int
    lib.ecg_preprocess_to_int8.argtypes = [pointer, C.POINTER(C.c_int8)]
    lib.ecg_preprocess_to_int8.restype = C.c_int
    lib.ecg_preprocess_workspace_bytes.restype = C.c_uint
    lib.ecg_preprocess_contract_sha256.restype = C.c_char_p
    return lib

def preprocess_float(raw, library=None):
    lib = library or load_library()
    raw = np.ascontiguousarray(raw, dtype=np.float32)
    if raw.shape != (12, 1000): raise ValueError('Expected [12,1000] raw mV.')
    output = np.empty_like(raw)
    for lead in range(12):
        if not lib.ecg_preprocess_lead(raw[lead].ctypes.data_as(C.POINTER(C.c_float)), lead,
                                     output[lead].ctypes.data_as(C.POINTER(C.c_float))):
            raise ValueError('C preprocessing rejected input.')
    return output

def preprocess_int8(raw, library=None):
    lib = library or load_library()
    raw = np.ascontiguousarray(raw, dtype=np.float32)
    if raw.shape != (12, 1000): raise ValueError('Expected [12,1000] raw mV.')
    output = np.empty((12, 1000), dtype=np.int8)
    if not lib.ecg_preprocess_to_int8(raw.ctypes.data_as(C.POINTER(C.c_float)), output.ctypes.data_as(C.POINTER(C.c_int8))):
        raise ValueError('C preprocessing rejected input.')
    return output
