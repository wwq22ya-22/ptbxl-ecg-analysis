"""Exercise the actual firmware parser on the PC; inference uses ONNX Runtime."""
from __future__ import annotations
import ctypes as C
import json
import os
import shutil
import subprocess
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEND = C.CFUNCTYPE(None, C.c_char_p, C.c_void_p)
RUN = C.CFUNCTYPE(C.c_int, C.POINTER(C.c_float), C.POINTER(C.c_uint32), C.c_void_p)


class Protocol(C.Structure):
    _fields_ = [
        ("input", C.POINTER(C.c_int8)), ("scale", C.c_float), ("zero_point", C.c_int),
        ("thresholds", C.POINTER(C.c_float)), ("model_sha256", C.c_char_p),
        ("send", SEND), ("run", RUN), ("context", C.c_void_p),
        ("sequence", C.c_uint32), ("received", C.c_uint32), ("crc", C.c_uint32),
        ("active", C.c_int),
    ]


def load_protocol_library():
    extension = ".dll" if os.name == "nt" else ".so"
    directory = ROOT / "outputs/cubeai/serial_protocol"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / ("ecg_protocol" + extension)
    source = ROOT / "firmware/ecg_serial/ecg_protocol.c"
    header = source.with_suffix(".h")
    if not target.exists() or target.stat().st_mtime < max(source.stat().st_mtime, header.stat().st_mtime):
        compiler = shutil.which("gcc") or (r"D:\MyProject\MinGW64\bin\gcc.exe" if os.name == "nt" else None)
        if not compiler or not Path(compiler).is_file():
            raise RuntimeError("模拟模式需要 GCC 来编译芯片协议；真实串口模式不需要。")
        flags = ["-shared", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror"]
        if os.name != "nt": flags.append("-fPIC")
        subprocess.run([compiler, *flags, str(source), "-o", str(target), "-lm"], check=True,
                       capture_output=True, creationflags=0x08000000 if os.name == "nt" else 0)
    library = C.CDLL(str(target))
    library.ecg_protocol_line.argtypes = [C.POINTER(Protocol), C.c_char_p]
    library.ecg_protocol_line.restype = None
    return library


class FirmwareSimulator:
    """Serial-shaped transport executing C parser + injected inference callback."""
    def __init__(self, manifest, infer):
        self.library = load_protocol_library()
        self.responses = deque()
        self.input = (C.c_int8 * 12000)()
        self.thresholds = (C.c_float * 5)(*manifest["thresholds"])
        self.model_id = manifest["model_sha256"].encode("ascii")
        self.is_open = True
        self.callback_error = None

        def send(message, _): self.responses.append(bytes(message))

        def run(output, elapsed, _):
            try:
                values = np.ctypeslib.as_array(self.input).copy()
                logits = np.asarray(infer(values), dtype=np.float32).reshape(5)
                for i, value in enumerate(logits): output[i] = value
                elapsed[0] = 0  # Simulation provides no MCU timing measurement.
                return 1
            except Exception as error:
                self.callback_error = error
                return 0

        self.send = SEND(send)
        self.run = RUN(run)
        self.protocol = Protocol(self.input, manifest["input_scale"], manifest["input_zero_point"],
                                 self.thresholds, self.model_id, self.send, self.run, None, 0, 0, 0, 0)

    def write(self, message):
        if not self.is_open: raise OSError("Transport closed")
        for line in message.splitlines():
            self.library.ecg_protocol_line(C.byref(self.protocol), C.create_string_buffer(line))
        return len(message)

    def flush(self): pass
    def readline(self): return self.responses.popleft() if self.responses else b""
    def close(self): self.is_open = False


def onnx_simulator(manifest, directory):
    import onnxruntime as ort
    from scripts.prepare_stm32 import digest
    model = Path(directory) / manifest["model_file"]
    if digest(model) != manifest["model_sha256"]: raise ValueError("模拟模型哈希不一致。")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])

    def infer(values):
        signal = ((values.astype(np.float32) - manifest["input_zero_point"])
                  * np.float32(manifest["input_scale"])).reshape(1, 12, 1000)
        return session.run(None, {"ecg": signal})[0][0]

    return FirmwareSimulator(manifest, infer)
