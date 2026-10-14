"""Raw serial protocol + actual C preprocessing, with injectable host inference."""
import ctypes as C
from collections import deque
import numpy as np
from src.inference.board_preprocessing import load_library
from src.inference.serial_simulator import SEND, RUN, Protocol

PREPARE = C.CFUNCTYPE(C.c_int, C.POINTER(C.c_float), C.POINTER(C.c_int8), C.POINTER(C.c_uint32), C.c_void_p)
class RawProtocol(C.Structure):
    _fields_ = Protocol._fields_ + [('raw_input', C.POINTER(C.c_float)), ('prepare', PREPARE)]

class RawFirmwareSimulator:
    def __init__(self, manifest, infer):
        self.library = load_library(protocol=True)
        self.library.ecg_protocol_line.argtypes = [C.POINTER(RawProtocol), C.c_char_p]
        self.library.ecg_protocol_line.restype = None
        if self.library.ecg_preprocess_contract_sha256().decode() != manifest['preprocessing_sha256']:
            raise ValueError('Simulator preprocessing source/config does not match manifest.')
        self.responses = deque()
        self.input = (C.c_int8 * 12000)()
        self.raw_input = (C.c_float * 12000)()
        self.thresholds = (C.c_float * 5)(*manifest['thresholds'])
        self.model_id = manifest['model_sha256'].encode('ascii')
        self.is_open = True
        self.callback_error = None
        self.preparation_calls = 0
        def send(message, _): self.responses.append(bytes(message))
        def prepare(raw, output, elapsed, _):
            self.preparation_calls += 1
            elapsed[0] = 0  # Host simulation is not MCU timing.
            return self.library.ecg_preprocess_to_int8(raw, output)
        def run(output, elapsed, _):
            try:
                logits = np.asarray(infer(np.ctypeslib.as_array(self.input).copy()), dtype=np.float32).reshape(5)
                for i, v in enumerate(logits): output[i] = v
                elapsed[0] = 0
                return 1
            except Exception as error:
                self.callback_error = error
                return 0
        self.send, self.run, self.prepare = SEND(send), RUN(run), PREPARE(prepare)
        self.protocol = RawProtocol(self.input, manifest['input_scale'], manifest['input_zero_point'],
                                    self.thresholds, self.model_id, self.send, self.run, None,
                                    0, 0, 0, 0, self.raw_input, self.prepare)
    def write(self, message):
        if not self.is_open: raise OSError('Transport closed')
        for line in message.splitlines():
            self.library.ecg_protocol_line(C.byref(self.protocol), C.create_string_buffer(line))
        return len(message)
    def flush(self): pass
    def readline(self): return self.responses.popleft() if self.responses else b''
    def close(self): self.is_open = False
