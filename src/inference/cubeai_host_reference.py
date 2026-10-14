"""Run the actual Cube.AI host DLL with exact firmware-quantized inputs."""
import ctypes as C
import hashlib
import os
import re
import subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BRIDGE = r'''
#include <stdint.h>
#include <string.h>
#include "ecg_network.h"
#include "ecg_network_data.h"
static AI_ALIGNED(32) ai_u8 activations[AI_ECG_NETWORK_DATA_ACTIVATIONS_SIZE];
static ai_handle network = AI_HANDLE_NULL;
__declspec(dllexport) int ecg_host_reference(const int8_t *input, float *output, const void *weights) {
    ai_buffer *inputs, *outputs;
    if (!network) {
        ai_handle pools[] = { activations };
        ai_handle weight_pools[] = { (ai_handle)weights };
        ai_error error = ai_ecg_network_create_and_init(&network, pools, weight_pools);
        if (error.type != AI_ERROR_NONE) return -1000 - error.type * 256 - error.code;
    }
    inputs = ai_ecg_network_inputs_get(network, NULL);
    outputs = ai_ecg_network_outputs_get(network, NULL);
    if (!inputs || !outputs) return -2;
    if (!inputs[0].data) return -3;
    if (!outputs[0].data) return -4;
    memcpy(inputs[0].data, input, 12000);
    if (ai_ecg_network_run(network, inputs, outputs) != 1) {
        ai_error error = ai_ecg_network_get_error(network);
        return -2000 - error.type * 256 - error.code;
    }
    memcpy(output, outputs[0].data, 5 * sizeof(float));
    return 1;
}
'''

class CubeAIHostReference:
    def __init__(self, workspace, output):
        workspace, output = Path(workspace), Path(output)
        # Host validation DLLs use caller-supplied weights. Read the identical,
        # already-generated seed44 firmware weights without modifying that file.
        self.weights_source = ROOT / 'outputs/distillation/distilled_export/generated/ecg_network_data_params.c'
        text = self.weights_source.read_text(encoding='utf-8')
        array = text.split('s_ecg_network_weights_array_u64[14035] = {', 1)[1].split('};', 1)[0]
        self.weights = np.asarray([int(value, 16) for value in re.findall(r'0x[0-9a-fA-F]+', array)], dtype='<u8')
        if self.weights.size != 14035: raise ValueError('Unexpected frozen firmware weight size.')
        dll = workspace / 'lib/libai_ecg_network.dll'
        source = output / 'cubeai_reference_bridge.c'
        source.write_text(BRIDGE, encoding='ascii')
        fingerprint = hashlib.sha256(BRIDGE.encode() + dll.read_bytes()).hexdigest()[:16]
        bridge = output / f'cubeai_reference_bridge_{fingerprint}.dll'
        if not bridge.exists():
            command = [r'D:\MyProject\MinGW64\bin\gcc.exe', '-shared', '-std=c99', '-O2',
                       '-I', str(workspace / 'generated'), '-I', str(workspace / 'include'),
                       str(source), str(dll), '-o', str(bridge)]
            result = subprocess.run(command, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode: raise RuntimeError(result.stderr.decode(errors='replace'))
        self.dll_directory = os.add_dll_directory(str(dll.parent.resolve()))
        self.library = C.CDLL(str(bridge.resolve()))
        self.library.ecg_host_reference.argtypes = [C.POINTER(C.c_int8), C.POINTER(C.c_float), C.c_void_p]
        self.library.ecg_host_reference.restype = C.c_int
        self.dll = dll

    def logits(self, quantized):
        values = np.ascontiguousarray(quantized, dtype=np.int8)
        if values.shape != (12, 1000): raise ValueError('Expected [12,1000] INT8 input.')
        output = np.empty(5, dtype=np.float32)
        status = self.library.ecg_host_reference(values.ctypes.data_as(C.POINTER(C.c_int8)),
                                                output.ctypes.data_as(C.POINTER(C.c_float)), self.weights.ctypes.data)
        if status != 1:
            raise RuntimeError(f'Cube.AI host C-model failed: {status}.')
        return output
