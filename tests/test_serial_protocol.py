"""Run PC transport against the actual compiled firmware parser, without a board."""
import json
import numpy as np
import pytest

from src.inference.serial_link import STM32Link, SerialProtocolError
from src.inference.serial_simulator import FirmwareSimulator


@pytest.fixture
def manifest():
    return {"protocol": 1, "model_sha256": "a" * 64,
            "labels": ["NORM", "MI", "STTC", "CD", "HYP"],
            "thresholds": [0.6, 0.65, 0.55, 0.75, 0.7],
            "input_scale": 0.1990550309419632, "input_zero_point": 5}


def test_full_waveform_roundtrip_quantization_and_multiple_labels(manifest):
    received = []
    signal = np.random.default_rng(42).normal(size=(12, 1000)).astype(np.float32)
    signal[0, :5] = [-1000, 1000, 0, -0.0, 1e-30]
    def infer(values):
        received.append(values)
        return np.array([3, -2, 3, -4, 3], dtype=np.float32)
    simulator = FirmwareSimulator(manifest, infer)
    link = STM32Link(simulator, manifest)
    assert link.handshake()["ready"]
    progress = []
    result = link.predict(signal, progress=progress.append)
    expected = np.clip(np.rint(signal.reshape(-1) / np.float32(manifest["input_scale"]))
                       + manifest["input_zero_point"], -128, 127).astype(np.int8)
    np.testing.assert_array_equal(received[0], expected)
    assert result["positive_labels"] == ["NORM", "STTC", "HYP"]
    assert progress[-1] == 1
    assert len(received) == 1


@pytest.mark.parametrize("command,code", [
    ("$D,7,1,1,0", "offset_or_count"),
    ("$D,7,0,1,nan", "float"),
    ("$D,7,0,1,inf", "float"),
    ("$D,7,0,2,1", "float"),
    ("$D,7,0,1,1,2", "extra_values"),
    ("$D,7,0,65,0", "offset_or_count"),
    ("$D,8,0,1,0", "sequence"),
    ("$R,7,0", "crc_or_incomplete"),
])
def test_corrupt_or_incomplete_transfer_does_not_infer(manifest, command, code):
    calls = []
    simulator = FirmwareSimulator(manifest, lambda x: calls.append(x))
    simulator.write(b"$B,7,12000\r\n")
    assert json.loads(simulator.readline())["type"] == "begin"
    simulator.write((command + "\r\n").encode())
    response = json.loads(simulator.readline())
    assert response["type"] == "error" and response["code"] == code
    assert not calls and not simulator.protocol.active


def test_bad_crc_after_complete_input_and_restart(manifest):
    calls = []
    simulator = FirmwareSimulator(manifest, lambda x: calls.append(x) or [0] * 5)
    class CorruptCRC:
        def write(self, value):
            original_length = len(value)
            if value.startswith(b"$R,"):
                value = b"$R,1,0\r\n"
            simulator.write(value)
            return original_length
        def flush(self): simulator.flush()
        def readline(self): return simulator.readline()
    link = STM32Link(CorruptCRC(), manifest)
    with pytest.raises(SerialProtocolError, match="crc_or_incomplete"):
        link.predict(np.zeros((12, 1000), dtype=np.float32))
    assert not calls
    STM32Link(simulator, manifest).predict(np.zeros((12, 1000), dtype=np.float32))
    assert len(calls) == 1


def test_mismatched_model_is_rejected_before_sending_waveform(manifest):
    simulator = FirmwareSimulator({**manifest, "model_sha256": "b" * 64}, lambda x: [0] * 5)
    with pytest.raises(SerialProtocolError, match="不一致"):
        STM32Link(simulator, manifest).handshake()


def test_stale_response_cannot_be_used_for_another_sample(manifest):
    simulator = FirmwareSimulator(manifest, lambda x: [0] * 5)
    simulator.responses.append(b'{"type":"result","seq":99,"probabilities":[1,1,1,1,1],"mask":31}\r\n')
    result = STM32Link(simulator, manifest).predict(np.zeros((12, 1000), dtype=np.float32))
    assert result["seq"] == 1 and result["mask"] == 0
