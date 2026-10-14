"""Golden SciPy comparison and safe raw/preprocessed protocol separation."""
import json
from pathlib import Path
import numpy as np
import pytest
from src.inference.board_preprocessing import CONFIG, load_library, preprocess_float, preprocess_int8
from src.inference.raw_serial_simulator import RawFirmwareSimulator
from src.inference.serial_simulator import FirmwareSimulator
from src.inference.serial_link import STM32Link, SerialProtocolError
from src.signal_processing.filters import FilterConfig, preprocess_ecg
from scripts.ecg_stm32_viz import transmission_sample

@pytest.fixture(scope='module')
def manifest():
    if not (CONFIG / 'serial_manifest.json').exists(): pytest.skip('Prepare board-preprocess export first.')
    return json.loads((CONFIG / 'serial_manifest.json').read_text(encoding='utf-8'))

@pytest.mark.parametrize('kind', ['constant', 'edges', 'ramp', 'noise', 'periodic'])
def test_scipy_equivalence_and_quantization(manifest, kind):
    x = np.zeros((12, 1000), dtype=np.float32)
    if kind == 'constant': x[:] = np.arange(12, dtype=np.float32)[:, None]
    if kind == 'edges': x[:, [0, 1, 26, 500, 998, 999]] = [12, -3, 2, 1, -9, 10]
    if kind == 'ramp': x[:] = np.linspace(-10, 10, 1000, dtype=np.float32)
    if kind == 'noise': x[:] = np.random.default_rng(8).normal(size=x.shape)
    if kind == 'periodic': x[:] = np.sin(np.arange(1000) * .2) + .2 * np.cos(np.arange(1000) * 1.7)
    lib = load_library()
    expected = preprocess_ecg(x.T, 100, config=FilterConfig.from_mapping(manifest['preprocessing'])).T
    np.testing.assert_allclose(preprocess_float(x, lib), expected, rtol=2e-6, atol=2e-6)
    q = np.clip(np.rint(expected / np.float32(manifest['input_scale'])) + manifest['input_zero_point'], -128, 127).astype(np.int8)
    np.testing.assert_array_equal(preprocess_int8(x, lib), q)

@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_input_rejected(manifest, value):
    x = np.zeros((12, 1000), dtype=np.float32); x[5, 40] = value
    with pytest.raises(ValueError, match='rejected'): preprocess_int8(x)

def test_raw_roundtrip_prepares_only_after_complete_crc_and_can_repeat(manifest):
    x = np.random.default_rng(7).normal(size=(12, 1000)).astype(np.float32)
    captured = []
    sim = RawFirmwareSimulator(manifest, lambda q: captured.append(q) or [2, -3, 2, -4, 3])
    link = STM32Link(sim, manifest)
    assert link.handshake()['preprocessing'] == 'board'
    result = link.predict(x)
    np.testing.assert_array_equal(captured[0], preprocess_int8(x).reshape(-1))
    assert sim.preparation_calls == 1 and result['preprocessing_ms'] == 0
    link.predict(x)
    assert sim.preparation_calls == 2 and len(captured) == 2

def test_bad_crc_does_not_preprocess_or_infer(manifest):
    sim = RawFirmwareSimulator(manifest, lambda _: pytest.fail('Incomplete input must not infer'))
    sim.write(b'$B,1,12000\r\n'); sim.readline()
    sim.write(b'$D,1,0,1,0\r\n'); sim.readline()
    sim.write(b'$R,1,0\r\n')
    assert json.loads(sim.readline())['code'] == 'crc_or_incomplete'
    assert sim.preparation_calls == 0

def test_both_directions_reject_wrong_pipeline_before_sending(manifest):
    old = dict(manifest, protocol=1, preprocessing_location='host')
    with pytest.raises(SerialProtocolError, match='不一致'):
        STM32Link(FirmwareSimulator(old, lambda _: [0] * 5), manifest).handshake()
    with pytest.raises(SerialProtocolError, match='不一致'):
        STM32Link(RawFirmwareSimulator(manifest, lambda _: [0] * 5), old).handshake()
    wrong = dict(manifest, preprocessing_sha256='0' * 64)
    with pytest.raises(SerialProtocolError, match='不一致'):
        STM32Link(RawFirmwareSimulator(manifest, lambda _: [0] * 5), wrong).handshake()

def test_gui_and_headless_choose_correct_payload(manifest):
    raw, prepared = np.ones((1, 12, 1000)), np.zeros((1, 12, 1000))
    samples = {'raw_mv': raw, 'ecg': prepared}
    assert transmission_sample(manifest, samples, 0) is not None
    np.testing.assert_array_equal(transmission_sample(manifest, samples, 0), raw[0])
    np.testing.assert_array_equal(transmission_sample({'preprocessing_location': 'host'}, samples, 0), prepared[0])
