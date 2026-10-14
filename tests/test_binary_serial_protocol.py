"""Exercise malformed traffic and exact inputs against the actual firmware C parser."""
import ctypes as C
import json
import struct
import zlib
import numpy as np
import pytest
from src.inference.binary_serial_link import BinarySTM32Link, encode_frame, HEADER, quantize_input
from src.inference.binary_serial_simulator import BinaryFirmwareSimulator
from src.inference.serial_simulator import FirmwareSimulator
from src.inference.serial_link import STM32Link, SerialProtocolError

@pytest.fixture
def manifest():
    return {'protocol':3,'model_sha256':'a'*64,'labels':['NORM','MI','STTC','CD','HYP'],
            'thresholds':[.55,.60,.55,.75,.65], 'input_scale':0.2008364498615265,'input_zero_point':0}

def read_frame(sim):
    header = sim.read(16)
    _, version, kind, length, seq, offset = HEADER.unpack(header)
    tail = sim.read(length+4)
    assert version == 3 and struct.unpack('<I',tail[-4:])[0] == zlib.crc32(header+tail[:-4])
    return kind, seq, offset, json.loads(tail[:-4]) if length else {}

def test_exact_inputs_same_as_old_c_parser_and_repeat(manifest):
    seen_fast, seen_old = [], []
    sim = BinaryFirmwareSimulator(manifest, lambda q: seen_fast.append(q) or [3,-2,3,-4,3])
    old_m = dict(manifest, protocol=1)
    old = FirmwareSimulator(old_m, lambda q: seen_old.append(q) or [3,-2,3,-4,3])
    fast_link, old_link = BinarySTM32Link(sim,manifest), STM32Link(old,old_m)
    fast_link.handshake(); old_link.handshake()
    signal = np.random.default_rng(44).normal(size=(12,1000)).astype(np.float32)
    signal[0,:6] = [-1000,1000,0,np.float32(manifest['input_scale'])*.5,
                    np.float32(manifest['input_scale'])*1.5,-np.float32(manifest['input_scale'])*2.5]
    progress = []
    result = fast_link.predict(signal,progress=progress.append)
    reference = old_link.predict(signal)
    np.testing.assert_array_equal(seen_fast[0],seen_old[0])
    np.testing.assert_array_equal(result['probabilities'],reference['probabilities'])
    assert result['positive_labels'] == ['NORM','STTC','HYP']
    assert result['chunks'] == 12 and result['tx_bytes'] == 12288 and progress[-1] == 1
    fast_link.predict(signal)
    assert sim.inference_calls == 2
    np.testing.assert_array_equal(seen_fast[0],seen_fast[1])

def test_all_byte_values_and_fragmented_frames(manifest):
    received = []
    sim = BinaryFirmwareSimulator(manifest,lambda q: received.append(q) or [0]*5)
    payload = (bytes(range(256))*47)[:12000]
    frames = [encode_frame(2,7,payload=struct.pack('<I',12000))]
    frames += [encode_frame(3,7,i,payload[i:i+1024]) for i in range(0,12000,1024)]
    frames += [encode_frame(4,7,12000,struct.pack('<I',zlib.crc32(payload)))]
    sim.write(b'boot\r\nEE')
    for frame in frames:
        for i in range(0,len(frame),7): sim.write(frame[i:i+7])
        read_frame(sim)
    assert sim.inference_calls == 1 and received[0].tobytes() == payload

@pytest.mark.parametrize('kind,seq,offset,payload,code',[
    (3,7,1,b'\0','offset_or_count'), (3,8,0,b'\0','sequence'),
    (3,7,0,b'','offset_or_count'), (4,7,12000,struct.pack('<I',0),'crc_or_incomplete'),
    (2,7,0,struct.pack('<I',11999),'sample_count'), (9,7,0,b'','command'),
])
def test_incomplete_or_invalid_does_not_infer(manifest,kind,seq,offset,payload,code):
    sim = BinaryFirmwareSimulator(manifest,lambda q: [0]*5)
    sim.write(encode_frame(2,7,payload=struct.pack('<I',12000))); read_frame(sim)
    sim.write(encode_frame(kind,seq,offset,payload))
    reply = read_frame(sim)
    assert reply[0] == 255 and reply[3]['code'] == code
    assert sim.inference_calls == 0 and not sim.protocol.active

def test_frame_crc_timeout_duplicate_and_recovery(manifest):
    sim = BinaryFirmwareSimulator(manifest,lambda q: [0]*5)
    begin = encode_frame(2,7,payload=struct.pack('<I',12000))
    bad = bytearray(begin); bad[-1] ^= 1
    sim.write(bad)
    assert read_frame(sim)[3]['code'] == 'frame_crc'
    sim.write(begin); read_frame(sim)
    chunk = encode_frame(3,7,0,b'\0'*1024)
    sim.write(chunk); read_frame(sim); sim.write(chunk)
    assert read_frame(sim)[3]['code'] == 'offset_or_count'
    sim.write(begin[:18])
    sim.library.ecg_binary_abort(C.byref(sim.protocol),b'receive_timeout')
    assert read_frame(sim)[3]['code'] == 'receive_timeout'
    link = BinarySTM32Link(sim,manifest)
    link.handshake(); link.predict(np.zeros((12,1000),np.float32))
    assert sim.inference_calls == 1

def test_full_input_bad_checksum_does_not_infer(manifest):
    sim = BinaryFirmwareSimulator(manifest,lambda q:[0]*5)
    link = BinarySTM32Link(sim,manifest); link.handshake()
    sim.write(encode_frame(2,7,payload=struct.pack('<I',12000))); read_frame(sim)
    for i in range(0,12000,1024):
        sim.write(encode_frame(3,7,i,b'\0'*min(1024,12000-i))); read_frame(sim)
    sim.write(encode_frame(4,7,12000,struct.pack('<I',123)))
    assert read_frame(sim)[3]['code'] == 'crc_or_incomplete' and sim.inference_calls == 0

@pytest.mark.parametrize('field,value',[('model_sha256','b'*64),('input_scale',.3),
    ('input_zero_point',1),('thresholds',[.1]*5),('labels',['HYP']*5)])
def test_wrong_contract_refused(manifest,field,value):
    sim = BinaryFirmwareSimulator(manifest,lambda q:[0]*5)
    with pytest.raises(SerialProtocolError): BinarySTM32Link(sim,dict(manifest,**{field:value})).handshake()

def test_oversized_header_rejected(manifest):
    sim = BinaryFirmwareSimulator(manifest,lambda q:[0]*5)
    sim.write(HEADER.pack(b'ECG3',3,3,1025,7,0))
    assert read_frame(sim)[3]['code'] == 'header'
    assert not sim.protocol.active and sim.inference_calls == 0

def test_corrupt_response_refused(manifest):
    sim = BinaryFirmwareSimulator(manifest,lambda q:[0]*5)
    original_write = sim.write
    def write(data):
        result = original_write(data)
        sim.responses[-1] ^= 1
        return result
    sim.write = write
    with pytest.raises(SerialProtocolError,match='CRC'): BinarySTM32Link(sim,manifest).handshake()

@pytest.mark.parametrize('signal',[np.zeros((1,12000)),np.full((12,1000),np.nan)])
def test_invalid_signal_refused(manifest,signal):
    with pytest.raises(ValueError): quantize_input(signal,manifest)
