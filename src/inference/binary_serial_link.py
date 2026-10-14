"""Protocol 3: transmit the unchanged seed44 INT8 tensor in twelve binary chunks."""
import json
import struct
import time
import zlib
import numpy as np
from src.inference.serial_link import SerialProtocolError

HEADER = struct.Struct('<4sBBHII')
CHUNK_BYTES = 1024

def quantize_input(signal, manifest):
    values = np.asarray(signal, dtype=np.float32)
    if values.shape != (12, 1000) or not np.isfinite(values).all():
        raise ValueError('发送输入必须是预处理后的有限浮点数，形状 [12,1000]。')
    # Float32 division and ties-to-even reproduce the old firmware's nearbyintf.
    with np.errstate(over='ignore'):
        q = np.rint(values / np.float32(manifest['input_scale']))
    return np.ascontiguousarray(np.clip(q + np.float32(manifest['input_zero_point']),
                                       -128, 127), dtype=np.int8)

def encode_frame(kind, sequence, offset=0, payload=b''):
    if len(payload) > CHUNK_BYTES:
        raise ValueError('二进制分包超过1024字节。')
    body = HEADER.pack(b'ECG3', 3, kind, len(payload), sequence, offset) + payload
    return body + struct.pack('<I', zlib.crc32(body) & 0xffffffff)

class BinarySTM32Link:
    def __init__(self, transport, manifest, *, timeout=5., inference_timeout=60.):
        if manifest.get('protocol') != 3:
            raise ValueError('快速串口需要协议3的配套清单。')
        self.transport, self.manifest = transport, manifest
        self.timeout, self.inference_timeout = timeout, inference_timeout
        self.sequence = 0
        self.last_transfer = {}

    def _read_exact(self, count, deadline):
        data = bytearray()
        while len(data) < count and time.monotonic() < deadline:
            data.extend(self.transport.read(count - len(data)))
        if len(data) != count:
            raise SerialProtocolError('等待快速串口回应超时。请确认已烧录A快速版固件，波特率为115200。')
        return bytes(data)

    def _request(self, kind, sequence, offset=0, payload=b'', *, timeout=None):
        data = encode_frame(kind, sequence, offset, payload)
        if self.transport.write(data) != len(data):
            raise SerialProtocolError('串口未完整发送数据，请重新连接。')
        self.transport.flush()
        self.tx_bytes += len(data)
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while time.monotonic() < deadline:
            prefix = self._read_exact(4, deadline)
            while prefix != b'ECG3':
                prefix = prefix[1:] + self._read_exact(1, deadline)
            header = prefix + self._read_exact(12, deadline)
            _, version, response_kind, length, response_seq, next_offset = HEADER.unpack(header)
            if version != 3 or length > CHUNK_BYTES:
                raise SerialProtocolError('快速串口回应头无效，请重新连接。')
            tail = self._read_exact(length + 4, deadline)
            self.rx_bytes += 20 + length
            if struct.unpack('<I', tail[-4:])[0] != (zlib.crc32(header + tail[:-4]) & 0xffffffff):
                raise SerialProtocolError('快速串口回应CRC校验失败，请重新连接后重发。')
            if response_seq != sequence:
                continue
            try:
                reply = json.loads(tail[:-4]) if length else {}
            except (ValueError, UnicodeDecodeError) as error:
                raise SerialProtocolError('快速串口回应内容无效。') from error
            if not isinstance(reply, dict):
                raise SerialProtocolError('快速串口回应内容无效。')
            if response_kind == 0xff:
                raise SerialProtocolError(f"芯片返回错误：{reply.get('code', 'unknown')}")
            if response_kind != (kind | 0x80):
                continue
            return reply, next_offset
        raise SerialProtocolError('快速串口回应超时。')

    def handshake(self):
        self.tx_bytes = self.rx_bytes = 0
        hello, offset = self._request(1, 0)
        m = self.manifest
        cutoffs = np.asarray(hello.get('thresholds', []), dtype=float)
        if (hello.get('protocol') != 3 or hello.get('ready') is not True or offset != 0
                or hello.get('model') != m['model_sha256']
                or hello.get('input_shape') != [12, 1000] or hello.get('labels') != m['labels']
                or hello.get('preprocessing') != 'host' or hello.get('input_kind') != 'int8'
                or hello.get('chunk_bytes') != CHUNK_BYTES
                or hello.get('input_zero_point') != m['input_zero_point']
                or not np.isclose(hello.get('input_scale', float('nan')), m['input_scale'], rtol=1e-7, atol=0)
                or cutoffs.shape != (5,) or not np.allclose(cutoffs, m['thresholds'], rtol=0, atol=1e-6)):
            raise SerialProtocolError('芯片模型、阈值、量化参数或协议与A快速部署包不一致。')
        return hello

    def predict(self, signal, *, progress=None):
        data = quantize_input(signal, self.manifest).tobytes()
        self.sequence = (self.sequence + 1) & 0x7fffffff
        seq = self.sequence
        self.tx_bytes = self.rx_bytes = 0
        begin_time = time.monotonic()
        _, offset = self._request(2, seq, payload=struct.pack('<I', len(data)))
        if offset != 0: raise SerialProtocolError('芯片初始输入位置错误。')
        for start in range(0, len(data), CHUNK_BYTES):
            chunk = data[start:start + CHUNK_BYTES]
            _, offset = self._request(3, seq, start, chunk)
            if offset != start + len(chunk):
                raise SerialProtocolError('芯片确认的数据位置错误，已停止发送。')
            if progress: progress(offset / len(data))
        upload_seconds = time.monotonic() - begin_time
        run_time = time.monotonic()
        result, offset = self._request(4, seq, len(data), struct.pack('<I', zlib.crc32(data) & 0xffffffff),
                                       timeout=self.inference_timeout)
        run_seconds = time.monotonic() - run_time
        probs = np.asarray(result.get('probabilities'), dtype=np.float64)
        if (offset != len(data) or result.get('seq') != seq or probs.shape != (5,)
                or not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1))):
            raise SerialProtocolError('芯片返回的五类结果无效。')
        mask = sum(1 << i for i, hit in enumerate(
            probs.astype(np.float32) >= np.asarray(self.manifest['thresholds'], np.float32)) if hit)
        if result.get('mask') != mask: raise SerialProtocolError('芯片分类掩码与部署阈值不一致。')
        result['positive_labels'] = [name for i, name in enumerate(self.manifest['labels']) if mask & (1 << i)]
        self.last_transfer = {'transport': 'binary_int8', 'payload_bytes': len(data), 'chunks': 12,
                              'tx_bytes': self.tx_bytes, 'rx_bytes': self.rx_bytes,
                              'upload_seconds': upload_seconds, 'run_roundtrip_seconds': run_seconds}
        result.update(self.last_transfer)
        return result
