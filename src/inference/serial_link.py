"""Chunked request/response link for twelve-lead STM32 inference."""
from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import numpy as np

from src.data.ptbxl import SUPERCLASSES


class SerialProtocolError(RuntimeError):
    pass


class STM32Link:
    def __init__(self, transport, manifest, *, timeout=5.0, inference_timeout=60.0):
        self.transport = transport
        self.manifest = manifest
        self.timeout = timeout
        self.inference_timeout = inference_timeout
        self.sequence = 0

    def _request(self, command, kind, sequence=None, *, timeout=None):
        data = (command + "\r\n").encode("ascii")
        if self.transport.write(data) != len(data):
            raise SerialProtocolError("串口未完整发送数据，请重新连接。")
        self.transport.flush()
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while time.monotonic() < deadline:
            raw = self.transport.readline()
            if not raw:
                continue
            try:
                reply = json.loads(raw.decode("ascii"))
            except (ValueError, UnicodeDecodeError):
                continue  # Ignore boot/debug messages.
            if not isinstance(reply, dict):
                continue
            if sequence is not None and reply.get("seq") != sequence:
                continue  # Never associate an older response with this waveform.
            if reply.get("type") == "error":
                raise SerialProtocolError(f"芯片返回错误：{reply.get('code', 'unknown')}")
            if reply.get("type") == kind:
                return reply
        raise SerialProtocolError(f"等待芯片 {kind} 超时，请检查接线、固件及波特率。")

    def handshake(self):
        hello = self._request("$P", "hello")
        expected = self.manifest
        cutoffs = np.asarray(hello.get("thresholds", []), dtype=float)
        if (hello.get("protocol") != 1 or hello.get("ready") is not True
                or hello.get("model") != expected["model_sha256"]
                or hello.get("input_shape") != [12, 1000]
                or hello.get("labels") != list(SUPERCLASSES)
                or hello.get("preprocessing") != "host"
                or cutoffs.shape != (5,)
                or not np.allclose(cutoffs, expected["thresholds"], atol=1e-6, rtol=0)):
            raise SerialProtocolError("芯片模型、阈值或输入协议与当前部署包不一致。")
        return hello

    def predict(self, signal, *, progress=None):
        values = np.asarray(signal, dtype=np.float32)
        if values.shape != (12, 1000) or not np.isfinite(values).all():
            raise ValueError("发送输入必须是预处理后的有限浮点数，形状 [12,1000]。")
        self.sequence = (self.sequence + 1) & 0x7FFFFFFF
        sequence = self.sequence
        flat = np.ascontiguousarray(values, dtype="<f4").reshape(-1)
        self._request(f"$B,{sequence},{flat.size}", "begin", sequence)
        for offset in range(0, flat.size, 64):
            chunk = flat[offset:offset + 64]
            payload = ",".join(format(float(value), ".9g") for value in chunk)
            reply = self._request(f"$D,{sequence},{offset},{len(chunk)},{payload}", "ack", sequence)
            if reply.get("next") != offset + len(chunk):
                raise SerialProtocolError("芯片确认的数据位置错误，已停止本次发送。")
            if progress:
                progress((offset + len(chunk)) / flat.size)
        crc = zlib.crc32(flat.tobytes()) & 0xFFFFFFFF
        result = self._request(f"$R,{sequence},{crc}", "result", sequence,
                               timeout=self.inference_timeout)
        probs = np.asarray(result.get("probabilities"), dtype=np.float64)
        if probs.shape != (5,) or not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
            raise SerialProtocolError("芯片返回的五类概率无效。")
        expected_mask = sum(1 << i for i, hit in enumerate(
            probs.astype(np.float32) >= np.asarray(self.manifest["thresholds"], dtype=np.float32)) if hit)
        if result.get("mask") != expected_mask:
            raise SerialProtocolError("芯片分类掩码与部署阈值不一致。")
        result["positive_labels"] = [name for i, name in enumerate(SUPERCLASSES) if expected_mask & (1 << i)]
        return result


def load_serial_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    thresholds = np.asarray(manifest.get("thresholds"), dtype=float)
    if (manifest.get("protocol") != 1 or manifest.get("labels") != list(SUPERCLASSES)
            or len(manifest.get("model_sha256", "")) != 64
            or thresholds.shape != (5,) or not np.isfinite(thresholds).all()
            or np.any((thresholds <= 0) | (thresholds >= 1))):
        raise ValueError("部署清单无效。")
    return manifest
