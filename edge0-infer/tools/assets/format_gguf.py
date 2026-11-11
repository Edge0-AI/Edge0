"""Small independent GGUF v3 implementation (little endian, F32/F16/Q4_1).

Format: https://github.com/ggml-org/llama.cpp/blob/7ab4ee7baad2d920464cbacfad4f4b07cf111fd2/ggml/docs/gguf.md
Shapes are in GGML order: the first dimension is contiguous.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import struct
from typing import BinaryIO

import numpy as np

SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
           6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
F32, F16, Q4_1 = 0, 1, 3


@dataclass(frozen=True)
class Value:
    type: int
    data: object
    element_type: int | None = None


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    type: int
    offset: int
    size: int


def align(n: int, alignment: int = 32) -> int:
    return (n + alignment - 1) // alignment * alignment


def tensor_size(shape: tuple[int, ...], type: int) -> int:
    if not 1 <= len(shape) <= 4 or any(d <= 0 for d in shape):
        raise ValueError('invalid tensor shape')
    if type == Q4_1:
        if shape[0] % 32:
            raise ValueError('Q4_1 row width must be a multiple of 32')
        return math.prod(shape) // 32 * 20
    if type not in (F32, F16):
        raise ValueError(f'unsupported tensor type {type}')
    return math.prod(shape) * (4 if type == F32 else 2)


def _read(f: BinaryIO, size: int) -> bytes:
    b = f.read(size)
    if len(b) != size:
        raise ValueError('truncated GGUF')
    return b


def _unpack(f: BinaryIO, fmt: str):
    return struct.unpack('<' + fmt, _read(f, struct.calcsize('<' + fmt)))[0]


def _string(f: BinaryIO) -> str:
    n = _unpack(f, 'Q')
    # Reject malicious lengths before allocating or reading.
    if n > 256 * 1024 * 1024:
        raise ValueError('GGUF string too large')
    return _read(f, n).decode('utf-8')


def _value(f: BinaryIO, type: int) -> Value:
    if type in SCALARS:
        return Value(type, _unpack(f, SCALARS[type]))
    if type == 8:
        return Value(type, _string(f))
    if type == 9:
        et = _unpack(f, 'I')
        n = _unpack(f, 'Q')
        if et == 9 or n > 10_000_000:
            raise ValueError('unsupported or oversized GGUF array')
        if et in SCALARS:
            # Numeric arrays can contain a vocabulary-sized score/type table.
            # Read bounded blocks instead of dispatching once per scalar.
            values = []
            fmt = SCALARS[et]
            width = struct.calcsize('<' + fmt)
            for start in range(0, n, 8192):
                count = min(8192, n - start)
                values.extend(struct.unpack('<' + str(count) + fmt, _read(f, count * width)))
            return Value(type, values, et)
        return Value(type, [_value(f, et).data for _ in range(n)], et)
    raise ValueError(f'unsupported metadata type {type}')


def _write_string(f: BinaryIO, s: str) -> None:
    b = s.encode('utf-8')
    f.write(struct.pack('<Q', len(b)))
    f.write(b)


def _write_value(f: BinaryIO, v: Value) -> None:
    if v.type in SCALARS:
        f.write(struct.pack('<' + SCALARS[v.type], v.data))
    elif v.type == 8:
        _write_string(f, v.data)
    elif v.type == 9 and v.element_type in (*SCALARS, 8):
        f.write(struct.pack('<IQ', v.element_type, len(v.data)))
        for item in v.data:
            _write_value(f, Value(v.element_type, item))
    else:
        raise ValueError('unsupported metadata value')


class Reader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.metadata: dict[str, Value] = {}
        self.tensors: dict[str, Tensor] = {}
        file_size = self.path.stat().st_size
        with self.path.open('rb') as f:
            if _read(f, 4) != b'GGUF' or _unpack(f, 'I') != 3:
                raise ValueError('expected little-endian GGUF v3')
            nt, nk = _unpack(f, 'Q'), _unpack(f, 'Q')
            if nt > 100_000 or nk > 100_000:
                raise ValueError('oversized GGUF index')
            for _ in range(nk):
                k = _string(f)
                if k in self.metadata:
                    raise ValueError(f'duplicate metadata key {k}')
                self.metadata[k] = _value(f, _unpack(f, 'I'))
            self.alignment = self.metadata.get('general.alignment', Value(4, 32)).data
            if not isinstance(self.alignment, int) or self.alignment <= 0 or self.alignment & (self.alignment - 1):
                raise ValueError('invalid GGUF alignment')
            for _ in range(nt):
                name = _string(f)
                nd = _unpack(f, 'I')
                if not 1 <= nd <= 4:
                    raise ValueError('invalid tensor dimensions')
                shape = tuple(_unpack(f, 'Q') for _ in range(nd))
                type, offset = _unpack(f, 'I'), _unpack(f, 'Q')
                if name in self.tensors:
                    raise ValueError(f'duplicate tensor {name}')
                self.tensors[name] = Tensor(name, shape, type, offset, tensor_size(shape, type))
            self.data_offset = align(f.tell(), self.alignment)
        ranges = sorted((t.offset, t.offset + t.size) for t in self.tensors.values())
        previous_end = 0
        for start, end in ranges:
            if start < previous_end or self.data_offset + end > file_size:
                raise ValueError('overlapping or truncated tensor data')
            previous_end = end

    def data(self, name: str) -> bytes:
        t = self.tensors[name]
        with self.path.open('rb') as f:
            f.seek(self.data_offset + t.offset)
            return _read(f, t.size)

    def digest(self, name: str) -> str:
        t = self.tensors[name]
        h = hashlib.sha256()
        with self.path.open('rb') as f:
            f.seek(self.data_offset + t.offset)
            remaining = t.size
            while remaining:
                chunk = _read(f, min(remaining, 8 * 1024 * 1024))
                h.update(chunk)
                remaining -= len(chunk)
        return h.hexdigest()


def write(path: str | Path, metadata: dict[str, Value],
          tensors: list[tuple[str, tuple[int, ...], int, bytes]]) -> None:
    alignment = metadata.get('general.alignment', Value(4, 32)).data
    if alignment != 32:
        raise ValueError('writer requires alignment 32')
    names = [t[0] for t in tensors]
    if len(names) != len(set(names)):
        raise ValueError('duplicate tensor names')
    offsets = []
    position = 0
    for name, shape, type, data in tensors:
        if len(data) != tensor_size(shape, type):
            raise ValueError(f'incorrect payload length: {name}')
        offsets.append(position)
        position = align(position + len(data), alignment)
    with Path(path).open('xb') as f:
        f.write(struct.pack('<4sIQQ', b'GGUF', 3, len(tensors), len(metadata)))
        for key, value in metadata.items():
            _write_string(f, key)
            f.write(struct.pack('<I', value.type))
            _write_value(f, value)
        for (name, shape, type, _), offset in zip(tensors, offsets):
            _write_string(f, name)
            f.write(struct.pack('<I', len(shape)))
            f.write(struct.pack('<' + 'Q' * len(shape), *shape))
            f.write(struct.pack('<IQ', type, offset))
        f.write(bytes(align(f.tell(), alignment) - f.tell()))
        base = f.tell()
        for (_, _, _, data), offset in zip(tensors, offsets):
            f.write(bytes(base + offset - f.tell()))
            f.write(data)


def pack_q4_1(values: np.ndarray) -> bytes:
    """Quantize consecutive blocks of 32; low nibbles precede high nibbles.

    Block layout: F16 scale, F16 minimum, 16 packed bytes.
    https://github.com/ggml-org/llama.cpp/blob/7ab4ee7baad2d920464cbacfad4f4b07cf111fd2/ggml/src/ggml-quants.c
    """
    x = np.asarray(values, dtype=np.float32)
    if x.size % 32 or not np.isfinite(x).all():
        raise ValueError('Q4_1 needs finite, complete blocks')
    x = x.reshape(-1, 32)
    m = x.min(axis=1)
    d = (x.max(axis=1) - m) / 15
    inv = np.divide(1, d, out=np.zeros_like(d), where=d != 0)
    q = np.floor((x - m[:, None]) * inv[:, None] + 0.5).clip(0, 15).astype(np.uint8)
    result = np.empty((len(x), 20), dtype=np.uint8)
    result[:, :2] = d.astype('<f2').view(np.uint8).reshape(-1, 2)
    result[:, 2:4] = m.astype('<f2').view(np.uint8).reshape(-1, 2)
    result[:, 4:] = q[:, :16] | (q[:, 16:] << 4)
    return result.tobytes()
