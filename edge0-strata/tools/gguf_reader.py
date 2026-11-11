"""Minimal GGUF v3 header reader - no ggml, no gguf-py.

Written because the `gguf` PyPI package cannot open this artifact: its `GGMLQuantizationType`
enum has no member for type 42 (`Q2_0`), which is newer than the library. That is itself a P0.S6
finding - the format this engine is specialized for is ahead of the standard tooling.

This is the reference implementation for `src/artifact/` (architecture §15: "GGUF v3 reader (mmap,
no ggml dependency)"). It parses the header only; tensor data is never read.
"""
from __future__ import annotations

import dataclasses
import os
import pathlib
import struct
from typing import Any

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

# ggml_type values. 42 = Q2_0, the GSQ-RCO routed-expert encoding, absent from gguf-py as of 0.19.
GGML_TYPES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_2", 5: "Q4_3", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
    21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64",
    28: "F64", 29: "IQ1_M", 30: "BF16", 31: "Q4_0_4_4", 32: "Q4_0_4_8", 33: "Q4_0_8_8",
    34: "TQ1_0", 35: "TQ2_0", 36: "IQ4_NL_4_4", 37: "IQ4_NL_4_8", 38: "IQ4_NL_8_8",
    39: "MXFP4", 40: "NVFP4", 41: "Q4_0_8_8", 42: "Q2_0",
    # Agention's llama.cpp fork (agentionai/llama.cpp): the APR trellis expert types (128-value blocks)
    143: "PTQ1_0", 144: "TQ2_T", 145: "TQK6", 146: "TQK7",
    # pair-trellis family added for Gyro-L (agentionai llama.cpp release/gyro-l)
    147: "TQP5", 148: "TQP6", 149: "TQP7",
}

# GGUF metadata value type ids
GGUF_META = {
    0: ("u8", 1), 1: ("i8", 1), 2: ("u16", 2), 3: ("i16", 2), 4: ("u32", 4), 5: ("i32", 4),
    6: ("f32", 4), 7: ("bool", 1), 8: ("string", None), 9: ("array", None),
    10: ("u64", 8), 11: ("i64", 8), 12: ("f64", 8),
}

# ggml block geometry for the encodings this model uses: (block elements, bytes per block)
# This is what makes a per-tensor byte count checkable, and it is the contract the kernels share.
BLOCK_GEOMETRY: dict[str, tuple[int, int]] = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "F64": (1, 8),
    # one value per block: the IST-DAS DeepSeek-V4.1 first shard carries the Engram's hash parameters as integer tensors
    "I8": (1, 1), "I16": (1, 2), "I32": (1, 4), "I64": (1, 8),
    "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24),
    "Q8_0": (32, 34), "Q8_1": (32, 36),
    "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144), "Q5_K": (256, 176),
    "Q6_K": (256, 210), "Q8_K": (256, 292),
    "IQ2_XXS": (256, 66), "IQ2_XS": (256, 74), "IQ3_XXS": (256, 98), "IQ1_S": (256, 50),
    "IQ4_NL": (32, 18), "IQ3_S": (256, 110), "IQ2_S": (256, 82), "IQ4_XS": (256, 136),
    "IQ1_M": (256, 56), "Q2_0": (64, 18), "MXFP4": (32, 17), "NVFP4": (64, 36),
    # fp16 scale + trellis stream (32 steps x 8 / 6 / 7 bits); PTQ1_0: 26 B of trits + fp16 scale
    "TQ2_T": (128, 34), "TQK6": (128, 26), "TQK7": (128, 30), "PTQ1_0": (128, 28),
    # TQP5/6/7: fp16 scale + 8*K bytes of a 64-step x K-bit circular stream (K = 5/6/7)
    "TQP5": (128, 42), "TQP6": (128, 50), "TQP7": (128, 58),
}


@dataclasses.dataclass
class TensorInfo:
    name: str
    shape: list[int]
    type_id: int
    type_name: str
    offset: int

    @property
    def elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def expected_bytes(self) -> int | None:
        geom = BLOCK_GEOMETRY.get(self.type_name)
        if geom is None:
            return None
        block_elems, block_bytes = geom
        if self.elements % block_elems:
            return None
        return self.elements // block_elems * block_bytes


_META_FMT = {"u8": "B", "i8": "b", "u16": "H", "i16": "h", "u32": "I", "i32": "i",
             "f32": "f", "bool": "?", "u64": "Q", "i64": "q", "f64": "d"}


class GGUFFile:
    def __init__(self, path: pathlib.Path):
        self.path = pathlib.Path(path)
        self.metadata: dict[str, Any] = {}
        self.tensors: list[TensorInfo] = []
        self.version = 0
        self.alignment = 32
        with self.path.open("rb") as fh:
            self._size = os.fstat(fh.fileno()).st_size
            self._pos = 0
            try:
                self._parse(fh)
            except ValueError as e:
                raise ValueError(f"{self.path.name}: {e}") from None
        self.data_start = self._data_start

    # ------------------------------------------------------------------ internals
    def _read(self, fh, n: int) -> bytes:
        left = self._size - self._pos
        if n < 0 or n > left:
            raise ValueError(f"header ends early: {n:,} bytes needed at offset {self._pos:,}, {left:,} left")
        raw = fh.read(n)
        self._pos += len(raw)
        if len(raw) != n:
            raise ValueError(f"header ends early at offset {self._pos:,}")
        return raw

    def _unpack(self, fh, fmt: str) -> tuple:
        return struct.unpack(fmt, self._read(fh, struct.calcsize(fmt)))

    def _count(self, fh, count: int, min_bytes: int, what: str) -> int:
        if count * min_bytes > self._size - self._pos:
            raise ValueError(f"{count:,} {what} cannot fit in the {self._size - self._pos:,} bytes left")
        return count

    def _parse(self, fh) -> None:
        (magic,) = self._unpack(fh, "<I")
        if magic != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file (magic {magic:#x})")
        self.version, n_tensors, n_kv = self._unpack(fh, "<IQQ")
        if self.version != 3:
            raise ValueError(f"GGUF v{self.version}, this reader handles v3")
        for _ in range(self._count(fh, n_kv, 12, "metadata entries")):      # key length (8) + value type (4)
            key = self._str(fh)
            self.metadata[key] = self._value(fh)
        for _ in range(self._count(fh, n_tensors, 24, "tensors")):           # name length, n_dims, type, offset
            name = self._str(fh)
            (n_dims,) = self._unpack(fh, "<I")
            shape = list(self._unpack(fh, f"<{self._count(fh, n_dims, 8, 'dimensions')}Q"))
            type_id, offset = self._unpack(fh, "<IQ")
            self.tensors.append(TensorInfo(name, shape, type_id,
                                           GGML_TYPES.get(type_id, f"type{type_id}"), offset))
        align = self.metadata.get("general.alignment")
        if align is not None:
            if type(align) is not int or align <= 0 or align % 8:
                raise ValueError("general.alignment must be a positive multiple of 8")
            self.alignment = align
        pos = self._pos
        self._data_start = (pos + self.alignment - 1) // self.alignment * self.alignment

    def _str(self, fh) -> str:
        (n,) = self._unpack(fh, "<Q")
        return self._read(fh, n).decode("utf-8", "replace")

    def _meta_type(self, t: int) -> tuple[str, int | None]:
        if t not in GGUF_META:
            raise ValueError(f"unknown metadata value type {t}")
        return GGUF_META[t]

    def _value(self, fh):
        (t,) = self._unpack(fh, "<I")
        name, size = self._meta_type(t)
        if name == "string":
            return self._str(fh)
        if name == "array":
            et, count = self._unpack(fh, "<IQ")
            ename, esize = self._meta_type(et)
            if ename == "array":
                raise ValueError("nested metadata arrays are not supported")
            if ename == "string":
                return [self._str(fh) for _ in range(self._count(fh, count, 8, "array strings"))]
            self._count(fh, count, esize, "array elements")
            return list(self._unpack(fh, f"<{count}{_META_FMT[ename]}"))
        return self._unpack(fh, f"<{_META_FMT[name]}")[0]

    # ------------------------------------------------------------------ helpers
    def by_type(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self.tensors:
            out[t.type_name] = out.get(t.type_name, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def find(self, needle: str) -> list[TensorInfo]:
        return [t for t in self.tensors if needle in t.name]
