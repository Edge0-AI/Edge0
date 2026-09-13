"""Metadata and bounded tensor access, independent of model execution."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]  # numpy order: last dimension contiguous
    encoding: int | str
    offset: int
    nbytes: int
    shard: int


class CheckpointSource(ABC):
    metadata: dict
    tensors: dict[str, TensorInfo]

    @abstractmethod
    def read_bytes(self, name: str, start: int, size: int) -> bytes:
        """Copy exactly the requested range; callers never hold mmap views."""

    @abstractmethod
    def read_rows(self, name: str, start: int, count: int) -> np.ndarray:
        """Read flattened outer rows, preserving the contiguous inner dimension."""

    @abstractmethod
    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class SafetensorsSource(CheckpointSource):
    def __init__(self, directory):
        from edge0.streaming.mmap import SafetensorsMmap
        self.shards = []
        self.tensors = {}
        self.metadata = {}
        try:
            directory = Path(directory)
            with (directory / "config.json").open() as f:
                self.metadata = json.load(f)
            for path in sorted(directory.glob("model*.safetensors")):
                shard = SafetensorsMmap(str(path))
                self.shards.append(shard)
                for name, e in shard.entries.items():
                    if name in self.tensors:
                        raise ValueError(f"duplicate tensor {name!r}")
                    self.tensors[name] = TensorInfo(name, e["shape"], e["dtype"],
                        e["offset"], e["size"], len(self.shards) - 1)
            if not self.shards:
                raise FileNotFoundError(f"no model*.safetensors under {directory}")
        except BaseException:
            self.close()
            raise

    def read_bytes(self, name, start, size):
        e = self.tensors[name]
        if start < 0 or size < 0 or start + size > e.nbytes:
            raise ValueError(f"byte range outside tensor {name!r}")
        return self.shards[e.shard]._mm[e.offset + start:e.offset + start + size]

    def read_rows(self, name, start, count):
        e = self.tensors[name]
        rows = math.prod(e.shape[:-1])
        if start < 0 or count < 0 or start + count > rows:
            raise ValueError(f"row range outside tensor {name!r}")
        dtype = {"F32": "<f4", "F16": "<f2", "BF16": "<u2", "U32": "<u4",
                 "I32": "<i4", "I64": "<i8", "U8": "u1", "I8": "i1"}[e.encoding]
        stride = e.shape[-1] * np.dtype(dtype).itemsize
        out = np.frombuffer(self.read_bytes(name, start * stride, count * stride), dtype=dtype)
        if e.encoding == "BF16":
            out = (out.astype(np.uint32) << 16).view(np.float32)
        return out.reshape(count, e.shape[-1])

    def close(self):
        for shard in self.shards:
            shard.close()
        self.shards.clear()


def open_checkpoint(path) -> CheckpointSource:
    path = Path(path).expanduser().resolve(strict=True)
    if path.is_dir():
        return SafetensorsSource(path)
    from .gguf import GGUFSource
    return GGUFSource(path)
