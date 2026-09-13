"""Strict GGUF v2/v3 reader; maps shards but never warms their payloads."""
from __future__ import annotations

import math
import mmap
from pathlib import Path
import re
import struct
import threading

from .source import CheckpointSource, TensorInfo

from .encodings import QUANT_SIZES
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
            6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_SPLIT = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")


class _Reader:
    def __init__(self, file):
        self.file = file
        self.size = file.seek(0, 2)
        file.seek(0)

    def read(self, n):
        if n < 0 or n > self.size - self.file.tell():
            raise ValueError("truncated GGUF header")
        if self.file.tell() + n > 256 * 1024**2:
            raise ValueError("GGUF header exceeds 256 MiB limit")
        data = self.file.read(n)
        if len(data) != n:
            raise ValueError("truncated GGUF header")
        return data

    def scalar(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self):
        n = self.scalar("Q")
        if n > 64 * 1024 * 1024:
            raise ValueError("GGUF string exceeds 64 MiB header limit")
        try:
            return self.read(n).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-8 in GGUF header") from exc

    def value(self, kind, depth=0):
        if kind in _SCALARS:
            return self.scalar(_SCALARS[kind])
        if kind == 8:
            return self.string()
        if kind == 9 and depth == 0:
            subtype, n = self.scalar("I"), self.scalar("Q")
            if n > 4_000_000 or n > self.size - self.file.tell():
                raise ValueError("GGUF metadata array exceeds header limit")
            return [self.value(subtype, depth + 1) for _ in range(n)]
        raise ValueError(f"unsupported GGUF metadata type {kind}")


def _header(file):
    r = _Reader(file)
    if r.read(4) != b"GGUF":
        raise ValueError("expected little-endian GGUF magic")
    version = r.scalar("I")
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}; expected 2 or 3")
    nt, nk = r.scalar("Q"), r.scalar("Q")
    if nt > 1_000_000 or nk > 1_000_000:
        raise ValueError("GGUF header counts exceed limit")
    meta = {}
    for _ in range(nk):
        key = r.string()
        if key in meta:
            raise ValueError(f"duplicate GGUF metadata key {key!r}")
        meta[key] = r.value(r.scalar("I"))
    alignment = meta.get("general.alignment", 32)
    if not isinstance(alignment, int) or alignment < 1 or alignment > 1 << 20 or alignment & (alignment - 1):
        raise ValueError("GGUF alignment must be a bounded power of two")
    tensors = []
    for _ in range(nt):
        name, ndim = r.string(), r.scalar("I")
        if not 1 <= ndim <= 4:
            raise ValueError(f"invalid dimension count for {name!r}")
        shape = tuple(reversed([r.scalar("Q") for _ in range(ndim)]))
        kind, offset = r.scalar("I"), r.scalar("Q")
        if kind not in QUANT_SIZES:
            raise ValueError(f"unsupported GGUF encoding {kind} in tensor {name!r}")
        block, size = QUANT_SIZES[kind]
        if any(d == 0 for d in shape) or shape[-1] % block:
            raise ValueError(f"invalid shape/block alignment for {name!r}: {shape}")
        if offset % alignment:
            raise ValueError(f"unaligned tensor {name!r}")
        tensors.append((name, shape, kind, offset, math.prod(shape) // block * size))
    base = (file.tell() + alignment - 1) // alignment * alignment
    # A metadata-only shard may end directly after its header, without padding.
    last = base
    for name, shape, kind, offset, size in sorted(tensors, key=lambda t: t[3]):
        if base + offset < last or base + offset + size > r.size:
            raise ValueError(f"overlapping or out-of-bounds GGUF tensor {name!r}")
        last = base + offset + size
    return meta, tensors, base


class GGUFSource(CheckpointSource):
    def __init__(self, path):
        path = Path(path).expanduser().resolve(strict=True)
        self.paths = []
        self._files = []
        self._maps = []
        self.metadata = {}
        self.tensors = {}
        self._read_lock = threading.Lock()
        self.bytes_read = 0
        self.largest_read_bytes = 0
        self.embedding_rows_read = 0
        try:
            match = _SPLIT.fullmatch(path.name)
            if match:
                prefix, number, total = match.groups()
                total, number = int(total), int(number)
                if not 1 <= number <= total <= 99999:
                    raise ValueError("invalid GGUF shard numbering")
                self.paths = [path.with_name(f"{prefix}-{i:05}-of-{total:05}.gguf")
                              for i in range(1, total + 1)]
            else:
                self.paths = [path]
            for shard, sibling in enumerate(self.paths):
                if not sibling.is_file():
                    raise FileNotFoundError(f"missing GGUF shard: {sibling}")
                f = sibling.open("rb")
                self._files.append(f)
                meta, entries, base = _header(f)
                count = meta.get("split.count", 1)
                if count != len(self.paths) or meta.get("split.no", 0) != shard:
                    raise ValueError(f"GGUF split metadata disagrees with filename: {sibling.name}")
                if shard == 0:
                    self.metadata = meta
                    arch = meta.get("general.architecture")
                    if arch not in ("qwen4exp", "qwen35moe"):
                        raise ValueError(f"unsupported GGUF architecture {arch!r}; expected qwen4exp or qwen35moe")
                else:
                    # Subsequent shards may carry split keys only.
                    for key, value in meta.items():
                        if key == "split.no":
                            continue
                        if key in self.metadata and self.metadata[key] != value:
                            raise ValueError(f"inconsistent shard metadata {key!r}")
                        if key.startswith(("qwen4exp.", "qwen35moe.", "tokenizer.")) and key not in self.metadata:
                            raise ValueError(f"architecture/tokenizer metadata only in later shard: {key}")
                for name, shape, kind, offset, size in entries:
                    if name in self.tensors:
                        raise ValueError(f"duplicate GGUF tensor {name!r}")
                    self.tensors[name] = TensorInfo(name, shape, kind, base + offset, size, shard)
                self._maps.append(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
            total_tensors = self.metadata.get("split.tensors.count", len(self.tensors))
            if len(self.paths) > 1 and "split.tensors.count" not in self.metadata:
                raise ValueError("split GGUF is missing split.tensors.count")
            if total_tensors != len(self.tensors):
                raise ValueError(f"GGUF tensor count mismatch: expected {total_tensors}, found {len(self.tensors)}")
        except BaseException:
            self.close()
            raise

    def read_bytes(self, name, start, size):
        e = self.tensors[name]
        if not self._maps:
            raise RuntimeError("checkpoint is closed")
        if start < 0 or size < 0 or start + size > e.nbytes:
            raise ValueError(f"byte range outside tensor {name!r}")
        with self._read_lock:
            self.bytes_read += size
            self.largest_read_bytes = max(self.largest_read_bytes, size)
        return self._maps[e.shard][e.offset + start:e.offset + start + size]

    def read_rows(self, name, start, count):
        from .quant import decode
        e = self.tensors[name]
        if start < 0 or count < 0 or start + count > math.prod(e.shape[:-1]):
            raise ValueError(f"row range outside tensor {name!r}")
        block, size = QUANT_SIZES[e.encoding]
        stride = e.shape[-1] // block * size
        if name == "per_layer_token_embd.weight":
            self.embedding_rows_read += count
        return decode(self.read_bytes(name, start * stride, count * stride), e.encoding).reshape(count, e.shape[-1])

    def close(self):
        for mm in self._maps:
            mm.close()
        self._maps.clear()
        for f in self._files:
            f.close()
        self._files.clear()
