"""Zero-copy byte-range access into safetensors shards.

A single expert is a contiguous byte slice of one stacked
[num_experts, ...] tensor; the streaming layer reads those slices straight
out of an mmap with no full-tensor materialization.
"""

from __future__ import annotations

import json
import mmap
import struct

import numpy as np


class SafetensorsMmap:
    """One safetensors shard, opened as a byte-range mmap.

    ``entries[name]`` holds ``{"offset", "size", "dtype", "shape"}``;
    ``raw(name)`` returns a zero-copy uint8 view of the tensor's bytes.
    """

    def __init__(self, path: str):
        self.path = path
        self._file = open(path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        header_len = struct.unpack("<Q", self._mm[:8])[0]
        header = json.loads(self._mm[8:8 + header_len])
        self.payload_base = 8 + header_len
        self.entries = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            begin, end = meta["data_offsets"]
            self.entries[name] = {
                "offset": self.payload_base + begin,
                "size": end - begin,
                "dtype": meta["dtype"],
                "shape": tuple(meta["shape"]),
            }

    def advise_willneed(self):
        """OS readahead hint over the whole shard (advisory)."""
        try:
            mmap_madv = getattr(mmap, "MADV_WILLNEED", None)
            if mmap_madv is not None:
                self._mm.madvise(mmap_madv)
        except Exception:  # noqa: BLE001 — advisory only
            pass

    def seq_read(self, chunk: int = 1 << 24):
        """Force every page resident: one sequential pass over the shard.

        macOS madvise only warms roughly half the file, so a full read is
        the reliable way to remove per-expert page-fault cost from the
        first request."""
        total = len(self._mm)
        for off in range(0, total, chunk):
            _ = self._mm[off:off + chunk]

    def raw(self, name: str) -> np.ndarray:
        e = self.entries[name]
        return np.frombuffer(
            self._mm, dtype=np.uint8, count=e["size"], offset=e["offset"]
        )

    def close(self):
        self._mm.close()
        self._file.close()


def bf16_bits_to_f32(data: np.ndarray) -> np.ndarray:
    """Raw BF16 bytes -> float32 array (bit pattern shifted into fp32)."""
    u16 = data.view("<u2")
    return (u16.astype(np.uint32) << 16).view(np.float32)


def u32_view(data: np.ndarray, shape) -> np.ndarray:
    """uint8 bytes -> uint32 view reshaped to ``shape`` (quantized payloads
    are stored as packed uint32 words)."""
    return data.view("<u4").reshape(shape)
