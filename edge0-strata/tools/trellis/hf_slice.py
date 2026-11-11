"""hf_slice.py - read single tensors (or slices of them) out of a remote safetensors repo with HTTP range requests, so a
quality test needs megabytes of a model, not the model.  No token is read or printed (public repos only; set HF_TOKEN in
the environment for gated ones - it is sent as a header and never printed).

  from hf_slice import Repo
  r = Repo("Qwen/Qwen3.8-Flash-Next")
  w = r.tensor("model.language_model.layers.0.mlp.experts.gate_up_proj", rows=slice(0, 1))   # expert 0 only
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import struct
import urllib.request

import numpy as np

_DT = {"BF16": (2, None), "F16": (2, np.float16), "F32": (4, np.float32), "F8_E4M3": (1, None), "I8": (1, np.int8),
       "U8": (1, np.uint8), "I32": (4, np.int32), "I64": (8, np.int64)}


def _get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    req = urllib.request.Request(url)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", "Bearer " + tok)
    if start is not None:
        req.add_header("Range", "bytes=%d-%d" % (start, end))
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def get_parallel(url: str, start: int, end: int, workers: int = 16, piece: int = 8 << 20) -> bytes:
    """bytes start..end (inclusive) of url, in `piece`-byte range requests on `workers` threads (one stream of a CDN
    object is slow, a dozen are not)."""
    spans = [(a, min(end, a + piece - 1)) for a in range(start, end + 1, piece)]
    if len(spans) == 1:
        return _get(url, start, end)

    def one(sp):
        for attempt in range(6):
            try:
                return _get(url, sp[0], sp[1])
            except Exception:
                if attempt == 5:
                    raise
        return b""
    with cf.ThreadPoolExecutor(workers) as ex:
        return b"".join(ex.map(one, spans))


def bf16_to_f32(raw: bytes | np.ndarray) -> np.ndarray:
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


class Repo:
    def __init__(self, repo: str, rev: str = "main"):
        self.repo, self.rev = repo, rev
        self.base = "https://huggingface.co/%s/resolve/%s/" % (repo, rev)
        self.map = json.loads(_get(self.base + "model.safetensors.index.json"))["weight_map"]
        self._hdr: dict[str, tuple[dict, int]] = {}

    def header(self, shard: str):
        if shard not in self._hdr:
            n = struct.unpack("<Q", _get(self.base + shard, 0, 7))[0]
            h = json.loads(_get(self.base + shard, 8, 8 + n - 1))
            self._hdr[shard] = (h, 8 + n)
        return self._hdr[shard]

    def info(self, name: str):
        shard = self.map[name]
        h, base = self.header(shard)
        return shard, h[name], base

    def tensor(self, name: str, lead: slice | int | None = None, workers: int = 16) -> np.ndarray:
        """The tensor `name` as float32.  lead: an index or slice on the first axis (e.g. an expert), fetched alone."""
        shard, t, base = self.info(name)
        shape = t["shape"]
        item = _DT[t["dtype"]][0]
        s, e = t["data_offsets"]
        inner = int(np.prod(shape[1:])) * item if len(shape) > 1 else item
        if lead is None:
            i0, i1 = 0, shape[0]
        elif isinstance(lead, int):
            i0, i1 = lead, lead + 1
        else:
            i0, i1 = lead.start or 0, lead.stop
        raw = get_parallel(self.base + shard, base + s + i0 * inner, base + s + i1 * inner - 1, workers)
        out_shape = [i1 - i0] + shape[1:]
        if t["dtype"] == "BF16":
            return bf16_to_f32(raw).reshape(out_shape)
        return np.frombuffer(raw, dtype=_DT[t["dtype"]][1]).astype(np.float32).reshape(out_shape)
