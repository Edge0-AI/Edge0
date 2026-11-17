# Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
# Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
"""tools/ds4/st.py - every tensor of a DeepSeek-V4.1 model directory, read in place.

    st = SafeTensors(Path("~/models/DSV4").expanduser())
    st.t["layers.0.attn.wq_a.trellis"]   -> (file, dtype, shape, absolute offset, bytes)
    st.raw(name), st.f32(name)            -> numpy views / float32 copies

The index's own files come first; any other .safetensors in the directory (the Engram shards 47 / 48 of the base
repo, model-0004[78]-of-00048.safetensors) only adds names the index does not have.
"""
from __future__ import annotations

import json
import pathlib
import struct

import numpy as np

DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "I32": 4, "I16": 2, "F8_E4M3": 1, "F8_E8M0": 1, "U8": 1, "I8": 1,
               "I64": 8}


class SafeTensors:
    def __init__(self, d: pathlib.Path):
        self.dir = pathlib.Path(d)
        ix = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        primary = sorted(set(ix.values()))
        others = sorted(p.name for p in self.dir.glob("*.safetensors") if p.name not in primary)
        self.files: list[str] = []
        self.t: dict[str, tuple] = {}
        self.mm: dict[str, np.memmap] = {}
        for fn in primary + others:
            p = self.dir / fn
            with p.open("rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            base = 8 + n
            self.files.append(fn)
            for k, v in h.items():
                if k == "__metadata__" or k in self.t:
                    continue
                a, b = v["data_offsets"]
                self.t[k] = (fn, v["dtype"], tuple(v["shape"]), base + a, b - a)
            self.mm[fn] = np.memmap(p, dtype=np.uint8, mode="r")

    def has(self, name: str) -> bool:
        return name in self.t

    def raw(self, name: str) -> np.ndarray:
        fn, _, _, off, nb = self.t[name]
        return self.mm[fn][off:off + nb]

    def f32(self, name: str) -> np.ndarray:
        """The tensor as float32 (exact for F32/F16/BF16), shaped as stored."""
        _, dt, shape, _, _ = self.t[name]
        r = self.raw(name)
        if dt == "F32":
            a = r.view("<f4")
        elif dt == "F16":
            a = r.view("<f2").astype(np.float32)
        elif dt == "BF16":
            a = (r.view("<u2").astype(np.uint32) << 16).view(np.float32)
        else:
            raise ValueError("%s is %s, not a float tensor" % (name, dt))
        return a.reshape(shape)
