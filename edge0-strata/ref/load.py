"""ref/load.py - P1.S4: GGUF -> FP32 tensors on demand, with LAZY per-expert dequantization.

The plan is explicit: "experts are dequantized lazily per (layer, expert) - do not materialize 120B
parameters". That is the whole design constraint here, and it is what `expert()` below exists for: the
routed-expert tensors are fused `[in, out, experts]`, and touching one expert must not touch the other
511.

Dequantization itself is NOT reimplemented. `gguf-py` handles the twelve standard encodings and is
already the reference this project validates its C++ against; `Q2_0` (type 42) is the one gguf-py
cannot represent at all, so it is served by this project's own decoder. That split is deliberate: it
means a bug in `tools/gguf_writer.dequantize_q2_0` cannot hide behind gguf-py and vice versa.

    from ref.load import Gguf
    g = Gguf("path.gguf")
    w = g.tensor("blk.0.attn_gate.weight")      # (out, in) fp32, whole tensor
    e = g.expert("blk.0.ffn_gate_exps.weight", 7)  # (out, in) fp32, ONE expert only
"""
from __future__ import annotations

import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
from gguf_reader import GGUFFile                     # noqa: E402
from gguf_writer import dequantize_q2_0              # noqa: E402

# gguf-py names differ from our reader's names in a few places; map only what differs.
_GGUF_PY_NAME = {"Q4_0": "Q4_0", "Q8_0": "Q8_0", "Q4_K": "Q4_K", "Q5_K": "Q5_K", "Q6_K": "Q6_K",
                 "Q3_K": "Q3_K", "IQ4_NL": "IQ4_NL", "IQ4_XS": "IQ4_XS", "Q5_0": "Q5_0"}

# (elements per block, bytes per block) - mirrors block_geometry() in src/artifact/gguf_reader.cpp
GEOM = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2),
        "Q4_0": (32, 18), "Q5_0": (32, 22), "Q8_0": (32, 34), "IQ4_NL": (32, 18),
        "Q3_K": (256, 110), "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210),
        "IQ4_XS": (256, 136), "Q2_0": (64, 18)}


def _dequant_flat(type_name: str, raw: np.ndarray) -> np.ndarray:
    """One dequantization dispatch, shared by tensor()/expert()/rows().

    Q2_0 is this project's own decoder because gguf-py's enum cannot represent type 42 at all;
    everything else goes through gguf-py, which is already the reference this project validates its
    C++ against.  Keeping one dispatch means a type added in one path cannot be forgotten in another.
    """
    if type_name == "Q2_0":
        return dequantize_q2_0(raw.tobytes())
    if type_name == "BF16":
        return _bf16_to_f32(raw)
    if type_name == "F32":
        return raw.view(np.float32)
    if type_name == "F16":
        return raw.view(np.float16).astype(np.float32)
    from gguf import quants, GGMLQuantizationType as Q
    return quants.dequantize(raw, Q[_GGUF_PY_NAME[type_name]]).astype(np.float32)


def _bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def _discover_shards(path) -> list[pathlib.Path]:
    """Resolve a GGUF argument to its shard list.

    The real artifact is TWO files and the split is not incidental: shard 2 holds exactly one tensor,
    `per_layer_token_embd.weight` (26.82 GiB). A reader that opens one file can therefore load the
    whole model EXCEPT the PLE table and will not fail until a forward pass reaches layer 1 - so
    shard discovery is not a convenience, it is a correctness requirement for the real model.

    Accepts a directory (all `*.gguf` in it), any single shard (siblings found by the
    `-NNNNN-of-MMMMM` suffix), or an explicit list.
    """
    if isinstance(path, (list, tuple)):
        return [pathlib.Path(p) for p in path]
    p = pathlib.Path(path)
    if p.is_dir():
        # A directory is ambiguous in practice: the project root holds BOTH the model shards (in a
        # subdirectory) and an unrelated `mmproj-*.gguf`.  Globbing `*.gguf` there silently loaded
        # the multimodal projector instead of the model.  So prefer an explicit shard set, accept a
        # single unambiguous file, and REFUSE to guess otherwise.
        shards = sorted(p.glob("*-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9].gguf"))
        if shards:
            totals = {re.match(r"^.*-(\d{5})-of-(\d{5})\.gguf$", s.name).group(2) for s in shards}
            if len(totals) != 1:
                raise ValueError(f"{p}: shard files from more than one split: {sorted(totals)}")
            return shards
        found = sorted(p.glob("*.gguf"))
        if len(found) == 1:
            return found
        if not found:
            raise FileNotFoundError(f"no .gguf files in {p}")
        raise ValueError(
            f"{p}: {len(found)} .gguf files and none is a shard set, so which is the model is "
            f"ambiguous: {[f.name for f in found]}. Pass one file explicitly.")
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", p.name)
    if m:
        stem, _, total = m.groups()
        sibs = sorted(p.parent.glob(f"{stem}-*-of-{total}.gguf"))
        if len(sibs) != int(total):
            raise FileNotFoundError(
                f"{p.name}: expected {total} shards, found {len(sibs)} in {p.parent}")
        return sibs
    return [p]


class Gguf:
    """Read-only, demand-paged view of a (possibly multi-shard) GGUF.

    Nothing is dequantized until asked for.  `g` is the FIRST shard's `GGUFFile`, so metadata and the
    tensor directory are reachable as before; reads route to whichever shard owns the tensor.
    """

    def __init__(self, path):
        self.paths = _discover_shards(path)
        self.path = self.paths[0]
        self.shards: list[tuple[GGUFFile, object]] = []
        self._by_name: dict[str, tuple[int, object]] = {}
        for i, sp in enumerate(self.paths):
            gf = GGUFFile(sp)
            self.shards.append((gf, sp.open("rb")))
            for t in gf.tensors:
                if t.name in self._by_name:
                    raise ValueError(f"{sp.name}: duplicate tensor {t.name} across shards")
                self._by_name[t.name] = (i, t)
        self.g = self.shards[0][0]           # metadata / directory / alignment live in shard 1

    # ---- raw bytes
    def _raw(self, name: str, nbytes: int, byte_offset: int = 0) -> bytes:
        i, t = self._by_name[name]
        gf, fh = self.shards[i]
        fh.seek(gf.data_start + t.offset + byte_offset)
        return fh.read(nbytes)

    def info(self, name: str):
        return self._by_name[name][1]

    def row_bytes(self, t) -> int:
        el, by = GEOM[t.type_name]
        return t.shape[0] // el * by

    # ---- full tensors
    def tensor(self, name: str) -> np.ndarray:
        """Dequantize a whole tensor to FP32, shape as GGUF order reversed to (out, in[, experts])."""
        t = self.info(name)
        el, by = GEOM[t.type_name]
        if t.elements % el:
            raise ValueError(f"{name}: {t.elements} not a multiple of block {el}")
        raw = np.frombuffer(self._raw(name, t.elements // el * by), dtype=np.uint8)
        flat = _dequant_flat(t.type_name, raw)
        # GGUF dim 0 varies fastest; numpy's last axis does, so reverse and transpose to (out, in).
        return flat.reshape(tuple(reversed(t.shape)))

    # ---- the lazy row path
    def rows(self, name: str, idx) -> np.ndarray:
        """Gather specific ROWS of a 2-D tensor without decoding the rest of it.

        This exists for `per_layer_token_embd.weight`, which is `[160, 320001536]` IQ4_NL = **26.82
        GiB** - 43% of the artifact, the entire second shard - and is read 16 rows per token by one
        layer (see round 98's finding). Calling `tensor()` on it would dequantize 320 million rows to
        FP32, about 205 GB, to use 2.5 KB of them.

        A row is contiguous in GGUF (ne0 varies fastest), so one row is `row_bytes` at
        `row * row_bytes`. The type's block size must divide the row width; for this table it does
        (160 = 5 x 32 for IQ4_NL).
        """
        t = self.info(name)
        if len(t.shape) != 2:
            raise ValueError(f"{name}: rows() needs a 2-D tensor, got shape {t.shape}")
        el, by = GEOM[t.type_name]
        width = t.shape[0]
        if width % el:
            raise ValueError(f"{name}: row width {width} is not a multiple of block {el}")
        rb = self.row_bytes(t)
        idx = np.asarray(idx, dtype=np.int64).ravel()
        if idx.size and (idx.min() < 0 or idx.max() >= t.shape[1]):
            raise ValueError(f"{name}: row index out of range (0..{t.shape[1] - 1})")
        out = np.empty((idx.size, width), dtype=np.float32)
        for k, r in enumerate(idx):
            raw = np.frombuffer(self._raw(name, rb, int(r) * rb), dtype=np.uint8)
            out[k] = _dequant_flat(t.type_name, raw)
        return out

    # ---- the lazy path
    def expert(self, name: str, expert_id: int) -> np.ndarray:
        """One expert out of a fused `[in, out, experts]` tensor, WITHOUT touching the others.

        In GGUF order the expert index is the LAST axis, so expert e's matrix occupies a contiguous
        run of `in*out` elements at offset e*in*out. Only that run is read and decoded - which is the
        plan's "do not materialize 120B parameters" requirement, and the reason the engine can run at
        all on a machine where the expert set is 34 GB.
        """
        t = self.info(name)
        if len(t.shape) != 3:
            raise ValueError(f"{name}: not a fused expert tensor (shape {t.shape})")
        in_f, out_f, n_exp = t.shape
        if not 0 <= expert_id < n_exp:
            raise ValueError(f"{name}: expert {expert_id} out of range (0..{n_exp - 1})")
        el, by = GEOM[t.type_name]
        per_expert = in_f * out_f
        if per_expert % el:
            raise ValueError(f"{name}: per-expert element count {per_expert} not a multiple of {el}")
        # byte offset of expert e: whole experts before it, in blocks
        start = expert_id * (per_expert // el) * by
        raw = np.frombuffer(self._raw(name, per_expert // el * by, start), dtype=np.uint8)
        flat = _dequant_flat(t.type_name, raw)
        return flat.reshape((out_f, in_f))

    def close(self) -> None:
        for _, fh in self.shards:
            fh.close()
