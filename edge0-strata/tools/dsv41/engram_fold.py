#!/usr/bin/env python3
"""engram_fold.py - the Engram tensors a pack needs, folded out of the quant's own shards, so a pack built around their
`engram-IQ4_NL.gguf` uses none of our substitutes.

The IST-DAS sets keep the Engram as three per-layer tensors (`blk.N.engram_k`, `engram_q`, `engram_wkv`); the engine's
layout wants `engram.gate` (k*q folded, [4, 5120] F32) and `engram.wkv` ([25600, 6144] BF16).  Theirs are stored the other
way round - their `engram_wkv` is `[6144, 25600]` and their `k/q` are `[5120, 4]` - so this tool dequantises, transposes and
writes the engine's layout:

  gate = transpose(k) * transpose(q)      F32
  wkv  = transpose(dequant(wkv))          BF16

Output is a `dense_requant.py`-style folder (dense.json + dense.bin) for `pack_gguf.py --dense-override`.

usage:
  engram_fold.py --model /data/helcig/set200/2.00bpw --out /data/helcig/engram_fold_200 [--layers 1,14]
  pack_gguf.py --model <set> --out <pack> --engram <their engram dir> \
      --skip-tensors 'attn_output_a|engram_(k|q|wkv)\\.' --dense-override <fold dir> ...
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pack_gguf  # noqa: E402  (its GGUF reader and its type table)


def shards(model: pathlib.Path):
    return [pack_gguf.Gguf(p) for p in sorted(model.glob("*.gguf"))]


def find(gs, name):
    for g in gs:
        for t in g.tensors:
            if t[0] == name:
                return g, t
    return None, None


def nbytes(typ: int, dims) -> int:
    tname, per, bpb = pack_gguf.TYPES[typ]
    nel = 1
    for d in dims:
        nel *= int(d)
    return nel // per * bpb if per > 1 else nel * bpb


def dequant_q5k(raw: np.ndarray, nblocks: int) -> np.ndarray:
    """ggml block_q5_K (176 B, 256 values): f16 d, f16 dmin, uint8 scales[12], qh[32], qs[128]."""
    b = raw.reshape(nblocks, 176)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
    dmin = b[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(-1)
    scales, qh, qs = b[:, 4:16].astype(np.int32), b[:, 16:48].astype(np.int32), b[:, 48:176].astype(np.int32)
    sc = np.empty((nblocks, 8), np.float32)
    mn = np.empty((nblocks, 8), np.float32)
    # ggml's get_scale_min_k4: the first four 6-bit scales are bytes 0..3 (low) and 4..7 (mins);
    # the last four take the top two bits of bytes 0..3 plus the low/high nibbles of 8..11
    for j in range(8):
        if j < 4:
            sc[:, j] = scales[:, j] & 63
            mn[:, j] = scales[:, j + 4] & 63
        else:
            sc[:, j] = (scales[:, j + 4] & 0xF) | ((scales[:, j - 4] >> 6) << 4)
            mn[:, j] = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
    out = np.empty((nblocks, 256), dtype=np.float32)
    for j in range(4):
        for half in range(2):
            d1 = d * sc[:, j * 2 + half]
            m1 = dmin * mn[:, j * 2 + half]
            q = qs[:, j * 32:(j + 1) * 32]
            h = (qh >> (j * 2 + half)) & 1
            nib = ((q & 0xF) if half == 0 else (q >> 4)) + (h << 4)
            out[:, j * 64 + half * 32: j * 64 + half * 32 + 32] = d1[:, None] * nib - m1[:, None]
    return out.reshape(-1)


def dequant(g, t) -> np.ndarray:
    name, typ, dims, off = t
    tname = pack_gguf.TYPES[typ][0]
    nb = nbytes(typ, dims)
    # Gguf's tensor offsets are relative to the shard's data section; pack_gguf adds data_start when it writes a pack
    raw = np.fromfile(g.path, dtype=np.uint8, count=nb, offset=g.data_start + off)
    if tname == "Q5_K":
        return dequant_q5k(raw, nb // 176).astype(np.float64)
    if tname == "Q4_K":
        raise SystemExit("%s: Q4_K not handled by the fold yet (Q5_K, F32, F16, BF16 are)" % name)
    if tname == "F32":
        return raw.view(np.float32).astype(np.float64)
    if tname == "BF16":
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    if tname == "F16":
        return raw.view(np.float16).astype(np.float64)
    raise SystemExit("%s: type %s is not one the fold reads" % (name, tname))


def to_bf16_bytes(v: np.ndarray) -> bytes:
    """F32 -> BF16, round to nearest even (what ggml's ggml_compute_fp32_to_bf16 does)."""
    u = v.astype(np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    rounded = (u + 0x7FFF + lsb) >> 16
    return rounded.astype(np.uint16).tobytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the folder of the quant's GGUF shards")
    ap.add_argument("--out", required=True, help="the override folder to write (dense.json + dense.bin)")
    ap.add_argument("--layers", default="", help="comma list of Engram layers; default: every layer with an engram_k")
    ap.add_argument("--gate-kv-from", default="", help="a known-good pack to copy engram.gate/wkv from (the ple table stays the only difference)")
    a = ap.parse_args()
    model = pathlib.Path(a.model).expanduser().resolve()
    out = pathlib.Path(a.out).expanduser().resolve()
    gs = shards(model)
    if not gs:
        raise SystemExit("%s: no .gguf files" % model)
    names = {t[0] for g in gs for t in g.tensors}
    if a.layers:
        layers = [int(x) for x in a.layers.split(",") if x.strip()]
    else:
        layers = sorted({int(n.split(".")[1]) for n in names if n.startswith("blk.") and ".engram_k." in n})
    out.mkdir(parents=True, exist_ok=True)
    idx: dict[str, list] = {}
    blobs: list[bytes] = []
    off = 0

    def add(name: str, tname: str, data: bytes, shape: list[int]) -> None:
        nonlocal off
        pad = (-off) % 64
        if pad:
            blobs.append(b"\0" * pad)
            off += pad
        blobs.append(data)
        idx[name] = [tname, "dense.bin", off, len(data), shape]
        off += len(data)

    # --gate-kv-from PACK: take the gate and the kv from a pack that is known good (the engine's own layout, already in the
    # right orientation) instead of deriving them from their k/q/wkv.  That leaves the ple table as the only difference
    # between the two packs, which is the only way to score their IQ4_NL Engram against our MXFP8 one.
    if a.gate_kv_from:
        src = pathlib.Path(a.gate_kv_from).expanduser().resolve()
        md = [l.split(None, 1)[1] for l in (src / "ds4.txt").read_text().splitlines() if l.startswith("model_dir ")][0]
        files, want = {}, {}
        for line in (src / "tensors.txt").read_text().splitlines():
            f = line.split()
            if not f:
                continue
            if f[0] == "file":
                files[int(f[1])] = " ".join(f[2:])
            elif f[0] == "t" and re.match(r"layers\.\d+\.engram\.(gate|wkv)\.weight$", f[1]):
                want[f[1]] = (f[2], int(f[3]), int(f[4]), int(f[5]), [int(x) for x in f[6:]])
        for name, (tname, fi, o, nb, shape) in sorted(want.items()):
            p = (pathlib.Path(md) / files[fi]).resolve()
            add(name, tname, np.fromfile(p, dtype=np.uint8, count=nb, offset=o).tobytes(), shape)
            print("%s: copied from %s (%s, %d bytes)" % (name, src.name, tname, nb))
        (out / "dense.bin").write_bytes(b"".join(blobs))
        json.dump(idx, open(out / "dense.json", "w"), indent=1)
        print("wrote %s (%.2f MB), dense.json with %d tensors" % (out / "dense.bin", os.path.getsize(out / "dense.bin") / 1e6, len(idx)))
        return 0

    for il in layers:
        kg, kt = find(gs, "blk.%d.engram_k.weight" % il)
        qg, qt = find(gs, "blk.%d.engram_q.weight" % il)
        wg, wt = find(gs, "blk.%d.engram_wkv.weight" % il)
        if kt is None or qt is None:
            print("layer %d: no engram_k/q in these shards, skipped" % il)
            continue
        # their k/q are [5120, 4] (ne0 = 5120): the engine's gate is [4, 5120]
        k = dequant(kg, kt).reshape(kt[2][1], kt[2][0]).T
        q = dequant(qg, qt).reshape(qt[2][1], qt[2][0]).T
        if k.shape != q.shape:
            raise SystemExit("layer %d: k %s and q %s differ after the transpose" % (il, k.shape, q.shape))
        gate = (k * q).astype(np.float32)
        add("layers.%d.engram.gate.weight" % il, "F32", gate.tobytes(), [gate.shape[1], gate.shape[0]])
        note = "MISSING"
        if wt is not None:
            w = dequant(wg, wt).reshape(wt[2][1], wt[2][0]).T.astype(np.float32)
            add("layers.%d.engram.wkv.weight" % il, "BF16", to_bf16_bytes(w), [w.shape[1], w.shape[0]])
            note = "BF16 [%d, %d]" % (w.shape[1], w.shape[0])
        print("layer %d: gate [%d, %d] F32, wkv %s" % (il, gate.shape[1], gate.shape[0], note))

    (out / "dense.bin").write_bytes(b"".join(blobs))
    json.dump(idx, open(out / "dense.json", "w"), indent=1)
    print("wrote %s (%.2f MB), dense.json with %d tensors" % (out / "dense.bin", os.path.getsize(out / "dense.bin") / 1e6, len(idx)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
