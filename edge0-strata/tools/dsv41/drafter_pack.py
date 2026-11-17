# Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
# Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
"""tools/ds4/drafter_pack.py - DeepSeek-V4.1's DSpark drafter (the speculative decoder the GGUFs leave out) for a GGUF
install: its three blocks from DeepSeek's own checkpoint (the shards that hold only them, model-00044..46-of-00048),
in the formats the GGUF path runs, as one file beside them that pack_gguf.py adds to the pack.

    python tools/ds4/drafter_pack.py --dir <folder with the three shards>     -> dspark.bin + dspark.json there

Every value is kept exactly: the FP8 (E4M3) matrices with their E8M0 scale a 32 x 32 block become BF16 (an E4M3 value
times a power of two has at most four significant bits); the routed experts (FP4 E2M1 codes two to a byte, the first in
the low nibble, an E8M0 scale a 32) become ggml's MXFP4 blocks (the scale byte, then value j's code in byte j's low
nibble and value j + 16's in its high one); BF16 and F32 tensors are copied.  dspark.json lists every tensor (name,
dtype, offset, bytes, shape) under the checkpoint's names, as tensors.txt does.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import struct
import sys

import numpy as np

SHARDS = ("model-00044-of-00048.safetensors", "model-00045-of-00048.safetensors", "model-00046-of-00048.safetensors")
ALIGN = 256


def e4m3_table() -> np.ndarray:
    t = np.zeros(256, np.float32)
    for c in range(256):
        s, e, m = c >> 7, (c >> 3) & 15, c & 7
        v = np.nan if (e == 15 and m == 7) else (m / 8.0) * 2.0 ** -6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)
        t[c] = -v if s else v
    return t


E4M3 = e4m3_table()


def shard(path: pathlib.Path):
    raw = np.memmap(path, np.uint8, mode="r")
    n = struct.unpack("<Q", bytes(raw[:8]))[0]
    head = json.loads(bytes(raw[8:8 + n]))
    head.pop("__metadata__", None)
    return raw, head, 8 + n


def data(raw, base: int, e: dict) -> np.ndarray:
    a, b = e["data_offsets"]
    return raw[base + a:base + b]


def bf16(x: np.ndarray) -> np.ndarray:
    """float32 -> BF16 bits, rounded to nearest even (exact here)."""
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def fp8_to_bf16(w: np.ndarray, s: np.ndarray, rows: int, cols: int) -> np.ndarray:
    w = w.reshape(rows, cols)
    scale = np.exp2(s.reshape((rows + 31) // 32, (cols + 31) // 32).astype(np.float32) - 127.0)
    v = E4M3[w] * np.repeat(np.repeat(scale, 32, 0), 32, 1)[:rows, :cols]
    return bf16(v)


def fp4_to_mxfp4(w: np.ndarray, s: np.ndarray, rows: int, k: int) -> np.ndarray:
    nb = k // 32
    p = w.reshape(rows, nb, 16)
    codes = np.empty((rows, nb, 32), np.uint8)
    codes[..., 0::2] = p & 15
    codes[..., 1::2] = p >> 4
    out = np.empty((rows, nb, 17), np.uint8)
    out[..., 0] = s.reshape(rows, nb)
    out[..., 1:] = codes[..., :16] | (codes[..., 16:] << 4)
    return out


QTYPES = {"Q8_0": (8, 32, 34), "Q2_K": (10, 256, 84), "Q3_K": (11, 256, 110), "Q4_K": (12, 256, 144),
          "Q5_K": (13, 256, 176), "Q6_K": (14, 256, 210), "IQ2_XXS": (16, 256, 66), "IQ3_XXS": (18, 256, 98),
          "IQ3_S": (21, 256, 110), "IQ2_S": (22, 256, 82), "IQ4_XS": (23, 256, 136)}
DENSE_KEEP = ("markov_head", "confidence_head", "norm", "hc_", "ffn.gate.", "attn_sink")
E2M1 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)


def requant(d: pathlib.Path, name: str, experts: str, dense: str, lib: str, jobs: int, out=None) -> int:
    """dspark.bin's tensors again with the routed experts in `experts` and the dense matrices (the ones the engine reads
    as matrices) in `dense`, through ggml's quantizers (libggml-base): dspark-<name>.bin / .json beside it - a smaller
    drafter (pack_gguf.py --drafter dspark-<name>)."""
    import ctypes
    from concurrent.futures import ThreadPoolExecutor
    g = ctypes.CDLL(lib)
    g.ggml_quantize_chunk.restype = ctypes.c_size_t
    g.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                      ctypes.c_int64, ctypes.c_void_p]
    src = json.loads((d / "dspark.json").read_text())
    raw = np.memmap(d / "dspark.bin", np.uint8, mode="r")

    def f32(dt, b):
        if dt == "BF16":
            return (np.frombuffer(b, np.uint16).astype(np.uint32) << 16).view(np.float32)
        if dt == "MXFP4":
            blk = np.frombuffer(b, np.uint8).reshape(-1, 17)
            v = np.empty((blk.shape[0], 32), np.float32)
            v[:, :16] = E2M1[blk[:, 1:] & 15]
            v[:, 16:] = E2M1[blk[:, 1:] >> 4]
            return (v * np.ldexp(np.float32(1), blk[:, :1].astype(np.int32) - 128)).reshape(-1)
        raise ValueError(dt)

    def quant(t, x, rows, ne0):   # (one tensor on the calling thread; the pool runs tensors side by side)
        tid, per, bpb = QTYPES[t]
        q = np.empty(rows * ne0 // per * bpb, np.uint8)
        ones = np.ones(ne0, np.float32)
        g.ggml_quantize_chunk(tid, x.ctypes.data, q.ctypes.data, 0, rows, ne0, ones.ctypes.data)
        return q

    role_re = re.compile(r"(mtp\.\d+\.ffn\.experts\.)\d+\.(w[123])\.weight$")

    def prep(item):
        t, dt, so, nb, shape = item
        b = raw[so:so + nb]
        m = role_re.match(t)
        role = (m.group(1), m.group(2)) if m else None
        want = experts if role else dense
        if (not role and (dt != "BF16" or len(shape) != 2 or shape[1] % 32 or any(k in t for k in DENSE_KEEP))) \
                or want in ("keep", dt):
            return t, dt, np.asarray(b), shape, role
        if shape[1] % QTYPES[want][1]:
            raise SystemExit("%s: rows of %d are not whole %s blocks" % (t, shape[1], want))
        return t, want, quant(want, np.ascontiguousarray(f32(dt, b)), shape[0], shape[1]), shape, role

    for t in {experts, dense} - {"keep"}:
        g.ggml_quantize_init(QTYPES[t][0])
    entries, off, prev = [], 0, None
    o = pathlib.Path(out) if out else d
    o.mkdir(parents=True, exist_ok=True)
    items = list(src["tensors"])
    with (o / ("dspark-%s.bin.part" % name)).open("wb") as f, ThreadPoolExecutor(jobs) as ex:
        for i0 in range(0, len(items), 2 * jobs):
            for t, odt, q, shape, role in ex.map(prep, items[i0:i0 + 2 * jobs]):
                if role is None or role != prev:   # (a role's experts back to back: expert e at the first's + e x size)
                    pad = (-off) % ALIGN
                    f.write(b"\0" * pad)
                    off += pad
                f.write(q.tobytes())
                entries.append([t, odt, off, q.nbytes, shape])
                off += q.nbytes
                prev = role
    (o / ("dspark-%s.bin.part" % name)).replace(o / ("dspark-%s.bin" % name))
    (o / ("dspark-%s.json" % name)).write_text(json.dumps({"from": "dspark.bin", "experts": experts, "dense": dense,
                                                          "tensors": entries}))
    print("dspark-%s.bin: %d tensors, %.2f GB (experts %s, dense %s)" % (name, len(entries), off / 1e9, experts, dense))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="the folder with the drafter's three shards")
    ap.add_argument("--requant", metavar="NAME",
                    help="from dspark.bin there: dspark-NAME.bin with --experts / --dense types (ggml's quantizers)")
    ap.add_argument("--experts", default="IQ4_XS", help="(--requant) the routed experts' type, or keep")
    ap.add_argument("--dense", default="Q8_0", help="(--requant) the dense matrices' type, or keep")
    ap.add_argument("--lib", default=str(pathlib.Path("~/llama.cpp/build/bin/libggml-base.so").expanduser()),
                    help="(--requant) libggml-base")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", help="(--requant) the folder to write to (default: --dir)")
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    if a.requant:
        return requant(d, a.requant, a.experts, a.dense, a.lib, a.jobs, a.out)
    entries, off = [], 0
    exp = re.compile(r"(mtp\.\d+\.ffn\.experts\.)(\d+)\.(w[123])\.weight$")

    def order(head):
        """The block's tensors, then each role's experts (w1, w3, w2) back to back in expert order - the engine reads
        expert e of a role at the first one's offset + e x its size, as the GGUF stacks them."""
        plain = sorted(t for t in head if t.startswith("mtp.") and not t.endswith(".scale") and not exp.match(t))
        ex = [exp.match(t) for t in head if exp.match(t)]
        roles = sorted({(m.group(1), m.group(3)) for m in ex}, key=lambda pr: (pr[0], "w1w3w2".index(pr[1])))
        n = {pr: 1 + max(int(m.group(2)) for m in ex if (m.group(1), m.group(3)) == pr) for pr in roles}
        return [(t, False) for t in plain] + [("%s%d.%s.weight" % (p, i, r), i < n[(p, r)] - 1)
                                              for p, r in roles for i in range(n[(p, r)])]

    with (d / "dspark.bin.part").open("wb") as f:
        for name in SHARDS:
            raw, head, base = shard(d / name)
            for t, packed in order(head):
                e, shape = head[t], list(head[t]["shape"])
                if e["dtype"] == "F8_E4M3":
                    s = head[t[:-len("weight")] + "scale"]
                    out, dt = fp8_to_bf16(data(raw, base, e), data(raw, base, s), shape[0], shape[1]), "BF16"
                elif e["dtype"] == "I8":
                    s = head[t[:-len("weight")] + "scale"]
                    shape = [shape[0], shape[1] * 2]
                    out, dt = fp4_to_mxfp4(data(raw, base, e), data(raw, base, s), shape[0], shape[1]), "MXFP4"
                elif e["dtype"] in ("BF16", "F32"):
                    out, dt = data(raw, base, e), e["dtype"]
                else:
                    raise SystemExit(f"{name}: {t} is {e['dtype']}, not a type the drafter's pack covers")
                b = out.tobytes()
                f.write(b)
                entries.append([t, dt, off, len(b), shape])
                off += len(b)
                if packed:   # (the next expert of the role follows at once)
                    continue
                pad = (-off) % ALIGN
                f.write(b"\0" * pad)
                off += pad
            del raw
    (d / "dspark.bin.part").replace(d / "dspark.bin")
    (d / "dspark.json").write_text(json.dumps({"from": list(SHARDS), "tensors": entries}))
    print(f"dspark.bin: {len(entries)} tensors, {off / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
