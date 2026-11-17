#!/usr/bin/env python3
"""dense_requant.py - the dense matrices of a DeepSeek-V4.1 pack (attention, compressor, indexer, shared experts; BF16 in the
unsloth GGUF) in a ggml K-quant, through ggml's own quantizer (libggml-base, ctypes), written beside the model as a small file the
pack reads instead of the originals - the experts, the Engram and the embedding/head are not touched, nothing big is copied:

    python tools/dsv41/dense_requant.py --pack PACK --out DIR --type Q5_K [--lib libggml-base.so] [--jobs 8]
    python tools/dsv41/pack_gguf.py --model GGUF_DIR --out PACK2 --dense-override DIR

DIR/dense.bin holds the quantized tensors (4096-aligned), DIR/dense.json {tensor name: [type, file, offset, bytes, shape]}.
Types: Q4_K (4.5 bpw), Q5_K (5.5), Q6_K (6.56), Q8_0 (8.5).  --include / --exclude are regexes on the checkpoint name
(default: layers.N.attn.* and layers.N.ffn.shared_experts.*, the main layers only, matrices whose rows are whole 256 blocks).
"""
import argparse
import ctypes
import json
import multiprocessing as mp
import os
import pathlib
import re
import sys
import time

import numpy as np

GT = {"Q8_0": (8, 32, 34), "Q4_K": (12, 256, 144), "Q5_K": (13, 256, 176), "Q6_K": (14, 256, 210)}
_lib = None


def _init(libpath):
    global _lib
    try:
        os.nice(10)
    except OSError:
        pass
    _lib = ctypes.CDLL(libpath)
    _lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    _lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                         ctypes.c_int64, ctypes.c_void_p]


def _job(j):
    name, path, off, nbytes, rows, cols, tid, qrb, out_path, out_off = j
    raw = np.memmap(path, dtype=np.uint16, mode="r", offset=off, shape=(rows * cols,))
    x = (raw.astype(np.uint32) << 16).view(np.float32).reshape(rows, cols)
    x = np.ascontiguousarray(x)
    dst = np.empty(rows * qrb, dtype=np.uint8)
    got = _lib.ggml_quantize_chunk(tid, x.ctypes.data, dst.ctypes.data, 0, rows, cols, None)
    if got != rows * qrb:
        raise RuntimeError("%s: ggml_quantize_chunk wrote %d of %d bytes" % (name, got, rows * qrb))
    # relative error of the result against the source (dequantize by the type's own reference is not exposed here:
    # the engine's tests compare the kernels; this tool only checks the size)
    fd = os.open(out_path, os.O_WRONLY)
    os.pwrite(fd, dst.tobytes(), out_off)
    os.close(fd)
    return name, rows * qrb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--type", required=True, choices=sorted(GT))
    ap.add_argument("--lib", default="/data/dsv41/build-hip/bin/libggml-base.so")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--include", default=r"^layers\.\d+\.(attn\.|ffn\.shared_experts\.)")
    ap.add_argument("--exclude", default=r"(norm|sink|gate\.bias|\.gate\.weight$|hc_|weights_proj)")
    a = ap.parse_args()
    tid, per, bpb = GT[a.type]
    files, tensors, cfg = {}, [], {}
    for line in open(os.path.join(a.pack, "tensors.txt"), encoding="utf-8"):
        f = line.split()
        if f and f[0] == "file":
            files[int(f[1])] = line.split(None, 2)[2].strip()
        elif f and f[0] == "t":
            tensors.append((f[1], f[2], int(f[3]), int(f[4]), int(f[5]), [int(x) for x in f[6:]]))
    for line in open(os.path.join(a.pack, "ds4.txt"), encoding="utf-8"):
        k, _, v = line.strip().partition(" ")
        cfg[k] = v
    model_dir = cfg["model_dir"]
    inc, exc = re.compile(a.include), re.compile(a.exclude)
    sel = [t for t in tensors if t[1] == "BF16" and len(t[5]) == 2 and t[5][1] % per == 0 and inc.search(t[0]) and not exc.search(t[0])]
    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, "dense.bin")
    jobs, idx, off = [], {}, 0
    for name, dt, fi, o, nb, shape in sel:
        rows, cols = shape
        qrb = cols // per * bpb
        jobs.append((name, os.path.join(model_dir, files[fi]), o, nb, rows, cols, tid, qrb, out_path, off))
        idx[name] = [a.type, "dense.bin", off, rows * qrb, shape]
        off += (rows * qrb + 4095) // 4096 * 4096
    with open(out_path, "wb") as f:
        f.truncate(off)
    src_b = sum(t[4] for t in sel)
    t0 = time.time()
    with mp.get_context("fork").Pool(a.jobs, initializer=_init, initargs=(a.lib,)) as pool:
        for i, (name, n) in enumerate(pool.imap_unordered(_job, jobs)):
            if i % 40 == 0:
                print("%d/%d %s %.0f s" % (i, len(jobs), name, time.time() - t0), flush=True)
    json.dump(idx, open(os.path.join(a.out, "dense.json"), "w"))
    print("dense %s: %d tensors, %.2f GB BF16 -> %.2f GB, %.0f s" % (a.type, len(sel), src_b / 1e9, sum(v[3] for v in idx.values()) / 1e9, time.time() - t0))


if __name__ == "__main__":
    main()
