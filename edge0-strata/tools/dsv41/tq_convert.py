#!/usr/bin/env python3
"""tq_convert.py - re-encode the routed experts of some layers of a DeepSeek-V4.1 pack with the trellis encoder
(tools/tqenc.py, HIP) and write them in the format the engine reads (strata-dsv41: ds4_tq.cu / ds4_tq_cpu.cpp):

    HIP_VISIBLE_DEVICES=2 TQENC_GPU_LOCK=2 python tools/dsv41/tq_convert.py --pack PACK --out DIR --layers 1,7 --type K5

PACK is the pack pack_gguf.py wrote (its tensors.txt names the source MXFP4 experts in the GGUF shards).  Per layer one file
DIR/tq-L<layer>.bin holds the three roles one after another (gate = w1, up = w3, down = w2), each stacked [n_exp][rows][row
bytes] in the trellis types' file layout (a row = k / 128 blocks of {fp16 d; 4 K stream bytes}, tools/trellis/tqenc.hpp), 4096-aligned;
DIR/tq.json lists them as {"layers.N.ffn.experts.w1": [type name, file, offset, bytes, [n_exp, rows, cols]], ...}.  Then

    python tools/dsv41/pack_gguf.py --model <gguf dir> --out PACK2 --tq-experts DIR

writes a pack whose layers listed there read these experts (the other layers keep their MXFP4).  The weights are encoded as they
are (plain error, no rotation, no activation weights): what the dense override experiments of gate 3 measure.  Types:
K4 = TQK4X, K5 = TQK5X (computed codebook, version 2), K4T / K5T (the table, version 1), K6 = TQK6, K7 = TQK7.
"""
import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tqenc  # noqa: E402

TYPES = {"K4": tqenc.TQK4X, "K5": tqenc.TQK5X, "K4T": tqenc.TQK4T, "K5T": tqenc.TQK5T, "K6": tqenc.TQK6, "K7": tqenc.TQK7}
CFGS = {"fast": dict(iters=0, ncand=1, scale_tries=1), "default": dict(iters=1, ncand=1, scale_tries=1)}
MXFP4_LUT = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)
ROLES = (("w1", "gate"), ("w3", "up"), ("w2", "down"))


def decode_mxfp4(raw, n_in):
    """ggml MXFP4 blocks (32 values in 17 bytes: E8M0 scale, 16 code bytes) -> float32 [rows, n_in]"""
    b = raw.reshape(-1, 17)
    e = b[:, 0].astype(np.int32)
    d = np.where(e == 0, np.float32(2.0 ** -127), np.ldexp(np.float32(1.0), e - 128)).astype(np.float32)
    q = b[:, 1:]
    return (np.concatenate([MXFP4_LUT[q & 15], MXFP4_LUT[q >> 4]], axis=1) * d[:, None]).reshape(-1, n_in)


def read_pack(path):
    files, tensors, cfg = {}, {}, {}
    for line in open(os.path.join(path, "tensors.txt"), encoding="utf-8"):
        f = line.split()
        if f and f[0] == "file":
            files[int(f[1])] = line.split(None, 2)[2].strip()
        elif f and f[0] == "t":
            tensors[f[1]] = (f[2], int(f[3]), int(f[4]), int(f[5]), [int(x) for x in f[6:]])
    for line in open(os.path.join(path, "ds4.txt"), encoding="utf-8"):
        k, _, v = line.strip().partition(" ")
        cfg[k] = v
    return files, tensors, cfg


def align(n, a=4096):
    return (n + a - 1) // a * a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", required=True, help="comma list")
    ap.add_argument("--type", required=True, choices=sorted(TYPES))
    ap.add_argument("--cfg", default="fast", choices=sorted(CFGS))
    ap.add_argument("--experts", type=int, default=0, help="only the first N experts (tests; the others stay zero streams)")
    ap.add_argument("--act", default="", help="calibration dir with L<l>_gu.f64 [n_exp][dim], L<l>_dn.f64 [n_exp][ff] (sums of x^2 of the "
                    "gate/up and down inputs per expert) and L<l>_cnt.u64 [n_exp]: an input-column weighted encode (error = sum h (w - dec)^2)")
    ap.add_argument("--prior", type=float, default=64.0, help="pseudo-count of the layer-mean prior for rarely used experts")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cpu", action="store_true", help="the CPU encoder instead of the card")
    a = ap.parse_args()
    t = TYPES[a.type]
    cfg = CFGS[a.cfg]
    files, tensors, pcfg = read_pack(a.pack)
    model_dir = pcfg["model_dir"]
    n_exp = int(pcfg["n_routed_experts"])
    ff, dim = int(pcfg["moe_intermediate_size"]), int(pcfg["hidden_size"])
    n_do = a.experts or n_exp
    os.makedirs(a.out, exist_ok=True)
    idx_path = os.path.join(a.out, "tq.json")
    idx = json.load(open(idx_path)) if os.path.exists(idx_path) else {}
    bb = tqenc.block_bytes(t)
    mm = {}
    gpu_lock = threading.Lock()
    for L in [int(x) for x in a.layers.split(",")]:
        path = os.path.join(a.out, "tq-L%d.bin" % L)
        if all("layers.%d.ffn.experts.%s" % (L, r) in idx for r, _ in ROLES) and os.path.exists(path):
            print("layer %d: already in %s, skipped" % (L, idx_path), flush=True)
            continue
        sizes = {}
        offs = {}
        off = 0
        for role, _ in ROLES:
            rows, cols = (ff, dim) if role != "w2" else (dim, ff)
            sizes[role] = (rows, cols, n_exp * rows * (cols // 128) * bb)
            offs[role] = off
            off += align(sizes[role][2])
        with open(path, "wb") as f:
            f.truncate(off)
        fd = os.open(path, os.O_RDWR)
        act = None
        if a.act:
            gu = np.fromfile("%s/L%d_gu.f64" % (a.act, L), dtype=np.float64).reshape(n_exp, dim)
            dn = np.fromfile("%s/L%d_dn.f64" % (a.act, L), dtype=np.float64).reshape(n_exp, ff)
            cnt = np.fromfile("%s/L%d_cnt.u64" % (a.act, L), dtype=np.uint64).astype(np.float64)
            tot = max(cnt.sum(), 1.0)
            act = (gu, dn, cnt, gu.sum(0) / tot, dn.sum(0) / tot)
        for role, rname in ROLES:
            rows, cols, total = sizes[role]
            t0 = time.time()
            per_out = rows * (cols // 128) * bb
            names = ["layers.%d.ffn.experts.%d.%s.weight" % (L, e, role) for e in range(n_do)]
            err2 = np.zeros(n_do)
            ref2 = np.zeros(n_do)

            def one(e):
                dt, fi, off_, nb, shape = tensors[names[e]]
                if dt != "MXFP4":
                    raise SystemExit("%s is %s, not MXFP4: only MXFP4 sources are re-encoded" % (names[e], dt))
                key = files[fi]
                if key not in mm:
                    mm[key] = np.memmap(os.path.join(model_dir, key), dtype=np.uint8, mode="r")
                w = decode_mxfp4(np.asarray(mm[key][off_:off_ + nb]), cols)
                h = None
                if act is not None:   # the input columns' mean x^2 of this expert, shrunk toward the layer's mean by the prior
                    gu_, dn_, cnt_, mg, md = act
                    sq, m = (dn_[e], md) if role == "w2" else (gu_[e], mg)
                    hc = (sq + a.prior * m) / (cnt_[e] + a.prior)
                    hc = np.maximum(hc / hc.mean(), 1e-3).astype(np.float32)
                    h = np.ascontiguousarray(np.broadcast_to(hc, (rows, cols)))
                with gpu_lock:
                    packed, err, tot, st = tqenc.encode(w, t, h=h, gpu=not a.cpu, **cfg)
                dec = tqenc.decode(packed, t).reshape(rows, cols)
                err2[e] = ((w.astype(np.float64) - dec) ** 2).sum()
                ref2[e] = (w.astype(np.float64) ** 2).sum()
                os.pwrite(fd, packed.tobytes(), offs[role] + e * per_out)
                return st["nonfinite_blocks"]

            with cf.ThreadPoolExecutor(a.threads) as ex:
                nonfin = sum(ex.map(one, range(n_do)))
            print("layer %d %s: %d experts %s %.0f s, rel mse %.4f (sqnr %.2f dB), nonfinite blocks %d" %
                  (L, rname, n_do, a.type, time.time() - t0, err2.sum() / ref2.sum(), 10 * np.log10(ref2.sum() / err2.sum()), nonfin), flush=True)
            idx["layers.%d.ffn.experts.%s" % (L, role)] = [tqenc.TYPE_NAMES[t], os.path.basename(path), offs[role], total, [n_exp, rows, cols]]
        os.fsync(fd)
        os.close(fd)
        json.dump(idx, open(idx_path, "w"), indent=1)


if __name__ == "__main__":
    main()
