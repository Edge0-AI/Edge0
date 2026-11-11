"""build_swap.py - Gyro-S with a few layers' experts swapped, without copying the 58 GB file.

Writes <var>/swap.bin (the new gate / up / down tensors of the chosen layers) and a pack <var>/pack whose native_experts.txt
points those layers at swap.bin (the shard column), everything else being the shipped Gyro-S pack and file: <var>/ holds a
symlink to the shipped GGUF, so `--native <var>/<gguf name>` finds swap.bin beside it.

Modes (all from the ORIGINAL Qwen weights, rotated into the file's basis with the file's own sign vectors):
  q8     Q8_0 (types 8/8): the near-lossless reference for the swapped layers
  plain  TQK6 gate/up + TQK7 down, plain-MSE trellis encode (tools/tqenc.py)
  diag   the same with per-input-column importance weights h = E[x^2]^power from the calibration dump (STRATA_ACT_DUMP)
  ldl    the same with block feedback from the full second-moment matrix of the dump (STRATA_ACT_DUMP_COV=1), blockldl.py

  python3 tools/trellis/build_swap.py --gguf SHIPPED --pack PACK --orig DIR --layers 4,16,28,40 --mode plain --var DIR [--gpu]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import gguf_reader  # noqa: E402
import blockldl  # noqa: E402
import tqenc  # noqa: E402

N_EXP, N_EMBD, N_FF = 512, 2560, 640
ALIGN = 4096


def q8_0_quantize(w: np.ndarray) -> np.ndarray:
    """float32 [..., n] (n % 32 == 0) -> uint8 [..., n/32 * 34] block_q8_0 (fp16 scale, 32 int8)."""
    shp = w.shape
    b = w.reshape(-1, 32).astype(np.float32)
    amax = np.abs(b).max(axis=1)
    d = amax / 127.0
    d16 = d.astype(np.float16)
    dq = d16.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(dq[:, None] > 0, b / dq[:, None], 0.0)
    q = np.clip(np.sign(q) * np.floor(np.abs(q) + 0.5), -127, 127).astype(np.int8)
    out = np.empty((b.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.reshape(*shp[:-1], shp[-1] // 32 * 34)


def bf16(raw: np.ndarray) -> np.ndarray:
    return (raw.astype(np.uint32) << 16).view(np.float32)


def load_h(dump_dir, layer, kind, power):
    p = pathlib.Path(dump_dir) / ("act_L%d_%s.bin" % (layer, kind))
    raw = p.read_bytes()
    n, count = np.frombuffer(raw[:16], dtype=np.int64)
    sq = np.frombuffer(raw[16:16 + 8 * n], dtype=np.float64)
    h = (sq / max(count, 1)) ** power
    h = np.maximum(h, h.mean() * 1e-4)    # no weight below 1e-4 of the mean
    return (h / h.mean()).astype(np.float32), int(count)


def load_hessian(dump_dir, layer, kind):
    p = pathlib.Path(dump_dir) / ("act_L%d_%s.bin" % (layer, kind))
    raw = p.read_bytes()
    n, count = np.frombuffer(raw[:16], dtype=np.int64)
    n = int(n)
    if len(raw) < 16 + 8 * n + 8 * n * n:
        raise SystemExit("%s has no covariance (run the calibration with STRATA_ACT_DUMP_COV=1)" % p)
    cov = np.frombuffer(raw[16 + 8 * n:16 + 8 * n + 8 * n * n], dtype=np.float64).reshape(n, n)
    return cov / max(int(count), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--mode", required=True, choices=["q8", "plain", "diag", "ldl"])
    ap.add_argument("--var", required=True)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--power", type=float, default=1.0)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--ncand", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--damp", type=float, default=0.3, help="ldl: damping, a fraction of the mean diagonal of H")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    var = pathlib.Path(a.var)
    var.mkdir(parents=True, exist_ok=True)
    gg = pathlib.Path(a.gguf)
    link = var / gg.name
    if not link.exists():
        link.symlink_to(gg)
    g = gguf_reader.GGUFFile(a.gguf)
    sw = [int(x) for x in g.metadata["prism.hadamard.sign_widths"]]
    sv = np.array(g.metadata["prism.hadamard.sign_values"], dtype=np.float32)
    signs, at = {}, 0
    for w in sw:
        signs[w] = sv[at:at + w]
        at += w
    if a.mode == "q8":
        gu_t = d_t = 8
        row_gu, row_d = N_EMBD // 32 * 34, N_FF // 32 * 34
    else:
        gu_t, d_t = tqenc.TQK6, tqenc.TQK7
        row_gu, row_d = N_EMBD // 128 * tqenc.block_bytes(gu_t), N_FF // 128 * tqenc.block_bytes(d_t)
    t_gate = N_EXP * N_FF * row_gu          # bytes of the gate (and up) tensor of a layer
    t_down = N_EXP * N_EMBD * row_d
    swap = open(var / "swap.bin", "wb")
    offs = {}
    pos = 0
    for l in layers:
        offs[l] = {}
        for role, nbytes in (("gate", t_gate), ("up", t_gate), ("down", t_down)):
            offs[l][role] = pos
            pos += (nbytes + ALIGN - 1) // ALIGN * ALIGN
    swap.truncate(pos)

    def enc(w2d, t, h):
        kw = dict(iters=a.iters, ncand=a.ncand, gpu=a.gpu)
        if not a.gpu:
            kw["threads"] = a.threads
        hh = None
        if h is not None:
            hh = np.broadcast_to(h, w2d.shape).astype(np.float32)
        packed, err, tot, st = tqenc.encode(w2d, t, h=hh, **kw)
        return packed, tot

    stats = {}
    t_start = time.time()
    for l in layers:
        gu_raw = np.memmap(pathlib.Path(a.orig) / ("L%d_gate_up.bf16" % l), dtype=np.uint16, mode="r").reshape(N_EXP, 2 * N_FF, N_EMBD)
        dn_raw = np.memmap(pathlib.Path(a.orig) / ("L%d_down.bf16" % l), dtype=np.uint16, mode="r").reshape(N_EXP, N_EMBD, N_FF)
        h_gu = h_dn = None
        fb_gu = fb_dn = None
        if a.mode == "ldl":
            fb_gu = blockldl.feedback_matrices(blockldl.hessian_factor(load_hessian(a.dump_dir, l, "gu"), a.damp))
            fb_dn = blockldl.feedback_matrices(blockldl.hessian_factor(load_hessian(a.dump_dir, l, "dn"), a.damp))
        if a.mode == "diag":
            h_gu, c1 = load_h(a.dump_dir, l, "gu", a.power)
            h_dn, c2 = load_h(a.dump_dir, l, "dn", a.power)
            print("layer %d: calibration counts gu %d dn %d; h cv gu %.2f dn %.2f" % (l, c1, c2, h_gu.std() / h_gu.mean(), h_dn.std() / h_dn.mean()))
        err_sum = {"gate": 0.0, "up": 0.0, "down": 0.0}
        en_sum = {"gate": 0.0, "up": 0.0, "down": 0.0}
        for e0 in range(0, N_EXP, a.chunk):
            e1 = min(N_EXP, e0 + a.chunk)
            gu = bf16(np.asarray(gu_raw[e0:e1]))
            dn = bf16(np.asarray(dn_raw[e0:e1]))
            mats = {"gate": (gu[:, :N_FF], signs[N_EMBD], h_gu, row_gu, N_FF, t_gate, gu_t),
                    "up": (gu[:, N_FF:], signs[N_EMBD], h_gu, row_gu, N_FF, t_gate, gu_t),
                    "down": (dn, signs[N_FF], h_dn, row_d, N_EMBD, t_down, d_t)}
            ldl_done = {}
            if a.mode == "ldl":
                # gate and up share H: one batch of rows
                both = np.concatenate([tqenc.rotate(np.ascontiguousarray(gu[:, :N_FF]), signs[N_EMBD]).reshape(-1, N_EMBD),
                                       tqenc.rotate(np.ascontiguousarray(gu[:, N_FF:]), signs[N_EMBD]).reshape(-1, N_EMBD)])
                pk, dec = blockldl.encode_ldl(both, gu_t, fb_gu, gpu=a.gpu, iters=a.iters, ncand=a.ncand, threads=a.threads)
                half = pk.shape[0] // 2
                ldl_done["gate"] = (pk[:half].reshape(half, -1), dec[:half])
                ldl_done["up"] = (pk[half:].reshape(half, -1), dec[half:])
                wrd = tqenc.rotate(np.ascontiguousarray(dn), signs[N_FF]).reshape(-1, N_FF)
                pk, dec = blockldl.encode_ldl(wrd, d_t, fb_dn, gpu=a.gpu, iters=a.iters, ncand=a.ncand, threads=a.threads)
                ldl_done["down"] = (pk.reshape(pk.shape[0], -1), dec)
            for role, (w, sg, h, row, n_out, tb, ty) in mats.items():
                wr = tqenc.rotate(np.ascontiguousarray(w), sg)           # [chunk, n_out, n_in]
                n_in = wr.shape[-1]
                if a.mode == "q8":
                    data = q8_0_quantize(wr)
                    dec = None
                elif a.mode == "ldl":
                    data, dec = ldl_done[role]
                    err_sum[role] += float(np.sum((wr.reshape(-1, n_in).astype(np.float64) - dec) ** 2))
                else:
                    packed, tot = enc(wr.reshape(-1, n_in), ty, h if a.mode == "diag" else None)
                    data = packed
                    err_sum[role] += tot
                en_sum[role] += float(np.sum(wr.astype(np.float64) ** 2))
                swap.seek(offs[l][role] + e0 * n_out * row)
                swap.write(np.ascontiguousarray(data).tobytes())
            if e0 % 128 == 0:
                print("layer %d experts %d-%d done  (%.0f s)" % (l, e0, e1, time.time() - t_start), flush=True)
        stats[l] = {r: (10 * np.log10(en_sum[r] / err_sum[r]) if err_sum[r] > 0 else None) for r in err_sum}
        print("layer %d weight SQNR (dB, unweighted): %s" % (l, {k: (round(v, 3) if v else v) for k, v in stats[l].items()}), flush=True)
    swap.close()
    # the pack: symlinks to the shipped pack's files, a new native_experts.txt
    pk = var / "pack"
    pk.mkdir(exist_ok=True)
    src = pathlib.Path(a.pack)
    for f in src.iterdir():
        if f.name == "native_experts.txt":
            continue
        d = pk / f.name
        if not d.exists():
            d.symlink_to(f.resolve())
    lines = (src / "native_experts.txt").read_text().splitlines()
    head, body = lines[0], lines[1:]
    out_lines, arena = [], 0
    for ln in body:
        f = ln.split()
        l = int(f[0])
        if l in offs:
            blob = row_gu * N_FF * 2 + row_d * N_EMBD
            f = [str(l), str(gu_t), str(d_t), str(arena), str(blob), str(offs[l]["gate"]), str(offs[l]["up"]), str(offs[l]["down"]), "swap.bin"]
        else:
            f[3] = str(arena)
        out_lines.append(" ".join(f))
        arena += N_EXP * int(f[4])
    import re
    head = re.sub(r"total \d+", "total %d" % arena, head)
    (pk / "native_experts.txt").write_text(head + "\n" + "\n".join(out_lines) + "\n")
    print("wrote", var, "swap.bin %.2f GB, pack %s" % (pos / 1e9, pk))
    return 0


if __name__ == "__main__":
    sys.exit(main())
