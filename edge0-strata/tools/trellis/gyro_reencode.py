"""gyro_reencode.py - gate 2/3 quality check of the trellis encoder against agentionai's Gyro files.

Decodes expert tensors of a Gyro GGUF (TQK6 gate/up, TQK7 down in Gyro-S; TQ2_T in Gyro-M) with the Strata decoder
(libtqenc's decode, bit-identical to tq_states.hpp, see tqenc_test), re-encodes the decoded weights at the same type
with tools/tqenc.py and reports the reconstruction error.  The decoded weights are a feasible trellis point, so an
ideal encoder recovers them exactly (error 0): the error measures the search, not the quantization.

  python3 tools/trellis/gyro_reencode.py --gguf /data/models/gyro/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf \
      --layers 0,20,39 --experts 0,7 --rows 256 [--cfg default,rot3,scale3]
"""
from __future__ import annotations

import argparse
import mmap
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import gguf_reader  # noqa: E402
import tqenc  # noqa: E402

CFGS = {
    "fast": dict(iters=0, ncand=1, scale_tries=1),
    "default": dict(iters=1, ncand=1, scale_tries=1),
    "iters3": dict(iters=3, ncand=1, scale_tries=1),
    "rot3": dict(iters=3, ncand=3, scale_tries=1),
    "scale3": dict(iters=3, ncand=3, scale_tries=3),
    "scale5": dict(iters=3, ncand=3, scale_tries=5),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--layers", default="0,20,39")
    ap.add_argument("--experts", default="0")
    ap.add_argument("--rows", type=int, default=256, help="rows (output channels) of each expert slice to test")
    ap.add_argument("--cfg", default="fast,default,rot3")
    ap.add_argument("--roles", default="gate,up,down")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()

    g = gguf_reader.GGUFFile(a.gguf)
    fh = open(a.gguf, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    by_name = {t.name: t for t in g.tensors}
    print("file", pathlib.Path(a.gguf).name, "tensors", len(g.tensors), "types", g.by_type())
    hk = {k: v for k, v in g.metadata.items() if k.startswith("prism.") and not isinstance(v, list)}
    print("prism metadata:", hk)
    cfgs = a.cfg.split(",")
    rows_total = {c: [0.0, 0.0, 0, 0, 0.0] for c in cfgs}   # err, energy, blocks, exact blocks, seconds
    for layer in [int(x) for x in a.layers.split(",")]:
        for role in a.roles.split(","):
            name = "blk.%d.ffn_%s_exps.weight" % (layer, role)
            ti = by_name.get(name)
            if ti is None:
                print("no tensor", name)
                continue
            t = tqenc.TYPE_BY_NAME.get(ti.type_name)
            if t is None:
                print(name, "type", ti.type_name, "is not a trellis type: skipped")
                continue
            n_in, n_out, n_exp = ti.shape[0], ti.shape[1], ti.shape[2]
            bb = tqenc.block_bytes(t)
            row_bytes = n_in // 128 * bb
            for e in [int(x) for x in a.experts.split(",")]:
                nrows = min(a.rows, n_out)
                off = g.data_start + ti.offset + (e * n_out) * row_bytes
                raw = np.frombuffer(mm, dtype=np.uint8, count=nrows * row_bytes, offset=off)
                wd = tqenc.decode(raw, t).reshape(nrows, n_in)
                en = float(np.sum(wd.astype(np.float64) ** 2))
                scales = raw.reshape(-1, bb)[:, :2].copy().view(np.float16).astype(np.float32).ravel()
                print("%s expert %d: %d rows x %d, %s %.3f bpw, |w| rms %.4g, agentionai block scales: median %.4g, "
                      "p1 %.4g, p99 %.4g, zero-scale blocks %d" % (
                          name, e, nrows, n_in, ti.type_name, tqenc.bpw(t), np.sqrt(en / wd.size),
                          np.median(scales), np.percentile(scales, 1), np.percentile(scales, 99), int((scales == 0).sum())))
                for c in cfgs:
                    t0 = time.time()
                    packed, err, tot, st = tqenc.encode(wd, t, threads=a.threads, **CFGS[c])
                    sec = time.time() - t0
                    dec = tqenc.decode(packed, t)
                    chk = float(np.sum((wd.ravel().astype(np.float64) - dec) ** 2))
                    nexact = int((err == 0).sum())
                    same_bytes = int((packed.reshape(-1) == raw).sum()) / raw.size
                    print("    %-8s rel MSE %.3e  SQNR %7.2f dB  exact blocks %5d/%d (%.1f%%)  scale==theirs %.1f%%  "
                          "%.0f weights/s  (recheck %.3e)" % (
                              c, tot / en, tqenc.sqnr_db(wd, dec), nexact, err.size, 100.0 * nexact / err.size,
                              100.0 * float(np.mean(packed[:, :2].reshape(-1, 2).view(np.uint16).ravel()
                                                    == raw.reshape(-1, bb)[:, :2].copy().view(np.uint16).ravel())),
                              wd.size / sec, chk / en))
                    r = rows_total[c]
                    r[0] += tot; r[1] += en; r[2] += err.size; r[3] += nexact; r[4] += sec
    print("\nTOTAL over all tested slices")
    for c in cfgs:
        r = rows_total[c]
        if r[1] > 0:
            print("  %-8s rel MSE %.3e  SQNR %.2f dB  exact blocks %.1f%%  %.0f weights/s" % (
                c, r[0] / r[1], 10 * np.log10(r[1] / max(r[0], 1e-300)), 100.0 * r[3] / r[2], r[2] * 128 / max(r[4], 1e-9)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
