# Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
# Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
"""tools/ds4/compare.py - the engine's logits (maya-ds4 --dump-logits) against the reference's (tools/ds4/ref.py --dump).

    python tools/ds4/compare.py --ref ref.npz --engine engine.f32 [--vocab 129280]

Per position: the cosine of the two logit rows, the largest difference, whether the top token agrees, and the KL
divergence of the engine's distribution from the reference's.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--vocab", type=int, default=129280)
    ap.add_argument("--layers", help="the engine's DS4_DUMP_H file: each layer's residual stream against the ref's")
    a = ap.parse_args()
    r = np.load(a.ref)
    ref = r["logits"].astype(np.float64)
    eng = np.fromfile(a.engine, dtype=np.float32).reshape(-1, a.vocab).astype(np.float64)
    n = min(len(ref), len(eng))
    agree = 0
    kls = []
    for i in range(n):
        x, y = ref[i], eng[i]
        cos = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
        px = np.exp(x - x.max())
        px /= px.sum()
        py = np.exp(y - y.max())
        py /= py.sum()
        kl = float(np.sum(px * (np.log(px + 1e-30) - np.log(py + 1e-30))))
        kls.append(kl)
        top = int(x.argmax()) == int(y.argmax())
        agree += top
        print("pos %3d  cos %.6f  max|d| %8.4f  top %s (ref %6d eng %6d)  KL %.2e" %
              (i, cos, np.abs(x - y).max(), "same" if top else "DIFF", x.argmax(), y.argmax(), kl))
    print("%d positions: top-1 agrees at %d, mean KL %.3e, max KL %.3e" % (n, agree, np.mean(kls), np.max(kls)))
    if a.layers:
        hs = r["hs"]                                                     # [layers, T, hc, dim]
        nl, T = hs.shape[0], hs.shape[1]
        eh = np.fromfile(a.layers, dtype=np.float32).reshape(-1, nl, *hs.shape[2:])   # [T, layers, hc, dim]
        T = min(T, len(eh))
        print("relative error of each layer's residual stream (rows: layers, columns: positions)")
        print("     " + " ".join("%6d" % t for t in range(T)))
        for li in range(nl):
            row = []
            for t in range(T):
                x, y = hs[li, t].astype(np.float64), eh[t, li].astype(np.float64)
                row.append(np.linalg.norm(x - y) / np.linalg.norm(x))
            print("L%-3d " % li + " ".join("%6.4f" % v for v in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
