# Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
# Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
"""dumpscan.py DUMP_L [--pos P] [--dim 5120] [--hc 4] - one run's DS4_DUMP_L records: each layer's residual stream at
position P against the other positions (its norm over the median norm, its largest |value| and where, non-finite
values) - where a position's stream blows up or goes non-finite."""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--pos", type=int, default=0)
    ap.add_argument("--dim", type=int, default=5120)
    ap.add_argument("--hc", type=int, default=4)
    a = ap.parse_args()
    n = a.dim * a.hc
    print("%-10s %10s %10s %12s %10s %10s %s" % ("record", "|h| pos", "median", "pos / med", "max|v| pos", "max|v| all",
                                                "non-finite (pos / all)"))
    with open(a.dump, "rb") as f:
        while True:
            h = f.read(24)
            if len(h) < 24:
                break
            il, p0, t = (int(v) for v in np.frombuffer(h, dtype=np.int64))
            v = np.frombuffer(f.read(t * n * 4), dtype=np.float32).reshape(t, n)
            p = a.pos - p0
            if not 0 <= p < t:
                continue
            fin = np.isfinite(v)
            vv = np.where(fin, v, 0).astype(np.float64)
            nr = np.linalg.norm(vv, axis=1)
            med = float(np.median(nr))
            mx = np.abs(vv[p])
            j = int(np.argmax(mx))
            print("L%-4d@%-3d %10.3e %10.3e %12.2f %10.3e %10.3e %d / %d   (max at stream %d dim %d)" % (
                il, p0, nr[p], med, nr[p] / max(med, 1e-30), mx[j], float(np.abs(vv).max()),
                int((~fin[p]).sum()), int((~fin).sum()), j // a.dim, j % a.dim))


if __name__ == "__main__":
    sys.exit(main())
