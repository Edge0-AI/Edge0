# Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
# Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
"""dumpscan_y.py DUMP_Y DUMP_R --pos P [--layers a-b] - one run's expert set rows (DS4_DUMP_Y, DS4_DUMP_R): position
P's rows a layer - its experts, their rows and P's place in them, |h| and |y| against the layer's median row."""
import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from dumpcmp import read_r, read_y, row_keys  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_y")
    ap.add_argument("dump_r")
    ap.add_argument("--pos", type=int, required=True)
    ap.add_argument("--layers", default="0-99")
    ap.add_argument("--dim", type=int, default=5120)
    ap.add_argument("--ff", type=int, default=2304)
    a = ap.parse_args()
    l0, l1 = (int(v) for v in a.layers.split("-"))
    R = read_r(a.dump_r)
    for il, pieces in read_y(a.dump_y, a.ff, a.dim):
        if not l0 <= il <= l1:
            continue
        r = [v for (l2, _), v in R.items() if l2 == il]
        if len(r) != 1:
            continue
        keys = row_keys(r[0])
        hn, yn, mine = [], [], []
        for d in pieces:
            for i in range(d["n"]):
                t, e, n, pl = keys[d["a"] + i]
                h = float(np.linalg.norm(d["h"][i].astype(np.float64)))
                y = float(np.linalg.norm(d["y"][i].astype(np.float64)))
                hm = float(np.abs(d["h"][i].astype(np.float32)).max())
                hn.append(h)
                yn.append(y)
                if t == a.pos:
                    mine.append((e, n, pl, h, hm, y, d["ring"], "%d+%d/%dx%d" % (d["a"], d["n"], d["ne"], d["mr"])))
        mh, my = float(np.median(hn)), float(np.median(yn))
        print("L%-3d median |h| %.3e |y| %.3e; pos %d: %s" % (il, mh, my, a.pos, "; ".join(
            "e%d (%d rows, #%d) |h| %.2e max %.2e |y| %.2e (x%.1f) ring %d %s" % (e, n, pl, h, hm, y, y / max(my, 1e-30), rg, pc)
            for e, n, pl, h, hm, y, rg, pc in mine)))


if __name__ == "__main__":
    sys.exit(main())
