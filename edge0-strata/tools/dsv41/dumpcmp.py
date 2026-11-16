# Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
# Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
"""dumpcmp.py - two maya-ds4 runs' debug dumps side by side (the prompt path's numerics: where two runs part).

  layers A B [--pos P]       DS4_DUMP_L files: each layer record's relative error between the runs - the median over
                             the positions, the worst position, and position P's
  route R --pos P            a DS4_DUMP_R file: position P's experts a layer (virtual slot, the expert's rows, P's row)
  gate R... --pos P          DS4_DUMP_R files: position P's router a layer - its picks and the margin between the k-th
                             pick's score and the next expert's (sqrt(softplus(logit)) + bias)
  sets A B --pos P --route RA RB
                             DS4_DUMP_Y files (and each run's DS4_DUMP_R): each layer's set rows' h / y error between
                             the runs (rows matched by token and expert), the worst row and position P's, with their
                             expert's rows and the piece they were computed in
"""
import argparse
import sys

import numpy as np


def read_l(path, dim_hc):
    out = []
    with open(path, "rb") as f:
        while True:
            h = f.read(24)
            if len(h) < 24:
                break
            il, p0, t = (int(v) for v in np.frombuffer(h, dtype=np.int64))
            v = np.frombuffer(f.read(t * dim_hc * 4), dtype=np.float32).reshape(t, dim_hc)
            out.append(((il, p0), v))
    return out


def rel(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-30)


def read_r(path):
    out = {}
    with open(path, "rb") as f:
        while True:
            h = f.read(64)
            if len(h) < 64:
                break
            il, p0, t, k, ne, nvs, nsl, kp = (int(v) for v in np.frombuffer(h, dtype=np.int64))
            ri = np.frombuffer(f.read(t * kp * 4), dtype=np.int32).reshape(t, kp)[:, :k]
            vs = np.frombuffer(f.read(ne * 4), dtype=np.int32)
            b = np.frombuffer(f.read((nvs + 1) * 4), dtype=np.int32)
            rt = np.frombuffer(f.read(t * k * 4), dtype=np.int32)
            gl = np.frombuffer(f.read(t * ne * 4), dtype=np.float32).reshape(t, ne)
            gb = np.frombuffer(f.read(ne * 4), dtype=np.float32)
            out[(il, p0)] = dict(t=t, k=k, ri=ri, vs=vs, b=b, rt=rt, nsl=nsl, glog=gl, bias=gb)
    return out


def read_y(path, ff, dim):   # a layer's pieces at a time (the dump holds the layers in order)
    cur, il0 = [], None
    with open(path, "rb") as f:
        while True:
            h = f.read(48)
            if len(h) < 48:
                break
            il, a, n, ne, ring, mr = (int(v) for v in np.frombuffer(h, dtype=np.int64))
            b = np.frombuffer(f.read((ne + 1) * 4), dtype=np.int32)
            hv = np.frombuffer(f.read(n * ff * 2), dtype=np.float16).reshape(n, ff)
            yv = np.frombuffer(f.read(n * dim * 4), dtype=np.float32).reshape(n, dim)
            if il0 is not None and il != il0:
                yield il0, cur
                cur = []
            il0 = il
            cur.append(dict(il=il, a=a, n=n, ne=ne, ring=ring, mr=mr, b=b, h=hv, y=yv))
    if cur:
        yield il0, cur


def row_keys(r):
    """A layer's sorted rows -> (token, expert, the expert's rows, the row's place among them)."""
    keys = {}
    for e, v in enumerate(r["vs"]):
        lo, hi = int(r["b"][v]), int(r["b"][v + 1])
        for g in range(lo, hi):
            keys[g] = (int(r["rt"][g]), e, hi - lo, g - lo)
    return keys


def sets(a):
    """Each layer: the set rows' y / h error between run A and run B, a row matched to the other run's by its (token,
    expert) - DS4_DUMP_R of each run (--route RA RB: the routing may differ) - the median, the worst row, position P's
    rows, with the expert's rows, the row's place in them and the piece it was computed in (row0+rows/experts x max)."""
    RA, RB = (read_r(f) for f in a.route)
    YB = read_y(a.files[1], a.ff, a.dim)
    print("%-3s %8s | %-52s | %s" % ("L", "median y", "worst: y err, h err, token, expert rows+place, ring, piece",
                                     "pos %d: y err, h err, expert rows+place, ring, piece" % a.pos))
    for il, pa in read_y(a.files[0], a.ff, a.dim):
        ilb, pb = next(YB)
        assert ilb == il, (il, ilb)
        ka = [row_keys(r) for (l2, _), r in RA.items() if l2 == il]
        kb = [row_keys(r) for (l2, _), r in RB.items() if l2 == il]
        if len(ka) != 1 or len(kb) != 1:   # (one chunk: p0 0)
            continue
        ka, kb = ka[0], kb[0]
        rows = {}
        for d in pb:
            for i in range(d["n"]):
                t, e, _, _ = kb[d["a"] + i]
                rows[(t, e)] = (d, i)
        errs = []
        for d in pa:
            for i in range(d["n"]):
                t, e, n, pl = ka[d["a"] + i]
                if (t, e) not in rows:
                    continue
                db, ib = rows[(t, e)]
                eh = float(rel(d["h"][i:i + 1].astype(np.float32), db["h"][ib:ib + 1].astype(np.float32))[0])
                ey = float(rel(d["y"][i:i + 1], db["y"][ib:ib + 1])[0])
                errs.append((ey, eh, t, e, n, pl, d["ring"], "%d+%d/%dx%d" % (d["a"], d["n"], d["ne"], d["mr"])))
        if not errs:
            continue
        w = max(errs)
        mine = sorted((x for x in errs if x[2] == a.pos), reverse=True)
        print("%-3d %8.1e | %7.1e %7.1e t%-4d e%-3d %3d+%-3d %d %-16s | %s" % (
            il, float(np.median([x[0] for x in errs])), w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7],
            "; ".join("%.1e %.1e e%d %d+%d %d %s" % (x[0], x[1], x[3], x[4], x[5], x[6], x[7]) for x in mine)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["layers", "route", "sets", "gate"])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--pos", type=int, default=-1)
    ap.add_argument("--dim", type=int, default=5120)
    ap.add_argument("--hc", type=int, default=4)
    ap.add_argument("--ff", type=int, default=2304)
    ap.add_argument("--route", nargs=2, metavar=("RA", "RB"))
    a = ap.parse_args()
    if a.mode == "layers":
        A = read_l(a.files[0], a.dim * a.hc)
        B = dict(read_l(a.files[1], a.dim * a.hc))
        print("%-12s %10s %10s %6s %10s" % ("record", "median", "worst", "at", "pos %d" % a.pos))
        for key, va in A:
            if key not in B:
                continue
            e = rel(va, B[key])
            w = int(np.argmax(e))
            p = a.pos - key[1]
            ep = "%.3e" % e[p] if 0 <= p < len(e) else "-"
            print("%-12s %10.3e %10.3e %6d %10s" % ("L%d@%d" % key, float(np.median(e)), float(e[w]), w + key[1], ep))
        return
    if a.mode == "gate":
        for f in a.files:   # each run's router at position P: the picks and the margin past the k-th
            print("== %s" % f)
            for (il, p0), r in sorted(read_r(f).items()):
                p = a.pos - p0
                if not 0 <= p < r["t"]:
                    continue
                lg = r["glog"][p].astype(np.float64)
                sc = np.sqrt(np.where(lg > 20, lg, np.log1p(np.exp(np.minimum(lg, 20)))))
                sel = sc + r["bias"]
                o = np.argsort(-sel, kind="stable")
                k = r["k"]
                print("L%-3d picks %s | k-th %.5f, next e%d %.5f: margin %.2e" % (
                    il, " ".join("e%d" % e for e in o[:k]), sel[o[k - 1]], o[k], sel[o[k]], sel[o[k - 1]] - sel[o[k]]))
        return
    if a.mode == "route":
        R = read_r(a.files[0])
        for (il, p0), r in sorted(R.items()):
            p = a.pos - p0
            if not 0 <= p < r["t"]:
                continue
            rows = np.nonzero(r["rt"] == p)[0]
            parts = []
            for e in r["ri"][p]:
                v = int(r["vs"][e])
                lo, hi = int(r["b"][v]), int(r["b"][v + 1])
                mine = [int(x) for x in rows if lo <= x < hi]
                parts.append("e%d v%d%s rows %d (row %s, %d into it)" % (e, v, "" if v < r["nsl"] else "*", hi - lo,
                             mine[0] if mine else "?", (mine[0] - lo) if mine else -1))
            print("L%d@%d: %s" % (il, p0, "; ".join(parts)))
        return
    sets(a)


if __name__ == "__main__":
    sys.exit(main())
