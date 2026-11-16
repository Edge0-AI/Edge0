"""swap_compare.py - KL / NLL of Gyro-S swap variants against the Q8_0-swap reference, with paired differences.

  python3 tools/trellis/swap_compare.py REF.tsv shipped=RUN.tsv plain=RUN.tsv diag=RUN.tsv [...] [--base shipped]

Per run: mean KL(REF || run) over REF's top-64 (tools/quant_quality.py's definition), top-1 agreement with REF, mean negative
log-likelihood of the text (and its difference to REF's), each with a standard error over segments.  Then the paired
difference of every run against --base (same positions): mean +- standard error over segments, and the share of segments
where the run is better.  Per text kind (english / multilingual / code / docs / agent) too.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import quant_quality as QQ  # noqa: E402


def per_position(ref, run):
    """{segment: (kl[], nll[], agree[])} for positions both runs have."""
    out = {}
    for name, rr in ref.items():
        ra = run.get(name, {})
        kls, nll, ag = [], [], []
        for q in sorted(set(rr) & set(ra)):
            tr, lr, topr, kr = rr[q]
            ta, la, topa, ka = ra[q]
            if tr != ta:
                raise SystemExit("%s pos %d: different text" % (name, q))
            da = dict(ka)
            floor = min(b for _, b in ka)
            pr = np.array([math.exp(b) for _, b in kr])
            pa = np.array([math.exp(da.get(i, floor)) for i, _ in kr])
            pr /= pr.sum()
            pa /= pa.sum()
            kls.append(float(np.sum(pr * np.log(pr / pa))))
            nll.append(-la)
            ag.append(float(topr == topa))
        if kls:
            out[name] = (np.array(kls), np.array(nll), np.array(ag))
    return out


def kind(name):
    return name.rsplit("_", 1)[0]


def se(x):
    return float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("runs", nargs="+", help="label=file.tsv")
    ap.add_argument("--base", default=None)
    a = ap.parse_args()
    ref = QQ.read_run(a.ref)
    runs = {}
    for r in a.runs:
        lab, _, path = r.partition("=")
        runs[lab] = per_position(ref, QQ.read_run(path))
    runs.setdefault('ref', per_position(ref, ref))
    base = a.base or next(iter(runs))
    segs = sorted(set.intersection(*[set(v) for v in runs.values()]))
    kinds = sorted({kind(s) for s in segs})
    print("segments %d, positions %d (reference %s)\n" % (len(segs), sum(len(runs[base][s][0]) for s in segs), a.ref))

    def seg_mean(lab, idx, ss):
        return np.array([runs[lab][s][idx].mean() for s in ss])

    for title, ss in [("ALL", segs)] + [(k, [s for s in segs if kind(s) == k]) for k in kinds]:
        if not ss:
            continue
        print("== %s (%d segments)" % (title, len(ss)))
        print("  %-10s %-22s %-14s %-20s %s" % ("run", "KL(ref||run)", "top-1 vs ref", "NLL of the text", "paired KL difference vs " + base))
        for lab in runs:
            kl, ag, nl = seg_mean(lab, 0, ss), seg_mean(lab, 2, ss), seg_mean(lab, 1, ss)
            kb = seg_mean(base, 0, ss)
            d = kl - kb
            nb = seg_mean(base, 1, ss)
            print("  %-10s %.5f +- %.5f      %.4f         %.4f +- %.4f   %s" % (
                lab, kl.mean(), se(kl), ag.mean(), nl.mean(), se(nl),
                "(base)" if lab == base else "%+.5f +- %.5f  better in %d/%d segments; NLL %+.4f +- %.4f" % (
                    d.mean(), se(d), int((d < 0).sum()), len(d), (nl - nb).mean(), se(nl - nb))))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
