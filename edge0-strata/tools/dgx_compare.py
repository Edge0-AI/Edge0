#!/usr/bin/env python3
"""dgx_compare - side-by-side medians and ratios of two dgx_bench result folders (or .jsonl files).

    python tools/dgx_compare.py bench/results/dgx-.../base.jsonl bench/results/dgx-.../gu.jsonl

Prints, per cell, the median prompt and decode tokens/s of both arms, their ratio, and the geometric mean of the
ratios over the cells (the "aggregate" the DGX Spark acceptance target is stated on).  A request whose prompt was
served from the conversation cache (cache_n > 0) is not counted.
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
from pathlib import Path


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def cells(rows: list[dict]) -> list[tuple]:
    return sorted({(r["len"], r["temp"], r["max_new"]) for r in rows})


def med(rows: list[dict], key: str):
    vals = [float(r[key]) for r in rows if r.get(key) and not r.get("cache_n")]
    return statistics.median(vals) if vals else None


def arm_stats(rows: list[dict]) -> dict:
    out = {}
    for c in cells(rows):
        sel = [r for r in rows if (r["len"], r["temp"], r["max_new"]) == c]
        out[c] = (med(sel, "prompt_per_second"), med(sel, "predicted_per_second"), med(sel, "ttft_s"),
                  len([r for r in sel if not r.get("cache_n")]))
    return out



def _by_cell(rows, key):
    out = {}
    for r in rows:
        if r.get(key) and not r.get("cache_n"):
            out.setdefault((r["len"], r["temp"], r["max_new"]), []).append(float(r[key]))
    return out


def bootstrap(rows_a, rows_b, iters, seed=7):
    """Per-cell and aggregate ratio confidence intervals, resampling each arm's rows within a cell."""
    rng = random.Random(seed)
    res = {}
    for key in ("prompt_per_second", "predicted_per_second"):
        a, b = _by_cell(rows_a, key), _by_cell(rows_b, key)
        cells = sorted(set(a) & set(b))
        if not cells:
            continue
        draws = {c: [] for c in cells}
        agg = []
        for _ in range(iters):
            logs = []
            for c in cells:
                va, vb = a[c], b[c]
                ra = statistics.median([rng.choice(va) for _ in va])
                rb = statistics.median([rng.choice(vb) for _ in vb])
                if ra <= 0 or rb <= 0:
                    continue
                draws[c].append(rb / ra)
                logs.append(math.log(rb / ra))
            if logs:
                agg.append(math.exp(sum(logs) / len(logs)))
        pct = lambda xs, p: sorted(xs)[min(len(xs) - 1, max(0, int(p * len(xs))))]
        res[key] = {c: (statistics.median(draws[c]), pct(draws[c], 0.05), pct(draws[c], 0.95)) for c in cells}
        res[key]["aggregate"] = (statistics.median(agg), pct(agg, 0.05), pct(agg, 0.95)) if agg else None
    return res

def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    ci = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    a_path, b_path = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = load(a_path), load(b_path)
    sa, sb = arm_stats(a), arm_stats(b)
    name_a, name_b = a_path.stem, b_path.stem
    print(f"A = {name_a}   B = {name_b}\n")
    print(f"| cell | A prompt | B prompt | ratio | A decode | B decode | ratio | A ttft | B ttft |")
    print("|---|---|---|---|---|---|---|---|---|")
    pr, dr = [], []
    for c in sorted(set(sa) | set(sb)):
        pa, da, ta, na = sa.get(c, (None, None, None, 0))
        pb, db, tb, nb = sb.get(c, (None, None, None, 0))
        f = lambda v, w="{:.1f}": "-" if v is None else w.format(v)
        rp = (pb / pa) if (pa and pb) else None
        rd = (db / da) if (da and db) else None
        if rp:
            pr.append(rp)
        if rd:
            dr.append(rd)
        print(f"| {c[0]} T={c[1]} n={c[2]} | {f(pa)} | {f(pb)} | {f(rp, '{:.3f}')}x | {f(da)} | {f(db)} | "
              f"{f(rd, '{:.3f}')}x | {f(ta, '{:.2f}')} | {f(tb, '{:.2f}')} |")
    gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")
    print(f"\nprompt aggregate (geomean of ratios over {len(pr)} cells): {gm(pr):.3f}x")
    print(f"decode aggregate (geomean of ratios over {len(dr)} cells): {gm(dr):.3f}x")
    if pr:
        print(f"lowest prompt ratio: {min(pr):.3f}x   lowest decode ratio: {min(dr) if dr else float('nan'):.3f}x")
    if ci > 0:
        bs = bootstrap(a, b, ci)
        print(f"\nbootstrap 90% intervals over {ci} resamples (5th .. 95th percentile of the ratio):")
        for key, label in (("prompt_per_second", "prompt"), ("predicted_per_second", "decode")):
            if key not in bs:
                continue
            for c, v in sorted(bs[key].items(), key=lambda kv: str(kv[0])):
                if c == "aggregate":
                    continue
                print(f"  {label:6s} {c[0]:6d}/{c[1]}/{c[2]:4d}: {v[0]:.3f}x  [{v[1]:.3f}, {v[2]:.3f}]")
            agg = bs[key].get("aggregate")
            if agg:
                print(f"  {label:6s} aggregate: {agg[0]:.3f}x  [{agg[1]:.3f}, {agg[2]:.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
