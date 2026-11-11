#!/usr/bin/env python3
"""Paired-block analysis for par_load.py ledgers (standard library only).

A pair = one A block and one B block of the same workload/level, run on the same hardware and cache regime and labelled
with the same --pair id (--arm A|B).  For every (workload, level, metric) it prints the per-pair ratios B/A, their
geometric mean and a paired bootstrap 95% interval (seeded), how many pairs B won, and the decision label from the
plan (screening = few pairs: <5% in every pair low priority, >=10% keep; confirmation = 20+ pairs, lower bound vs --target).

    python3 tools/par_analyze.py a1.json b1.json a2.json b2.json ...           (arm/pair read from the ledgers)
    python3 tools/par_analyze.py runs/*.json --metric agg_tps --target 2.0 --json out.json

Metrics where lower is better (ttft_*, tpot_*, max_gap_*, latency_*) are inverted so ratio > 1 always means B is better.
`--baseline-arm`/`--cand-arm` choose the arms (default A and B).  A pair missing either arm, an invalid ledger
(no server usage) or a block with errors is reported and left out.
"""
import argparse, glob, json, math, random, sys
from collections import defaultdict

LOWER_BETTER = ("ttft", "tpot", "max_gap", "latency", "lag", "joiner_ttft", "incumbent_gap", "makespan")


def lower_is_better(metric):
    return metric.startswith(LOWER_BETTER)


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def boot(ratios, n=10000, seed=1):
    rng = random.Random(seed)
    k = len(ratios)
    gm = sorted(geomean([ratios[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    return gm[int(0.025 * n)], gm[int(0.975 * n) - 1]


def load(paths):
    rows, skipped = [], []
    for p in paths:
        try:
            L = json.load(open(p))
        except Exception as e:  # noqa: BLE001
            skipped.append((p, f"unreadable {e}"))
            continue
        if not L.get("valid", True):
            skipped.append((p, "invalid ledger: " + str(L.get("invalid_reason"))))
            continue
        for b in L.get("blocks", []):
            if b.get("errors"):
                skipped.append((p, f"block has {b['errors']} errors"))
                continue
            rows.append((p, b))
    return rows, skipped


def analyze(rows, metric, a_arm="A", b_arm="B", target=None, min_confirm=20, boot_n=10000):
    groups = defaultdict(lambda: defaultdict(dict))
    for p, b in rows:
        if metric not in b or b[metric] is None or not b.get("pair"):
            continue
        key = (b.get("workload"), b.get("level", b.get("offered_rate")), b.get("share"), b.get("round"))
        groups[key][b["pair"]][b["arm"]] = b[metric]
    out = []
    for key, pairs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        ratios, used = [], []
        for pid, arms in pairs.items():
            if a_arm in arms and b_arm in arms and arms[a_arm] and arms[b_arm]:
                r = arms[b_arm] / arms[a_arm]
                ratios.append(1 / r if lower_is_better(metric) else r)
                used.append(pid)
        if not ratios:
            continue
        gm = geomean(ratios)
        lo, hi = boot(ratios, boot_n) if len(ratios) > 1 else (gm, gm)
        n = len(ratios)
        faster = sum(1 for r in ratios if r > 1.0)
        if n < min_confirm:
            kind = "screen"
            if all(r < 1.05 for r in ratios):
                label = "low priority (<5% in every pair)" if all(r > 0.95 for r in ratios) else "reject/regression in every pair" if all(r < 1.0 for r in ratios) else "low priority"
            elif gm >= 1.10 and faster == n:
                label = "keep (>=10%)"
            elif gm >= 1.05:
                label = "investigate (5-10%)"
            else:
                label = "mixed, investigate"
        else:
            kind = "confirm"
            label = "n/a"
            if target is not None:
                label = f"PASS (lower bound {lo:.3f} >= {target})" if lo >= target else f"NOT PROVEN (lower bound {lo:.3f} < {target})"
        out.append({"workload": key[0], "level": key[1], "share": key[2], "round": key[3], "metric": metric, "pairs": n,
                    "ratios": [round(r, 4) for r in ratios], "geomean": gm, "ci95": [lo, hi], "pairs_better": faster,
                    "kind": kind, "decision": label, "mean_A": sum(pairs[p][a_arm] for p in used) / n, "mean_B": sum(pairs[p][b_arm] for p in used) / n})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ledgers", nargs="+")
    ap.add_argument("--metric", action="append", default=None, help="repeatable; default agg_tps ttft_p95 max_gap_p95")
    ap.add_argument("--baseline-arm", default="A")
    ap.add_argument("--cand-arm", default="B")
    ap.add_argument("--target", type=float, default=None, help="confirmation: the lower bound must clear this ratio")
    ap.add_argument("--min-confirm", type=int, default=20)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    paths = [p for g in a.ledgers for p in (glob.glob(g) or [g])]
    rows, skipped = load(paths)
    res = []
    for m in a.metric or ["agg_tps", "ttft_p95", "max_gap_p95"]:
        res += analyze(rows, m, a.baseline_arm, a.cand_arm, a.target, a.min_confirm)
    for s in skipped:
        print("skipped:", *s)
    print(f"{'workload':8} {'level':>6} {'metric':14} {'pairs':>5} {'geomean':>8} {'95% CI':>17} {'better':>7}  decision")
    for r in res:
        lvl = r["level"] if r["share"] is None else f"{r['level']}/s{r['share']}"
        print(f"{r['workload']:8} {str(lvl):>6} {r['metric']:14} {r['pairs']:>5} {r['geomean']:>8.3f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] "
              f"{r['pairs_better']:>3}/{r['pairs']:<3}  {r['kind']}: {r['decision']}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
