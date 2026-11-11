"""pilot_summary.py - tables from the jsonl of pilot.py.
  python3 tools/trellis/pilot_summary.py pilot1.jsonl
"""
import collections
import json
import math
import sys

rows = [json.loads(l) for l in open(sys.argv[1])]
print("records:", len(rows), "layers:", sorted({r["layer"] for r in rows}), "experts:", sorted({r["expert"] for r in rows}),
      "rows per matrix:", sorted({r["rows"] for r in rows}))
print("activation columns are PLACEHOLDERS (iso gaussian / synthetic outlier channels), not calibration data\n")


def db(x):
    return -10 * math.log10(x)


def table(keyf, title, cols=("mse", "act_iso", "act_syn")):
    g = collections.defaultdict(list)
    for r in rows:
        g[keyf(r)].append(r)
    print(title)
    for k in sorted(g):
        v = g[k]
        print("  %-40s n=%-3d " % (k, len(v)) + "  ".join("%s %.4f (%.2f dB)" % (c, sum(x[c] for x in v) / len(v), db(sum(x[c] for x in v) / len(v))) for c in cols))
    print()


rate_bpw = {"K4": 1.125, "K5": 1.375, "K6": 1.625, "K7": 1.875}
table(lambda r: "%s %.3f bpw %s" % (r["rate"], rate_bpw[r["rate"]], "hadamard" if r["had"] else "plain   "), "mean over everything")
table(lambda r: "%s %s %s" % (r["role"], r["rate"], "had" if r["had"] else "plain"), "by projection")
table(lambda r: "L%02d %s %s" % (r["layer"], r["rate"], "had" if r["had"] else "plain"), "by layer")
# spread of the weight mse across experts for K5 plain
v = [r["mse"] for r in rows if r["rate"] == "K5" and not r["had"]]
print("K5 plain weight mse over all %d matrices: min %.4f max %.4f" % (len(v), min(v), max(v)))
# hadamard gain on the synthetic outlier activations
print("\nHadamard effect on act_syn (relative error, plain / hadamard), per rate")
for rate in ("K4", "K5", "K6", "K7"):
    a = [r["act_syn"] for r in rows if r["rate"] == rate and not r["had"]]
    b = [r["act_syn"] for r in rows if r["rate"] == rate and r["had"]]
    print("  %s plain %.4f had %.4f ratio %.3f" % (rate, sum(a) / len(a), sum(b) / len(b), (sum(a) / len(a)) / (sum(b) / len(b))))
