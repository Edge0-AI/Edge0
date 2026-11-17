#!/usr/bin/env python3
"""ab.py (ARC-W): interleaved A/B of engine arms on the Arc box (one engine load per arm per round, order rotated).

  ab.py --name N --rounds 2 --plan "short:128:2,p4k:8:2" --arm "A=/path/strata" --arm "B=/path/strata|STRATA_X=1;STRATA_Y=2|--spec 2 ..."

arm = name=engine[|env;env[|extra args]].  Summary: median per case of the LAST repeat of each load (the first read of a
prompt in an engine is low), ratio to the first arm, and whether the greedy tokens equal the first arm's (same round).
"""
import argparse, json, os, statistics, subprocess, sys

W = "/mnt/models/strata-test/arc037"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--plan", default="short:128:2,p1k:8:3,p4k:8:2")
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--common", default="")
    a = ap.parse_args()
    arms = []
    for s in a.arm:
        n, _, rest = s.partition("=")
        parts = rest.split("|")
        arms.append({"name": n, "engine": parts[0], "env": [e for e in (parts[1].split(";") if len(parts) > 1 else []) if e],
                     "extra": parts[2] if len(parts) > 2 else None})
    data = {x["name"]: [] for x in arms}
    for r in range(a.rounds):
        order = arms[r % len(arms):] + arms[:r % len(arms)]
        for x in order:
            tag = f"{a.name}-{x['name']}-r{r}"
            cmd = [sys.executable, f"{W}/src/tools/arc/b.py", "--engine", x["engine"], "--tag", tag, "--plan", a.plan]
            for e in x["env"]: cmd += ["--env", e]
            if x["extra"] is not None: cmd += ["--extra", x["extra"]]
            if a.common: cmd += a.common.split()
            subprocess.run(cmd, stderr=sys.stderr)
            try:
                data[x["name"]].append(json.load(open(f"{W}/out/{tag}.json"))["results"])
            except Exception as e:
                print("missing", tag, e, file=sys.stderr)
    base = arms[0]["name"]
    cases = []
    for rr in data[base][:1]:
        for rec in rr:
            if rec["case"] not in cases: cases.append(rec["case"])
    print(f"== {a.name}  rounds={a.rounds}  plan={a.plan}")
    print(f"{'arm':<14}{'case':<8}{'pp med':>9}{'pp all':>26}{'tg med':>9}{'tg all':>22}{'x pp':>7}{'x tg':>7}{'ident':>7}")
    def last(rr, c):
        v = [x for x in rr if x["case"] == c]
        return v[-1] if v else None
    for c in cases:
        bpp = bt = None
        for x in arms:
            pp = [last(rr, c)["pp"] for rr in data[x["name"]] if last(rr, c)]
            tg = [last(rr, c)["tg"] for rr in data[x["name"]] if last(rr, c)]
            if not pp: continue
            mpp, mtg = statistics.median(pp), statistics.median(tg)
            if x["name"] == base: bpp, bt = mpp, mtg
            ident = ""
            if x["name"] != base:
                ok = 0; tot = 0
                for ra, rb in zip(data[base], data[x["name"]]):
                    la, lb = last(ra, c), last(rb, c)
                    if la and lb and la["generated"] > 8:
                        tot += 1; ok += la["tokens"] == lb["tokens"]
                ident = f"{ok}/{tot}" if tot else ""
            print(f"{x['name']:<14}{c:<8}{mpp:9.1f}{' '.join('%.0f' % v for v in pp):>26}{mtg:9.1f}{' '.join('%.1f' % v for v in tg):>22}"
                  f"{mpp / bpp if bpp else 0:7.3f}{mtg / bt if bt else 0:7.3f}{ident:>7}")
    json.dump(data, open(f"{W}/out/{a.name}.ab.json", "w"))


main()
