#!/usr/bin/env python3
"""sweep.py - run par_run.py over a list of variants (one server start each) and print a one-line summary per variant.

variants file, one per line, '#' comments:   tag | gpus | parallel | groups | env (K=V;K=V) | engine/server extras | work
  s8g1 | 0,1 | 8 | 1 | STRATA_BATCH_MTP=1 | | w2:4,8@--max-tokens 128
Empty fields = unset.  Run on the box:  python3 sweep.py --box boston --root ~/par29 --exe ~/par29/main/b86/strata --out RES variants.txt
"""
import argparse, json, os, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("variants")
ap.add_argument("--box", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--exe", required=True)
ap.add_argument("--repo", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--rounds", default="2")
ap.add_argument("--default-work", default="w1;w2:4,8@--max-tokens 128")
ap.add_argument("--exe-base", default=None)
a = ap.parse_args()
repo = a.repo or a.root + "/main"
here = os.path.dirname(os.path.abspath(__file__))
for line in open(a.variants):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    f = [x.strip() for x in line.split("|")] + [""] * 8
    tag, gpus, par, groups, env, extra, work, srv = f[:8]
    cmd = [sys.executable, f"{here}/../par_run.py", "--box", a.box, "--tag", tag, "--exe", a.exe, "--repo", repo, "--gpus", gpus or "0,1",
           "--out", a.out, "--rounds", a.rounds, "--work", work or a.default_work]
    if "nomtp" in tag:
        cmd += ["--no-mtp"]
    if par:
        cmd += ["--parallel", par]
    if groups:
        cmd += ["--batch-groups" if False else "--groups", groups]
    if env:
        cmd += ["--env", env]
    if extra:
        cmd += ["--extra", extra]
    if srv:
        cmd += ["--server-extra", srv]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = [l for l in r.stdout.splitlines() if l.startswith(("w1 ", "w2 ", "w4 ", "w6 ", "SERVER", "ABORT", "DONE", "[", "par_load"))]
    print(f"## {tag}: rc={r.returncode}", flush=True)
    for l in out:
        print("   " + l[:200], flush=True)
