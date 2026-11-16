#!/usr/bin/env python3
"""kl_logits.py DIR PREFIX REF ARM... - first-window logits dumps (STRATA_DUMP_FIRST_LOGITS=DIR/PREFIX-<arm>, files .0 .1 ...):
KL(REF || ARM) per request, mean/median/max, and top-1 agreement (the klcmp.py of phaseA-tests/026q, for dumps of one arm per label)."""
import sys
from pathlib import Path
import numpy as np
d = Path(sys.argv[1]); pre = sys.argv[2]; ref = sys.argv[3]; arms = sys.argv[4:]
def ls(x):
    x = x.astype(np.float64); m = x.max(); return x - m - np.log(np.exp(x - m).sum())
for arm in arms:
    kls, top, n = [], 0, 0
    i = 0
    while (d / f"{pre}-{ref}.{i}").exists() and (d / f"{pre}-{arm}.{i}").exists():
        a = np.fromfile(d / f"{pre}-{ref}.{i}", "<f4"); b = np.fromfile(d / f"{pre}-{arm}.{i}", "<f4")
        la, lb = ls(a), ls(b)
        kls.append(float((np.exp(la) * (la - lb)).sum())); top += int(a.argmax() == b.argmax()); n += 1; i += 1
    if not n:
        print(arm, "no dumps"); continue
    k = np.array(kls)
    print(f"{arm:4s} vs {ref}: requests {n}  mean KL {k.mean():.6f}  median {np.median(k):.6f}  max {k.max():.6f}  top-1 same {top}/{n}")
