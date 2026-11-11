#!/usr/bin/env python3
"""kl.py A.bin B.bin ... : KL(A||B), top-1 agreement and max |dlogit| between first-token logit dumps (STRATA_DUMP_FIRST_LOGITS)."""
import sys
import numpy as np


def load(p):
    return np.fromfile(p, dtype=np.float32).astype(np.float64)


def lsm(x):
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


a = load(sys.argv[1])
la = lsm(a)
pa = np.exp(la)
for p in sys.argv[2:]:
    b = load(p)
    lb = lsm(b)
    kl = float((pa * (la - lb)).sum())
    top = np.argsort(-a)[:5]
    print(f"{p}: KL(A||B)={kl:.3e}  top1 {'same' if a.argmax() == b.argmax() else 'DIFF'}  max|dlogit|={np.abs(a - b).max():.4f}  "
          f"max|dlogit| top5={np.abs(a[top] - b[top]).max():.4f}  A top1 p={pa.max():.3f}")
