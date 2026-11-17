"""merge_dumps.py - add STRATA_ACT_DUMP files (int64 n, int64 count, double sum x^2 [n]) of several runs into one directory, and
print the per-layer summary.

  python3 tools/trellis/merge_dumps.py OUT_DIR IN_DIR [IN_DIR ...]
"""
import pathlib
import sys

import numpy as np

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
names = sorted({p.name for d in sys.argv[2:] for p in pathlib.Path(d).glob("act_L*_*.bin") if p.name.startswith("act_L") and p.stat().st_size < 400_000_000})
for name in names:
    tot, cnt, n0, cov = None, 0, None, None
    for d in sys.argv[2:]:
        p = pathlib.Path(d) / name
        if not p.exists():
            continue
        raw = p.read_bytes()
        n, c = np.frombuffer(raw[:16], dtype=np.int64)
        sq = np.frombuffer(raw[16:16 + 8 * n], dtype=np.float64)
        tot = sq.copy() if tot is None else tot + sq
        if len(raw) >= 16 + 8 * n + 8 * n * n:
            cv = np.frombuffer(raw[16 + 8 * n:16 + 8 * n + 8 * n * n], dtype=np.float64)
            cov = cv.copy() if cov is None else cov + cv
        cnt += int(c)
        n0 = int(n)
    (out / name).write_bytes(np.array([n0, cnt], dtype=np.int64).tobytes() + tot.astype(np.float64).tobytes()
                             + (cov.tobytes() if cov is not None else b""))
    m = tot / cnt
    print("%-16s n %d count %d  mean E[x^2] %.4g  cv over channels %.3f  max/mean %.1f" % (name, n0, cnt, m.mean(), m.std() / m.mean(), m.max() / m.mean()))
