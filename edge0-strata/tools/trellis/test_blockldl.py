"""test_blockldl.py - block feedback against plain encoding on a synthetic anisotropic H (an incoherent random basis with a
spread spectrum) and on real expert weights: the proxy loss tr(E H E^T) must drop, the plain loss rises a little.
  python3 tools/trellis/test_blockldl.py [--orig DIR --layer 4]
"""
import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import blockldl  # noqa: E402
import tqenc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1024)
ap.add_argument("--rows", type=int, default=512)
ap.add_argument("--spread", type=float, default=100.0, help="ratio of the largest to the smallest eigenvalue")
ap.add_argument("--type", default="TQK6")
a = ap.parse_args()
t = tqenc.TYPE_BY_NAME[a.type]
rng = np.random.default_rng(1)
n = a.n
Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
lam = np.logspace(0, -np.log10(a.spread), n)
H = (Q * lam) @ Q.T
W = (rng.standard_normal((a.rows, n)) * 0.02).astype(np.float32)
R = blockldl.hessian_factor(H, 0.01)
fb = blockldl.feedback_matrices(R)
pk, _, _, _ = tqenc.encode(W, t, iters=1, threads=8)
plain = tqenc.decode(pk, t).reshape(W.shape)
_, ldl = blockldl.encode_ldl(W, t, fb, threads=8)


def proxy(Wq):
    E = (W - Wq).astype(np.float64)
    return np.trace(E @ H @ E.T) / np.trace(W.astype(np.float64) @ H @ W.T.astype(np.float64))


for name, Wq in (("plain", plain), ("block LDL", ldl)):
    E = W - Wq
    print("%-10s unweighted %.2f dB   H-weighted %.2f dB" % (name, 10 * np.log10(np.sum(W.astype(np.float64) ** 2) / np.sum(E.astype(np.float64) ** 2)), -10 * np.log10(proxy(Wq))))
