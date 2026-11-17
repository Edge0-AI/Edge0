"""Python-level checks of tools/tqenc.py (run on the box with libtqenc.so built): Hadamard orthonormality and the
Q(Ux) == Q_orig x identity through the encoder, signs and .tqx round trips, K5/K4 sizes."""
import os
import sys
import tempfile
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tqenc  # noqa: E402

fails = 0


def check(c, msg):
    global fails
    if not c:
        fails += 1
        print("FAIL", msg)


H = tqenc.hadamard_matrix()
check(np.abs(H @ H.T - np.eye(128)).max() < 1e-6 and np.abs(H - H.T).max() == 0, "H128 orthonormal and symmetric")
rng = np.random.default_rng(3)
for n in (2304, 5120):
    s = tqenc.make_signs(n, 5)
    check(set(np.unique(s)) <= {-1.0, 1.0}, "signs +-1")
    check(np.array_equal(tqenc.unpack_signs(tqenc.pack_signs(s), n), s), "sign pack round trip")
    W = rng.standard_normal((8, n)).astype(np.float32)
    x = rng.standard_normal(n).astype(np.float32)
    Wr = tqenc.rotate(W, s)
    check(np.abs(tqenc.unrotate(Wr, s) - W).max() < 1e-5, "unrotate(rotate(W)) == W")
    Ux = tqenc.rotate(x[None, :], s)[0]
    check(np.abs(Wr @ Ux - W @ x).max() < 1e-3, "Wrot (U x) == W x before quantization")
    for t in (tqenc.TQK4X, tqenc.TQK5X, tqenc.TQK4T, tqenc.TQK5T, tqenc.TQK6, tqenc.TQK7, tqenc.TQ2_T):
        packed, err, tot, st = tqenc.encode(Wr, t, iters=0)
        check(packed.shape == (8 * n // 128, tqenc.block_bytes(t)), "packed shape")
        Q = tqenc.decode(packed, t).reshape(8, n)
        y1 = Q @ Ux
        y2 = tqenc.unrotate(Q, s) @ x
        check(np.abs(y1 - y2).max() < 1e-3 * max(1.0, np.abs(y1).max()), "Q(Ux) == unrotate(Q) x  type %d" % t)
        check(abs(tot - np.sum((Wr.astype(np.float64) - Q) ** 2)) < 1e-4 * tot, "encoder error == decoded error")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.tqx")
            tqenc.write_tqx(p, t, 8, n, packed, s)
            t2, r2, c2, pk2, s2 = tqenc.read_tqx(p)
            check(t2 == t and r2 == 8 and c2 == n and np.array_equal(pk2, packed) and np.array_equal(s2, s), "tqx round trip")
check(tqenc.block_bytes(tqenc.TQK5X) == 22 and tqenc.block_bytes(tqenc.TQK4X) == 18, "K5 22 bytes, K4 18 bytes")
check(abs(tqenc.bpw(tqenc.TQK5X) - 1.375) < 1e-9 and abs(tqenc.bpw(tqenc.TQK4X) - 1.125) < 1e-9, "bpw")
print("python checks:", "OK" if not fails else "%d FAILED" % fails)
sys.exit(1 if fails else 0)
