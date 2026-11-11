"""blockldl.py - Hessian-aware trellis encoding by block feedback (the block form of GPTQ / LDLQ, as QTIP does it).

Weights W [m, n] in the rotated basis, H = E[x x^T] [n, n] of the rotated layer input.  Columns are cut into 128-blocks (the
trellis blocks).  Block b is encoded on its own (m blocks of 128, plain-MSE trellis search), its error E_b = W_b - Q_b is
pushed onto the not yet coded columns, exactly as the OBS update does for a whole block:

    W[:, c1:] -= E_b  R_bb^{-1}  R[c0:c1, c1:]          R upper, H^-1 = R^T R

so the later blocks absorb what the earlier ones could not represent, in the directions that matter for ||(W - Q) X||.
The result is a normal TQK6 / TQK7 / TQK5X / TQK4X tensor; only the encoder changed.
"""
from __future__ import annotations

import numpy as np

import tqenc

try:
    from scipy.linalg import solve_triangular
except Exception:   # numpy only: a small back substitution
    solve_triangular = None


def _solve_upper(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if solve_triangular is not None:
        return solve_triangular(A, B, lower=False)
    return np.linalg.solve(A, B)


def hessian_factor(H: np.ndarray, damp: float = 0.01) -> np.ndarray:
    """R (upper) with (H + damp * mean(diag H) I)^-1 = R^T R."""
    H = np.asarray(H, dtype=np.float64).copy()
    d = damp * float(np.mean(np.diag(H)))
    H[np.diag_indices_from(H)] += d
    Hinv = np.linalg.inv(H)
    Hinv = (Hinv + Hinv.T) * 0.5
    L = np.linalg.cholesky(Hinv)
    return L.T


def feedback_matrices(R: np.ndarray, g: int = 128):
    """For every block b the matrix R_bb^{-1} R[b, b+1:] ([g, n - (b+1) g]); the last block has none."""
    n = R.shape[0]
    out = []
    for c0 in range(0, n, g):
        c1 = c0 + g
        out.append(_solve_upper(R[c0:c1, c0:c1], R[c0:c1, c1:]).astype(np.float32) if c1 < n else None)
    return out


def encode_ldl(W: np.ndarray, t: int, fb, gpu: bool = False, iters: int = 1, ncand: int = 1, threads: int = 0):
    """W [m, n] float32 (n % 128 == 0), fb from feedback_matrices.  Returns (packed [m, n/128, block_bytes], decoded [m, n])."""
    m, n = W.shape
    B = n // 128
    bb = tqenc.block_bytes(t)
    Wc = np.ascontiguousarray(W, dtype=np.float32).copy()
    packed = np.empty((m, B, bb), dtype=np.uint8)
    dec = np.empty((m, n), dtype=np.float32)
    kw = dict(iters=iters, ncand=ncand)
    if gpu:
        kw["gpu"] = True
    else:
        kw["threads"] = threads
    for b in range(B):
        c0, c1 = b * 128, (b + 1) * 128
        blk = np.ascontiguousarray(Wc[:, c0:c1])
        pk, _, _, _ = tqenc.encode(blk, t, **kw)
        q = tqenc.decode(pk, t).reshape(m, 128)
        packed[:, b] = pk
        dec[:, c0:c1] = q
        if fb[b] is not None:
            Wc[:, c1:] -= (blk - q) @ fb[b]
    return packed, dec
