"""CUDA (torch reference) backend checked against real MLX.

MLX is the ground truth here: every test builds inputs with ``mx.quantize``
and compares the torch implementation against the MLX op it replaces.
Skipped unless both ``mlx`` and ``torch`` are importable (e.g. Apple
Silicon with torch installed); the torch side runs on CPU there.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from edge0.backends.cuda import quant as cq  # noqa: E402

E, O, D = 8, 16, 128


def _t(a):
    if a.dtype == mx.bfloat16:  # numpy has no bfloat16: move the raw bits
        return torch.from_numpy(np.array(a.view(mx.uint16))).view(torch.bfloat16)
    return torch.from_numpy(np.array(a))


def _quantized(bits=4, group_size=64, dtype=mx.float32, shape=(E, O, D)):
    mx.random.seed(0)
    w = mx.random.normal(shape).astype(dtype)
    wq, s, b = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(wq, s, b)
    return wq, s, b


def _both(x, idx, wq, s, b, **kw):
    ref = mx.gather_qmm(x, wq, s, b, rhs_indices=idx, **kw)
    mx.eval(ref)
    got = cq.gather_qmm(_t(x), _t(wq), _t(s), _t(b), _t(idx), **kw)
    return np.array(ref.astype(mx.float32)), got.float().numpy()


@pytest.mark.parametrize("x_shape,idx_shape", [
    ((3, D), (3,)),            # default convention: broadcasts to (3, 3, O)
    ((5, 1, 1, D), (5, 3)),    # streaming/layer.py unsorted path
    ((15, 1, D), (15,)),       # streaming/layer.py sorted path (post _gather_sort)
    ((2, 1, 4, D), (2, 3)),
    ((1, 2, D), (4, 1)),
])
def test_gather_qmm_matches_mlx(x_shape, idx_shape):
    wq, s, b = _quantized()
    mx.random.seed(1)
    x = mx.random.normal(x_shape)
    idx = mx.random.randint(0, E, idx_shape).astype(mx.uint32)
    ref, got = _both(x, idx, wq, s, b, transpose=True, group_size=64, bits=4)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_gather_qmm_bits(bits):
    wq, s, b = _quantized(bits=bits)
    x = mx.random.normal((4, 1, 1, D))
    idx = mx.random.randint(0, E, (4, 2)).astype(mx.uint32)
    ref, got = _both(x, idx, wq, s, b, transpose=True, group_size=64, bits=bits)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_gather_qmm_no_transpose():
    wq, s, b = _quantized(shape=(E, D, O * 4))
    x = mx.random.normal((3, 1, 1, D))
    idx = mx.random.randint(0, E, (3, 2)).astype(mx.uint32)
    ref, got = _both(x, idx, wq, s, b, transpose=False, group_size=64, bits=4)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_gather_qmm_sorted_flag_is_only_a_hint():
    wq, s, b = _quantized()
    x = mx.random.normal((6, 1, D))
    idx = mx.array(sorted(np.random.default_rng(0).integers(0, E, 6).tolist()),
                   dtype=mx.uint32)
    ref, got = _both(x, idx, wq, s, b, transpose=True, group_size=64, bits=4,
                     sorted_indices=True)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_gather_qmm_bf16_checkpoint_dtypes():
    # Checkpoint scales/biases are bf16 and activations are bf16 in the
    # real models; compare in float32 with a bf16-sized tolerance.
    wq, s, b = _quantized(dtype=mx.bfloat16)
    x = mx.random.normal((4, 1, 1, D)).astype(mx.bfloat16)
    idx = mx.random.randint(0, E, (4, 2)).astype(mx.uint32)
    ref, got = _both(x, idx, wq, s, b, transpose=True, group_size=64, bits=4)
    np.testing.assert_allclose(got, ref, rtol=2e-2, atol=1e-1)


def test_swiglu_matches_mlx():
    import mlx.nn as mnn
    up, gate = mx.random.normal((4, 32)), mx.random.normal((4, 32))
    ref = np.array(mnn.silu(gate) * up)
    got = cq.swiglu(_t(up), _t(gate)).numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)
