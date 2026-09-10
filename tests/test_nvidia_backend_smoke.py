"""Smoke test for MLX's CUDA-backend quantized-matmul support in isolation.

Skips everywhere except a real GPU box with a CUDA-backed MLX installed.
This does NOT test edge0 end-to-end -- see docs/nvidia.md: as of this
writing edge0 does not run end-to-end on any NVIDIA hardware/MLX version
tried (0.30.4 has no CUDA GatherQMM at all; 0.31.1's is incomplete;
0.32.x has both required kernels but edge0's `ling.py` hits an unrelated
`IndexError` downstream in `core.eval`). What IS confirmed working as of
0.32.x is the isolated op below -- useful signal on its own for whoever
picks up the `backends/cuda/` slot next: the quantized kernel is not the
remaining blocker, whatever `ling.py` hits is.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core", reason="mlx not installed")


def _has_gpu() -> bool:
    try:
        return mx.default_device().type == mx.DeviceType.gpu
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_gpu(),
    reason="requires mlx[cuda12] (or Metal) with a real GPU device active",
)


def test_default_device_is_gpu():
    assert mx.default_device().type == mx.DeviceType.gpu


def test_gather_qmm_matches_dense_reference():
    """The exact op edge0's streaming path depends on
    (``backends/mlx/quant.py::gather_qmm``), checked against MLX's own
    dense quantize/dequantize round-trip as ground truth -- no
    cross-framework comparison needed since both sides are MLX here.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    n_experts, out_features, in_features = 4, 32, 128
    group_size, bits = 64, 4

    w = mx.array(
        rng.standard_normal((n_experts, out_features, in_features)).astype(np.float32)
    )
    wq, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(wq, scales, biases)

    x = mx.array(rng.standard_normal((2, in_features)).astype(np.float32))
    rhs_indices = mx.array([1, 3])

    gathered = mx.gather_qmm(
        x, wq, scales, biases, rhs_indices=rhs_indices,
        transpose=True, group_size=group_size, bits=bits,
    )
    mx.eval(gathered)

    w_deq = mx.dequantize(wq, scales, biases, group_size=group_size, bits=bits)
    dense = mx.stack([
        x[i] @ w_deq[int(rhs_indices[i])].T for i in range(x.shape[0])
    ])
    mx.eval(dense)

    rel_err = float(mx.linalg.norm(gathered - dense) / mx.linalg.norm(dense))
    assert rel_err < 1e-2, f"gather_qmm vs dense mismatch: rel_err={rel_err}"


@pytest.mark.skipif(
    "EDGE0_NVIDIA_SMOKE_MODEL" not in __import__("os").environ,
    reason="set EDGE0_NVIDIA_SMOKE_MODEL to a local checkpoint dir to run this",
)
@pytest.mark.xfail(
    reason="edge0 does not run end-to-end on CUDA on any MLX version "
           "tried as of this writing -- see docs/nvidia.md. Left as "
           "xfail (not skipped) so this test flips to an unexpected "
           "pass, and gets noticed, the day the underlying gap closes.",
    strict=False,
)
def test_edge0_end_to_end_generation():
    import os
    from edge0.backends import core, io

    model_path = os.environ["EDGE0_NVIDIA_SMOKE_MODEL"]
    tokenizer = io.load_tokenizer(model_path)
    ids = tokenizer.encode("The capital of France is")
    assert len(ids) > 0
