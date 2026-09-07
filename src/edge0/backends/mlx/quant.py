"""MLX backend: quantized gather kernels (bit-compatible with mlx-lm)."""

from __future__ import annotations

import mlx.core as mx


def gather_qmm(x, w, scales, biases, rhs_indices, transpose=True,
               group_size=64, bits=4, mode="affine",
               sorted_indices=False):
    """Quantized matmul over a gathered subset of expert rows.

    ``w`` is a stacked [num_experts, out, in] weight tensor; rows are
    gathered by ``rhs_indices`` before dequantizing.  Same kernel and
    defaults as the resident (non-streaming) MoE path, so outputs are
    bit-identical to a resident ``SwitchGLU``.
    """
    return mx.gather_qmm(
        x, w, scales, biases, rhs_indices=rhs_indices, transpose=transpose,
        group_size=group_size, bits=bits, mode=mode,
        sorted_indices=sorted_indices)


def swiglu(up: mx.array, gate: mx.array) -> mx.array:
    """SiLU gated activation: silu(gate) * up."""
    return mx.nn.silu(gate) * up
