"""MLX backend: quantized gather kernels (bit-compatible with mlx-lm)."""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort


def gather_sort(x, indices):
    """Sort token rows by expert id for ``gather_qmm(sorted_indices=True)``:
    returns ``(x_sorted, indices_sorted, inv_order)`` (mlx-lm's helper)."""
    return _gather_sort(x, indices)


def scatter_unsort(x, inv_order, shape=None):
    """Undo ``gather_sort``; ``shape`` re-splits the leading axis."""
    return _scatter_unsort(x, inv_order, shape)


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
