"""CUDA backend: quantized gather kernels (torch reference implementation).

Semantics match ``mx.gather_qmm`` as checked against real MLX 0.30.4
(Metal, the version this repo pins) in ``tests/test_cuda_backend.py``:

* Packing: each uint32 word holds ``32 // bits`` codes, least significant
  bits first, along the last (input) axis. Codes are unsigned; a group of
  ``group_size`` codes dequantizes as ``w = code * scale + bias``. Scales
  can be negative -- nothing here assumes otherwise.
* Broadcasting: the output batch shape is
  ``broadcast(x.shape[:-2], rhs_indices.shape)`` and
  ``out[b] = x[b] @ W[rhs_indices[b]].T``. Both call patterns in
  ``streaming/layer.py`` rely on this: the unsorted path passes
  ``x[..., 1, 1, D]`` against ``rhs_indices[..., K]``, the sorted path
  passes ``x[T*K, 1, D]`` against ``rhs_indices[T*K]``.
* ``sorted_indices`` is a kernel hint in MLX; it never changes the values.
  Ignored here.

Not fast: every call dequantizes the gathered experts in full. The point
is a correct, testable reference for the streaming path.
"""

from __future__ import annotations

import torch


def _dequantize(w: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor,
                group_size: int, bits: int) -> torch.Tensor:
    """Packed ``[..., rows, in * bits / 32]`` uint32 -> float32 ``[..., rows, in]``."""
    per_word = 32 // bits
    # int32 is enough: >> sign-extends, but the mask keeps only the low
    # ``bits`` bits, which the sign bits never reach (int64 doubled the
    # transient memory -- 1.9 GB of codes for edge0-8b's lm_head).
    words = w.view(torch.int32)
    shifts = torch.arange(per_word, device=w.device, dtype=torch.int32) * bits
    codes = (words.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    codes = codes.reshape(*w.shape[:-1], w.shape[-1] * per_word)
    grouped = codes.reshape(*codes.shape[:-1], -1, group_size).to(torch.float32)
    deq = (grouped * scales.to(torch.float32).unsqueeze(-1)
           + biases.to(torch.float32).unsqueeze(-1))
    return deq.reshape(codes.shape)


def gather_qmm(x, w, scales, biases, rhs_indices, transpose=True,
               group_size=64, bits=4, mode="affine",
               sorted_indices=False):
    """Quantized matmul over a gathered subset of experts (``mx.gather_qmm``)."""
    if mode != "affine" or bits not in (2, 4, 8):
        raise NotImplementedError(
            f"reference gather_qmm covers affine 2/4/8-bit only "
            f"(got mode={mode!r}, bits={bits!r})")
    idx = rhs_indices.to(torch.long)
    bshape = torch.broadcast_shapes(x.shape[:-2], idx.shape)
    M = x.shape[-2]
    xf = x.to(torch.float32).expand(*bshape, *x.shape[-2:]).reshape(-1, M, x.shape[-1])
    flat = idx.expand(bshape).reshape(-1)
    n_out = w.shape[-2] if transpose else w.shape[-1] * (32 // bits)
    out = torch.empty(flat.numel(), M, n_out, dtype=torch.float32, device=x.device)
    # One distinct expert at a time: dequantizing a copy per (token, expert)
    # pair peaked at several GB per MoE layer during prefill. An index past
    # the last expert raises here; in MLX it silently reads out of bounds.
    for e in torch.unique(flat).tolist():
        rows = (flat == e).nonzero().squeeze(-1)
        deq = _dequantize(w[e], scales[e], biases[e], group_size, bits)
        out[rows] = torch.matmul(xf[rows], deq.T if transpose else deq)
    return out.reshape(*bshape, M, n_out).to(x.dtype)


def gather_sort(x, indices):
    """Sort token rows by expert id for ``gather_qmm(sorted_indices=True)``:
    returns ``(x_sorted, indices_sorted, inv_order)``, as mlx-lm's
    ``_gather_sort`` does."""
    m = indices.shape[-1]
    flat = indices.reshape(-1).to(torch.long)
    order = torch.argsort(flat, stable=True)
    inv_order = torch.argsort(order)
    return x.flatten(0, -3)[order // m], flat[order], inv_order


def scatter_unsort(x, inv_order, shape=None):
    """Undo ``gather_sort``; ``shape`` re-splits the leading axis."""
    x = x[inv_order]
    if shape is not None:
        x = x.unflatten(0, tuple(shape))
    return x


def swiglu(up: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """SiLU gated activation: silu(gate) * up."""
    return torch.nn.functional.silu(gate) * up
