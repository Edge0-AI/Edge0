"""CUDA backend: quantized gather kernel (contract: ``gather_qmm``).

Deprioritized by a discovery made while building this skeleton: MLX
itself ships a real CUDA backend (``pip install mlx[cuda12]``, checked
against ``ml-explore/mlx`` source) with a native
``GatherQMM::eval_gpu`` in ``mlx/backend/cuda/quantized/quantized.cpp``
-- the exact op this module exists to replace. That path was tested
end-to-end on real NVIDIA hardware and does not work today at any MLX
version tried (see docs/nvidia.md) -- this reference implementation is
Plan A again, not a fallback.

Packing/formula VERIFIED against real ``mx.quantize`` output (DGX
Spark, GB10, mlx 0.32.2, CPU execution -- packing is backend-
independent, defined by ``mx.quantize`` itself): nibble ``j`` sits in
bits ``4j..4j+3`` of each uint32 word (8 per word, LSB-first) --
dequant error 0.0 with this order, up to 4.15 with the reverse order,
tested against ``mx.dequantize``'s own reference output. Codes are
unsigned 0..15, no zero-point offset; the formula is ``w = code*scale +
bias`` per group, and ``scale`` CAN be negative (confirmed, e.g.
expert 0 row 0: scale -0.2907, bias +2.3726) -- _unpack_affine_u4 below
already made no positivity assumption, so this needed no code change,
only removing the "unverified" label.

STILL OPEN, found by that same test and NOT yet fixed here: the
verification script called ``gather_qmm`` with the DEFAULT calling
convention (unexpanded ``x``, default ``sorted_indices``) and got a
full ``[len(rhs_indices), len(x), out_features]`` broadcast back (e.g.
``(3,3,4)`` for 3 experts-selected x 3 x-rows), with the real
"row i -> expert idx[i]" mapping living on the diagonal -- NOT the
one-row-per-expert ``[T, out]`` shape ``gather_qmm`` below assumes.
``streaming/layer.py``'s real call sites always pass
``sorted_indices=True`` plus pre-expanded ``x`` (``expand_dims(x_flat,
(-2,-3))``) -- whether that combination changes the OUTPUT SEMANTICS
(not just kernel selection) to the aligned one-to-one shape this file
assumes is UNVERIFIED; the test done so far didn't exercise
``sorted_indices=True`` at all. Don't trust this file's shape handling
for the real streaming call pattern until that's checked specifically
-- the dequant math is solid, the calling convention isn't yet.
"""

from __future__ import annotations

import torch

from edge0.backends.cuda.core import DEVICE


def _unpack_affine_u4(packed: torch.Tensor, out_features: int,
                       in_features: int) -> torch.Tensor:
    """Unpack ``bits=4`` affine-quantized weights from packed uint32 words
    into a ``[out_features, in_features]`` tensor of 0..15 integer codes.

    Packing order VERIFIED (see module docstring, dated confirmation
    against real ``mx.quantize`` output): 8 nibbles per uint32 word,
    LSB-first along the ``in_features`` axis.
    """
    w32 = packed.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, device=packed.device) * 4
    nibbles = (w32.unsqueeze(-1) >> shifts) & 0xF  # [..., 8]
    return nibbles.reshape(out_features, in_features).to(torch.float32)


def gather_qmm(x, w, scales, biases, rhs_indices, transpose=True,
                group_size=64, bits=4, mode="affine",
                sorted_indices=False):
    """Reference gather + affine dequant + matmul. Dequant math VERIFIED
    (see module docstring); the ``sorted_indices=True`` calling
    convention ``streaming/layer.py`` actually uses is NOT verified to
    produce the right shape -- see module docstring before wiring this
    into the streaming installer. Not wired for speed either way
    (materializes full dequantized weights per call, no fused kernel).
    """
    if mode != "affine" or bits != 4:
        raise NotImplementedError(
            f"reference gather_qmm only covers affine/4-bit "
            f"(got mode={mode!r}, bits={bits!r})")

    gathered_w = w.index_select(0, rhs_indices.reshape(-1).to(torch.long))
    gathered_s = scales.index_select(0, rhs_indices.reshape(-1).to(torch.long))
    gathered_b = biases.index_select(0, rhs_indices.reshape(-1).to(torch.long))

    n_rows = gathered_w.shape[0]
    out_features, packed_in = gathered_w.shape[-2], gathered_w.shape[-1]
    in_features = packed_in * 8  # 8 int4 values per uint32 word

    codes = _unpack_affine_u4(
        gathered_w.reshape(-1, packed_in), out_features, in_features
    ).reshape(n_rows, out_features, in_features)

    n_groups = in_features // group_size
    codes_g = codes.reshape(n_rows, out_features, n_groups, group_size)
    s = gathered_s.reshape(n_rows, out_features, n_groups, 1).to(torch.float32)
    b = gathered_b.reshape(n_rows, out_features, n_groups, 1).to(torch.float32)
    deq = (codes_g * s + b).reshape(n_rows, out_features, in_features)

    if transpose:
        return torch.matmul(x, deq.transpose(-1, -2))
    return torch.matmul(x, deq)


def swiglu(up: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate) * up
    if mode != "affine" or bits != 4:
        raise NotImplementedError(
            f"reference gather_qmm only covers affine/4-bit "
            f"(got mode={mode!r}, bits={bits!r})")

    gathered_w = w.index_select(0, rhs_indices.reshape(-1).to(torch.long))
    gathered_s = scales.index_select(0, rhs_indices.reshape(-1).to(torch.long))
    gathered_b = biases.index_select(0, rhs_indices.reshape(-1).to(torch.long))

    n_rows = gathered_w.shape[0]
    out_features, packed_in = gathered_w.shape[-2], gathered_w.shape[-1]
    in_features = packed_in * 8  # 8 int4 values per uint32 word

    codes = _unpack_affine_u4(
        gathered_w.reshape(-1, packed_in), out_features, in_features
    ).reshape(n_rows, out_features, in_features)

    n_groups = in_features // group_size
    codes_g = codes.reshape(n_rows, out_features, n_groups, group_size)
    s = gathered_s.reshape(n_rows, out_features, n_groups, 1).to(torch.float32)
    b = gathered_b.reshape(n_rows, out_features, n_groups, 1).to(torch.float32)
    deq = (codes_g * s + b).reshape(n_rows, out_features, in_features)

    if transpose:
        return torch.matmul(x, deq.transpose(-1, -2))
    return torch.matmul(x, deq)


def swiglu(up: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate) * up
