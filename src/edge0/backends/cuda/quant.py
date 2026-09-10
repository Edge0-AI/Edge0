"""CUDA backend: quantized gather kernel (contract: ``gather_qmm``).

Deprioritized by a discovery made while building this skeleton: MLX
itself ships a real CUDA backend (``pip install mlx[cuda12]``, checked
against ``ml-explore/mlx`` source) with a native
``GatherQMM::eval_gpu`` in ``mlx/backend/cuda/quantized/quantized.cpp``
-- the exact op this module exists to replace. Testing
``EDGE0_BACKEND=mlx`` with ``mlx[cuda12]`` installed, unmodified, on
real NVIDIA hardware is the next action, not finishing this file (see
the chat for the full finding and the "route 0" experiment).

This reference implementation stays here for two honest reasons, not
as a stand-in for that test:

1. It is a *fallback if the MLX-CUDA path has a real gap* (e.g. an op
   edge0 needs that the CUDA backend hasn't ported from Metal yet --
   plausible, since MLX's own docs describe CUDA as the newer backend).
2. Its bit-packing/dequant math is UNVERIFIED against ``mx.quantize``'s
   actual affine int4 layout -- built from reading
   ``mlx/backend/cpu/quantized.cpp`` reference logic, not from running
   both side by side (no Mac in this sandbox to generate ground
   truth). Treat every number this produces as unverified until
   checked against ``mx.quantize`` output on real hardware.  Do NOT
   promote this to the fast path without that check even if route 1
   (hand-tuned staging) ends up being the right call later.
"""

from __future__ import annotations

import torch

from edge0.backends.cuda.core import DEVICE


def _unpack_affine_u4(packed: torch.Tensor, out_features: int,
                       in_features: int) -> torch.Tensor:
    """Unpack ``bits=4`` affine-quantized weights from packed uint32 words
    into a ``[out_features, in_features]`` tensor of 0..15 integer codes.

    UNVERIFIED packing-order assumption (see module docstring): 8 nibbles
    per uint32 word, packed least-significant-nibble-first along the
    ``in_features`` axis (standard convention, matches
    ``mlx/backend/cpu/quantized.cpp``'s scalar unpack loop as read, not
    as executed against real output).
    """
    w32 = packed.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, device=packed.device) * 4
    nibbles = (w32.unsqueeze(-1) >> shifts) & 0xF  # [..., 8]
    return nibbles.reshape(out_features, in_features).to(torch.float32)


def gather_qmm(x, w, scales, biases, rhs_indices, transpose=True,
                group_size=64, bits=4, mode="affine",
                sorted_indices=False):
    """Reference gather + affine dequant + matmul. Correct shape/data-flow,
    UNVERIFIED numerics (see module docstring) -- not wired for speed
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
