"""CUDA backend: module-factory namespace (contract: Module, Linear,
RMSNorm, silu, gelu).

``torch.nn`` already provides all four natively (``nn.Linear``,
``nn.functional.silu``/``gelu``); the only real gap is ``RMSNorm``,
which is standard ``torch.nn.RMSNorm`` since torch 2.4 -- re-exported
here under the contract's names rather than assuming call sites import
``torch.nn`` directly (see ``backends/__init__.py``'s enforcement that
framework code only ever imports ``edge0.backends.{core,nn,io,quant}``).
"""

from __future__ import annotations

import torch
import torch.nn as _tnn
import torch.nn.functional as F

Module = _tnn.Module
Linear = _tnn.Linear


class RMSNorm(_tnn.RMSNorm):
    """``mlx.nn.RMSNorm(dims, eps)`` signature on top of ``torch.nn.RMSNorm``.

    Subclassed rather than wrapped so the parameter keeps the name
    ``weight``, as in MLX and the checkpoints. Numerics match
    ``mx.fast.rms_norm`` (eps inside the sqrt); see
    ``tests/test_cuda_backend.py``.
    """

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__(dims, eps=eps)


def silu(x):
    return F.silu(x)


def gelu(x):
    return F.gelu(x)
