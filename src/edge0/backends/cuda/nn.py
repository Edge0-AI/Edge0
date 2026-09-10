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


class RMSNorm(_tnn.Module):
    """Matches ``mlx.nn.RMSNorm(dims, eps)`` call signature; wraps
    ``torch.nn.RMSNorm`` (normalized_shape=dims) rather than
    hand-rolling the reduction, since torch's built-in already matches
    the standard eps-inside-sqrt convention the vendored models assume
    (worth a numerical spot-check against ``mx.fast.rms_norm`` before
    trusting this for anything beyond shape/wiring tests -- see the
    mapping doc, gated-delta callers pass a raw eps positionally that
    MLX's ``mx.fast.rms_norm`` treats identically, but this has not
    been cross-checked digit-for-digit).
    """

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.norm = _tnn.RMSNorm(dims, eps=eps)

    def forward(self, x):
        return self.norm(x)


def silu(x):
    return F.silu(x)


def gelu(x):
    return F.gelu(x)
