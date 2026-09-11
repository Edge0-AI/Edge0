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


def _quant_params(weight, scales, in_features):
    """(bits, group_size) of an MLX affine-quantized tensor, from shapes:
    packed width = in * bits / 32, scales width = in / group_size. Works
    for per-path overrides (the edge0-35b routers are 8-bit)."""
    return (weight.shape[-1] * 32 // in_features,
            in_features // scales.shape[-1])


class QuantizedLinear(_tnn.Module):
    """``mlx.nn.QuantizedLinear`` layout (``weight`` packed uint32,
    ``scales``, ``biases``) dequantized on the fly -- the 4-bit payload
    stays resident, not a bf16 copy."""

    def __init__(self, weight, scales, biases, in_features, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = weight.shape[0]
        self.bits, self.group_size = _quant_params(weight, scales, in_features)
        self.register_buffer("weight", weight)
        self.register_buffer("scales", scales)
        self.register_buffer("biases", biases)
        self.bias = None if bias is None else _tnn.Parameter(bias, False)

    def forward(self, x):
        from edge0.backends.cuda.quant import _dequantize
        w = _dequantize(self.weight, self.scales, self.biases,
                        self.group_size, self.bits).to(x.dtype)
        return F.linear(x, w, None if self.bias is None
                        else self.bias.to(x.dtype))


class QuantizedEmbedding(_tnn.Module):
    """``mlx.nn.QuantizedEmbedding`` layout; only the looked-up rows are
    dequantized. Output dtype: ``dtype``, else the scales' (the
    checkpoint's)."""

    def __init__(self, weight, scales, biases, embedding_dim, dtype=None):
        super().__init__()
        self.num_embeddings = weight.shape[0]
        self.embedding_dim = embedding_dim
        self.bits, self.group_size = _quant_params(weight, scales,
                                                   embedding_dim)
        self.out_dtype = dtype or scales.dtype
        self.register_buffer("weight", weight)
        self.register_buffer("scales", scales)
        self.register_buffer("biases", biases)

    def forward(self, ids):
        from edge0.backends.cuda.quant import _dequantize
        rows = _dequantize(self.weight[ids], self.scales[ids],
                           self.biases[ids], self.group_size, self.bits)
        return rows.to(self.out_dtype)


def silu(x):
    return F.silu(x)


def gelu(x):
    return F.gelu(x)
