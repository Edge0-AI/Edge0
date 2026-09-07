"""Prerouter head math — faithful port of the qwen trained-head
implementation (the legacy ``v7_trained`` file) with identifiers
renamed.

Semantics (cross-token):
  * features = concat[hidden, this-token executed top-k one-hot,
    previous-token executed top-k one-hot];
  * head = fc1(features) -> exact(erf) gelu -> fc2 + linear_init on the
    SAME concat features;
  * fp16 weights by default (the training export precision, and the
    precision ling v7 production ran the heads at); fp32 for debugging.
"""

from __future__ import annotations

import math

from edge0.backends import core
from edge0.backends import nn

_EYE_CACHE: dict[int, core.array] = {}


def _eye(num_experts: int) -> core.array:
    e = _EYE_CACHE.get(num_experts)
    if e is None:
        e = core.eye(num_experts)
        _EYE_CACHE[num_experts] = e
    return e


def topk_onehot(inds: core.array, num_experts: int) -> core.array:
    """inds [B,T,k] -> one-hot [B,T,num_experts]."""
    idm = _eye(num_experts)
    oh = core.take(idm, inds, axis=0)  # [B,T,k,E]
    return oh.sum(axis=-2)


def gelu_erf(x: core.array) -> core.array:
    """Exact GELU (erf) matching torch ``F.gelu`` default."""
    return 0.5 * x * (1.0 + core.erf(x / math.sqrt(2.0)))


def select_from_logits(logits: core.array, top_k: int):
    """Mirror the qwen HF router math: precise softmax -> top-k ->
    renormalize.  Returns (inds [..., k], scores [..., k])."""
    gates = core.softmax(logits, axis=-1, precise=True)
    inds = core.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
    scores = core.take_along_axis(gates, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    return inds, scores


class PrerouterHead(nn.Module):
    """Trained prerouter head (fp16 by default; fp32 via ``dtype``)."""

    def __init__(self, hidden: int, num_experts: int, prerouter_hidden: int,
                 dtype=core.float16):
        super().__init__()
        self._dtype = dtype
        f_in = hidden + 2 * num_experts
        self.fc1 = nn.Linear(f_in, prerouter_hidden, bias=False)
        self.fc2 = nn.Linear(prerouter_hidden, num_experts, bias=False)
        self.linear_init = nn.Linear(f_in, num_experts, bias=False)

    def __call__(self, h: core.array, executed_oh: core.array,
                 prev_oh: core.array) -> core.array:
        feats = core.concatenate([h, executed_oh, prev_oh], axis=-1)
        if feats.dtype != self._dtype:
            feats = feats.astype(self._dtype)
        # linear_init consumes the full concat features, same as training.
        return self.linear_init(feats) + self.fc2(
            gelu_erf(self.fc1(feats)))
