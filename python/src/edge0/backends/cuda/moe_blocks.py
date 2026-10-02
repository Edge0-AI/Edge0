"""CUDA backend: adapters between StreamingSwitchGLU and transformers' MoE
blocks.

The vendored MLX models call ``switch_mlp(x, inds)`` and weight the
per-expert outputs themselves -- the streaming twin's native interface.
transformers' blocks call their experts differently, so the twin is
wrapped before ``install_streaming_experts`` swaps it in (its ``wrap=``).
"""

from __future__ import annotations

import torch


class TransformersExpertsAdapter(torch.nn.Module):
    """``experts(hidden [T, H], top_k_index [T, K], top_k_weights [T, K])
    -> [T, H]``, as ``Qwen3_5MoeSparseMoeBlock`` calls it, on top of a
    twin that returns the raw per-expert outputs ``[T, K, H]``.

    Not usable for blocks that index or iterate ``self.experts`` as a
    ``ModuleList`` (the edge0-8b checkpoint's own ``modeling_bailing_moe_v3.py``
    does): those need their forward replaced, not their experts.
    """

    def __init__(self, twin):
        super().__init__()
        object.__setattr__(self, "twin", twin)  # not an nn.Module

    def forward(self, hidden_states, top_k_index, top_k_weights):
        y = self.twin(hidden_states, top_k_index)                 # [T, K, H]
        return (y * top_k_weights[..., None].to(y.dtype)).sum(dim=-2)
