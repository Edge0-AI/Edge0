"""Routing math, shared by resident and streaming MoE paths.

These functions must stay bit-identical to the vendored base
implementations (they are extracted from them verbatim); parity tests pin
this.
"""

from __future__ import annotations

from edge0.backends import core


def select_from_logits(logits, top_k: int, norm: bool = True):
    """Softmax-topk router: precise softmax -> top-k -> renormalize.

    Mirrors the Qwen3.5-MoE HF router semantics (norm_topk_prob=True).
    Returns ``(inds [..., k], scores [..., k])``.
    """
    gates = core.softmax(logits, axis=-1, precise=True)
    inds = core.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
    scores = core.take_along_axis(gates, inds, axis=-1)
    if norm:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    return inds, scores


def group_select_from_logits(logits, top_k: int, n_group: int,
                             topk_group: int, routed_scaling: float,
                             norm: bool = True, expert_bias=None):
    """Sigmoid + group-limited top-k router (DeepSeek-V3 / Bailing law).

    Selection scores are ``sigmoid(logits) + expert_bias``; the
    ``topk_group`` best groups (by sum of each group's top-2 selection
    scores) survive; the top-k experts within surviving groups are chosen
    by selection score; the WEIGHTS are the raw sigmoid scores of the
    chosen experts (bias excluded), normalized if ``norm`` and scaled by
    ``routed_scaling``.

    Returns ``(inds [..., k], scores [..., k])``.
    """
    scores = core.sigmoid(logits.astype(core.float32))
    select = scores
    if expert_bias is not None:
        select = scores + expert_bias

    b_shape = select.shape[:-1]
    k_drop = n_group - topk_group
    if k_drop > 0:
        grouped = select.reshape(
            *b_shape, n_group, select.shape[-1] // n_group)
        # Group score = sum of the top-2 selection scores in the group.
        top2 = core.topk(grouped, 2, axis=-1)
        group_scores = top2.sum(axis=-1)
        drop = core.argpartition(group_scores, kth=k_drop - 1,
                                 axis=-1)[..., :k_drop]
        masked = core.put_along_axis(
            grouped,
            core.expand_dims(drop, -1),
            core.array(-float("inf"), grouped.dtype),
            axis=-2,
        )
        select = masked.reshape(*b_shape, select.shape[-1])
    # topk_group == n_group keeps every group — nothing to drop.

    idx = core.argpartition(-select, kth=top_k - 1, axis=-1)[..., :top_k]
    w = core.take_along_axis(scores, idx, axis=-1)
    if norm:
        w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
    w = w * routed_scaling
    return idx, w
