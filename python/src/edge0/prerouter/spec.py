"""Prerouter abstraction: the trained cross-token routing predictor.

A prerouter head is OWNED by layer N: it runs on layer N's MoE-input
hidden state (post-attention norm output) at token t and predicts layer
N+1's routing.  Layer N+1 consumes the prediction from layer N's head
at token t-1 (cross-token double shift: prev-layer + prev-token), so the
staged expert set for a decode step IS the prerouter's routing — zero
drops by construction.

Feature vector per head: ``concat[hidden, this-token top-k one-hot,
prev-token top-k one-hot]``; head = ``fc1 -> exact(erf) gelu -> fc2 +
linear_init`` on the same features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from edge0.moe.spec import RouterKind


@dataclass(frozen=True)
class PrerouterSpec:
    """Configuration of one model's prerouter.

    Attributes:
        kind: routing math family the prerouter head's logits are turned
            into expert selections with.  ``SOFTMAX_TOPK`` (qwen:
            precise softmax -> top-k -> renormalize) or
            ``SIGMOID_GROUP`` (ling: sigmoid + group-limited top-k, the
            model's own ``_select_from_logits``).
        start_layer: first CONSUMING layer; the first owner is
            ``start_layer - 1`` and the last owner ``n_layers - 2``.
        hidden: prerouter hidden width (512 for both shipped models).
        dtype: head compute dtype — "fp16" (default, training export
            precision) or "fp32" (debug only).
        feature_topk: which one-hot feeds the "this-token top-k" feature.
            "executed": the top-k the block actually routed (qwen
            trained semantics — executed == prerouter-selected at decode);
            "teacher": the original gate's top-k, recomputed per layer
            (training semantics).
        owners: explicit owner list override (defaults to
            ``range(start_layer - 1, n_layers - 1)``).
        weights_file: safetensors path with per-owner head weights
            (keys ``layers.<N>.fc1.weight`` etc.); may be provided later
            via ``install_prerouter``.
        patch_call: for models whose MoE block is NOT prerouter-aware
            (qwen3_next), install a class-level ``__call__`` patch so
            decode routes through ``pred_inds``.  Models with baked hooks
            (bailing_hybrid) set this False.
    """

    kind: RouterKind = RouterKind.SOFTMAX_TOPK
    start_layer: int = 7
    hidden: int = 512
    dtype: str = "fp16"
    feature_topk: str = "executed"  # "executed" | "teacher"
    owners: tuple[int, ...] | None = None
    weights_file: str = ""
    patch_call: bool = False

    def owner_layers(self, n_layers: int) -> list[int]:
        if self.owners is not None:
            return list(self.owners)
        return list(range(self.start_layer - 1, n_layers - 1))
