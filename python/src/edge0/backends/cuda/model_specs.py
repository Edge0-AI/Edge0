"""MoESpec instances for the CUDA backend's two shipped tiers.

Values are the SAME family facts already verified and shipped in
``models/edge0_35b/__init__.py`` / ``models/edge0_8b/__init__.py``
(num_experts, top_k, router kind, quantization, ...) -- not
re-derived, since those are already checked against the real
checkpoints. What differs here is ``block_path``: the MLX specs' path
navigates MLX's OWN post-sanitize module tree (``language_model.model.
layers.{layer}.mlp``, from ``_impl/qwen3_5_moe.py`` wrapping
``TextModel`` under ``self.language_model``); ``transformers.
Qwen3_5MoeForCausalLM`` has no such wrapper, so its real attribute path
is one level shallower. ``key_template`` (which resolves SAFETENSORS
KEYS, not Python attributes) keeps the ``language_model.`` prefix,
because that prefix IS present in the real on-disk checkpoint --
confirmed via the published ``model.safetensors.index.json``, not
assumed. edge0-8b needs no such split: its on-disk keys and
``transformers.BailingMoeV3ForCausalLM``'s attribute path already
agree (confirmed the same way), so its spec is byte-for-byte what
``models/edge0_8b/__init__.py`` already uses.
"""

from __future__ import annotations

from edge0.moe.spec import MoESpec, QuantSpec, RouterKind, WeightLayout

QWEN35_MOE_SPEC = MoESpec(
    num_experts=256, top_k=4, intermediate_size=512,
    router=RouterKind.SOFTMAX_TOPK, norm_topk_prob=True,
    shared_experts=1,
    quant=QuantSpec(bits=4, group_size=64, mode="affine"),
    layout=WeightLayout.SEPARATE,
    key_template="language_model.model.layers.{layer}.mlp.switch_mlp",
    block_path="model.layers.{layer}.mlp",
    layer_path="model.layers.{layer}",
)

BAILING_V3_MOE_SPEC = MoESpec(
    num_experts=128, top_k=8, intermediate_size=512,
    router=RouterKind.SIGMOID_GROUP, norm_topk_prob=True,
    routed_scaling=2.5, n_group=8, topk_group=4,
    shared_experts=1,
    quant=QuantSpec(bits=4, group_size=64, mode="affine"),
    layout=WeightLayout.SEPARATE,
    key_template="model.layers.{layer}.mlp.experts",
    block_path="model.layers.{layer}.mlp",
    layer_path="model.layers.{layer}",
)
