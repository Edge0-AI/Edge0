"""MoE abstraction: routing kinds, quantized weight layouts, and the spec
that drives both the resident math and the streaming layer.

Every MoE block in a supported model is described by one ``MoESpec``; the
streaming subsystem consumes the spec (layout + key template) so one
generic ``StreamingSwitchGLU`` serves all models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class RouterKind(Enum):
    """Router math families supported by edge0."""

    SOFTMAX_TOPK = "softmax_topk"
    """precise softmax -> top-k -> renormalize (Qwen3.5-MoE style,
    norm_topk_prob=True)."""

    SIGMOID_GROUP = "sigmoid_group"
    """sigmoid scores + group-limited top-k + routed scaling
    (DeepSeek-V3 / Bailing style)."""


class WeightLayout(Enum):
    """How expert projections are stored in the checkpoint."""

    SEPARATE = "separate"
    """gate_proj / up_proj / down_proj as three stacked tensors."""

    FUSED_GATE_UP = "fused_gate_up"
    """gate_proj+up_proj fused into one stacked tensor (rows doubled
    along the out axis); down_proj separate."""


@dataclass(frozen=True)
class QuantSpec:
    """Weight quantization description (per-tensor defaults)."""

    bits: int = 4
    group_size: int = 64
    mode: str = "affine"


@dataclass(frozen=True)
class MoESpec:
    """Everything the framework needs to know about a model's MoE blocks.

    Attributes:
        num_experts: routed experts per layer.
        top_k: experts selected per token (routing width).
        intermediate_size: expert FFN hidden size (documentation only;
            the math derives sizes from the weight tensors).
        router: routing math family.
        norm_topk_prob: renormalize selection weights.
        routed_scaling: multiplier applied to selection weights
            (sigmoid-group routers); None for softmax-topk.
        n_group / topk_group: group-limited routing parameters
            (sigmoid-group routers only).
        shared_experts: number of resident shared experts (never streamed).
        quant: weight quantization.
        layout: weight layout.
        key_template: safetensors key prefix for layer ``{layer}``, with
            ``{proj}`` / ``{part}`` placeholders appended by the consumer,
            e.g. ``"language_model.model.layers.{layer}.mlp.switch_mlp"``.
        block_path: dotted attribute path from the loaded model object to
            the MoE block, with ``{layer}``, e.g.
            ``"language_model.model.layers.{layer}.mlp.switch_mlp"``.
        layer_path: dotted attribute path to the DECODER LAYER object
            (the block's owner), with ``{layer}``, e.g.
            ``"language_model.model.layers.{layer}"``.  Used by prerouter
            stagers to read per-layer caches; default derived from
            ``block_path`` by stripping the trailing attribute parts.
        expert_row_axis: axis of the stacked [num_experts, ...] tensors
            along which one expert is a contiguous slice (always 0 for the
            supported layouts; kept for documentation).
    """

    num_experts: int
    top_k: int
    intermediate_size: int
    router: RouterKind = RouterKind.SOFTMAX_TOPK
    norm_topk_prob: bool = True
    routed_scaling: float | None = None
    n_group: int | None = None
    topk_group: int | None = None
    shared_experts: int = 0
    quant: QuantSpec = field(default_factory=QuantSpec)
    layout: WeightLayout = WeightLayout.SEPARATE
    key_template: str = ""
    block_path: str = ""
    layer_path: str = ""
    expert_row_axis: int = 0

    def keys(self, layer: int, proj: str, part: str) -> str:
        """Resolve one safetensors key for a layer's projection part."""
        prefix = self.key_template.format(layer=layer)
        return f"{prefix}.{proj}.{part}"

    def block_of(self, model, layer: int):
        """Resolve the MoE block object inside a loaded model."""
        obj = model
        for part in self.block_path.format(layer=layer).split("."):
            if part.isdigit():
                obj = obj[int(part)]  # layers are plain lists in most families
            else:
                obj = getattr(obj, part)
        return obj

    def layer_of(self, model, layer: int):
        """Resolve the decoder layer object (the block's owner).

        Falls back to ``block_path`` minus the trailing attribute
        segments (the convention: a block lives at
        ``<layer>.<mlp>.<block>``, so the layer is two segments up).
        """
        if self.layer_path:
            path = self.layer_path.format(layer=layer)
        else:
            path = ".".join(
                self.block_path.format(layer=layer).split(".")[:-2])
        obj = model
        for part in path.split("."):
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj

    @property
    def bundle_projs(self) -> tuple[str, ...]:
        """Projections a cached expert bundle contains, in stack order.

        Order matches the math signature directly: up, gate, down
        (``_swiglu(up, gate)``); fused layout packs gate rows on top of
        up rows so ``split(x_gu, 2)`` yields ``(gate, up)``.
        """
        if self.layout is WeightLayout.FUSED_GATE_UP:
            return ("gate_up_proj", "down_proj")
        return ("up_proj", "gate_proj", "down_proj")

    @property
    def fuse_gu(self) -> bool:
        return self.layout is WeightLayout.FUSED_GATE_UP
