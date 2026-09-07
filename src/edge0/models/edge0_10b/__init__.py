"""edge0-10b adapter: Ling 3.0 hybrid (MLA + MoE, 10B tier).

Family facts (production profile, verified against the v7 checkpoint):

* 24 layers (layer 0 dense), 128 routed experts (4-bit affine,
  group 64), K=8 native, sigmoid + group-limited routing
  (n_group 8, topk_group 4, routed scaling 2.5, norm_topk_prob),
  1 resident shared expert, separate gate/up/down stacked tensors
  under ``model.layers.N.mlp.experts``.
* hybrid prerouter baked into the vendored model: 22 heads
  (explicit owners 1..22), start_layer 1, hidden 512, fp16,
  feature_topk "executed", consumed inside ``BailingSparseMoE``
  from ``prerouter_cache`` logits (patch_call=False).
* serving: port 8083; acceptance ≈33 tok/s, peak active ≈1.4 GB.
"""

from __future__ import annotations

import sys

from edge0.config import GenerationConfig
from edge0.models.base import ModelConfig, artifact
from edge0.moe.spec import MoESpec, QuantSpec, RouterKind, WeightLayout
from edge0.prerouter.spec import PrerouterSpec
from edge0.registry import register_model
from edge0.streaming.options import LayerOptions


class Ling10BConfig(ModelConfig):
    """edge0-10b family config (Ling 3.0 hybrid, K=8)."""

    @classmethod
    def _defaults(cls, model_dir: str) -> "Ling10BConfig":
        return cls(
            name="edge0-10b",
            model_dir=model_dir,
            moe_spec=MoESpec(
                num_experts=128, top_k=8, intermediate_size=512,
                router=RouterKind.SIGMOID_GROUP, norm_topk_prob=True,
                routed_scaling=2.5, n_group=8, topk_group=4,
                shared_experts=1,
                quant=QuantSpec(bits=4, group_size=64, mode="affine"),
                layout=WeightLayout.SEPARATE,
                key_template="model.layers.{layer}.mlp.experts",
                block_path="model.layers.{layer}.mlp",
                layer_path="model.layers.{layer}",
            ),
            options=LayerOptions.prod_k8(),
            prerouter=PrerouterSpec(
                kind=RouterKind.SIGMOID_GROUP,
                start_layer=7, hidden=512, dtype="fp16",
                feature_topk="executed",
                owners=tuple(range(6, 23)),
                weights_file=artifact("prerouter_edge0_10b.safetensors"),
                patch_call=False,
            ),
            prerouter_top_k=8,
            lora=artifact("lora_edge0_10b.safetensors"),
            lora_r=16, lora_alpha=32.0,
            gen=GenerationConfig(
                temperature=0.7, top_p=0.95, top_k=64,
                repetition_penalty=1.1, max_new_tokens=2048,
                eos_ids=(156895,)),
            prefill_chunk=2048,
            hot_window=1,
            intra_staging=False,
            prefetch_history=True,
            port=8083,
            target_tok_s=33.0,
            peak_active_mem_mb=1400.0,
        )


def build_model(model_dir: str | None = None, **overrides):
    """Load the ling skeleton with streaming twins / LoRA / prerouter
    installed (the engine's own build path, DRY-shared)."""
    from edge0.engine.ling import load_installed

    cfg = Ling10BConfig.from_pretrained(model_dir, **overrides)
    model, _mcfg, _shards, _installs = load_installed(cfg.model_dir, cfg)
    return model


def build_engine(model_dir: str | None = None, **overrides):
    """Build a ready-to-generate Ling10BEngine.

    ``think`` (THINK_MODE parity, default False) selects the
    chat-template thinking mode for ``encode_chat``.
    """
    from edge0.engine.ling import Ling10BEngine

    think = overrides.pop("think", False)
    cfg = Ling10BConfig.from_pretrained(model_dir, **overrides)
    return Ling10BEngine(cfg.model_dir, cfg, think=think)


register_model("edge0-10b", sys.modules[__name__])


Config = Ling10BConfig  # registry contract: adapter.Config
