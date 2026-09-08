"""edge0-35b adapter: Qwen3.5-MoE (K=4 tier, 35B-class).

Family facts (production profile, verified against the round-7
checkpoint):

* 40 layers, 256 routed experts (4-bit affine, group 64), K=4,
  softmax->top-k routing with norm_topk_prob, 1 resident shared
  expert, separate gate/up/down stacked tensors under
  ``language_model.model.layers.N.mlp.switch_mlp``.
* trained prerouter: 33 heads (owners 6..38), start_layer 7,
  hidden 512, fp16, consumed directly by the patched MoE block at
  decode (patch_call=True, cross-token staged decode, K=4).
* serving: port 8085; acceptance ≈13 tok/s, peak active ≈3.3 GB.
"""

from __future__ import annotations

import sys

from edge0.config import GenerationConfig
from edge0.models.base import ModelConfig, artifact
from edge0.moe.spec import MoESpec, QuantSpec, RouterKind, WeightLayout
from edge0.prerouter.spec import PrerouterSpec
from edge0.registry import register_model
from edge0.streaming.options import LayerOptions


class Qwen35Config(ModelConfig):
    """edge0-35b family config (Qwen3.5-MoE, K=4)."""

    @classmethod
    def _defaults(cls, model_dir: str) -> "Qwen35Config":
        return cls(
            name="edge0-35b",
            model_dir=model_dir,
            moe_spec=MoESpec(
                num_experts=256, top_k=4, intermediate_size=512,
                router=RouterKind.SOFTMAX_TOPK, norm_topk_prob=True,
                shared_experts=1,
                quant=QuantSpec(bits=4, group_size=64, mode="affine"),
                layout=WeightLayout.SEPARATE,
                key_template="language_model.model.layers.{layer}."
                             "mlp.switch_mlp",
                block_path="language_model.model.layers.{layer}.mlp",
                layer_path="language_model.model.layers.{layer}",
            ),
            options=LayerOptions.staged_k4(),
            prerouter=PrerouterSpec(
                kind=RouterKind.SOFTMAX_TOPK,
                start_layer=7, hidden=512, dtype="fp16",
                feature_topk="executed",
                weights_file=artifact(
                    "prerouter_edge0_35b.safetensors", model_dir),
                patch_call=True,
            ),
            prerouter_top_k=4,
            lora=artifact("lora_edge0_35b.safetensors", model_dir),
            lora_r=16, lora_alpha=32.0,
            gen=GenerationConfig(
                temperature=0.6, top_p=0.95, top_k=64,
                repetition_penalty=1.0, max_new_tokens=2048,
                eos_ids=(248046, 248044)),
            prefill_chunk=2048,
            hot_window=4,
            intra_staging=False,
            prefetch_history=True,
            port=8085,
            target_tok_s=13.0,
            peak_active_mem_mb=3400.0,
        )


def build_model(model_dir: str | None = None, **overrides):
    """Load the qwen skeleton with streaming twins / LoRA / prerouter
    installed (the engine's own build path, DRY-shared)."""
    from edge0.engine.qwen import load_installed

    cfg = Qwen35Config.from_pretrained(model_dir, **overrides)
    model, _mcfg, _shards, _installs = load_installed(cfg.model_dir, cfg)
    return model


def build_engine(model_dir: str | None = None, **overrides):
    """Build a ready-to-generate Qwen35Engine."""
    from edge0.engine.qwen import Qwen35Engine

    cfg = Qwen35Config.from_pretrained(model_dir, **overrides)
    return Qwen35Engine(cfg.model_dir, cfg)


register_model("edge0-35b", sys.modules[__name__])


Config = Qwen35Config  # registry contract: adapter.Config
