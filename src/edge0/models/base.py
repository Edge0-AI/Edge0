"""Model adapters: per-model configs, profiles and build entry points.

The registry contract (``edge0.registry``) is: an adapter module exposes
``Config`` (a ``ModelConfig`` subclass with ``from_pretrained``),
``build_model`` and ``build_engine``.  ``Config`` carries everything the
engines need — MoE spec, streaming options, prerouter spec, LoRA, sampling
defaults, server port and the acceptance profile (tokens/s + peak-active
memory targets measured on the production machine).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from edge0.config import GenerationConfig
from edge0.moe.spec import MoESpec
from edge0.prerouter.spec import PrerouterSpec
from edge0.streaming.options import LayerOptions

# Repository-root artifacts (gitignored; produced once by
# scripts/convert_adapters_legacy.py from the training npz exports, then
# the source npz are discarded).
ARTIFACTS_DIR = Path(__file__).resolve().parents[3] / "artifacts"


def artifact(name: str, model_dir: str | None = None) -> str:
    """Path of one adapter file: the model's own directory first
    (v7-deploy layout — model and its adapters side by side in ONE
    dir), falling back to ``artifacts/`` (legacy conversion cache)."""
    if model_dir:
        cand = Path(model_dir) / name
        if cand.is_file():
            return str(cand)
    return str(ARTIFACTS_DIR / name)


@dataclass
class ModelConfig:
    """Everything one edge0 model tier needs to run.

    Subclasses fix the family defaults (specs, presets, generation, port,
    acceptance targets); ``from_pretrained`` merges user overrides.
    """

    name: str
    model_dir: str
    moe_spec: MoESpec
    options: LayerOptions
    prerouter: PrerouterSpec | None = None
    prerouter_top_k: int = 0          # 0 -> options.top_k
    lora: str = ""                    # safetensors path; "" disables
    lora_r: int = 16
    lora_alpha: float = 32.0
    gen: GenerationConfig = field(default_factory=GenerationConfig)
    prefill_chunk: int = 2048
    hot_window: int = 4
    intra_staging: bool = False
    prefetch_history: bool = True
    port: int = 8000
    # acceptance profile (measured on the production Mac)
    target_tok_s: float = 0.0
    peak_active_mem_mb: float = 0.0

    @classmethod
    def from_pretrained(cls, model_dir: str | None = None, **overrides):
        """Build the family config, resolving artifact paths and applying
        per-field overrides (any public attribute may be overridden)."""
        if model_dir is None:
            model_dir = overrides.pop("model_dir", None) or ""
        base = cls._defaults(model_dir)
        for key, value in overrides.items():
            if not hasattr(base, key):
                raise TypeError(
                    f"unknown {cls.__name__} override {key!r} "
                    f"(known: {sorted(f.name for f in fields(base))})")
            base = replace(base, **{key: value})
        return base

    @classmethod
    def _defaults(cls, model_dir: str) -> "ModelConfig":
        raise NotImplementedError


def resolve_prerouter(pspec: PrerouterSpec, weights_file: str) -> PrerouterSpec:
    """Fill the prerouter weights path (specs are frozen; only used when
    the caller didn't already set one)."""
    if pspec.weights_file:
        return pspec
    return replace(pspec, weights_file=weights_file)


def resolve_lora(lora: str, model_dir: str, r: int, alpha: float):
    """Sanitize the LoRA override: a bare model name resolves to the
    artifact of the same tier, ``model_dir`` means "keep the training
    weights in place", empty string disables."""
    if lora == "model_dir":
        return (os.path.join(model_dir, "lora.safetensors")
                if os.path.isfile(os.path.join(model_dir, "lora.safetensors"))
                else "")
    return lora
