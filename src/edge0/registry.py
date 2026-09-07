"""Transformers-style model registry: name -> adapter class.

Adapters live in ``edge0.models`` and register themselves on import;
``AutoConfig`` / ``AutoModel`` / ``AutoEngine`` resolve a model by
explicit name, by the checkpoint's ``config.json`` ``model_type`` (via
``TYPE_ALIASES``), or by the checkpoint directory basename.  Unknown
names raise ``KeyError`` listing the registered models.
"""

from __future__ import annotations

from dataclasses import dataclass

MODEL_REGISTRY: dict[str, type] = {}
TYPE_ALIASES: dict[str, str] = {
    # qwen3_5_moe family
    "qwen3_5_moe_text": "edge0-35b",
    "qwen3_5_moe": "edge0-35b",
    # ling family (config.json carries architectures, not model_type)
    "bailing_hybrid": "edge0-10b",
    "bailing_moe_linear": "edge0-10b",
    "BailingMoeV3ForCausalLM": "edge0-10b",
}


def register_model(name: str, adapter: type) -> None:
    if name in MODEL_REGISTRY:
        raise ValueError(f"model {name!r} already registered")
    MODEL_REGISTRY[name] = adapter


@dataclass(frozen=True)
class _RegisteredModel:
    """One entry of the registry, for the KeyError listing."""

    name: str
    adapter: type

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name} ({self.adapter.__module__}.{self.adapter.__name__})"


def _resolve_name(model_dir: str | None, name: str | None) -> str:
    if name:
        key = name
    elif model_dir:
        key = _model_type_from_dir(model_dir)
    else:
        raise ValueError("AutoX.from_pretrained needs model_dir or name")
    if key in MODEL_REGISTRY:
        return key
    if key in TYPE_ALIASES:
        return TYPE_ALIASES[key]
    known = ", ".join(sorted(MODEL_REGISTRY))
    raise KeyError(
        f"unknown model {key!r} (resolved from name={name!r}, "
        f"model_dir={model_dir!r}); registered: {known}")

def _model_type_from_dir(model_dir: str) -> str:
    """config.json model_type (or architectures[0] for checkpoints that
    omit model_type, e.g. the ling family), falling back to the
    directory basename."""
    import json
    import os
    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            mt = str(cfg.get("model_type", "") or "")
            if not mt:
                archs = cfg.get("architectures") or []
                if archs:
                    mt = str(archs[0])
            if mt:
                return mt
        except Exception:
            pass
    base = os.path.basename(os.path.normpath(model_dir))
    return base if base not in ("", ".") else ""


class AutoConfig:
    """Resolve a model's config class from the registry."""

    @classmethod
    def from_pretrained(cls, model_dir=None, name=None, **kwargs):
        from edge0 import models  # noqa: F401  (populates MODEL_REGISTRY)
        key = _resolve_name(model_dir, name)
        return MODEL_REGISTRY[key].Config.from_pretrained(
            model_dir, **kwargs)


class AutoModel:
    """Load a base model + install streaming experts / prerouter / LoRA."""

    @classmethod
    def from_pretrained(cls, model_dir=None, name=None, **kwargs):
        from edge0 import models  # noqa: F401
        key = _resolve_name(model_dir, name)
        return MODEL_REGISTRY[key].build_model(model_dir, **kwargs)


class AutoEngine:
    """Build a ready-to-generate engine for a model tier."""

    @classmethod
    def from_pretrained(cls, model_dir=None, name=None, **kwargs):
        from edge0 import models  # noqa: F401
        key = _resolve_name(model_dir, name)
        return MODEL_REGISTRY[key].build_engine(model_dir, **kwargs)
