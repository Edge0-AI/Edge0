"""edge0: streaming MoE inference for Apple Silicon.

SSD expert offload + parallel LoRA + prerouter routing prediction, with a
transformers-style AutoModel registry and a backend-isolated design (all
MLX code lives under ``edge0.backends.mlx``; a CUDA backend can be added
as a sibling package implementing the same interface).

Public API::

    from edge0 import AutoConfig, AutoModel, AutoEngine

    cfg  = AutoConfig.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, name="edge0-35b", ...)
    engine = AutoEngine.from_pretrained(model_dir, ...)
"""

from edge0 import registry  # noqa: F401  (populates the model registry)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AutoConfig",
    "AutoModel",
    "AutoEngine",
]


def __getattr__(name: str):
    # Deferred imports so `import edge0` stays light and backend selection
    # (EDGE0_BACKEND) happens on first use.
    if name in ("AutoConfig", "AutoModel", "AutoEngine"):
        from edge0 import registry as _reg
        return getattr(_reg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
