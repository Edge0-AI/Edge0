"""Model adapters: one package per tier, registered on import.

``edge0.models.base`` carries the shared ``ModelConfig``; the tier
packages (``edge0.models.edge0_35b``, ``edge0.models.edge0_10b``)
define their family configs and the ``build_model`` / ``build_engine``
entry points, then call ``register_model`` so ``AutoConfig`` /
``AutoModel`` / ``AutoEngine`` can resolve them by name.
"""

from __future__ import annotations

from edge0.models.base import ModelConfig

# Importing the tier packages populates edge0.registry.MODEL_REGISTRY.
from edge0.models import edge0_10b, edge0_35b  # noqa: F401

__all__ = ["ModelConfig", "edge0_10b", "edge0_35b"]
