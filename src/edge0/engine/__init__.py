"""edge0 engines: streaming prefill/decode lifecycles per model family.

``edge0.engine.base.Edge0Engine`` owns the shared prefill / step /
generate loops; ``edge0.engine.qwen.Qwen35Engine`` and
``edge0.engine.ling.Ling10BEngine`` implement the family hooks
(streaming install, prerouter wiring, step-boundary staging).
"""

from edge0.engine.base import Edge0Engine
from edge0.engine.hooks import (
    make_history_prefetch,
    make_intra_after_layer,
    make_prefill_before_layer,
)
from edge0.engine.ling import Ling10BEngine
from edge0.engine.qwen import Qwen35Engine

__all__ = [
    "Edge0Engine",
    "Qwen35Engine",
    "Ling10BEngine",
    "make_history_prefetch",
    "make_intra_after_layer",
    "make_prefill_before_layer",
]
