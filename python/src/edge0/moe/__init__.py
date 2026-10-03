"""MoE specs and routing math (shared by resident and streaming paths)."""

from edge0.moe.routing import group_select_from_logits, select_from_logits
from edge0.moe.spec import (
    MoESpec,
    QuantSpec,
    RouterKind,
    WeightLayout,
)

__all__ = [
    "MoESpec",
    "QuantSpec",
    "RouterKind",
    "WeightLayout",
    "select_from_logits",
    "group_select_from_logits",
]
