"""Attention taxonomy.

edge0 does not re-implement attention kernels: the vendored base model
implementations (``edge0.backends.mlx._impl``) carry the kernels (GQA /
GatedDeltaNet for edge0-35b, MLA / DeltaNet for edge0-10b).  This module
describes attention so the framework can introspect a model — cache
sizing, layer roles, documentation, and future kernels — without knowing
the concrete implementation.

A new attention kind = a new ``AttentionKind`` member + an
``AttentionSpec`` description + a kernel implementation (in a vendored
base or a backend ``_impl`` module).  See docs/attention.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AttentionKind(Enum):
    """Attention / sequence-mixing families supported by edge0 models."""

    GQA = "gqa"
    """Grouped-query attention with softmax and KV cache."""

    MLA = "mla"
    """Multi-head latent attention (compressed KV)."""

    DELTANET = "deltanet"
    """DeltaNet linear attention (gated delta rule, no KV cache)."""

    GATED_DELTANET = "gated_deltanet"
    """GatedDeltaNet (gated delta rule with decay / linear attention)."""

    DENSE_MLP = "dense_mlp"
    """No sequence mixing (mlp-only layer)."""


@dataclass(frozen=True)
class AttentionSpec:
    """Description of one layer's sequence-mixing block.

    Attributes:
        kind: attention family.
        layer_indices: 0-based layer indices covered by this spec.
        num_heads / num_kv_heads / head_dim: head geometry
            (GQA/MLA only; None otherwise).
        cache: whether the block maintains a KV-style cache
            (False for linear-attention layers).
        notes: free-form (e.g. hybrid schedules).
    """

    kind: AttentionKind
    layer_indices: tuple[int, ...]
    num_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    cache: bool = True
    notes: str = ""

    def __repr__(self) -> str:  # compact one-liner for logs/docs
        return (f"AttentionSpec({self.kind.value}, layers="
                f"{self.layer_indices[0]}..{self.layer_indices[-1]}, "
                f"cache={self.cache})")


def summarize(specs: list[AttentionSpec]) -> str:
    """Human-readable one-line summary, e.g. for the CLI banner."""
    return ", ".join(
        f"{s.kind.value}x{len(s.layer_indices)}" for s in specs)
