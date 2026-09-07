"""Tensor-store protocol: where adapter weights (LoRA, prerouter heads)
are loaded from.

The canonical format is safetensors (mmap-backed, lazy, with provenance
metadata).  No other format is supported.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TensorStore(ABC):
    """Read-only view over a weights file (adapter artifacts).

    Implementations must be lazily opened and must not materialize the
    whole file into memory.
    """

    @abstractmethod
    def keys(self) -> list[str]:
        """Sorted tensor names in the store."""

    @abstractmethod
    def get(self, name: str):
        """Return one tensor as an array (backend-native).

        Raises KeyError for unknown names.
        """

    @abstractmethod
    def metadata(self) -> dict:
        """File-level provenance metadata (safetensors __metadata__)."""

    @property
    @abstractmethod
    def path(self) -> str:
        """Source file path (for error messages / logging)."""


def open_tensor_store(path: str) -> TensorStore:
    """Open a tensor store by extension (safetensors)."""
    if path.endswith(".safetensors"):
        from edge0.backends import io  # deferred: backend selection
        return io.SafeTensorsStore(path)
    raise ValueError(
        f"unsupported adapter format: {path!r} (only .safetensors is "
        f"supported)")
