"""CUDA backend: namespace assembly (torch-backed reference implementation)."""

from __future__ import annotations

from edge0.backends.cuda import core, io, nn, quant  # noqa: F401


class BackendImpl:
    """Handle to the active backend implementation."""

    name = "cuda"
    description = "PyTorch / CUDA (reference implementation)"

    @property
    def version(self) -> str:
        import torch
        return torch.__version__
