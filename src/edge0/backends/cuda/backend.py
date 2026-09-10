"""CUDA backend: namespace assembly (torch-backed reference implementation).

See ``edge0/backends/cuda/quant.py`` for why this is Plan B, not the
first thing to try -- test ``EDGE0_BACKEND=mlx`` with
``pip install mlx[cuda12]`` on real NVIDIA hardware first.
"""

from __future__ import annotations

from edge0.backends.cuda import core, io, nn, quant  # noqa: F401


class BackendImpl:
    """Handle to the active backend implementation."""

    name = "cuda"
    description = "PyTorch / CUDA (reference implementation, unverified " \
                   "quant numerics -- see backends/cuda/quant.py)"

    @property
    def version(self) -> str:
        import torch
        return torch.__version__
