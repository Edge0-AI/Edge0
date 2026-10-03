"""MLX backend: namespace assembly.

``core`` is mlx.core (arrays, ops, eval/compile, random), ``nn`` is
mlx.nn (module factory), ``io`` and ``quant`` are edge0 wrappers around
mlx-lm and mlx's quantized gather kernels.
"""

from __future__ import annotations

import mlx.core as core  # noqa: F401
import mlx.nn as nn  # noqa: F401

from edge0.backends.mlx import io, quant  # noqa: F401


class BackendImpl:
    """Handle to the active backend implementation."""

    name = "mlx"
    description = "MLX / Metal on Apple Silicon"

    @property
    def version(self) -> str:
        import mlx
        return mlx.__version__
