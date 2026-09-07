"""MLX backend — the reference backend implementation.

Everything that touches ``mlx`` lives under this package.  Nothing else in
the framework may import mlx directly.
"""

from edge0.backends.mlx.backend import BackendImpl, core, io, nn, quant

__all__ = ["BackendImpl", "core", "io", "nn", "quant"]
