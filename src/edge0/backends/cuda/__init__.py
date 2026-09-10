"""CUDA backend (torch reference implementation).

Everything that touches ``torch`` for array/nn/quant ops lives under
this package, mirroring ``edge0/backends/mlx/``'s isolation rule.
"""

from edge0.backends.cuda.backend import BackendImpl, core, io, nn, quant

__all__ = ["BackendImpl", "core", "io", "nn", "quant"]
