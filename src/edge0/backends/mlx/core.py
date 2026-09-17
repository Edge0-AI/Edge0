"""MLX backend: array ops namespace.

Everything from ``mlx.core`` (same objects, so ``core.array``,
``core.compile`` ... are the real MLX ones), plus the few contract names
that MLX only offers as array methods or indexing syntax. Framework code
calls these instead of the method forms so it also runs on backends whose
arrays lack them (torch has no ``.astype`` or ``.at[]``, and its ``.size``
is a method).
"""

from __future__ import annotations

from mlx.core import *  # noqa: F401,F403


def astype(x, dtype):
    return x.astype(dtype)


def size(x) -> int:
    """Number of elements (``x.size`` in MLX)."""
    return x.size


def index_add(a, indices, values):
    """``a.at[indices].add(values)``: out-of-place, duplicates accumulate."""
    return a.at[indices].add(values)
