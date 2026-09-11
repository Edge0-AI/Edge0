"""CUDA backend: array ops namespace (mirrors ``mlx.core`` by name).

The contract in ``edge0/backends/__init__.py`` documents this namespace
against MLX's naming, not PyTorch's idiomatic API -- so this module wraps
torch under MLX's names rather than exposing torch as-is.  Framework code
(``streaming/layer.py``, ``moe/routing.py``, ``sampling.py``, ...) calls
``core.take_along_axis(...)``, not ``torch.gather(...)``.

Known semantic gaps vs. MLX, flagged rather than silently papered over:

* MLX arrays are copy-on-write / functionally immutable; every op here
  that MLX documents as returning a new array (``put_along_axis``, ...)
  is implemented with torch's out-of-place variant even where an
  in-place ``_`` variant would be faster.  ``streaming/layer.py``'s
  incremental-stack path (sticky slots, see its module docstring) is
  exactly the place that later wants an in-place
  ``put_along_axis_``-style fast path once the CUDA staging design is
  set -- deliberately not pre-empted here.
* ``eval`` / ``compile`` exist because MLX is lazy-by-default; torch is
  eager, so ``eval`` is a no-op and ``compile`` wraps ``torch.compile``
  (itself opt-in-able, since it re-traces on shape changes -- exactly
  the staged/exact path shape variability in ``streaming/layer.py``).
* ``take`` / ``take_along_axis`` follow numpy/MLX gather semantics
  (index array shape composes with the source's remaining axes), not
  ``torch.take``'s flat-index semantics -- implemented via
  ``index_select`` + reshape.
"""

from __future__ import annotations

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- dtypes (contract: float16/float32/bfloat16/int32/uint32/int64) ------

float16 = torch.float16
float32 = torch.float32
bfloat16 = torch.bfloat16
int32 = torch.int32
uint32 = torch.uint32
int64 = torch.int64


def _as_tensor(x, dtype=None):
    if isinstance(x, torch.Tensor):
        t = x
    else:
        # numpy arrays (incl. non-writeable mmap views) and python lists
        t = torch.as_tensor(x)
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    if t.device != DEVICE:
        t = t.to(DEVICE, non_blocking=True)
    return t


# ---- construction ----------------------------------------------------

def array(x, dtype=None):
    """``mx.array`` equivalent: numpy/list/scalar -> device tensor.

    This is the single conversion point the streaming layer calls once
    per expert bundle (see ``streaming/layer.py::_build``); the actual
    host->device transfer for the SSD-offload path happens here.  A
    synchronous ``.to(DEVICE)`` is a correct placeholder -- the staging
    strategy (pinned double-buffer vs. GPUDirect Storage) is the part
    intentionally deferred, and replacing this function's body is the
    entire diff either route needs; nothing in streaming/layer.py calls
    torch directly.
    """
    return _as_tensor(x, dtype)


def zeros(shape, dtype=float32):
    return torch.zeros(shape, dtype=dtype, device=DEVICE)


def zeros_like(x):
    return torch.zeros_like(x)


def eye(n, dtype=float32):
    return torch.eye(n, dtype=dtype, device=DEVICE)


def arange(*args, dtype=None):
    return torch.arange(*args, dtype=dtype, device=DEVICE)


def full(shape, value, dtype=float32):
    return torch.full(shape, value, dtype=dtype, device=DEVICE)


# ---- shape ops ----------------------------------------------------------

def expand_dims(x, axes):
    if isinstance(axes, int):
        axes = (axes,)
    for ax in sorted(axes):
        x = torch.unsqueeze(x, ax)
    return x


def squeeze(x, axis=None):
    return torch.squeeze(x) if axis is None else torch.squeeze(x, axis)


def reshape(x, shape):
    return torch.reshape(x, tuple(shape))


def transpose(x, axes=None):
    return x.permute(*axes) if axes is not None else x.t()


def concatenate(arrays, axis=0):
    return torch.cat(list(arrays), dim=axis)


def stack(arrays, axis=0):
    return torch.stack(list(arrays), dim=axis)


def split(x, indices_or_sections, axis=0):
    # mx.split(x, [i, j], axis) takes SPLIT POINTS, like numpy.split --
    # torch.split takes SECTION SIZES. Convert when a list is given; an
    # int section count (used e.g. in streaming/layer.py's
    # split(x_gu, 2, axis=-1)) is already the torch convention.
    if isinstance(indices_or_sections, int):
        return list(torch.chunk(x, indices_or_sections, dim=axis))
    points = list(indices_or_sections)
    sizes = []
    prev = 0
    n = x.shape[axis]
    for p in points:
        sizes.append(p - prev)
        prev = p
    sizes.append(n - prev)
    return list(torch.split(x, sizes, dim=axis))


# ---- math / reductions ---------------------------------------------------

def matmul(a, b):
    return torch.matmul(a, b)


def softmax(x, axis=-1, precise=False):
    """``precise=True`` accumulates in float32, as MLX does, and returns
    the input dtype."""
    if precise:
        return torch.softmax(x.float(), dim=axis).to(x.dtype)
    return torch.softmax(x, dim=axis)


def sigmoid(x):
    return torch.sigmoid(x)


def erf(x):
    return torch.erf(x)


def where(cond, a, b):
    return torch.where(cond, a, b)


def sum(x, axis=None, keepdims=False):
    return torch.sum(x) if axis is None else torch.sum(x, dim=axis, keepdim=keepdims)


def cumsum(x, axis=None):
    return torch.cumsum(x, dim=-1 if axis is None else axis)


def sort(x, axis=-1):
    return torch.sort(x, dim=axis).values


def topk(x, k, axis=-1):
    """The ``k`` largest VALUES in ascending order, like ``mx.topk``
    (``sampling.py`` reads the threshold from ``[..., :1]``). Unlike
    ``torch.topk`` there are no indices."""
    vals = torch.topk(x, k, dim=axis).values
    return torch.sort(vals, dim=axis).values


def argpartition(x, kth, axis=-1):
    """Full index permutation with every element at or before ``kth``
    (negative counts from the end) no larger than the rest, like
    ``mx.argpartition``. A stable sort satisfies that for any ``kth``;
    call sites slice ``[..., :k]`` / ``[..., -k:]``. O(n log n) is fine
    at expert counts (<= 256)."""
    return torch.argsort(x, dim=axis, stable=True)


def take(a, indices, axis=None):
    """numpy/MLX ``take`` semantics (index shape composes with the
    source's remaining axes), NOT ``torch.take``'s flat-index semantics.
    """
    if not isinstance(indices, torch.Tensor):
        indices = torch.as_tensor(indices, device=DEVICE)
    if axis is None:
        flat = a.reshape(-1)
        return flat[indices.reshape(-1)].reshape(indices.shape)
    idx_flat = indices.reshape(-1).to(torch.long)
    out = torch.index_select(a, axis, idx_flat)
    rest = a.shape[:axis] + a.shape[axis + 1:]
    return out.reshape(tuple(indices.shape) + rest)


def _along_axis_index(a, indices, axis):
    """numpy/MLX ``*_along_axis`` broadcast ``indices`` against ``a`` on
    every axis except ``axis``; torch.gather/scatter do not."""
    shape = list(a.shape)
    shape[axis] = indices.shape[axis]
    return torch.broadcast_to(indices.to(torch.long), shape)


def take_along_axis(a, indices, axis):
    return torch.gather(a, axis, _along_axis_index(a, indices, axis))


def put_along_axis(a, indices, values, axis):
    """Functional (out-of-place) scatter, matching MLX's immutable-array
    semantics -- see the module docstring re: an in-place fast path.
    Indices broadcast against ``a`` and ``values`` against the indices
    (``moe/routing.py`` masks whole groups with ``[..., k, 1]`` indices and
    a scalar -inf).
    """
    idx = _along_axis_index(a, indices, axis)
    src = torch.as_tensor(values, dtype=a.dtype, device=a.device)
    return torch.scatter(a, axis, idx, torch.broadcast_to(src, idx.shape))


def index_add(a, indices, values):
    """``a.at[indices].add(values)`` in MLX: out-of-place, rows given by
    ``indices`` along axis 0, duplicates accumulate."""
    return a.index_add(0, indices.to(torch.long), values.to(a.dtype))


def size(x) -> int:
    """Number of elements (``x.size`` in MLX; a method in torch)."""
    return x.numel()


def max(x, axis=None, keepdims=False):
    if axis is None:
        return torch.amax(x)
    return torch.amax(x, dim=axis, keepdim=keepdims)


def maximum(a, b):
    a = torch.as_tensor(a, device=DEVICE)
    return torch.maximum(a, torch.as_tensor(b, dtype=a.dtype, device=a.device))


def argmax(x, axis=None, keepdims=False):
    return torch.argmax(x, dim=axis, keepdim=keepdims)


def set_cache_limit(limit):
    """MLX buffer-cache cap; torch's caching allocator has no equivalent
    knob here, so this is a no-op that returns the previous limit (0)."""
    return 0


def astype(x, dtype):
    return x.to(dtype)


def item(x):
    return x.item()


def tolist(x):
    return x.tolist()


# ---- lazy-graph hooks (MLX-specific; no-ops / thin wraps under eager torch)

def eval(*arrays):
    """MLX forces lazy-graph materialization here; torch is eager, so
    there is nothing to force. Kept as a no-op rather than removed so
    call sites (``streaming/layer.py``, ``prerouter/*``) need no
    backend-conditional code.
    """
    return None


def compile(fn):
    """Identity: runs eagerly. ``mx.compile`` is cheap to re-trace, but
    ``torch.compile`` recompiles on every new shape, and the streaming
    path changes shapes nearly every call (staged vs. exact, varying
    expert counts). Revisit with profiling on real hardware."""
    return fn


class random:
    """``mx.random`` submodule equivalent (contract: seed, categorical)."""

    @staticmethod
    def seed(s):
        torch.manual_seed(s)

    @staticmethod
    def categorical(logits, axis=-1):
        # mx.random.categorical draws ONE sample per row from unnormalized
        # logits along `axis`; torch.multinomial wants probabilities on
        # the LAST axis with shape [..., num_classes] -> 1 sample each.
        if axis != -1 and axis != logits.ndim - 1:
            logits = torch.movedim(logits, axis, -1)
        probs = torch.softmax(logits.float(), dim=-1)
        shape = probs.shape[:-1]
        flat = probs.reshape(-1, probs.shape[-1])
        draw = torch.multinomial(flat, 1).reshape(shape)
        return draw
