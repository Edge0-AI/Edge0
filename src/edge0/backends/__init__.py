"""Backend abstraction.

The framework core (engine, streaming, prerouter, adapters, model
adapters, server) imports ONLY the backend namespaces re-exported here::

    from edge0.backends import core, nn, io, quant

A backend is a directory under ``edge0/backends/<name>/`` exposing:

``backend.py``
    ``core``  — array ops namespace (matmul, softmax, argpartition, take,
                 put_along_axis, eval, compile, ...)
    ``nn``    — module factory namespace (Module, Linear, RMSNorm, silu, ...)
    ``io``    — model/tokenizer/tensor-store loading
    ``quant`` — quantized gather kernels (``gather_qmm``)
    ``BackendImpl`` — small object with ``name`` and metadata

The MLX backend is the reference implementation.  A future CUDA backend
implements the same surface with eager semantics; engine / prerouter /
server / model-adapter code is shared verbatim.  The op list below is the
contract the framework currently exercises (grow it as models are added).

Contract (documented, enforced by tests/grep in CI):
  * The MLX package (``mlx`` / ``mlx_lm``) must never be imported
    outside ``edge0/backends/mlx/`` — framework code reaches the backend
    only through the namespaces re-exported here.
  * ``core`` must provide (for the MLX backend these are mlx.core names):
      array, zeros, eye, arange, full, expand_dims, squeeze, reshape,
      transpose, concatenate, stack, split, matmul, softmax, silu (via nn),
      sigmoid, erf, where, sum, cumsum, sort, topk, argpartition, take,
      take_along_axis, put_along_axis, astype, item, tolist, eval, compile,
      random.seed, random.categorical, and the dtypes float16 / float32 /
      bfloat16 / int32 / uint32 / int64.
  * ``nn`` must provide Module, Linear, RMSNorm, silu, gelu.
  * ``io`` must provide load_model, load_tokenizer, open_tensor_store.
  * ``quant`` must provide gather_qmm.
"""

import os

_BACKEND = os.environ.get("EDGE0_BACKEND", "mlx")

if _BACKEND == "mlx":
    from edge0.backends.mlx.backend import (  # noqa: F401
        BackendImpl,
        core,
        io,
        nn,
        quant,
    )
elif _BACKEND == "cuda":
    # Reference/fallback implementation (torch-backed). Try
    # EDGE0_BACKEND=mlx with `pip install mlx[cuda12]` on real NVIDIA
    # hardware FIRST -- MLX's own CUDA backend natively implements
    # every op this facade calls (confirmed against ml-explore/mlx
    # source: GatherQMM::eval_gpu exists in
    # mlx/backend/cuda/quantized/quantized.cpp, and its allocator uses
    # cudaMallocManaged to keep the unified-memory model on GPUs with
    # concurrentManagedAccess) -- that path needs zero new code here.
    from edge0.backends.cuda.backend import (  # noqa: F401
        BackendImpl,
        core,
        io,
        nn,
        quant,
    )
else:
    raise ImportError(
        f"unknown backend {_BACKEND!r}; available: mlx, cuda")

backend = BackendImpl()

__all__ = ["backend", "core", "io", "nn", "quant"]
