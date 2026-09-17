"""Install streaming MoE blocks into a loaded base model.

Generalizes the deployment's ``install_streaming_experts``: instead of
hard-coding a walk of ``model.layers``, the walk is driven by ``MoESpec``
(``block_path`` template + num layers).  The returned list of
``StreamingSwitchGLU`` objects is what the engine, prerouter stager, and
prefill hooks drive.
"""

from __future__ import annotations

from edge0.backends import core

from edge0.moe.spec import MoESpec
from edge0.streaming.cache import PrefetchBuffer, SharedExpertCache
from edge0.streaming.layer import StreamingSwitchGLU
from edge0.streaming.mmap import SafetensorsMmap
from edge0.streaming.options import LayerOptions


def _swap_submodule(block, name: str, value) -> None:
    """``setattr`` that also works on torch modules, which refuse to put a
    non-Module where a child module is registered."""
    try:
        setattr(block, name, value)
    except TypeError:
        delattr(block, name)
        object.__setattr__(block, name, value)


def install_streaming_experts(
    model,
    shards: list[SafetensorsMmap],
    spec: MoESpec,
    options: LayerOptions | None = None,
    num_layers: int | None = None,
    shared_cache: SharedExpertCache | None = None,
    prefetch_buffer: PrefetchBuffer | None = None,
    wrap=None,
) -> list[StreamingSwitchGLU]:
    """Replace every routed-expert block in ``model`` with its streaming
    twin and return the twins in layer order.

    ``num_layers`` defaults to ``spec``-independent discovery: the block
    path template is probed with increasing layer indices until it stops
    resolving.  The original blocks are kept on the objects as
    ``_edge0_resident`` (used by the bit-parity tests); nothing else in the
    base model is touched.

    ``wrap(twin)`` adapts the twin to the block's own calling convention
    before it is swapped in. The vendored MLX blocks call
    ``switch_mlp(x, inds)`` / ``experts(x, idx)`` and weight the per-expert
    outputs themselves, which is the twin's native interface, so the
    default is no wrapping.
    """
    if num_layers is None:
        n = 0
        while True:
            try:
                spec.block_of(model, n)
            except (AttributeError, IndexError):  # past the last layer
                break
            n += 1
        if n == 0:
            raise ValueError(
                f"block_path {spec.block_path!r} resolves no layer 0 — "
                "check MoESpec.block_path")
        num_layers = n

    opts = options or LayerOptions()
    cache = shared_cache or SharedExpertCache(opts.cache_slots)
    buf = prefetch_buffer or PrefetchBuffer(opts.prefetch_cap)

    stream_layers: list[StreamingSwitchGLU | None] = []
    for i in range(num_layers):
        block = spec.block_of(model, i)
        if not hasattr(block, "experts") and not hasattr(
                block, "switch_mlp"):
            # Dense (non-routed) layer — nothing to stream.
            stream_layers.append(None)
            continue
        twin = StreamingSwitchGLU(
            shards, i, spec, options=opts,
            shared_cache=cache, prefetch_buffer=buf)
        # keep the resident block reachable (bit-parity tests, debugging);
        # a plain attribute, not a child module -- as a child it would make
        # the block its own descendant (torch's state_dict recurses forever)
        object.__setattr__(block, "_edge0_resident", block)
        # swap the twin into the block, following the family convention:
        # qwen-style blocks call ``switch_mlp(x, inds)``; ling-style blocks
        # hold ``experts = SwitchGLU(...)`` and call ``experts(x, idx)``.
        # The original submodule is stashed for parity tests.
        name = "switch_mlp" if hasattr(block, "switch_mlp") else "experts"
        # object.__setattr__: keep the stash out of torch's child registry
        object.__setattr__(block, "_edge0_resident_switch",
                           getattr(block, name))
        _swap_submodule(block, name, wrap(twin) if wrap else twin)
        if opts.top_k is not None:
            res = getattr(block, "_edge0_resident", block)
            if hasattr(res, "top_k"):
                res.top_k = opts.top_k
        stream_layers.append(twin)
    return stream_layers


def attach_streaming(block, twin) -> None:
    """Point a MoE block's ``__call__`` at its streaming twin.

    For vendored models whose layer code calls ``self.mlp(...)`` directly,
    the installer swaps the block object itself (``setattr`` on the layer);
    for models whose blocks carry a ``__call__`` the engine may patch it
    class-level.  This helper exists so both paths share one spelling.
    """
    twin._edge0_resident = getattr(block, "_edge0_resident", block)
    block._edge0_stream = twin


def route_indices(mx_array) -> core.array:
    """Sanity guard: indices must be a GPU mx array when they leave the
    block (the whole staged design assumes no per-layer host syncs)."""
    assert isinstance(mx_array, core.array), (
        "router indices must stay on GPU (core.array), got "
        f"{type(mx_array).__name__}")
    return mx_array
