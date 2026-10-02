"""Streaming layer options.

One ``LayerOptions`` instance describes the streaming behavior of one MoE
layer (all layers usually share one instance).  This is the typed
replacement for the deployment's env-var knobs; profiles (see
``edge0.models``) pick the option sets for a given tier.

Defaults reproduce the deployment's production behavior for the edge0-35b
K=4 tier (staged decode on, sync fills, hot pins on prefill).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LayerOptions:
    """Options for one streaming MoE layer.

    Attributes:
        staged: enable fixed-slot staged decode (routing through staged
            slots with zero per-layer host syncs).
        staged_replace: staged set REPLACES the router (routing set ==
            staged set; used with a prerouter head — no drops).
        staged_n: number of staged slots.
        staged_trigger: router top-k that activates the staged path
            (usually the routed top_k).
        staged_sync: fill staged slots synchronously at the step boundary
            (as opposed to purely async double-buffered fills).
        asm_cache: cache assembled (slot table, stacked-wargs) per staged
            expert set so repeated sets don't rebuild graph nodes.
        incr_stack: incremental sticky-slot stack (in-place row writes
            instead of re-stacking; sync staged mode only).
        incr_writeback: write evicted bundles back into the shared LRU
            (dedups the staged set from the LRU while staged).
        history_prefetch: prefetch the union of recent actual expert sets.
        hot_per_layer: resident hot-expert pins per layer (0 disables).
        hot_update_interval: refresh hot pins every N forward calls.
        hot_decay: decay factor for hot-expert usage counts.
        pin_bonus: extra score for prerouter-predicted experts when
            selecting hot pins.
        cache_slots: shared LRU capacity across all layers, keyed by
            ``(layer_idx, expert)`` with a single global budget (see
            ``SharedExpertCache``).  Size it for the WHOLE model, not per
            layer: a decode step touches ``top_k`` experts on every MoE layer
            (23 x 8 = 184 distinct keys on edge0-8b), so a budget smaller than
            that cannot hold even one token's working set and the hit rate
            collapses to ~0 (the released 64 slots = 2.8 slots/layer measured
            exactly 0 hits, rebuilding 184 bundles/step).
        prefetch_cap: prefetch buffer capacity.
        load_threads: threads for on-demand expert builds.
        prefetch_threads: threads for eager prefetch builds.
        use_compile: wrap the MoE math in ``mx.compile``.
        top_k: override the resident block's routed top-k (both prefill
            and decode); None leaves the model config value.
        full_layer_prefill: whole-layer load-drop prefill (E3b).
        prefill_full_layers: number of LEADING layers that get whole-layer
            loads (N*~310MB of page cache for qwen); 0 = every layer.
        prefill_hot: resident hot-stack size used during prefill
            (0 disables the hot-stack path).
    """

    staged: bool = False
    staged_replace: bool = False
    staged_n: int = 8
    staged_trigger: int = 8
    staged_sync: bool = True
    asm_cache: bool = True
    incr_stack: bool = False
    incr_writeback: bool = False
    history_prefetch: bool = True
    hot_per_layer: int = 0
    hot_update_interval: int = 4
    hot_decay: float = 0.75
    pin_bonus: float = 2.0
    cache_slots: int = 64
    prefetch_cap: int = 48
    load_threads: int = 8
    prefetch_threads: int = 4
    use_compile: bool = True
    top_k: int | None = None
    full_layer_prefill: bool = False
    prefill_full_layers: int = 0
    prefill_hot: int = 0
    #: Issue madvise(MADV_WILLNEED) over the byte ranges of the experts a
    #: prefetch/stage is about to touch, before building bundles.  Lets the
    #: kernel do bulk ASYNC readahead (one syscall per tensor range) instead
    #: of the loader DEMAND-faulting page by page — on a memory-starved host
    #: a step otherwise degrades into "cold pages x per-fault latency", with
    #: those faults serializing on the VM map lock.  Retains no MLX arrays,
    #: so it does not displace the page cache.
    warm_willneed: bool = False

    # ---- presets ----------------------------------------------------------

    @classmethod
    def staged_k4(cls, **overrides) -> "LayerOptions":
        """edge0-35b K=4 tier: staged decode + hot pins + whole-layer
        prefill (production profile).

        ``staged_replace`` stays False: the trained prerouter head supplies
        the routing (the MoE block routes via the prerouter logits), so the
        staged set == the routing set exactly — the slot table maps without
        drops.

        Memory-starved hosts (measured on a 16 GiB M2, 19.7 GB checkpoint,
        60-step replay, 5 rounds, same-run pairing):

        * ``staged_sync=True`` + ``incr_stack=True`` +
          ``incr_writeback=True`` — the incremental stack replaces the
          per-step ``core.stack`` with persistent in-place tensors and keeps
          the staged bundles out of the shared LRU.  Peak MLX 2.92 -> 2.60
          GiB, file-backed page cache 4.25 -> 4.72 GiB, minor faults/step
          5162 -> 1925, CPU 143 -> 109 ms/step.  (Requires sync staged.)
        * ``warm_willneed=True`` — issue ``madvise(MADV_WILLNEED)`` over the
          predicted experts' byte ranges before building.  Staging wall
          162 -> 27 ms/step, because the kernel does bulk async readahead
          instead of the loader demand-faulting page by page (faults
          serialize on the VM map lock).

        Together: step 244.25 -> 187.58 ms, 4.09 -> 5.33 tok/s (+31.0%,
        paired 4/5), and the prerouter's own gain over ``prerouter=None``
        rises from +8.0% to +37.8% (paired 5/5) with bit-identical output.
        """
        return cls(
            staged=True, staged_replace=False, staged_n=4,
            staged_trigger=4, staged_sync=True, history_prefetch=True,
            incr_stack=True, incr_writeback=True, warm_willneed=True,
            hot_per_layer=0, hot_update_interval=4, hot_decay=0.75,
            cache_slots=64, prefetch_cap=48, load_threads=8,
            prefetch_threads=4, use_compile=True, top_k=4,
            full_layer_prefill=False, prefill_full_layers=0,
            prefill_hot=32, **overrides)

    @classmethod
    def staged_k8(cls, **overrides) -> "LayerOptions":
        """edge0-8b tier: staged decode with K=8 (native routing width),
        no hot pins, E3b whole-layer prefill."""
        return cls(
            staged=True, staged_replace=False, staged_n=8,
            staged_trigger=8, staged_sync=True, history_prefetch=True,
            hot_per_layer=0, cache_slots=64, prefetch_cap=48,
            load_threads=8, prefetch_threads=4, use_compile=True, top_k=8,
            full_layer_prefill=True, prefill_hot=0, **overrides)

    @classmethod
    def prod_k8(cls, **overrides) -> "LayerOptions":
        """edge0-8b tier: staged decode ON for the prerouter consumers.

        Only the layers whose ROUTE is a prerouter prediction keep staged
        slots (consumer ``li >= start_layer + 1``, i.e. 8-23): the stager
        submits exactly the set the consuming block re-selects from the same
        cached logits, so ``slots == routing`` and nothing is dropped
        (measured ``staged_dropped = 0``; output bit-identical to the exact
        path).  The layers below ``start_layer`` route with their own gate and
        run the exact path (``ling.load_installed`` sets
        ``_staged_mode = False``; legacy history staging there dropped 17-26%
        of their experts and degraded output -- opt back in with
        ``history_slots=True`` / ``--history-slots``).

        Measured on M4 Pro (same-process paired A/B): slots cut expert loads
        from 184 to 56 per step (8-23 served entirely by the prediction), and
        the prerouter routes ~5% faster than gate routing with the same
        loading path.

        ``cache_slots`` stays at the low-memory default 64 (~81 MiB): this
        tier's contract is a small resident footprint and every cached bundle
        is 1.27 MiB of RAM.  The 64 is NOT enough to reuse anything across
        tokens (2.8 slots/layer for a 184-key/step working set, measured 0
        hits): every predicted expert whose set changed since the previous
        token (44% of the 128 staged uses/step -- the other 56% come from the
        previous token's slots) plus all 56 gate-routed uses on L1-7 are
        re-sliced and re-packed every step.  That is the price of the small
        footprint, and the staged slots are how this tier avoids paying it for
        the consumer layers.  Enlarging the LRU buys speed with RAM (1024 =
        +1.3 GiB, peak 1.4 -> 2.6 GiB -> 40.02 -> 31.44 ms/step, paired 3/3
        rounds, output identical) -- do it only when the memory budget allows;
        at cache_slots=64 the scoped staged path is the fastest arm measured
        (37.99 vs 39.52 ms/step for gate routing + exact loads, 6/8 rounds) and
        under a load-bound regime (build latency throttled to 200 us) it wins
        32-40% (4/4 rounds)."""
        return cls(
            staged=True, staged_replace=False, staged_n=8,
            staged_trigger=8, staged_sync=True, history_prefetch=False,
            hot_per_layer=0,
            cache_slots=64, prefetch_cap=48,
            load_threads=8, prefetch_threads=4, use_compile=True, top_k=8,
            full_layer_prefill=True, prefill_hot=0, **overrides)

    def for_layer(self, layer_idx: int) -> "LayerOptions":
        """Per-layer copy (staged slots are per-layer state, but options
        are shared; this hook exists for future per-layer overrides)."""
        return self
