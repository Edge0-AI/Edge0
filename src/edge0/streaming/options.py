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
        cache_slots: shared LRU capacity across all layers.
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

    # ---- presets ----------------------------------------------------------

    @classmethod
    def staged_k4(cls, **overrides) -> "LayerOptions":
        """edge0-35b K=4 tier: staged decode + hot pins + whole-layer
        prefill (production profile).

        ``staged_replace`` stays False: the trained prerouter head supplies
        the routing (the MoE block routes via the prerouter logits), so the
        staged set == the routing set exactly — the slot table maps without
        drops."""
        return cls(
            staged=True, staged_replace=False, staged_n=4,
            staged_trigger=4, staged_sync=True, history_prefetch=True,
            hot_per_layer=0, hot_update_interval=4, hot_decay=0.75,
            cache_slots=64, prefetch_cap=48, load_threads=8,
            prefetch_threads=4, use_compile=True, top_k=4,
            full_layer_prefill=False, prefill_full_layers=0,
            prefill_hot=32, **overrides)

    @classmethod
    def staged_k8(cls, **overrides) -> "LayerOptions":
        """edge0-10b tier: staged decode with K=8 (native routing width),
        no hot pins, E3b whole-layer prefill."""
        return cls(
            staged=True, staged_replace=False, staged_n=8,
            staged_trigger=8, staged_sync=True, history_prefetch=True,
            hot_per_layer=0, cache_slots=64, prefetch_cap=48,
            load_threads=8, prefetch_threads=4, use_compile=True, top_k=8,
            full_layer_prefill=True, prefill_hot=0, **overrides)

    @classmethod
    def prod_k8(cls, **overrides) -> "LayerOptions":
        """edge0-10b tier: deployment production profile.

        Mirrors start_server.sh on the ling v7 deploy: STAGED_DECODE=0
        (staged decode is OFF — the deployment verified staged decode on
        ling degrades output, see the comment in start_server.sh),
        STAGED_SYNC=0, STAGED_SLOTS=8, EXPERT_CACHE_SLOTS=64,
        MLX_COMPILE_MOE=1, PREFILL_FULL_LAYER=1.  The hybrid prerouter
        still stages each next token's expert set into the shared LRU
        (engine.py: stage_all at the step boundary AND right after
        prefill), and the vendored MoE blocks re-select from the cached
        prerouter logits."""
        return cls(
            staged=False, staged_replace=False, staged_n=8,
            staged_trigger=8, staged_sync=False, history_prefetch=False,
            hot_per_layer=0, cache_slots=64, prefetch_cap=48,
            load_threads=8, prefetch_threads=4, use_compile=True, top_k=8,
            full_layer_prefill=True, prefill_hot=0, **overrides)

    def for_layer(self, layer_idx: int) -> "LayerOptions":
        """Per-layer copy (staged slots are per-layer state, but options
        are shared; this hook exists for future per-layer overrides)."""
        return self
