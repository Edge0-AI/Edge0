"""StreamingSwitchGLU: the SSD-offloaded MoE expert block.

Faithful port of the deployment's ``streaming_experts_qwen35.py``
(itself a port of the ling ``stream_experts.py`` design), generalized so
ONE class serves every supported model:

* expert weight layout (separate vs fused gate+up projections) and
  safetensors key prefix come from ``MoESpec``;
* every behavior knob (staged decode, hot pins, prefetch, compile,
  incremental stack, whole-layer prefill) comes from ``LayerOptions``
  instead of env vars.

Public surface consumed by the engine / prerouter / prefill hooks:

    load_full_layer / clear_full_layer / load_hot_layer /
    materialize_hot / dematerialize_hot / clear_hot_layer
    stage_experts / wait_staged / swap_staged / sync_actuals /
    stage_from_prefill / stage_from_history
    prefetch / prefetch_all / warm_pages
    stats / reset

The ``__call__(x, indices)`` routing:

1. whole-layer prefill path   — full stacked weights resident
   (``_full_weights``), sorted gather, bit-identical to the resident
   math; also records usage + last-token top-k for staging.
2. hot-stack prefill path     — resident top-N stack
   (``_hot_weights``) with exact per-expert scatter-add for misses.
3. staged decode path         — routing through fixed staged slots with
   a single GPU-side expert->slot table (``core.take``), no per-layer
   host syncs; missing experts map to an all-zero overflow row
   (production prerouter consumers have zero misses by construction:
   the staged set IS the prerouter's prediction).
4. exact decode path          — per-expert bundles (pinned / shared LRU
   / prefetch buffer / pool builds), sorted gather for large batches.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from edge0.backends import core, quant
from edge0.backends import nn
import numpy as np

from edge0.moe.spec import MoESpec
from edge0.streaming.cache import PrefetchBuffer, SharedExpertCache
from edge0.streaming.mmap import SafetensorsMmap, u32_view
from edge0.streaming.options import LayerOptions


def _swiglu(up, gate):
    return nn.silu(gate) * up


class StreamingSwitchGLU:
    """Streaming version of a routed-expert GLU block.

    Only the routed experts are streamed.  Shared experts, routers, and
    attention stay resident in the base model.
    """

    def __init__(
        self,
        shards: list[SafetensorsMmap],
        layer_idx: int,
        spec: MoESpec,
        options: LayerOptions | None = None,
        shared_cache: SharedExpertCache | None = None,
        prefetch_buffer: PrefetchBuffer | None = None,
    ):
        self.layer_idx = layer_idx
        self.spec = spec
        self.num_experts = spec.num_experts
        self.group_size = spec.quant.group_size
        self.bits = spec.quant.bits
        self.mode = spec.quant.mode
        self._fuse_gu = spec.fuse_gu
        self._bundle_projs = spec.bundle_projs

        o = options or LayerOptions()
        self._staged_mode = o.staged
        self._staged_replace = o.staged_replace
        self._staged_n = max(1, o.staged_n)
        # Router top-k that triggers the staged path (single-token decode);
        # set by the engine to the routed top_k (usually the staged_n).
        self._staged_trigger = o.staged_trigger
        self._staged_sync = o.staged_sync
        self.hot_per_layer = o.hot_per_layer
        self.hot_update_interval = o.hot_update_interval
        self.hot_decay = o.hot_decay
        self._pin_bonus = o.pin_bonus
        self._use_compile = o.use_compile

        self._pool = ThreadPoolExecutor(max_workers=o.load_threads)
        self._prefetch_pool = ThreadPoolExecutor(
            max_workers=max(1, o.prefetch_threads))

        self._shards = shards
        self._shard_idx = {
            name: i
            for i, s in enumerate(shards)
            for name in s.entries
        }
        self._prefix = spec.key_template.format(layer=layer_idx)
        self._shape = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for part in ("weight", "scales", "biases"):
                name = f"{self._prefix}.{proj}.{part}"
                self._shape[(proj, part)] = self._shards[
                    self._shard_idx[name]].entries[name]["shape"]
        # Bundle-format shapes: what _build produces / caches store / math
        # consumes. Fused gate_up = gate rows on top of up rows along axis 1.
        if self._fuse_gu:
            self._bundle_shape = {}
            for part in ("weight", "scales", "biases"):
                gs = self._shape[("gate_proj", part)]
                self._bundle_shape[("gate_up_proj", part)] = (
                    gs[0], 2 * gs[1]) + tuple(gs[2:])
                self._bundle_shape[("down_proj", part)] = self._shape[
                    ("down_proj", part)]
        else:
            self._bundle_shape = self._shape

        self._pinned = {}
        self._hot_counts = {}
        self._calls = 0
        self._full_weights: dict | None = None
        self._hot_weights: dict | None = None
        self._hot_backing: dict | None = None
        self._hot_key: tuple | None = None
        self._hot_set: set | None = None
        self.predicted_experts: set[int] = set()
        self._lock = threading.Lock()
        self.last_used: list[int] = []
        self._stats = {"loads": 0, "hits": 0, "load_wall": 0.0}
        self.shared_cache = shared_cache or SharedExpertCache(o.cache_slots)
        self._prefetch_buf = prefetch_buffer or PrefetchBuffer(o.prefetch_cap)
        self._inflight: dict[int, object] = {}
        self._stats.update({
            "prefetch_submitted": 0,
            "prefetch_hits": 0,
            "prefetch_wasted": 0,
            "prefetch_wait": 0.0,
        })

        # ---- MoE math (core.compile variants) ------------------------------

        def _make_moe_math():
            gs, bt, md = self.group_size, self.bits, self.mode

            @core.compile
            def fn(x, w_u, s_u, b_u, w_g, s_g, b_g, w_d, s_d, b_d, local):
                xe = core.expand_dims(x, (-2, -3))
                x_up = quant.gather_qmm(
                    xe, w_u, s_u, b_u, rhs_indices=local,
                    transpose=True, group_size=gs, bits=bt, mode=md,
                    sorted_indices=False)
                x_gate = quant.gather_qmm(
                    xe, w_g, s_g, b_g, rhs_indices=local,
                    transpose=True, group_size=gs, bits=bt, mode=md,
                    sorted_indices=False)
                z = quant.gather_qmm(
                    _swiglu(x_up, x_gate), w_d, s_d, b_d,
                    rhs_indices=local, transpose=True, group_size=gs,
                    bits=bt, mode=md, sorted_indices=False)
                return z.squeeze(-2)
            return fn

        def _make_moe_math_sorted():
            gs, bt, md = self.group_size, self.bits, self.mode

            @core.compile
            def fn(x, w_u, s_u, b_u, w_g, s_g, b_g, w_d, s_d, b_d,
                   local, inv_order, out_shape):
                x_up = quant.gather_qmm(
                    x, w_u, s_u, b_u, rhs_indices=local,
                    transpose=True, group_size=gs, bits=bt, mode=md,
                    sorted_indices=True)
                x_gate = quant.gather_qmm(
                    x, w_g, s_g, b_g, rhs_indices=local,
                    transpose=True, group_size=gs, bits=bt, mode=md,
                    sorted_indices=True)
                z = quant.gather_qmm(
                    _swiglu(x_up, x_gate), w_d, s_d, b_d,
                    rhs_indices=local, transpose=True, group_size=gs,
                    bits=bt, mode=md, sorted_indices=True)
                return quant.scatter_unsort(z, inv_order, out_shape).squeeze(-2)
            return fn

        def _make_moe_math_fused():
            gs, bt, md = self.group_size, self.bits, self.mode

            @core.compile
            def fn(x, w_gu, s_gu, b_gu, w_d, s_d, b_d, local):
                xe = core.expand_dims(x, (-2, -3))
                x_gu = quant.gather_qmm(
                    xe, w_gu, s_gu, b_gu, rhs_indices=local,
                    transpose=True, group_size=gs, bits=bt, mode=md,
                    sorted_indices=False)
                x_gate, x_up = core.split(x_gu, 2, axis=-1)
                z = quant.gather_qmm(
                    _swiglu(x_up, x_gate), w_d, s_d, b_d,
                    rhs_indices=local, transpose=True, group_size=gs,
                    bits=bt, mode=md, sorted_indices=False)
                return z.squeeze(-2)
            return fn

        def _make_moe_math_sorted_fused():
            gs, bt, md = self.group_size, self.bits, self.mode

            @core.compile
            def fn(x, w_gu, s_gu, b_gu, w_d, s_d, b_d,
                   local, inv_order, out_shape):
                x_gu = quant.gather_qmm(
                    x, w_gu, s_gu, b_gu, rhs_indices=local,
                    transpose=True, group_size=gs, bits=bt, mode=md,
                    sorted_indices=True)
                x_gate, x_up = core.split(x_gu, 2, axis=-1)
                z = quant.gather_qmm(
                    _swiglu(x_up, x_gate), w_d, s_d, b_d,
                    rhs_indices=local, transpose=True, group_size=gs,
                    bits=bt, mode=md, sorted_indices=True)
                return quant.scatter_unsort(z, inv_order, out_shape).squeeze(-2)
            return fn

        self._moe_math = _make_moe_math()
        self._moe_math_sorted = _make_moe_math_sorted()
        if self._fuse_gu:
            # Fused gate+up: replace the math fns with the fused variants
            # (same pattern as the original streaming_experts_qwen35.py) —
            # every call site passes its wargs through unchanged.
            self._moe_math = _make_moe_math_fused()
            self._moe_math_sorted = _make_moe_math_sorted_fused()

        # ---- staged decode ------------------------------------------------
        # Port of ling stream_experts.py R1 fixed-slot staging. The router's
        # indices stay on GPU: a CPU-built [num_experts] expert->slot table
        # maps them into the staged tensor via core.take, so a whole decode
        # step builds/evaluates with a single sync (the final logits eval)
        # instead of 40 per-layer tolist barriers. Missing experts map to
        # the overflow zero slot (contribution dropped) — with a prerouter
        # the staged set IS the routing, so this never fires in production.
        # Double-buffered staged slots: `_staged_state` is the ACTIVE set
        # read by the current decode step's __call__ (never disturbed
        # mid-step); `_staged_state_next` is the fill target written by
        # stage_experts, swapped in at the step boundary by swap_staged().
        self._staged_state: tuple | None = None
        self._staged_state_next: tuple | None = None
        self._staging_lock = threading.Lock()
        self._staging_future = None
        self._staged_pending: list[int] | None = None
        self._pending_indices: core.array | None = None
        self._staged_at_use: set[int] | None = None
        self._last_prefill_topk: list[int] | None = None
        # Small cache of already-materialized (GPU) expert->slot tables.
        self._slot_table_cache: dict[tuple, core.array] = {}
        # Cache of the assembled (slot_of, stacked-wargs) pair per staged
        # expert set, so decode steps whose staged set repeats don't rebuild
        # the 9 core.stack nodes + slot table on every layer every step.
        if not o.asm_cache:
            class _NoAsmCache(dict):
                def __setitem__(self, k, v):  # never store
                    pass
            self._staged_asm_cache = _NoAsmCache()
        else:
            self._staged_asm_cache = {}
        # Persistent all-zero overflow slot rows (slot index = _staged_n),
        # so missing experts map to a real zero row.
        self._zero_slot = {}
        for proj in self._bundle_projs:
            for part in ("weight", "scales", "biases"):
                shape = self._bundle_shape[(proj, part)][1:]
                dtype = core.uint32 if part == "weight" else core.bfloat16
                self._zero_slot[(proj, part)] = core.zeros(shape, dtype)
        # ---- incremental stack --------------------------------------------
        # Sticky-slot persistent staged tensors: the [n+1, ...] stacked
        # weights are allocated ONCE per layer and only the slots whose
        # expert changed are rewritten in place each step, replacing the
        # 9 core.stack graph nodes per layer per step. Safe only when the
        # previous forward's eval has finished (staged_sync, the
        # production default) — the async double-buffered fill would race
        # the writes.
        self._incr_mode = (
            o.incr_stack and self._staged_mode and self._staged_sync
            and not self._staged_replace)
        # Write-back LRU (dedup staged set from the shared LRU): while
        # staged, an expert's bundle is held only by the incremental stack
        # machinery; it enters the LRU only when evicted from the stack.
        self._incr_writeback = (
            self._incr_mode and o.incr_writeback)
        self._incr_tensors = None      # {(proj,part): core.array [n+1, ...]}
        self._incr_slot_table = None   # core.array [num_experts] int32
        self._incr_occupancy = None    # list[slot] -> expert id | None
        self._incr_bundles = {}        # expert -> bundle (current set)
        self._stats.update({
            "staged_used": 0,
            "staged_dropped": 0,
            "staged_fallback": 0,
            "stage_wall": 0.0,
            "stage_wait_ms": 0.0,
            "stage_delta_hits": 0,
            "stage_cache_hits": 0,
            "stage_builds": 0,
            "incr_writes": 0,
        })

    # ---- loading ---------------------------------------------------------

    def _shard_for(self, name: str) -> SafetensorsMmap:
        return self._shards[self._shard_idx[name]]

    def _build(self, expert: int):
        """Load one expert's 9 tensors (gate/up/down x weight/scales/biases).

        Hot-stack fast path: if the expert is a member of the resident
        hot stack, slice its rows from the stack instead of reading the
        shard — zero IO, zero page-cache pressure."""
        if self._hot_backing is not None and expert in self._hot_set:
            idx = self._hot_key.index(expert)
            shape = self._bundle_shape
            b = {}
            for (proj, part), buf in self._hot_backing.items():
                sh = shape[(proj, part)]
                per = buf.size // (len(self._hot_key) + 1)  # + zero row
                sl = buf[idx * per:(idx + 1) * per]
                if part == "weight":
                    b[(proj, part)] = core.array(
                        sl.view("<u4").reshape(sh[1:]))
                else:
                    b[(proj, part)] = core.array(sl.view("<u2")).view(
                        core.bfloat16).reshape(sh[1:])
            return b
        per_w = {}

        def _read_rows(proj, part):
            name = f"{self._prefix}.{proj}.{part}"
            raw = self._shard_for(name).raw(name)
            shape = self._shape[(proj, part)]
            per = raw.size // self.num_experts
            return raw[expert * per:(expert + 1) * per]

        def _to_mx(sl, key):
            part = key[1]
            shape = self._bundle_shape[key]
            if part == "weight":
                return core.array(u32_view(sl, shape[1:]))
            return core.array(sl.view("<u2")).view(
                core.bfloat16).reshape(shape[1:])

        if self._fuse_gu:
            # gate rows on top of up rows (matches split(x_gu, 2) order);
            # numpy-level concat of the raw byte slices before the single
            # core.array per part (pool thread, off the critical path).
            for part in ("weight", "scales", "biases"):
                sl = np.concatenate([
                    _read_rows("gate_proj", part),
                    _read_rows("up_proj", part)])
                per_w[("gate_up_proj", part)] = _to_mx(
                    sl, ("gate_up_proj", part))
            for part in ("weight", "scales", "biases"):
                per_w[("down_proj", part)] = _to_mx(
                    _read_rows("down_proj", part), ("down_proj", part))
            return per_w
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for part in ("weight", "scales", "biases"):
                per_w[(proj, part)] = _to_mx(
                    _read_rows(proj, part), (proj, part))
        return per_w

    def warm_pages(self, experts):
        """Fault expert mmap pages into the OS page cache without retaining
        mx arrays (MLX active memory stays flat)."""
        for e in experts:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for part in ("weight", "scales", "biases"):
                    name = f"{self._prefix}.{proj}.{part}"
                    raw = self._shard_for(name).raw(name)
                    per = raw.size // self.num_experts
                    sl = raw[e * per:(e + 1) * per]
                    _ = sl.sum()

    def _get_bundles(self, experts):
        """Resolve bundles: pinned/LRU hits synchronously, misses via pool."""
        bundles = {}
        missing = []
        with self._lock:
            for e in experts:
                p = self._pinned.get(e)
                if p is not None:
                    bundles[e] = p
                    continue
                bundles[e] = None
                missing.append(e)
        still_missing = []
        for e in missing:
            cached = self.shared_cache.get((self.layer_idx, e))
            if cached is not None:
                self._stats["hits"] += 1
                bundles[e] = cached
            else:
                still_missing.append(e)
        missing = still_missing
        resolved = set()
        for e in missing:
            with self._lock:
                bundle = self._prefetch_buf.pop((self.layer_idx, e))
            if bundle is not None:
                self._stats["prefetch_hits"] += 1
                self._stats["hits"] += 1
                with self._lock:
                    self._hot_counts[e] = (
                        self._hot_counts.get(e, 0.0) * self.hot_decay + 1.0)
                self.shared_cache.put((self.layer_idx, e), bundle)
                bundles[e] = bundle
                resolved.add(e)
        missing = [e for e in missing if e not in resolved]
        resolved = set()
        for e in missing:
            with self._lock:
                fut = self._inflight.get(e)
            if fut is not None:
                t0 = time.perf_counter()
                tensors = fut.result()
                with self._lock:
                    self._inflight.pop(e, None)
                    self._stats["prefetch_wait"] += time.perf_counter() - t0
                    self._hot_counts[e] = (
                        self._hot_counts.get(e, 0.0) * self.hot_decay + 1.0)
                self._stats["prefetch_hits"] += 1
                self._stats["hits"] += 1
                self.shared_cache.put((self.layer_idx, e), tensors)
                bundles[e] = tensors
                resolved.add(e)
        missing = [e for e in missing if e not in resolved]
        if missing:
            t0 = time.perf_counter()
            built = list(self._pool.map(self._build, missing))
            wall = time.perf_counter() - t0
            with self._lock:
                for e, tensors in zip(missing, built):
                    self._stats["loads"] += 1
                    self._hot_counts[e] = (
                        self._hot_counts.get(e, 0.0) * self.hot_decay + 1.0)
                    bundles[e] = tensors
                self._stats["load_wall"] += wall
                for e, tensors in zip(missing, built):
                    self.shared_cache.put((self.layer_idx, e), tensors)
        return bundles

    # ---- prefetch --------------------------------------------------------

    def prefetch(self, experts):
        unique = sorted({int(e) for e in experts})
        if not unique:
            return
        submitted = []
        with self._lock:
            missing = []
            for e in unique:
                if (
                    e in self._pinned
                    or e in self._inflight
                    or self.shared_cache.peek((self.layer_idx, e)) is not None
                    or self._prefetch_buf.contains((self.layer_idx, e))
                ):
                    continue
                missing.append(e)
            for e in missing:
                fut = self._prefetch_pool.submit(self._build, e)
                self._inflight[e] = fut
                submitted.append((e, fut))
            self._stats["prefetch_submitted"] += len(missing)
        # Register the done-callback OUTSIDE the lock: if the build finished
        # before submit() returned, add_done_callback fires synchronously on
        # this thread, and _on_prefetch_done re-acquires the same non-reentrant
        # lock — deadlock. Moving the registration out fixes it.
        for e, fut in submitted:
            fut.add_done_callback(partial(self._on_prefetch_done, e))

    def prefetch_all(self):
        n = self.num_experts
        if self._prefetch_buf.cap < n + 32:
            self._prefetch_buf.set_cap(n + 32)
        self.prefetch(range(n))

    def _on_prefetch_done(self, expert, fut):
        try:
            bundle = fut.result()
        except Exception:
            with self._lock:
                self._inflight.pop(expert, None)
            return
        with self._lock:
            self._inflight.pop(expert, None)
            if self.shared_cache.peek((self.layer_idx, expert)) is not None:
                return
            evicted = self._prefetch_buf.put((self.layer_idx, expert), bundle)
            if evicted is not None:
                self._stats["prefetch_wasted"] += 1

    # ---- fixed-slot staging (staged decode) ------------------------------

    def _build_staged_bundles(self, experts: list[int]):
        """Resolve per-expert bundles for the next staged slots.

        Runs on the thread pool. Reuses the previous staged state by
        reference (delta reuse), then pinned / shared-LRU mx arrays; only
        genuinely cold experts go through the raw mmap build. The core.stack
        into fixed [n+1, ...] tensors is done on the main thread in __call__
        because MLX lazy-op construction is not safe concurrently with eval.
        """
        _t0 = time.perf_counter()
        n = self._staged_n
        exp = sorted({int(e) for e in experts})[:n]
        prev = {}
        st = self._staged_state
        if st is not None:
            pbundles, _, pexp = st
            for i, e in enumerate(pexp):
                prev[e] = pbundles[i]
        slot_of = [n] * self.num_experts
        for i, e in enumerate(exp):
            slot_of[e] = i
        bundles = []
        for e in exp:
            b = prev.get(e)
            if b is not None:
                with self._lock:
                    self._stats["stage_delta_hits"] += 1
            if b is None:
                with self._lock:
                    b = self._pinned.get(e)
            if b is None:
                b = self.shared_cache.get((self.layer_idx, e))
                if b is not None:
                    with self._lock:
                        self._stats["stage_cache_hits"] += 1
            if b is None:
                b = self._build(e)
                # put the fresh build into the shared LRU so later steps
                # hit the cache instead of rebuilding: the staged resolver
                # above checks shared_cache BEFORE _build, so this closes
                # the loop (delta miss -> LRU hit, no rebuild).
                self.shared_cache.put((self.layer_idx, e), b)
                with self._lock:
                    self._stats["stage_builds"] += 1
            bundles.append(b)
        wall = time.perf_counter() - _t0
        return tuple(bundles), slot_of, exp, wall

    def _stage_incr(self, experts: list[int]):
        """Incremental-stack resolver: sticky slot assignment + in-place
        delta writes into the persistent stacked tensors.

        Keeps every surviving expert in its current slot (zero work), frees
        the slots of evicted experts, and assigns newly arriving experts to
        the freed slots — writing ONLY those slots' rows in place (one
        __setitem__ per (proj, part) per changed slot) plus slot-table
        updates. Bundle resolution cascade is identical to
        _build_staged_bundles (pinned -> shared LRU -> raw build).
        Runs on the main thread (sync staged mode only)."""
        _t0 = time.perf_counter()
        n = self._staged_n
        exp = sorted({int(e) for e in experts})[:n]
        exp_set = set(exp)

        # Lazy one-time allocation of the persistent tensors. Slot n is the
        # all-zero overflow row and is NEVER written after this init.
        if self._incr_tensors is None:
            self._incr_tensors = {}
            for proj in self._bundle_projs:
                for part in ("weight", "scales", "biases"):
                    shape = self._bundle_shape[(proj, part)][1:]
                    dtype = core.uint32 if part == "weight" else core.bfloat16
                    t = core.zeros((n + 1, *shape), dtype)
                    core.eval(t)
                    self._incr_tensors[(proj, part)] = t
            self._incr_slot_table = core.full(
                (self.num_experts,), n, dtype=core.int32)
            core.eval(self._incr_slot_table)
            self._incr_occupancy = [None] * n

        occ = self._incr_occupancy
        # Sticky assignment: survivors keep slots; evicted slots free up.
        free = [i for i in range(n) if occ[i] is not None
                and occ[i] not in exp_set]
        for i in free:
            occ[i] = None
        free = [i for i in range(n) if occ[i] is None]
        # slot-of list for compatibility consumers (_staged_state / stats).
        slot_of_list = [n] * self.num_experts
        for i in range(n):
            e = occ[i]
            if e is not None:
                slot_of_list[e] = i

        changed = [e for e in exp if e not in self._incr_bundles]
        new_slots = []
        for j, e in enumerate(changed):
            slot = free[j] if j < len(free) else None
            new_slots.append((e, slot))

        writes = []
        # Mirror the occupancy exactly: only the CURRENT staged set keeps
        # bundles. A stale entry would hide a returning expert from
        # `changed` (no slot assigned, no row write -> its contribution
        # silently maps to the zero row).
        bundles = {e: self._incr_bundles[e] for e in exp
                   if e in self._incr_bundles}
        for e, slot in new_slots:
            if slot is None:
                # |exp| <= n guarantees a free slot for every changed
                # expert (survivors <= |exp|), so this is unreachable;
                # guard anyway: drop the expert rather than corrupt a slot.
                continue
            b = self._pinned.get(e)
            if b is None:
                b = self.shared_cache.get((self.layer_idx, e))
                if b is not None:
                    with self._lock:
                        self._stats["stage_cache_hits"] += 1
            if b is None:
                b = self._build(e)
                if not self._incr_writeback:
                    # eager mode: share the fresh build via the LRU
                    self.shared_cache.put((self.layer_idx, e), b)
                with self._lock:
                    self._stats["stage_builds"] += 1
            bundles[e] = b
            occ[slot] = e
            slot_of_list[e] = slot
            writes.append((slot, b))
        # Evicted experts: table back to the zero overflow slot.
        evicted = [e for e in self._incr_bundles if e not in exp_set]
        if self._incr_writeback:
            # Write-back LRU: while an expert is staged its canonical bundle
            # lives ONLY in _incr_bundles (the persistent stack rows are the
            # math copy) — no LRU duplication of the staged set. On eviction
            # from the stack the bundle is written back so returning experts
            # still hit the LRU.
            for e in evicted:
                self.shared_cache.put((self.layer_idx, e),
                                      self._incr_bundles[e])
        self._incr_bundles = bundles

        # ---- in-place delta writes (main thread, previous graph already
        # evaluated: stage runs at the step boundary in sync mode) ----
        for slot, b in writes:
            for proj in self._bundle_projs:
                for part in ("weight", "scales", "biases"):
                    self._incr_tensors[(proj, part)][slot] = b[(proj, part)]
        tbl = self._incr_slot_table
        for e in evicted:
            tbl[e] = n
        for e, slot in new_slots:
            if slot is not None:
                tbl[e] = slot
        # FIX(2026-09-02): force-evaluate the slot table after in-place
        # mutations so the consuming forward's lazy graph (core.take /
        # _moe_math) reads the UPDATED mapping — without this, near-100%
        # slot churn reads stale slot tables and the output collapses.
        if writes or evicted or new_slots:
            core.eval(self._incr_slot_table)
        with self._lock:
            self._stats["incr_writes"] += len(writes)
            self._stats["stage_delta_hits"] += len(exp) - len(writes)
            self._stats["stage_wall"] += time.perf_counter() - _t0

        # Compatibility state: bundles tuple ordered like the old resolver
        # (sync_actuals promotion + fallback paths read it).
        return (tuple(bundles[e] for e in exp), slot_of_list, exp)

    def stage_experts(self, experts):
        """Async-fill the NEXT slot set for the following decode step.

        Writes to `_staged_state_next` (never the active `_staged_state`), so
        the fill for token t+1 can run during token t without disturbing the
        set token t is reading. swap_staged() promotes it at the step
        boundary. If the fill cannot land in time, the consuming __call__
        falls back to the exact per-layer path (staged_fallback).
        """
        if not self._staged_mode:
            return
        unique = sorted({int(e) for e in experts})
        if not unique:
            return
        self._drain_rearm()
        with self._staging_lock:
            if self._staging_future is not None:
                self._staged_pending = unique
                return
            if self._staged_sync:
                if self._incr_mode:
                    self._staged_state_next = self._stage_incr(unique)
                else:
                    bundles, slot_of, exp, wall = self._build_staged_bundles(
                        unique)
                    self._staged_state_next = (tuple(bundles), slot_of, exp)
                    with self._lock:
                        self._stats["stage_wall"] += wall
                return
            self._submit_stage_locked(unique)

    def wait_staged(self):
        """Block until the in-flight staged fill lands, then promote it to
        the active state (ling stream_experts.py:441-455 semantics: the
        fill's completion IS the promotion — no separate blocking
        step-boundary swap; each layer waits lazily at consume time so the
        pool's SSD builds overlap the forward's graph building)."""
        self._drain_rearm()
        with self._staging_lock:
            fut = self._staging_future
        if fut is not None:
            t0 = time.perf_counter()
            try:
                fut.result()
            except Exception:
                pass
            with self._lock:
                self._stats["stage_wait_ms"] += (
                    time.perf_counter() - t0) * 1000
        with self._staging_lock:
            if self._staged_state_next is not None:
                self._staged_state = self._staged_state_next
                self._staged_state_next = None

    def swap_staged(self):
        """Promote the filled next set to active at the step boundary.

        Blocks until the in-flight fill lands (the exposed remainder of the
        hidden fill), then promotes. Layers with no pending fill keep their
        current active set (or None on the very first step)."""
        self.wait_staged()
        with self._staging_lock:
            if self._staged_state_next is not None:
                self._staged_state = self._staged_state_next
                self._staged_state_next = None
            self._staged_pending = None

    def _submit_stage_locked(self, unique):
        try:
            fut = self._pool.submit(self._build_staged_bundles, unique)
        except RuntimeError:
            self._staging_future = None
            return
        self._staging_future = fut
        # If the future is ALREADY done (idle pool + tiny build), the done
        # callback fires synchronously INSIDE this call — but we hold
        # _staging_lock here and _on_stage_done needs it => deadlock.
        # Fix: only attach the callback while still holding the lock if the
        # future is pending; for an already-done one, detach-then-handle
        # after release via _staged_rearm (picked up at the next entry).
        if not fut.done():
            fut.add_done_callback(self._on_stage_done)
            return
        self._staged_rearm = fut

    def _drain_rearm(self):
        """Handle a synchronously-completed fill outside _staging_lock."""
        fut = getattr(self, "_staged_rearm", None)
        if fut is None:
            return
        self._staged_rearm = None
        self._on_stage_done(fut)

    def _on_stage_done(self, fut):
        try:
            arrays, slot_of, exp, wall = fut.result()
        except Exception:
            with self._staging_lock:
                self._staging_future = None
            return
        with self._staging_lock:
            self._staged_state_next = (arrays, slot_of, exp)
            self._staging_future = None
            with self._lock:
                self._stats["stage_wall"] += wall
            if self._staged_pending:
                p = self._staged_pending
                self._staged_pending = None
                self._submit_stage_locked(p)

    def sync_actuals(self):
        """After a decode step: record this step's actual router indices so
        they can be staged for the next step (the "current router as the
        prerouter" substitution). No correctness handling: missing experts are
        simply dropped by the staged path."""
        idx = self._pending_indices
        if idx is None:
            return
        try:
            flat = idx.reshape(-1).tolist()
        except Exception:
            return
        self._pending_indices = None
        unique = sorted(set(int(v) for v in flat))
        if not unique:
            return
        with self._lock:
            self.last_used = unique
            for e in unique:
                self._hot_counts[e] = (
                    self._hot_counts.get(e, 0.0) * self.hot_decay + 1.0)
            used_exp = self._staged_at_use
            if used_exp is not None:
                dropped = [e for e in unique if e not in used_exp]
                self._stats["staged_dropped"] += len(dropped)
            # Promote bundles the router actually used into the shared LRU
            # so they survive beyond the staging state. Skipped in
            # write-back mode: staged bundles enter the LRU only on stack
            # eviction (avoiding staged-set duplication).
            if self._incr_writeback:
                return
            st = self._staged_state
            if st is not None:
                sbundles, _, sexp = st
                smap = {e: sbundles[i] for i, e in enumerate(sexp)}
                for e in unique:
                    if e in self._pinned:
                        continue
                    b = smap.get(e)
                    if b is not None:
                        self.shared_cache.put((self.layer_idx, e), b)

    def stage_from_prefill(self):
        """After prefill: stage the last prefill token's actual top-k so the
        first decode step is already staged (router used as the predictor)."""
        if not self._staged_mode:
            return
        topk = getattr(self, "_last_prefill_topk", None)
        if not topk:
            return
        unique = sorted(set(int(v) for v in topk))
        if unique:
            self.last_used = unique
            self.stage_experts(unique)

    def _refresh_hot_pins(self):
        if self.hot_per_layer <= 0:
            return
        cands = set(self._hot_counts) | set(self.predicted_experts)
        scores = {e: self._hot_counts.get(e, 0.0) for e in cands}
        for e in self.predicted_experts:
            scores[e] = scores.get(e, 0.0) + self._pin_bonus
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top = top[:self.hot_per_layer]
        target = {e for e, _ in top}
        with self._lock:
            self._pinned = {e: t for e, t in self._pinned.items()
                            if e in target}
            for e in target:
                cached = self.shared_cache.pop((self.layer_idx, e))
                if cached is not None:
                    self._pinned[e] = cached
                elif e not in self._pinned:
                    self._pinned[e] = self._build(e)
            self._hot_target = target

    def refresh_hot_pins(self):
        if self.hot_per_layer <= 0:
            return
        try:
            self._refresh_hot_pins()
        except Exception:  # best-effort
            pass

    def stats(self):
        with self._lock:
            s = dict(self._stats)
            s["pinned"] = len(self._pinned)
        s["shared_size"] = self.shared_cache.size()
        s["calls"] = self._calls
        return s

    # ---- whole-layer prefill (load full layer once, then drop) ------------

    def load_full_layer(self) -> None:
        if self._full_weights is not None:
            return
        # The checkpoint stores experts ALREADY STACKED per layer
        # (switch_mlp.gate_proj.weight [256,512,256], one tensor) — so
        # this is 9 direct whole-tensor loads, NOT 256x9 per-expert
        # builds. The core.array.view bitcast makes it ~9ms/layer; with
        # before_layer_cb + async_eval_per_layer, the CPU load hides
        # under the previous layer's GPU execution.
        w = {}

        def _read_full(proj, part):
            name = f"{self._prefix}.{proj}.{part}"
            raw = self._shard_for(name).raw(name)
            return raw

        for proj in self._bundle_projs:
            for part in ("weight", "scales", "biases"):
                if self._fuse_gu and proj == "gate_up_proj":
                    # Fused rows = gate rows on top of up rows PER EXPERT
                    # (the [E, 2*INTER, in] layout _build produces): the
                    # shard stores gate/up as two separate [E, ...]
                    # tensors, so interleave per-expert slices — a flat
                    # concatenate of both tensors would misalign rows.
                    raw_g = _read_full("gate_proj", part)
                    raw_u = _read_full("up_proj", part)
                    per = raw_g.size // self.num_experts
                    shape = self._bundle_shape[(proj, part)]
                    rows = np.concatenate([
                        np.concatenate([
                            raw_g[e * per:(e + 1) * per],
                            raw_u[e * per:(e + 1) * per]])
                        for e in range(self.num_experts)])
                    if part == "weight":
                        arr = core.array(u32_view(rows, shape))
                    else:
                        arr = core.array(rows.view("<u2")).view(
                            core.bfloat16).reshape(shape)
                else:
                    raw = _read_full(proj, part)
                    shape = self._bundle_shape[(proj, part)]
                    if part == "weight":
                        arr = core.array(u32_view(raw, shape))
                    else:
                        arr = core.array(raw.view("<u2")).view(
                            core.bfloat16).reshape(shape)
                w[(proj, part)] = arr
        self._full_weights = w

    def clear_full_layer(self) -> None:
        self._full_weights = None

    # ---- hot-layer prefill (LRU-resident partial stack) -------------------
    # Middle state between on-demand (re-reads the union every request)
    # and whole-layer E3b (thrashes the page cache): keep the layer's
    # top-N most frequent experts as a resident stacked tensor; misses go
    # through the exact per-expert path and are scatter-added.

    def load_hot_layer(self, n_hot: int = 128) -> None:
        """Load/refresh the layer's hot-expert weights.

        Backing store = one NUMPY array per (proj, part) holding the
        stacked rows of the top-n_hot experts (page cache, survives across
        requests, counts ZERO toward MLX active). The mx window
        (materialize_hot) exposes a few layers at a time so peak MLX
        memory stays at window_size * n_hot * ~1.2MB instead of
        40 * n_hot * 1.2MB.

        Selection = top-n_hot by _hot_counts (decayed frequency). The
        backing is rebuilt only when membership changes by more than
        12.5%; small churn is absorbed by the exact miss path."""
        counts = self._hot_counts
        if counts:
            hot = sorted(counts, key=lambda e: counts[e], reverse=True)
            hot = hot[:n_hot]
        else:
            hot = list(range(n_hot))
        key = tuple(hot)
        if self._hot_backing is not None:
            if self._hot_key == key:
                return
            old_set = set(self._hot_key)
            changed = len([e for e in key if e not in old_set])
            if changed <= max(1, n_hot // 8):
                return
        backing = {}
        for proj in self._bundle_projs:
            for part in ("weight", "scales", "biases"):
                if self._fuse_gu and proj == "gate_up_proj":
                    name_g = f"{self._prefix}.gate_proj.{part}"
                    name_u = f"{self._prefix}.up_proj.{part}"
                    raw_g = self._shard_for(name_g).raw(name_g)
                    raw_u = self._shard_for(name_u).raw(name_u)
                    per = raw_g.size // self.num_experts
                    rows = []
                    for e in hot:
                        rows.append(np.concatenate([
                            raw_g[e * per:(e + 1) * per],
                            raw_u[e * per:(e + 1) * per]]))
                else:
                    name = f"{self._prefix}.{proj}.{part}"
                    raw = self._shard_for(name).raw(name)
                    per = raw.size // self.num_experts
                    rows = [raw[e * per:(e + 1) * per] for e in hot]
                # concatenated numpy backing (page cache, not GPU), plus
                # one all-zero row: the prefill path sends misses to row
                # n_hot ("overflow row") and needs it to contribute zero.
                # Without it that gather reads past the end of the stack,
                # which only happened to be zero-filled memory.
                rows.append(np.zeros_like(rows[0]))
                backing[(proj, part)] = np.concatenate(rows)
        self._hot_backing = backing
        self._hot_key = key
        self._hot_set = set(hot)
        # invalidate any materialized mx window content
        self._hot_weights = None

    def materialize_hot(self) -> None:
        """Materialize the numpy backing into mx arrays (GPU) for this
        layer. Called by the engine's before_layer_cb under the sliding
        window; the engine dematerializes layers outside the window."""
        if self._hot_weights is not None or self._hot_backing is None:
            return
        w = {}
        shape0 = self._bundle_shape
        n_e = len(self._hot_key) + 1                 # + zero overflow row
        for (proj, part), buf in self._hot_backing.items():
            shape = shape0[(proj, part)]
            if part == "weight":
                # uint8 bytes -> u32 view, then [N, ...] stack
                arr = core.array(buf.view("<u4")).reshape(
                    (n_e,) + shape[1:])
            else:
                # uint8 bytes -> bf16 bitcast view, then [N, ...] stack
                arr = core.array(buf.view("<u2")).view(
                    core.bfloat16).reshape((n_e,) + shape[1:])
            w[(proj, part)] = arr
        self._hot_weights = w

    def dematerialize_hot(self) -> None:
        """Release the GPU copy; the numpy backing survives."""
        self._hot_weights = None

    def clear_hot_layer(self) -> None:
        self._hot_weights = None
        self._hot_backing = None
        self._hot_key = None
        self._hot_set = None

    # ---- forward ---------------------------------------------------------

    def _gather_gate_up(self, x, w, local, sorted_indices):
        """Up/gate projections over a stacked weight dict, fused or not."""
        if self._fuse_gu:
            x_gu = quant.gather_qmm(
                x, w[("gate_up_proj", "weight")],
                w[("gate_up_proj", "scales")],
                w[("gate_up_proj", "biases")], rhs_indices=local,
                transpose=True, group_size=self.group_size,
                bits=self.bits, mode=self.mode,
                sorted_indices=sorted_indices)
            x_gate, x_up = core.split(x_gu, 2, axis=-1)
            return x_up, x_gate
        x_up = quant.gather_qmm(
            x, w[("up_proj", "weight")], w[("up_proj", "scales")],
            w[("up_proj", "biases")], rhs_indices=local, transpose=True,
            group_size=self.group_size, bits=self.bits, mode=self.mode,
            sorted_indices=sorted_indices)
        x_gate = quant.gather_qmm(
            x, w[("gate_proj", "weight")], w[("gate_proj", "scales")],
            w[("gate_proj", "biases")], rhs_indices=local, transpose=True,
            group_size=self.group_size, bits=self.bits, mode=self.mode,
            sorted_indices=sorted_indices)
        return x_up, x_gate

    def __call__(self, x: core.array, indices: core.array | None = None) -> core.array:
        self._calls += 1
        # Record the router indices regardless of which path runs below, so
        # sync_actuals() can stage the next step's slots even after
        # fallbacks. In real-prerouter mode there are no router indices (the
        # staged set is the routing), so sync_actuals records the staged set.
        self._pending_indices = indices
        if self._calls % self.hot_update_interval == 0:
            self._refresh_hot_pins()

        # HOT-layer prefill path (multi-token chunks only): indices remapped
        # to the resident top-N stack; misses (~3%) contribute zero from the
        # fast gather (overflow row) and are computed separately below via
        # the exact per-expert path, then scatter-added. Numerically exact.
        if (self._hot_weights is not None and core.size(indices) > 8
                and self._full_weights is None):
            with self._lock:
                flat = indices.reshape(-1).tolist()
                for e in set(int(v) for v in flat):
                    self._hot_counts[e] = (
                        self._hot_counts.get(e, 0.0) * self.hot_decay + 1.0)
                if self._staged_mode:
                    k = indices.shape[-1]
                    self._last_prefill_topk = flat[-k:]
            w = self._hot_weights
            pos = {e: i for i, e in enumerate(self._hot_key)}
            n_rows = len(self._hot_key)
            # normalize shapes: x may arrive [1, T, H] (batched) — flatten
            # to [T, H]; indices likewise [1, T, k] -> [T, k].
            x_flat = x.reshape(-1, x.shape[-1])
            ind_flat = indices.reshape(-1, indices.shape[-1])
            tok_n = ind_flat.shape[0]
            k_n = ind_flat.shape[-1]
            flat = ind_flat.reshape(-1).tolist()
            flat_idx = [int(v) for v in flat]
            g_list = []
            miss_rows = []      # flat positions needing exact add
            for i, e in enumerate(flat_idx):
                p = pos.get(e)
                if p is None:
                    g_list.append(n_rows)   # overflow zero row
                    miss_rows.append(i)
                else:
                    g_list.append(p)
            remapped = core.array(g_list, dtype=core.int32).reshape(
                ind_flat.shape)
            x3 = core.expand_dims(x_flat, (-2, -3))
            xs, local, inv_order = quant.gather_sort(x3, remapped)
            x_up, x_gate = self._gather_gate_up(xs, w, local, True)
            z = quant.gather_qmm(
                _swiglu(x_up, x_gate),
                w[("down_proj", "weight")], w[("down_proj", "scales")],
                w[("down_proj", "biases")], rhs_indices=local, transpose=True,
                group_size=self.group_size, bits=self.bits, mode=self.mode,
                sorted_indices=True)
            out = quant.scatter_unsort(z, inv_order)
            # Miss contribution: switch_mlp's caller applies scores after
            # us, so only RAW per-expert outputs [T, k, H] at miss slots are
            # needed. Reuse _moe_math (same kernel as the staged path) with
            # a single-expert "stack" built from the bundle.
            if miss_rows:
                miss_pairs = [(i // k_n, i % k_n, flat_idx[i])
                              for i in miss_rows]
                by_expert = {}
                for t, kk, e in miss_pairs:
                    by_expert.setdefault(e, []).append((t, kk))
                for e, tk in by_expert.items():
                    b = self._build(e)
                    rows = core.array([t for t, _ in tk], dtype=core.int32)
                    xe = core.take(x_flat, rows, axis=0)     # [m, H]
                    xe2 = core.expand_dims(xe, (-2, -3))     # [m, 1, 1, H]
                    # bundle tensors lack the expert dim -> add it once
                    # (view-like reshape, no copy of the 4bit payload)
                    def _e1(t):
                        return t.reshape((1,) + t.shape)
                    one = core.zeros((1,), dtype=core.int32)
                    if self._fuse_gu:
                        ggu = quant.gather_qmm(
                            xe2, _e1(b[("gate_up_proj", "weight")]),
                            _e1(b[("gate_up_proj", "scales")]),
                            _e1(b[("gate_up_proj", "biases")]),
                            rhs_indices=one, transpose=True,
                            group_size=self.group_size, bits=self.bits,
                            mode=self.mode, sorted_indices=True)
                        gg, gu = core.split(ggu, 2, axis=-1)
                    else:
                        gu = quant.gather_qmm(
                            xe2, _e1(b[("up_proj", "weight")]),
                            _e1(b[("up_proj", "scales")]),
                            _e1(b[("up_proj", "biases")]),
                            rhs_indices=one, transpose=True,
                            group_size=self.group_size, bits=self.bits,
                            mode=self.mode, sorted_indices=True)
                        gg = quant.gather_qmm(
                            xe2, _e1(b[("gate_proj", "weight")]),
                            _e1(b[("gate_proj", "scales")]),
                            _e1(b[("gate_proj", "biases")]),
                            rhs_indices=one, transpose=True,
                            group_size=self.group_size, bits=self.bits,
                            mode=self.mode, sorted_indices=True)
                    ze = quant.gather_qmm(
                        _swiglu(gu, gg),
                        _e1(b[("down_proj", "weight")]),
                        _e1(b[("down_proj", "scales")]),
                        _e1(b[("down_proj", "biases")]),
                        rhs_indices=one, transpose=True,
                        group_size=self.group_size, bits=self.bits,
                        mode=self.mode, sorted_indices=True
                    )                                        # [m, 1, 1, H]
                    ze = ze.squeeze(-2).squeeze(-2)          # [m, H]
                    flat_pos = core.array(
                        [t * k_n + kk for t, kk in tk], dtype=core.int32)
                    out = core.index_add(out, flat_pos, ze[:, None, :])
            out = out.reshape(*ind_flat.shape, *out.shape[-2:])
            out = out.squeeze(-2)                    # [T, k, H]
            if x.ndim == 3:                          # restore batch dim
                out = out.reshape(x.shape[0], tok_n, k_n, -1)
            return out

        # Whole-layer prefill path (multi-token chunks only): the full
        # stacked weights are already loaded (by the engine's
        # before_layer_cb load_full_layer / clear_full_layer), so the router
        # indices go straight into the sorted gather — same ops as the exact
        # path (bit-identical), no tolist / no bundles / no remap.
        if self._full_weights is not None and core.size(indices) > 8:
            # Record usage so hot pins survive prefill (the whole-layer path
            # never touches _get_bundles, so without this _hot_counts stays
            # empty and PREFILL_PIN has nothing to select from). Also keep
            # the last token's top-k so the first decode step can be staged.
            with self._lock:
                flat = indices.reshape(-1).tolist()
                unique = set(int(v) for v in flat)
                for e in unique:
                    self._hot_counts[e] = (
                        self._hot_counts.get(e, 0.0) * self.hot_decay + 1.0)
                if self._staged_mode:
                    k = indices.shape[-1]
                    self._last_prefill_topk = flat[-k:]
            w = self._full_weights
            x = core.expand_dims(x, (-2, -3))
            x, local, inv_order = quant.gather_sort(x, indices)
            x_up, x_gate = self._gather_gate_up(x, w, local, True)
            z = quant.gather_qmm(
                _swiglu(x_up, x_gate),
                w[("down_proj", "weight")], w[("down_proj", "scales")],
                w[("down_proj", "biases")], rhs_indices=local, transpose=True,
                group_size=self.group_size, bits=self.bits, mode=self.mode,
                sorted_indices=True)
            # Matches the resident SwitchGLU contract: 4D [B, T, k, H] (the
            # caller's SparseMoeBlock sums over axis -2).  NO batch restore
            # here — gather_sort/scatter_unsort already keep [B, T, k, H]
            # (verified against the deployment's whole-layer path, which
            # returns the same 4D shape).
            return quant.scatter_unsort(z, inv_order, indices.shape).squeeze(-2)

        # Staged decode path (prerouter-style, single-token): consume the
        # fixed staged slots; router indices never leave the GPU (expert->slot
        # via core.take). Missing experts map to the overflow zero slot and
        # their contribution is dropped. Falls back to the exact path only if
        # the slots are not ready yet.

        def _build_asm(bundles, slot_of_list, exp_key):
            slot_of = self._slot_table_cache.get(exp_key)
            if slot_of is None:
                slot_of = core.array(slot_of_list, dtype=core.int32)
                core.eval(slot_of)
                if len(self._slot_table_cache) > 8:
                    self._slot_table_cache.pop(next(
                        iter(self._slot_table_cache)))
                self._slot_table_cache[exp_key] = slot_of
            # Stack on the main thread (MLX lazy-op construction must not
            # race eval). Order matches the math fn's signature directly:
            # fused (w_gu, s_gu, b_gu, w_d, s_d, b_d); unfused
            # (w_u, s_u, b_u, w_g, s_g, b_g, w_d, s_d, b_d).
            wargs = []
            for proj in self._bundle_projs:
                for part in ("weight", "scales", "biases"):
                    rows = [b[(proj, part)] for b in bundles]
                    rows.extend([self._zero_slot[(proj, part)]] * (
                        self._staged_n + 1 - len(rows)))
                    wargs.append(core.stack(rows))
            return slot_of, tuple(wargs)

        # Real-prerouter routing: no router indices, route through the WHOLE
        # staged set (routing == prediction, no drops).
        if self._staged_replace and indices is None:
            self.wait_staged()
            st = self._staged_state
            if st is None:
                with self._lock:
                    self._stats["staged_fallback"] += 1
                self._staged_at_use = None
                # First-step guard: zero contribution (nothing staged yet).
                return core.expand_dims(core.zeros_like(x), -2)
            bundles, slot_of_list, staged_exp = st
            self._staged_at_use = staged_exp
            self.last_used = list(staged_exp)
            exp_key = tuple(staged_exp)
            asm = self._staged_asm_cache.get(exp_key)
            if asm is None:
                slot_of, wargs = _build_asm(bundles, slot_of_list, exp_key)
                if len(self._staged_asm_cache) > 1:
                    self._staged_asm_cache.pop(next(
                        iter(self._staged_asm_cache)))
                self._staged_asm_cache[exp_key] = (slot_of, wargs)
            else:
                slot_of, wargs = asm
            k = len(staged_exp)
            local2d = core.arange(k, dtype=core.int32)[None, None, :]
            with self._lock:
                self._stats["staged_used"] += k
            return self._moe_math(x, *wargs, local2d)

        if self._staged_mode and indices is not None and core.size(indices) == self._staged_trigger:
            self.wait_staged()
            st = self._staged_state
            if st is None:
                with self._lock:
                    self._stats["staged_fallback"] += 1
                self._staged_at_use = None
            elif self._incr_mode and self._incr_tensors is not None:
                # Incremental stack: the persistent tensors were updated
                # in place at stage time (sticky slots); consume them
                # directly — no core.stack, no asm cache. Order matches the
                # math fn's signature exactly as _build_asm produces it.
                _, _, staged_exp = st
                self._staged_at_use = staged_exp
                wargs = tuple(
                    self._incr_tensors[(proj, part)]
                    for proj in self._bundle_projs
                    for part in ("weight", "scales", "biases"))
                local2d = core.take(self._incr_slot_table, indices)
                with self._lock:
                    self._stats["staged_used"] += core.size(indices)
                return self._moe_math(x, *wargs, local2d)
            else:
                bundles, slot_of_list, staged_exp = st
                self._staged_at_use = staged_exp
                exp_key = tuple(staged_exp)
                asm = self._staged_asm_cache.get(exp_key)
                if asm is None:
                    slot_of, wargs = _build_asm(
                        bundles, slot_of_list, exp_key)
                    if len(self._staged_asm_cache) > 1:
                        self._staged_asm_cache.pop(next(
                            iter(self._staged_asm_cache)))
                    self._staged_asm_cache[exp_key] = (slot_of, wargs)
                else:
                    slot_of, wargs = asm
                local2d = core.take(slot_of, indices)
                with self._lock:
                    self._stats["staged_used"] += core.size(indices)
                return self._moe_math(x, *wargs, local2d)

        # Exact decode path: resolve bundles for unique experts.
        _t0 = time.perf_counter()
        flat = indices.reshape(-1)
        unique = sorted(set(int(v) for v in flat.tolist()))
        with self._lock:
            self.last_used = unique
        _t1 = time.perf_counter()
        bundles = self._get_bundles(unique)
        _t2 = time.perf_counter()
        remap = {e: i for i, e in enumerate(unique)}
        local2d = core.array(
            [remap[e] for e in flat.tolist()]
        ).reshape(indices.shape)
        _t3 = time.perf_counter()

        vals = list(bundles.values())
        wargs = []
        for proj in self._bundle_projs:
            for part in ("weight", "scales", "biases"):
                wargs.append(core.stack([b[(proj, part)] for b in vals]))

        orig_x = x
        x = core.expand_dims(x, (-2, -3))
        do_sort = core.size(local2d) >= 64
        inv_order = None
        if do_sort:
            if self._use_compile:
                x_s, local, inv_order = quant.gather_sort(x, local2d)
                return self._moe_math_sorted(
                    x_s, *wargs, local, inv_order, indices.shape)
            x, local, inv_order = quant.gather_sort(x, local2d)
        elif self._use_compile:
            return self._moe_math(orig_x, *wargs, local2d)
        else:
            local = local2d

        if self._fuse_gu:
            w_gu, s_gu, b_gu = wargs[0], wargs[1], wargs[2]
            w_d, s_d, b_d = wargs[3], wargs[4], wargs[5]
            x_gu = quant.gather_qmm(
                x, w_gu, s_gu, b_gu, rhs_indices=local,
                transpose=True, group_size=self.group_size, bits=self.bits,
                mode=self.mode, sorted_indices=do_sort)
            x_gate, x_up = core.split(x_gu, 2, axis=-1)
        else:
            # wargs order matches the math contract: up first, then gate,
            # then down (see spec.bundle_projs).
            w_u, s_u, b_u = wargs[0], wargs[1], wargs[2]
            w_g, s_g, b_g = wargs[3], wargs[4], wargs[5]
            w_d, s_d, b_d = wargs[6], wargs[7], wargs[8]
            x_up = quant.gather_qmm(
                x, w_u, s_u, b_u, rhs_indices=local,
                transpose=True, group_size=self.group_size, bits=self.bits,
                mode=self.mode, sorted_indices=do_sort)
            x_gate = quant.gather_qmm(
                x, w_g, s_g, b_g, rhs_indices=local,
                transpose=True, group_size=self.group_size, bits=self.bits,
                mode=self.mode, sorted_indices=do_sort)
        z = quant.gather_qmm(
            _swiglu(x_up, x_gate), w_d, s_d, b_d, rhs_indices=local,
            transpose=True, group_size=self.group_size, bits=self.bits,
            mode=self.mode, sorted_indices=do_sort)
        if do_sort:
            z = quant.scatter_unsort(z, inv_order, indices.shape)
        return z.squeeze(-2)

    # ---- lifecycle --------------------------------------------------------

    def reset(self):
        """Per-request reset: drop staged/prefill working state so the next
        request starts clean (the shared LRU, hot counts, and pools survive
        across requests — they are the cross-request memory)."""
        self._pending_indices = None
        self._staged_at_use = None
        self._last_prefill_topk = None
        self._staged_state = None
        self._staged_state_next = None
        with self._staging_lock:
            self._staged_pending = None
        self.last_used = []
        self._full_weights = None

    def close(self):
        """Release thread pools (idempotent)."""
        for pool in (self._pool, self._prefetch_pool):
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
