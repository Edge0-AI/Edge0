"""edge0-35b engine: Qwen3.6-35B-A3B (K=4 tier).

Port of the deployment's ``engine_qwen.py`` trained-prerouter path:

* prefill — E3b whole-layer load-drop (with the sliding hot-expert
  window), driven by the callback hooks added to the vendored
  ``qwen3_next.py``.
* decode — the prerouter head (class-level block patch) supplies the
  routing; at the step boundary ONE stacked head batch is evaluated,
  ONE tolist extracts every layer's next expert set, and the staged
  fills overlap the next forward (0 drops: staged set == routing set).
* router layers (< start, >= n-1) are staged from their actuals.
"""

from __future__ import annotations

import os

from edge0.backends import core

from edge0.backends.mlx._impl.qwen3_5_moe import Model as Qwen35Model
from edge0.backends.mlx._impl.qwen3_5_moe import ModelArgs as Qwen35Args
from edge0.backends import io
from edge0.backends.mlx.io import load_model, load_tokenizer, open_shards
from edge0.engine.base import Edge0Engine
from edge0.engine.hooks import (
    make_history_prefetch,
    make_intra_after_layer,
    make_prefill_before_layer,
)
from edge0.prerouter.install import install_prerouter
from edge0.prerouter.stager import CrossTokenStager
from edge0.streaming.install import install_streaming_experts


def _get_model_classes(config):
    """mlx-lm class hook: serve the vendored qwen3_5_moe backbone."""
    return Qwen35Model, Qwen35Args


def load_installed(model_dir: str, cfg):
    """Load the qwen skeleton and install streaming twins, LoRA and the
    trained prerouter (shared by ``build_model`` and ``build_engine``).

    Returns ``(model, model_config, shards, installs)`` where
    ``installs`` carries the layer maps and prerouter state the engine
    drives at the step boundary.
    """
    model, model_config = load_model(
        model_dir, lazy=True, strict=False,
        model_config={"model_type": "qwen3_5_moe"},
        get_model_classes=_get_model_classes)
    shards = open_shards(model_dir)
    spec = cfg.moe_spec
    opts = cfg.options
    # qwen config.json nests the text params under ``text_config``; mlx-lm
    # returns the raw dict, so resolve defensively.
    _tc = model_config.get("text_config", model_config)
    n_layers = int(_tc.get("num_hidden_layers", 40))
    installed = install_streaming_experts(
        model, shards, spec, options=opts, num_layers=n_layers)
    all_stream = {li: t for li, t in enumerate(installed) if t is not None}
    # Zero-drop layer scoping (quality invariant).  Staged slots are only
    # coherent for a layer whose ROUTE is a prerouter prediction (consumer
    # ``start_layer+1 .. n_layers-2``): the head's submission is exactly the
    # set that consumer itself re-selects from the same cached state, so
    # slots == routing and ``staged_dropped == 0``.  Every other layer routes
    # with its own gate, and slots filled from history (the previous token's
    # actuals) silently zero every routed expert outside that set -- measured
    # 3.0-3.75 of 4 per step for L0-6/L39, and the same class of misalignment
    # in the 8b tier.  Those layers run the exact path (``_step_pre`` /
    # ``_step_post``) instead.
    #
    # Switch-off is the same rule taken to its limit: with
    # ``--no-prerouter`` there is no prediction anywhere, so no layer keeps
    # slots and the off path is the exact path (gate routing, on-demand
    # loads), not history staging.  ``history_slots=True`` opts back into the
    # legacy behaviour on purpose (``--history-slots``).
    if not getattr(cfg, "history_slots", False):
        if cfg.prerouter is None:
            for t in all_stream.values():
                t._staged_mode = False
        else:
            lo, hi = cfg.prerouter.start_layer, n_layers - 2
            for li, t in all_stream.items():
                if not lo <= li <= hi:
                    t._staged_mode = False
    stream_layers = {li: t for li, t in all_stream.items() if t._staged_mode}

    if cfg.lora:
        from edge0.adapters.lora import install_lora
        install_lora(model, cfg.lora, r=cfg.lora_r, alpha=cfg.lora_alpha)

    pg_state = pg_stager = None
    if cfg.prerouter and cfg.prerouter.weights_file:
        pg_state, heads = install_prerouter(
            model=model, spec=spec, pspec=cfg.prerouter, n_layers=n_layers)
        pg_stager = CrossTokenStager(
            model=model, spec=spec, pspec=cfg.prerouter,
            state=pg_state, stream_layers=stream_layers,
            top_k=cfg.prerouter_top_k)
        print(f"[edge0-35b] prerouter installed: {len(heads)} heads, "
              f"start={cfg.prerouter.start_layer}, "
              f"K={cfg.prerouter_top_k}", flush=True)
    installs = dict(all_stream_layers=all_stream,
                    stream_layers=stream_layers,
                    pg_state=pg_state, pg_stager=pg_stager)
    return model, model_config, shards, installs


class Qwen35Engine(Edge0Engine):
    """Streaming Qwen3.6-35B-A3B engine (staged decode + trained prerouter)."""

    name = "edge0-35b"

    def __init__(self, model_dir: str, cfg, tokenizer=None):
        # NAN_BANG_COLLAPSE_FIX parity (engine/ling.py): hidden clip default
        # 1000 unless the deployer overrides QWEN_HIDDEN_CLIP explicitly.
        # One fp16 overflow inside a layer otherwise poisons the whole net
        # into all-NaN logits -> argmax fallback token 0 ('!') collapse,
        # which does not recover until the process restarts.
        if "QWEN_HIDDEN_CLIP" not in os.environ:
            os.environ["QWEN_HIDDEN_CLIP"] = "1000"
        super().__init__(model_dir, cfg, tokenizer=tokenizer)

    def _build(self):
        cfg = self.cfg
        (self.model, self.model_config, self.shards,
         inst) = load_installed(cfg.model_dir, cfg)
        self._all_stream_layers = inst["all_stream_layers"]
        self._stream_layers = inst["stream_layers"]
        self._pg_state = inst["pg_state"]
        self._pg_stager = inst["pg_stager"]
        self._lm = self.model.language_model
        opts = cfg.options
        if self._tok is None:
            try:
                self._tok = load_tokenizer(cfg.model_dir)
            except Exception:  # noqa: BLE001 — tokenizer optional for CLI
                pass

        self._prefill_before_layer = make_prefill_before_layer(
            self._all_stream_layers,
            full_layer=bool(getattr(opts, "full_layer_prefill", False)),
            full_n=getattr(opts, "prefill_full_layers", 0),
            hot_n=opts.prefill_hot, hot_window=cfg.hot_window)
        self._intra_after_layer = make_intra_after_layer(
            self._all_stream_layers, enabled=cfg.intra_staging)
        self._history_prefetch = make_history_prefetch(
            self._all_stream_layers, enabled=cfg.prefetch_history)

        self.cache = self._lm.make_cache()

    # ---- forward ----------------------------------------------------------

    def _forward(self, ids, intra_stage: bool = True) -> core.array:
        inputs = core.array(ids)[None, :]
        opts = self.cfg.options
        prefill_multi = len(ids) > 1 and self._prefill_active
        full_layer = bool(
            prefill_multi and self._all_stream_layers
            and opts.full_layer_prefill)
        before_cb = (self._prefill_before_layer
                     if (prefill_multi and (full_layer or opts.prefill_hot))
                     else None)
        if full_layer:
            after_cb = None
        else:
            after_cb = (self._intra_after_layer
                        if (intra_stage and self._intra_after_layer is not None
                            and len(ids) == 1)
                        else None)
        h = self._lm.model(
            inputs, cache=self.cache, before_layer_cb=before_cb,
            after_layer_cb=after_cb,
            async_eval_per_layer=bool(prefill_multi and full_layer))
        logits = self._lm.lm_head(h[0, -1])
        core.eval(logits)
        # Step boundary: ONE stacked head batch -> ONE tolist -> fills.
        # Demo parity (engine_qwen._forward: the deployment's pre-routing /
        # trained-state condition and not full_layer): stage_all+swap run
        # after EVERY forward, including prefill chunks.  During prefill the
        # per-layer features (block.prerouter_m_in/oh) are not captured (the
        # patch only fills them on single-token forwards), so stage_all
        # submits nothing and the first decode step consumes no prediction --
        # it routes with the router, the pos-0 fallback used in training.
        if (self._pg_stager is not None and not full_layer):
            self._pg_stager.stage_all()
            self._pg_state.swap()
        return logits

    # ---- step hooks -------------------------------------------------------

    def _step_pre(self, token_id: int) -> None:
        if self._pg_stager is not None:
            # Layers without an incoming prediction (L0-6, L39, and every
            # layer switched off by the history_slots scoping): they route
            # with their own gate this step, so a staged slot set filled from
            # history would silently zero every routed expert outside it.
            # They run the exact path -- staging them is the legacy
            # history_slots=True behaviour.  No prefetch either: the previous
            # token's set overlaps the next route by only ~6%, so the reads
            # are wasted (measured 11.7 -> 16.7 tok/s when removed).
            st = self._pg_state
            for li, exp in self._all_stream_layers.items():
                if li < st.start or li >= st.n - 1:
                    if exp._staged_mode and exp.last_used:
                        exp.stage_experts(list(exp.last_used))
        elif self._history_prefetch is not None:
            self._history_prefetch()

    def _step_post(self, token_id: int) -> None:
        if self._pg_stager is not None:
            st = self._pg_state
            for li, exp in self._all_stream_layers.items():
                if li < st.start or li >= st.n - 1:
                    # Non-staged layers stage nothing from their actuals, so
                    # the bookkeeping (and its per-layer host sync) is dead
                    # weight -- unless the layer keeps hot pins, whose counts
                    # it feeds.
                    if (exp._staged_mode
                            or getattr(exp, "hot_per_layer", 0) > 0):
                        exp.sync_actuals()
            return
        if self._stream_layers:
            for exp in self._all_stream_layers.values():
                exp.sync_actuals()
                exp.stage_experts(list(exp.last_used))
                exp.swap_staged()

    # ---- prefill / reset --------------------------------------------------

    def _prefill_end(self) -> None:
        for exp in self._all_stream_layers.values():
            exp.clear_full_layer()
        for exp in self._all_stream_layers.values():
            exp.refresh_hot_pins()
        if self._stream_layers:
            if self._pg_stager is None or getattr(self.cfg, "history_slots",
                                                  False):
                for exp in self._stream_layers.values():
                    exp.stage_from_prefill()
            else:
                # With a prerouter the first decode step routes with the
                # router (pos-0 fallback, as in training): staging the last
                # prefill token's actual top-k there zeroes every routed
                # expert outside that set (measured 4/4 for L7).  Prefetch
                # them instead; the step then runs the exact path.
                for exp in self._stream_layers.values():
                    exp.prefetch_from_prefill()

    def _reset_state(self) -> None:
        self.cache = self._lm.make_cache()
        if self._pg_state is not None:
            self._pg_state.reset()
            # Drop the per-block feature captures too.  The MoE patch writes
            # ``prerouter_m_in`` / ``prerouter_oh`` only on single-token
            # forwards and nothing else touches them, so without this the
            # NEXT request's prefill would re-run every head on the previous
            # request's last hidden state: its stage_all would submit stale
            # predictions that the first decode step then routes with (the
            # fresh-process behaviour is the gate fallback at pos 0).
            for owner in self._pg_state.owners:
                block = self.cfg.moe_spec.block_of(self.model, owner)
                block.prerouter_m_in = None
                block.prerouter_oh = None
        # Per-request reset of the streaming layers' staged state
        # (``StreamingSwitchGLU.reset``): a leftover ``_staged_state`` from
        # the previous request defeats the staged path's "fill not ready ->
        # exact path" fallback (``st`` would never be None) and would serve
        # the previous request's expert bundles.
        for exp in getattr(self, "_all_stream_layers", {}).values():
            exp.reset()

    def _lm_logits(self, h: core.array) -> core.array:
        return self._lm.lm_head(h[0, -1])
