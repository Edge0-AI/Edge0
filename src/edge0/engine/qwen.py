"""edge0-35b engine: Qwen3.5-MoE (K=4 tier).

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
    """Streaming Qwen3.5-MoE engine (staged decode + trained prerouter)."""

    name = "edge0-35b"

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
        before_cb = self._prefill_before_layer if prefill_multi else None
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
        if (self._pg_stager is not None and not self._prefill_active
                and len(ids) == 1):
            self._pg_stager.stage_all()
            self._pg_state.swap()
        return logits

    # ---- step hooks -------------------------------------------------------

    def _step_pre(self, token_id: int) -> None:
        if self._pg_stager is not None:
            # Router layers (no owner): fill from their previous actuals.
            st = self._pg_state
            for li, exp in self._all_stream_layers.items():
                if li < st.start or li >= st.n - 1:
                    if exp.last_used:
                        exp.stage_experts(list(exp.last_used))
        elif self._history_prefetch is not None:
            self._history_prefetch()

    def _step_post(self, token_id: int) -> None:
        if self._pg_stager is not None:
            st = self._pg_state
            for li, exp in self._all_stream_layers.items():
                if li < st.start or li >= st.n - 1:
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
            for exp in self._stream_layers.values():
                exp.stage_from_prefill()

    def _reset_state(self) -> None:
        self.cache = self._lm.make_cache()
        if self._pg_state is not None:
            self._pg_state.reset()

    def _lm_logits(self, h: core.array) -> core.array:
        return self._lm.lm_head(h[0, -1])
