"""Prefill / decode callback factories.

All callbacks are ports of the deployment engines' proven hooks
(``engine_qwen.py`` / ling ``engine.py``); each is a pure function
building a callback from a ``dict[int, StreamingSwitchGLU]`` so the
same machinery serves both model families.
"""

from __future__ import annotations


def make_prefill_before_layer(all_stream_layers, *, full_n: int = 0,
                              hot_n: int = 0, hot_window: int = 4):
    """Before-layer prefill hook: whole-layer load-drop (E3b) + the
    sliding hot-expert window.

    Before layer ``li`` runs: drop layer ``li-1``'s whole-layer set (its
    async_eval was already submitted, so the GPU queue keeps the arrays
    alive until evaluated), then either load layer ``li``'s full stacked
    tensors (``full_n`` unset / ``li < full_n``) or build the hot-stack
    window (``hot_n``): numpy backing for layers ``li..li+ahead``,
    GPU materialization for the window, dematerialize trailing layers.
    """
    w = max(1, hot_window)
    ahead = max(1, w // 2)

    def before_layer(li: int) -> None:
        exp_old = all_stream_layers.get(li - 1)
        if exp_old is not None:
            exp_old.clear_full_layer()
        if full_n and li >= full_n:
            return
        if hot_n:
            for lj in range(li, min(li + ahead + 1,
                                    len(all_stream_layers))):
                exp_w = all_stream_layers.get(lj)
                if exp_w is None:
                    continue
                exp_w.load_hot_layer(hot_n)   # numpy backing (cheap)
                exp_w.materialize_hot()       # GPU window
            for lj in range(0, max(0, li - (w - ahead))):
                exp_t = all_stream_layers.get(lj)
                if exp_t is not None:
                    exp_t.dematerialize_hot()
            if not full_n:
                return
        exp = all_stream_layers.get(li)
        if exp is not None:
            exp.load_full_layer()

    return before_layer


def make_intra_after_layer(all_stream_layers, enabled: bool):
    """Intra-step staging: right after layer ``li`` runs, submit the next
    step's layer-``li`` fill from ``last_used`` (the router's actuals from
    the previous step, already on the CPU — no mid-forward tolist, which
    segfaults in mlx 0.30.4 when a pool thread forces a GPU sync while the
    main thread is still building the graph)."""
    if not enabled:
        return None

    def after_layer(li: int, hidden_states=None) -> None:
        exp = all_stream_layers.get(li)
        if exp is None or not exp.last_used:
            return
        exp.stage_experts(list(exp.last_used))

    return after_layer


def make_history_prefetch(all_stream_layers, enabled: bool):
    """Zero-cost decode prefetch: submit each layer's previous-token
    expert set (``last_used``) so the SSD loads overlap this step's GPU
    compute.  The no-prerouter replacement for next-token prediction —
    exploits adjacent-token expert locality (~0.4 overlap measured)."""
    if not enabled:
        return None

    def history_prefetch() -> None:
        for li in range(1, len(all_stream_layers)):
            e = all_stream_layers.get(li)
            if e is None or not e.last_used:
                continue
            e.prefetch(list(e.last_used))

    return history_prefetch
