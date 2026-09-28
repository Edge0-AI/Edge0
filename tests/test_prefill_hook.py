"""The prefill hook must honor ``full_layer=False``.

Regression guard for a silent no-op: ``make_prefill_before_layer`` used
``full_n=0`` to mean "every layer" *and* as the falsy "unset" value, so a
hook built with ``full_n=0`` fell straight through to
``load_full_layer()``.  Because the engines installed the hook for every
multi-token prefill regardless of ``full_layer_prefill``, that flag did
nothing on the tier that ships ``prefill_hot=0``: disabling it still
streamed all 128 experts of all 23 layers (~4.1 GiB for a 27-token
prompt) instead of the routed experts only (~0.4 GiB).
"""

from __future__ import annotations

from edge0.engine.hooks import make_prefill_before_layer


class _FakeLayer:
    def __init__(self) -> None:
        self.calls: list = []

    def clear_full_layer(self) -> None:
        self.calls.append("clear")

    def load_full_layer(self) -> None:
        self.calls.append("full")

    def load_hot_layer(self, n: int) -> None:
        self.calls.append(("hot_load", n))

    def materialize_hot(self) -> None:
        self.calls.append("hot_mat")

    def dematerialize_hot(self) -> None:
        self.calls.append("hot_dem")


def _run(n: int = 4, **kwargs):
    layers = {i: _FakeLayer() for i in range(n)}
    before_layer = make_prefill_before_layer(layers, **kwargs)
    for li in range(n):
        before_layer(li)
    return layers


def _full_loads(layers) -> list[int]:
    return sorted(li for li, exp in layers.items() if "full" in exp.calls)


def test_full_layer_disabled_never_loads_a_whole_layer():
    assert _full_loads(_run(full_layer=False)) == []


def test_full_layer_default_still_loads_every_layer():
    # full_n=0 means "every layer" (the prod_k8 default).
    assert _full_loads(_run()) == [0, 1, 2, 3]


def test_full_layer_honors_the_leading_layer_count():
    assert _full_loads(_run(full_layer=True, full_n=2)) == [0, 1]


def test_hot_window_survives_full_layer_disabled():
    layers = _run(full_layer=False, hot_n=8)
    assert _full_loads(layers) == []
    assert any("hot_mat" in exp.calls for exp in layers.values())


def test_hot_stack_layer_is_not_also_loaded_whole():
    # staged_k4 shape: no full layers, hot stack only.
    assert _full_loads(_run(full_layer=False, hot_n=32)) == []
