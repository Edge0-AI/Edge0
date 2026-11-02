"""The shared prefill/generate loop's stop contract, against a scripted family
engine (no checkpoints)."""

from __future__ import annotations

from types import SimpleNamespace

from edge0.backends import core
from edge0.config import GenerationConfig
from edge0.engine.base import Edge0Engine

GREEDY = GenerationConfig(temperature=0.0, max_new_tokens=5)


class ScriptedEngine(Edge0Engine):
    """Always predicts token 5 and records what the family hooks saw."""

    def __init__(self):
        self.forwards = []
        self.prefill_ends = 0
        super().__init__("unused", SimpleNamespace(prefill_chunk=2))

    def _build(self):
        pass

    def _forward(self, ids, intra_stage=True):
        self.forwards.append(list(ids))
        logits = [0.0] * 8
        logits[5] = 10.0
        return core.array(logits)

    def _prefill_end(self):
        self.prefill_ends += 1


def test_generate_runs_to_the_cap_without_should_stop():
    eng = ScriptedEngine()
    assert eng.generate([1, 2, 3], GREEDY) == [5] * 5
    assert eng.forwards == [[1, 2], [3]] + [[5]] * 5


def test_should_stop_ends_generation_at_a_token_boundary():
    eng = ScriptedEngine()
    seen = []
    out = eng.generate([1, 2, 3], GREEDY, on_token=seen.append,
                       should_stop=lambda: len(seen) >= 2)
    assert out == [5, 5]
    assert seen == [5, 5]
    assert eng.forwards == [[1, 2], [3], [5], [5]]


def test_should_stop_ends_prefill_between_chunks():
    eng = ScriptedEngine()
    out = eng.generate([1, 2, 3, 4, 5, 6], GREEDY,
                       should_stop=lambda: len(eng.forwards) >= 1)
    assert out == []
    assert eng.forwards == [[1, 2]]
    # The prefill working set is still released.
    assert eng.prefill_ends == 1


def test_should_stop_before_the_start_never_touches_the_model():
    eng = ScriptedEngine()
    assert eng.generate([1, 2, 3], GREEDY, should_stop=lambda: True) == []
    assert eng.forwards == []
    assert eng.prefill_ends == 0
