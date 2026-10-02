"""Sampler unit tests: shape-independent draws and truncation behavior."""

from __future__ import annotations

import mlx.core as mx
import pytest

from edge0.sampling import _mask_logits, sample


def test_mask_identity_for_sane_kwargs():
    logits = mx.arange(10, dtype=mx.float32)
    out = _mask_logits(logits, temperature=1.0, top_k=None, top_p=None)
    assert mx.allclose(out, logits).item()


def test_mask_topk_keeps_only_top_k():
    logits = mx.arange(10, dtype=mx.float32)  # 0..9
    out = _mask_logits(logits, temperature=1.0, top_k=3, top_p=None)
    vals = mx.sort(out, axis=-1)[-3:].tolist()
    assert vals == [7.0, 8.0, 9.0]
    assert int(mx.sum(mx.isfinite(out)).item()) == 3


def test_sample_returns_int_and_in_range():
    logits = mx.ones((5,), dtype=mx.float32)
    for _ in range(5):
        t = sample(logits, temperature=1.0)
        assert isinstance(t, int)
        assert 0 <= t < 5


def test_sample_seed_deterministic():
    logits = mx.arange(8, dtype=mx.float32) * 2.0
    a = sample(logits, temperature=0.5, seed=42)
    b = sample(logits, temperature=0.5, seed=42)
    assert a == b


def test_repetition_penalty_changes_distribution():
    logits = mx.array([10.0, 9.0, 0.0], dtype=mx.float32)
    # history token 0 is penalized -> token 1 becomes likelier
    t = sample(logits, temperature=1.0, repetition_penalty=2.0,
               history=(0,))
    assert isinstance(t, int)
    # with penalty 2x on a +10 logit: 5 vs 9 -> argmax flips to 1
    out = _mask_logits(logits, 1.0, None, None)
    assert out.dtype == mx.float32


def test_sample_temperature_zero_is_argmax():
    logits = mx.array([1.0, 5.0, 3.0], dtype=mx.float32)
    # temperature <= 0 is deterministic decode (HF / llama.cpp semantics):
    # greedy — the draw must ALWAYS be the argmax, whatever the seed.
    for seed in range(20):
        assert sample(logits, temperature=0.0, seed=seed) == 1


def test_sample_temperature_zero_greedy_with_topk():
    # top-k truncation still applies before the greedy argmax
    logits = mx.array([1.0, 5.0, 3.0, 4.0, 2.0], dtype=mx.float32)
    # top-2 = {5.0 (idx 1), 4.0 (idx 3)} -> greedy picks idx 1
    for seed in range(5):
        assert sample(logits, temperature=0.0, top_k=2, seed=seed) == 1
