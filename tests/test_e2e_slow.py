"""Real-checkpoint smoke tests for the shipped model tiers.

Set ``EDGE0_8B_MODEL`` and/or ``EDGE0_35B_MODEL`` to checkpoint directories
before running ``pytest -m slow``.  A missing tier is an explicit skip so a
machine without every large checkpoint can still exercise the available one.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from edge0 import AutoEngine
from edge0.config import GenerationConfig


pytestmark = pytest.mark.slow

MAX_NEW_TOKENS = 8


def _checkpoint(env_name: str, tier: str) -> str:
    model_dir = os.environ.get(env_name)
    if not model_dir:
        pytest.skip(f"{env_name} is not set; skipping {tier} checkpoint")
    path = Path(model_dir).expanduser()
    if not path.is_dir():
        pytest.skip(f"{env_name} checkpoint not found: {path}")
    return str(path)


def _chat_ids(engine, messages: list[dict[str, str]]) -> list[int]:
    if hasattr(engine, "encode_chat"):
        return list(engine.encode_chat(messages, think=False))

    tok = engine._tok
    rendered = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    return list(tok(rendered)["input_ids"])


def _generate(env_name: str, tier: str,
              messages: list[dict[str, str]]) -> list[int]:
    model_dir = _checkpoint(env_name, tier)
    engine = AutoEngine.from_pretrained(model_dir, name=tier)
    try:
        prompt_ids = _chat_ids(engine, messages)
        started = time.perf_counter()
        output = engine.generate(
            prompt_ids,
            GenerationConfig(
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                max_new_tokens=MAX_NEW_TOKENS,
            ),
            max_new_tokens=MAX_NEW_TOKENS,
        )
        elapsed = time.perf_counter() - started
        tok_s = len(output) / elapsed if elapsed else 0.0
        print(f"[slow] {tier}: {len(output)} tokens in {elapsed:.2f}s "
              f"({tok_s:.1f} tok/s)", flush=True)
    finally:
        engine.close()
    return output


def test_edge0_8b_real_checkpoint():
    output = _generate(
        "EDGE0_8B_MODEL",
        "edge0-8b",
        [{"role": "user", "content": "你好，请用一句话介绍海滨城市。"}],
    )
    assert output
    assert len(output) <= MAX_NEW_TOKENS


def test_edge0_35b_real_checkpoint():
    output = _generate(
        "EDGE0_35B_MODEL",
        "edge0-35b",
        [{"role": "user", "content":
          "Hello! Write one short sentence about the seaside."}],
    )
    assert output
    assert len(output) <= MAX_NEW_TOKENS
