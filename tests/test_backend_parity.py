"""Shared framework code must give the same answers on both backends.

Each case in tests/backend_parity_worker.py runs twice in a subprocess --
EDGE0_BACKEND=mlx and EDGE0_BACKEND=cuda -- on identical inputs, and the
outputs are compared here. MLX is the reference. Skipped unless both mlx
and torch are importable.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("torch")

WORKER = pathlib.Path(__file__).with_name("backend_parity_worker.py")


def _run(case, inputs, tmp_path):
    inp = tmp_path / f"{case}.in.npz"
    np.savez(inp, **inputs)
    results = {}
    for backend in ("mlx", "cuda"):
        out = tmp_path / f"{case}.{backend}.npz"
        env = dict(os.environ, EDGE0_BACKEND=backend)
        proc = subprocess.run(
            [sys.executable, str(WORKER), case, str(inp), str(out)],
            env=env, capture_output=True, text=True)
        assert proc.returncode == 0, (
            f"{case} failed on {backend}:\n{proc.stderr[-4000:]}")
        results[backend] = dict(np.load(out))
    return results["mlx"], results["cuda"]


def _logits(shape, seed=0, scale=3.0):
    return (np.random.default_rng(seed).standard_normal(shape) * scale
            ).astype(np.float32)


def test_select_from_logits(tmp_path):
    ref, got = _run("select_from_logits", {"logits": _logits((6, 256))},
                    tmp_path)
    np.testing.assert_array_equal(got["inds"], ref["inds"])
    np.testing.assert_allclose(got["scores"], ref["scores"], rtol=1e-5,
                               atol=1e-6)


def test_group_select_from_logits(tmp_path):
    rng = np.random.default_rng(1)
    inputs = {"logits": _logits((6, 128), seed=1),
              "expert_bias": (rng.standard_normal(128) * 0.1
                              ).astype(np.float32)}
    ref, got = _run("group_select_from_logits", inputs, tmp_path)
    np.testing.assert_array_equal(got["inds"], ref["inds"])
    np.testing.assert_allclose(got["scores"], ref["scores"], rtol=1e-5,
                               atol=1e-6)


def test_mask_logits(tmp_path):
    ref, got = _run("mask_logits", {"logits": _logits((1000,), seed=2)},
                    tmp_path)
    for name in ref:
        np.testing.assert_array_equal(np.isinf(got[name]),
                                      np.isinf(ref[name]), err_msg=name)
        keep = ~np.isinf(ref[name])
        np.testing.assert_allclose(got[name][keep], ref[name][keep],
                                   rtol=1e-6, err_msg=name)
