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


def _run(case, inputs, tmp_path, backends=("mlx", "cuda")):
    inp = tmp_path / f"{case}.in.npz"
    np.savez(inp, **inputs)
    results = {}
    for backend in backends:
        out = tmp_path / f"{case}.{backend}.npz"
        env = dict(os.environ, EDGE0_BACKEND=backend)
        proc = subprocess.run(
            [sys.executable, str(WORKER), case, str(inp), str(out)],
            env=env, capture_output=True, text=True)
        assert proc.returncode == 0, (
            f"{case} failed on {backend}:\n{proc.stderr[-4000:]}")
        results[backend] = dict(np.load(out))
    if len(backends) == 1:
        return results[backends[0]]
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


def _model_8b():
    path = os.environ.get("EDGE0_8B_MODEL")
    if not path or not os.path.isfile(os.path.join(path, "model.safetensors")):
        pytest.skip("set EDGE0_8B_MODEL to an edge0-8b checkpoint directory")
    return path


def test_streaming_layer_real_experts(tmp_path):
    rng = np.random.default_rng(3)
    i16 = np.stack([rng.choice(128, 8, replace=False) for _ in range(16)])
    i1 = i16[:1]
    inputs = {
        "model_dir": np.array(_model_8b()),
        "x1": rng.standard_normal((1, 1536)).astype(np.float32),
        "x16": rng.standard_normal((16, 1536)).astype(np.float32),
        "i1": i1.astype(np.int32), "i16": i16.astype(np.int32),
        # half of token 0's experts plus unrelated ones: exercises drops
        "staged_set": np.concatenate([i1[0, :4], [e for e in range(128)
                                      if e not in i1[0]][:4]]).astype(np.int32),
    }
    ref, got = _run("streaming", inputs, tmp_path)
    assert set(got) == set(ref)
    for name in sorted(ref):
        assert got[name].shape == ref[name].shape, name
        # bf16 activations through three 4-bit matmuls: compare at bf16
        # resolution, relative to the output scale.
        scale = np.abs(ref[name]).max()
        np.testing.assert_allclose(got[name], ref[name], rtol=0,
                                   atol=2e-2 * scale, err_msg=name)
    # the staged run must actually drop the experts outside the staged set
    assert not np.allclose(ref["staged_t1_e"], ref["exact_t1_e"])
    # Hot-stack prefill is documented as numerically exact, misses included
    # (half the experts are misses here): same answer as the exact path,
    # on each backend. Before the stack got its zero overflow row, misses
    # gathered past the end of the stack.
    for res in (ref, got):
        for tag in ("c", "e"):
            np.testing.assert_allclose(res[f"hot_t16_{tag}"],
                                       res[f"exact_t16_{tag}"], rtol=0,
                                       atol=1e-2 * np.abs(res[f"exact_t16_{tag}"]).max())


def test_install_streaming_experts_into_transformers_qwen35(tmp_path):
    """install_streaming_experts end to end on the torch backend: a real
    transformers Qwen3.5-MoE model, the edge0-35b MoESpec paths, experts
    streamed from a real safetensors shard. Same logits as the model's own
    dense experts on the weights the shard encodes."""
    rng = np.random.default_rng(4)
    res = _run("install_qwen35_tiny", {
        "shard_path": np.array(str(tmp_path / "experts.safetensors")),
        "ids6": rng.integers(0, 128, (1, 6)),
        "ids40": rng.integers(0, 128, (1, 40)),
    }, tmp_path, backends=("cuda",))
    assert int(res["n_twins"]) == 4
    assert str(res["experts_type"]) == "TransformersExpertsAdapter"
    for name in ("ids6", "ids40"):
        ref, got = res[f"ref_{name}"], res[f"got_{name}"]
        np.testing.assert_allclose(got, ref, rtol=0,
                                   atol=1e-5 * np.abs(ref).max(), err_msg=name)


def test_load_model_edge0_35b_format(tmp_path):
    """cuda load_model on a checkpoint in the published edge0-35b on-disk
    format (MLX-quantized throughout, MLX-sanitized norms and conv1d,
    language_model. prefix), then streaming experts on top: same logits as
    the source model on the weights the checkpoint encodes."""
    rng = np.random.default_rng(5)
    res = _run("load_model_qwen35_tiny", {
        "ckpt_dir": np.array(str(tmp_path / "ckpt")),
        "ids6": rng.integers(0, 128, (1, 6)),
        "ids40": rng.integers(0, 128, (1, 40)),
    }, tmp_path, backends=("cuda",))
    assert str(res["embed_type"]) == "QuantizedEmbedding"
    assert str(res["q_proj_type"]) == "QuantizedLinear"
    assert str(res["lm_head_type"]) == "QuantizedLinear"
    assert str(res["router_dtype"]) == "torch.float32"   # 8-bit, dequantized
    assert int(res["n_meta_params"]) == 0   # experts replaced by the twins
    for key in ("ids6", "ids40"):
        ref, got = res[f"ref_{key}"], res[f"got_{key}"]
        np.testing.assert_allclose(got, ref, rtol=0,
                                   atol=1e-5 * np.abs(ref).max(), err_msg=key)
        assert (got.argmax(-1) == ref.argmax(-1)).all(), key


def test_engines_import_without_mlx_and_refuse_other_backends(tmp_path):
    res = _run("engine_guard", {"_": np.zeros(1)}, tmp_path,
               backends=("cuda",))
    assert res["loaded_mlx"].size == 0, list(res["loaded_mlx"])
    for msg, tier in zip(res["errors"], ("edge0-35b", "edge0-8b")):
        assert msg.startswith(f"{tier}: the engine runs only on "
                              "EDGE0_BACKEND=mlx"), msg


def test_mask_logits(tmp_path):
    ref, got = _run("mask_logits", {"logits": _logits((1000,), seed=2)},
                    tmp_path)
    for name in ref:
        np.testing.assert_array_equal(np.isinf(got[name]),
                                      np.isinf(ref[name]), err_msg=name)
        keep = ~np.isinf(ref[name])
        np.testing.assert_allclose(got[name][keep], ref[name][keep],
                                   rtol=1e-6, err_msg=name)
