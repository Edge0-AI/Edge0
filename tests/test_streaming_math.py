"""StreamingSwitchGLU path math tests: every streaming path must produce
outputs identical to the exact per-expert path (same gather_qmm kernel),
and the exact path must match a plain fp16 reference implementation.

No real checkpoints needed: small random 4-bit weights are written to a
temp safetensors shard with the exact layout the checkpoint uses
(gate_proj / up_proj / down_proj stacked along the expert axis).
"""

from __future__ import annotations

import numpy as np
import pytest
import mlx.core as mx
import mlx.nn as nn
from safetensors.numpy import save_file

from edge0.moe.spec import MoESpec, QuantSpec, WeightLayout
from edge0.streaming.cache import SharedExpertCache
from edge0.streaming.layer import StreamingSwitchGLU
from edge0.streaming.mmap import SafetensorsMmap
from edge0.streaming.options import LayerOptions

N_EXPERTS = 8
DIM = 64
INTER = 128


def _write_shard(path, fuse_gu: bool, seed: int = 0):
    """Random fp16 weights -> 4-bit shard; returns the fp16 originals.

    Layout matches the checkpoint convention [out, in/8] packed u32:
    gate/up project hidden (DIM) to intermediate (INTER); down projects
    INTER back to DIM — so down rows run along DIM with INTER as the
    packed axis.
    """
    mx.random.seed(seed)
    w = {}
    w["gate_proj"] = mx.random.normal((N_EXPERTS, INTER, DIM)).astype(mx.float16)
    w["up_proj"] = mx.random.normal((N_EXPERTS, INTER, DIM)).astype(mx.float16)
    w["down_proj"] = mx.random.normal((N_EXPERTS, DIM, INTER)).astype(mx.float16)

    def _parts(proj):
        w16 = w[proj]
        # transpose to [E, out, in] -> quantize per-expert rows along last
        # axis; scales are stored as bf16 bit patterns (u16), weights as
        # packed u32.
        q, scales, biases = mx.quantize(w16, group_size=64, bits=4)
        # mx.quantize returns fp16 scales; the checkpoint stores bf16 bit
        # patterns — convert numerically, then bitcast to u16.
        return {
            "weight": np.asarray(q),                            # uint32
            "scales": np.asarray(scales.astype(mx.bfloat16).view(mx.uint16)),
            "biases": np.asarray(biases.astype(mx.bfloat16).view(mx.uint16)),
        }

    tensors = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for part, arr in _parts(proj).items():
            tensors[f"layers.0.mlp.switch_mlp.{proj}.{part}"] = arr
    save_file(tensors, str(path))
    return {k: np.asarray(v) for k, v in w.items()}


def _spec(fuse_gu: bool) -> MoESpec:
    return MoESpec(
        num_experts=N_EXPERTS, top_k=4, intermediate_size=INTER,
        quant=QuantSpec(bits=4, group_size=64),
        layout=(WeightLayout.FUSED_GATE_UP if fuse_gu
                else WeightLayout.SEPARATE),
        key_template="layers.{layer}.mlp.switch_mlp",
        block_path="layers.{layer}.mlp.switch_mlp",
    )


def _options(**kw) -> LayerOptions:
    base = dict(staged=False, staged_n=4, staged_trigger=4,
                staged_sync=True, load_threads=2, prefetch_threads=1)
    base.update(kw)
    return LayerOptions(**base)


@pytest.fixture(scope="module", params=[False, True],
                ids=["separate", "fused_gate_up"])
def layer(request, tmp_path_factory):
    fuse = request.param
    path = tmp_path_factory.mktemp("shard") / "w.safetensors"
    _write_shard(path, fuse)
    mm = SafetensorsMmap(str(path))
    lay = StreamingSwitchGLU(
        [mm], 0, _spec(fuse), _options(),
        shared_cache=SharedExpertCache(64))
    return lay, mm, fuse


def ref_moe(x, inds, shard: SafetensorsMmap, fuse_gu: bool):
    """Bit-level reference: dequantize the SAME 4-bit payloads the
    streaming paths consume, then compute the GLU in float32.  This pins
    the file layout + dequant math; kernel rounding aside it must match
    every streaming path exactly."""
    xf = x.astype(mx.float32)
    wcache = {}

    def w(e, proj):
        key = (e, proj)
        if key not in wcache:
            name = f"layers.0.mlp.switch_mlp.{proj}.weight"
            raw = shard.raw(name)
            per = raw.size // N_EXPERTS
            sl = raw[e * per:(e + 1) * per]
            # [out, in/8] packed u32: gate/up out=INTER in=DIM;
            # down out=DIM in=INTER.
            if proj == "down_proj":
                q = mx.array(sl.view("<u4")).reshape(DIM, INTER // 8)
            else:
                q = mx.array(sl.view("<u4")).reshape(INTER, DIM // 8)
            parts = {"weight": q}
            n_in = INTER if proj == "down_proj" else DIM
            for part in ("scales", "biases"):
                name = f"layers.0.mlp.switch_mlp.{proj}.{part}"
                raw = shard.raw(name)
                per = raw.size // N_EXPERTS
                sl = raw[e * per:(e + 1) * per]
                parts[part] = mx.array(sl.view("<u2")).view(
                    mx.bfloat16).reshape(-1, n_in // 64)
            wcache[key] = mx.dequantize(
                parts["weight"], parts["scales"], parts["biases"],
                group_size=64, bits=4)  # [out, in]
        return wcache[key]

    T, k = inds.shape
    rows = []
    for t in range(T):
        per = []
        for j in range(k):
            e = int(inds[t, j])
            gate = xf[t] @ w(e, "gate_proj").T
            up = xf[t] @ w(e, "up_proj").T
            z = nn.silu(gate) * up  # swiglu
            per.append(z @ w(e, "down_proj").T)
        rows.append(mx.stack(per))
    return mx.stack(rows)


def _inds(tokens=1, k=4):
    return mx.array(np.random.default_rng(0).integers(
        0, N_EXPERTS, size=(tokens, k)), dtype=mx.int32)


def test_exact_matches_reference(layer):
    lay, mm, fuse = layer
    x = mx.random.normal((1, 2, DIM)).astype(mx.float16)
    inds = _inds(tokens=2)
    out = lay(x, inds)
    assert out.shape == (1, 2, 4, DIM)
    ref = ref_moe(x[0], inds, mm, fuse)
    # gather_qmm evaluates with internal bf16 precision: a few elements
    # carry ~4.0 absolute error where |ref| is small, while the overall
    # error is only ≈0.24% relative L2. A per-element allclose would need
    # a tolerance loose enough to mask real bugs, so assert on relative
    # L2 instead: a gate/up swap or misaligned row layout (the bugs this
    # pins) shows up as ≈100% error, ~400x above the kernel noise floor.
    err = mx.sqrt(mx.sum((out[0].astype(mx.float32) - ref) ** 2)
                  / mx.sum(ref ** 2))
    assert err < 1e-2, f"exact path diverges from dequant reference: {err}"


def test_staged_matches_exact(layer):
    lay, _, _ = layer
    opts = _options(staged=True)
    # fresh layer, staged mode
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "w.safetensors")
        fp16 = _write_shard(path, lay._fuse_gu)
        mm = SafetensorsMmap(path)
        s = StreamingSwitchGLU([mm], 0, _spec(lay._fuse_gu), opts,
                               shared_cache=SharedExpertCache(64))
        x = mx.random.normal((1, 1, DIM)).astype(mx.float16)
        inds = _inds(tokens=1)
        # exact reference from the same layer
        exact = lay(x, inds)
        # stage the same expert set synchronously, then consume via slots
        s.stage_experts(list(map(int, inds[0].tolist())))
        s.wait_staged()
        out = s(x, inds)
        assert out.shape == exact.shape
        assert mx.allclose(out, exact).item(), "staged != exact"


def test_staged_replace_matches_exact(layer):
    lay, _, _ = layer
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "w.safetensors")
        _write_shard(path, lay._fuse_gu)
        mm = SafetensorsMmap(path)
        s = StreamingSwitchGLU(
            [mm], 0, _spec(lay._fuse_gu),
            _options(staged=True, staged_replace=True),
            shared_cache=SharedExpertCache(64))
        x = mx.random.normal((1, 1, DIM)).astype(mx.float16)
        staged = [0, 2, 5, 7]
        s.stage_experts(staged)
        s.wait_staged()
        out = s(x, None)                      # route through the whole set
        inds = mx.array([staged], dtype=mx.int32)
        exact = lay(x, inds)
        assert out.shape == exact.shape
        assert mx.allclose(out, exact).item(), "staged_replace != exact"


def test_staged_fallback_when_empty(layer):
    lay, _, _ = layer
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "w.safetensors")
        _write_shard(path, lay._fuse_gu)
        mm = SafetensorsMmap(path)
        s = StreamingSwitchGLU([mm], 0, _spec(lay._fuse_gu),
                               _options(staged=True),
                               shared_cache=SharedExpertCache(64))
        x = mx.random.normal((1, 1, DIM)).astype(mx.float16)
        inds = _inds(tokens=1)
        # nothing staged -> falls back to the exact path
        out = s(x, inds)
        exact = lay(x, inds)
        assert mx.allclose(out, exact).item()
        st = s.stats()
        assert st["staged_fallback"] == 1


def test_full_layer_prefill_matches_exact(layer):
    lay, _, _ = layer
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "w.safetensors")
        _write_shard(path, lay._fuse_gu)
        mm = SafetensorsMmap(path)
        s = StreamingSwitchGLU([mm], 0, _spec(lay._fuse_gu),
                               _options(staged=True),
                               shared_cache=SharedExpertCache(64))
        x = mx.random.normal((1, 4, DIM)).astype(mx.float16)
        # 3D router indices like the model forward ([B, T, k]): the FULL
        # branch keeps the [B, T, k, H] SwitchGLU contract without a batch
        # restore (matching the deployment's whole-layer path), so the
        # comparison must use the router's 3D shape, not the 2D _inds form.
        inds = mx.expand_dims(_inds(tokens=4), 0)  # size 16 > 8 -> prefill path
        s.load_full_layer()
        out = s(x, inds)
        s.clear_full_layer()
        exact = lay(x, inds)
        assert out.shape == exact.shape
        assert mx.allclose(out, exact).item(), "full-layer != exact"


def test_hot_stack_matches_exact(layer):
    lay, _, _ = layer
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "w.safetensors")
        _write_shard(path, lay._fuse_gu)
        mm = SafetensorsMmap(path)
        s = StreamingSwitchGLU([mm], 0, _spec(lay._fuse_gu),
                               _options(staged=True, hot_per_layer=0),
                               shared_cache=SharedExpertCache(64))
        x = mx.random.normal((1, 4, DIM)).astype(mx.float16)
        inds = _inds(tokens=4)
        s.load_hot_layer(n_hot=N_EXPERTS)     # full coverage -> no misses
        s.materialize_hot()
        out = s(x, inds)
        exact = lay(x, inds)
        assert out.shape == exact.shape
        assert mx.allclose(out, exact).item(), "hot path != exact"


def test_double_buffered_swap(layer):
    """stage_experts fills next; swap promotes; consume uses the new set."""
    lay, _, _ = layer
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "w.safetensors")
        _write_shard(path, lay._fuse_gu)
        mm = SafetensorsMmap(path)
        s = StreamingSwitchGLU(
            [mm], 0, _spec(lay._fuse_gu),
            _options(staged=True, staged_sync=False),
            shared_cache=SharedExpertCache(64))
        x = mx.random.normal((1, 1, DIM)).astype(mx.float16)
        first = [1, 3, 4, 6]
        second = [0, 2, 5, 7]
        s.stage_experts(first)
        s.swap_staged()
        out1 = s(x, mx.array([first], dtype=mx.int32))
        s.stage_experts(second)
        s.swap_staged()
        out2 = s(x, mx.array([second], dtype=mx.int32))
        ref1 = lay(x, mx.array([first], dtype=mx.int32))
        ref2 = lay(x, mx.array([second], dtype=mx.int32))
        assert mx.allclose(out1, ref1).item()
        assert mx.allclose(out2, ref2).item()
