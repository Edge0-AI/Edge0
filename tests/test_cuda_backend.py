"""CUDA (torch reference) backend checked against real MLX.

MLX is the ground truth here: every test builds inputs with ``mx.quantize``
and compares the torch implementation against the MLX op it replaces.
Skipped unless both ``mlx`` and ``torch`` are importable (e.g. Apple
Silicon with torch installed); the torch side runs on CPU there.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from edge0.backends.cuda import quant as cq  # noqa: E402

E, O, D = 8, 16, 128


def _t(a):
    if a.dtype == mx.bfloat16:  # numpy has no bfloat16: move the raw bits
        return torch.from_numpy(np.array(a.view(mx.uint16))).view(torch.bfloat16)
    return torch.from_numpy(np.array(a))


def _quantized(bits=4, group_size=64, dtype=mx.float32, shape=(E, O, D)):
    mx.random.seed(0)
    w = mx.random.normal(shape).astype(dtype)
    wq, s, b = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(wq, s, b)
    return wq, s, b


def _both(x, idx, wq, s, b, **kw):
    ref = mx.gather_qmm(x, wq, s, b, rhs_indices=idx, **kw)
    mx.eval(ref)
    got = cq.gather_qmm(_t(x), _t(wq), _t(s), _t(b), _t(idx), **kw)
    return np.array(ref.astype(mx.float32)), got.float().numpy()


@pytest.mark.parametrize("x_shape,idx_shape", [
    ((3, D), (3,)),            # default convention: broadcasts to (3, 3, O)
    ((5, 1, 1, D), (5, 3)),    # streaming/layer.py unsorted path
    ((15, 1, D), (15,)),       # streaming/layer.py sorted path (post _gather_sort)
    ((2, 1, 4, D), (2, 3)),
    ((1, 2, D), (4, 1)),
])
def test_gather_qmm_matches_mlx(x_shape, idx_shape):
    wq, s, b = _quantized()
    mx.random.seed(1)
    x = mx.random.normal(x_shape)
    idx = mx.random.randint(0, E, idx_shape).astype(mx.uint32)
    ref, got = _both(x, idx, wq, s, b, transpose=True, group_size=64, bits=4)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_gather_qmm_bits(bits):
    wq, s, b = _quantized(bits=bits)
    x = mx.random.normal((4, 1, 1, D))
    idx = mx.random.randint(0, E, (4, 2)).astype(mx.uint32)
    ref, got = _both(x, idx, wq, s, b, transpose=True, group_size=64, bits=bits)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_gather_qmm_no_transpose():
    wq, s, b = _quantized(shape=(E, D, O * 4))
    x = mx.random.normal((3, 1, 1, D))
    idx = mx.random.randint(0, E, (3, 2)).astype(mx.uint32)
    ref, got = _both(x, idx, wq, s, b, transpose=False, group_size=64, bits=4)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_gather_qmm_sorted_flag_is_only_a_hint():
    wq, s, b = _quantized()
    x = mx.random.normal((6, 1, D))
    idx = mx.array(sorted(np.random.default_rng(0).integers(0, E, 6).tolist()),
                   dtype=mx.uint32)
    ref, got = _both(x, idx, wq, s, b, transpose=True, group_size=64, bits=4,
                     sorted_indices=True)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_gather_qmm_bf16_checkpoint_dtypes():
    # Checkpoint scales/biases are bf16 and activations are bf16 in the
    # real models; compare in float32 with a bf16-sized tolerance.
    wq, s, b = _quantized(dtype=mx.bfloat16)
    x = mx.random.normal((4, 1, 1, D)).astype(mx.bfloat16)
    idx = mx.random.randint(0, E, (4, 2)).astype(mx.uint32)
    ref, got = _both(x, idx, wq, s, b, transpose=True, group_size=64, bits=4)
    np.testing.assert_allclose(got, ref, rtol=2e-2, atol=1e-1)


@pytest.mark.parametrize("shape,axes", [
    ((5, 7), (-2, -3)), ((5, 7), (0, 1)), ((5, 7), -1), ((2, 3, 4), (1, -1)),
])
def test_expand_dims_matches_mlx(shape, axes):
    from edge0.backends.cuda import core as cc
    x = np.zeros(shape, dtype=np.float32)
    assert tuple(cc.expand_dims(torch.from_numpy(x), axes).shape) == \
        tuple(mx.expand_dims(mx.array(x), axes).shape)


def test_rmsnorm_matches_mlx():
    from edge0.backends.cuda import nn as cnn
    x = mx.random.normal((3, 64))
    w = mx.random.normal((64,))
    ref = np.array(mx.fast.rms_norm(x, w, 1e-6))
    norm = cnn.RMSNorm(64, eps=1e-6)
    assert list(norm.state_dict()) == ["weight"]
    norm.load_state_dict({"weight": _t(w)})
    got = norm(_t(x)).detach().numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_gelu_matches_mlx():
    import mlx.nn as mnn
    from edge0.backends.cuda import nn as cnn
    x = mx.random.normal((4, 32)) * 3
    np.testing.assert_allclose(cnn.gelu(_t(x)).numpy(), np.array(mnn.gelu(x)),
                               rtol=1e-5, atol=1e-5)


# ---- bailing_hybrid port, piece by piece (no checkpoint needed) -------------

def _bailing():
    from edge0.backends.cuda._impl import bailing_hybrid as tb
    from edge0.backends.mlx._impl import bailing_hybrid as mb
    return tb, mb


def _f32(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def test_bailing_kda_recurrence_matches_mlx_gated_delta_ops():
    from mlx_lm.models.gated_delta import gated_delta_ops
    tb, _ = _bailing()
    B, T, H, Dk, Dv = 1, 6, 4, 16, 16
    q, k, v = _f32((B, T, H, Dk), 0), _f32((B, T, H, Dk), 1), _f32((B, T, H, Dv), 2)
    g_log = -np.abs(_f32((B, T, H, Dk), 3))          # log-decay <= 0
    beta = 1 / (1 + np.exp(-_f32((B, T, H), 4)))
    s0 = _f32((B, H, Dv, Dk), 5)
    y_ref, s_ref = gated_delta_ops(mx.array(q), mx.array(k), mx.array(v),
                                   mx.exp(mx.array(g_log)), mx.array(beta),
                                   mx.array(s0))
    y, s = tb._kda_update(*(torch.from_numpy(a) for a in (q, k, v, g_log, beta, s0)))
    np.testing.assert_allclose(y.numpy(), np.array(y_ref), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(s.numpy(), np.array(s_ref), rtol=1e-5, atol=1e-5)


def test_bailing_rope_interleave_matches_mlx():
    tb, mb = _bailing()
    x = _f32((1, 4, 5, 64), 6)
    pos = np.arange(5) + 7                            # a cache offset of 7
    ref = mb._rope_interleave_torch(mx.array(x), mx.array(pos), 6e6)
    got = tb._rope_interleave_torch(torch.from_numpy(x), torch.from_numpy(pos), 6e6)
    np.testing.assert_allclose(got.numpy(), np.array(ref), rtol=1e-5, atol=1e-5)


def test_bailing_short_conv_with_state_matches_mlx():
    tb, mb = _bailing()
    C, K = 8, 4
    w = _f32((C, 1, K), 7)                            # checkpoint layout
    mc, tc = mb.ShortConv1d(C, K), tb.ShortConv1d(C, K)
    mc.conv.weight = mx.array(np.swapaxes(w, 1, 2))   # what MLX's sanitize does
    tc.weight.data = torch.from_numpy(w)
    x1, x2 = _f32((1, 5, C), 8), _f32((1, 1, C), 9)   # prefill, then one step
    m1, ms = mc(mx.array(x1))
    m2, _ = mc(mx.array(x2), ms)
    with torch.no_grad():
        t1, ts = tc(torch.from_numpy(x1))
        t2, _ = tc(torch.from_numpy(x2), ts)
    for got, ref in ((t1, m1), (t2, m2)):
        np.testing.assert_allclose(got.numpy(), np.array(ref), rtol=1e-5, atol=1e-5)


def test_bailing_gate_matches_mlx():
    tb, mb = _bailing()
    ta = tb.ModelArgs(hidden_size=64, num_experts=128)
    ma = mb.ModelArgs(hidden_size=64, num_experts=128)
    w, bias, x = _f32((128, 64), 10), _f32((128,), 11) * 0.1, _f32((3, 5, 64), 12)
    mg, tg = mb.BailingGate(ma), tb.BailingGate(ta)
    mg.weight, mg.expert_bias = mx.array(w), mx.array(bias)
    tg.weight.data, tg.expert_bias.data = torch.from_numpy(w), torch.from_numpy(bias)
    # MLX on the CPU: float32 matmul on some Apple GPUs runs at reduced
    # precision (~7e-4 from float64 on an M5 Max), the CPU path does not.
    with mx.stream(mx.cpu):
        mi, mw = mg(mx.array(x))
        mx.eval(mi, mw)
    with torch.no_grad():
        ti, tw = tg(torch.from_numpy(x))
    mi, mw = np.array(mi), np.array(mw)
    ti, tw = ti.numpy(), tw.numpy()
    om, ot = np.argsort(mi, -1), np.argsort(ti, -1)   # order within top-k is free
    np.testing.assert_array_equal(np.take_along_axis(ti, ot, -1),
                                  np.take_along_axis(mi, om, -1))
    np.testing.assert_allclose(np.take_along_axis(tw, ot, -1),
                               np.take_along_axis(mw, om, -1), rtol=1e-5)


def test_bailing_sdpa_causal_with_offset_matches_float64():
    """A second prefill chunk over a cache: query i of L must see keys up to
    offset + i (end-aligned), which is what mx.fast's "causal" does and not
    what torch's is_causal does."""
    tb, _ = _bailing()
    L, off, D = 5, 7, 32
    q = np.random.default_rng(13).standard_normal((1, 2, L, D))
    k = np.random.default_rng(14).standard_normal((1, 2, off + L, D))
    v = np.random.default_rng(15).standard_normal((1, 2, off + L, D))
    s = np.einsum("bhqd,bhkd->bhqk", q, k) * D ** -0.5
    visible = np.arange(off + L)[None, :] <= (off + np.arange(L))[:, None]
    s = np.where(visible, s, -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True))
    ref = np.einsum("bhqk,bhkd->bhqd", p / p.sum(-1, keepdims=True), v)
    got = tb._sdpa(*(torch.from_numpy(a.astype(np.float32)) for a in (q, k, v)),
                   D ** -0.5, "causal")
    np.testing.assert_allclose(got.numpy(), ref, rtol=1e-5, atol=1e-5)


# ---- qwen3_5_moe port, piece by piece (no checkpoint needed) ----------------

def _qwen():
    from edge0.backends.cuda._impl import qwen3_5_moe as tq
    from edge0.backends.mlx._impl import qwen3_5 as mq5
    from edge0.backends.mlx._impl import qwen3_next as mqn
    return tq, mq5, mqn


def test_qwen_gated_delta_update_matches_mlx():
    """Scalar per-head decay from (a, A_log, dt_bias), beta = sigmoid(b),
    k heads repeated to the value heads -- vs mlx-lm's ops path."""
    from mlx_lm.models.gated_delta import gated_delta_update
    tq, _, _ = _qwen()
    B, T, Hk, Hv, Dk, Dv = 1, 6, 2, 4, 16, 16
    q, k, v = _f32((B, T, Hk, Dk), 20), _f32((B, T, Hk, Dk), 21), _f32((B, T, Hv, Dv), 22)
    a, b = _f32((B, T, Hv), 23), _f32((B, T, Hv), 24)
    A_log, dt_bias = _f32((Hv,), 25), _f32((Hv,), 26)
    s0 = _f32((B, Hv, Dv, Dk), 27) * 0.1
    with mx.stream(mx.cpu):
        y_ref, s_ref = gated_delta_update(
            *(mx.array(t) for t in (q, k, v, a, b, A_log, dt_bias, s0)),
            use_kernel=False)
        mx.eval(y_ref, s_ref)
    y, s = tq._gated_delta_update(*(torch.from_numpy(t) for t in
                                    (q, k, v, a, b, A_log, dt_bias, s0)))
    np.testing.assert_allclose(y.numpy(), np.array(y_ref), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(s.numpy(), np.array(s_ref), rtol=1e-5, atol=1e-5)


def test_qwen_partial_rope_matches_mlx_fast_rope():
    tq, _, _ = _qwen()
    x = _f32((1, 4, 5, 256), 28)
    for offset in (0, 7):
        with mx.stream(mx.cpu):
            ref = mx.fast.rope(mx.array(x), 64, traditional=False, base=1e7,
                               scale=1.0, offset=offset)
            mx.eval(ref)
        got = tq._rope(torch.from_numpy(x), 64, 1e7, offset)
        np.testing.assert_allclose(got.numpy(), np.array(ref), rtol=1e-5,
                                   atol=1e-5, err_msg=f"offset={offset}")


def test_qwen_gated_norm_and_conv_match_mlx():
    tq, _, mqn = _qwen()
    w, x, z = _f32((16,), 29), _f32((1, 5, 4, 16), 30), _f32((1, 5, 4, 16), 31)
    mn = mqn.Qwen3NextRMSNormGated(16)
    mn.weight = mx.array(w)
    tn = tq.RMSNormGated(16)
    tn.weight.data = torch.from_numpy(w)
    with mx.stream(mx.cpu):
        ref = mn(mx.array(x), mx.array(z))
        mx.eval(ref)
    with torch.no_grad():
        np.testing.assert_allclose(tn(torch.from_numpy(x), torch.from_numpy(z)).numpy(),
                                   np.array(ref), rtol=1e-5, atol=1e-5)
    import mlx.nn as mnn
    C, K = 12, 4
    cw, cx = _f32((C, K, 1), 32), _f32((1, 9, C), 33)          # MLX layout
    mc = mnn.Conv1d(C, C, K, groups=C, bias=False, padding=0)
    mc.weight = mx.array(cw)
    tc = tq._DepthwiseConv(C, K)
    tc.weight.data = torch.from_numpy(cw)
    with mx.stream(mx.cpu):
        cref = mc(mx.array(cx))
        mx.eval(cref)
    with torch.no_grad():
        np.testing.assert_allclose(tc(torch.from_numpy(cx)).numpy(), np.array(cref),
                                   rtol=1e-5, atol=1e-5)


def test_qwen_attention_with_cache_matches_mlx():
    """Gated GQA attention with partial RoPE, prefill then decode steps
    through the KV cache (so RoPE offsets and the end-aligned causal mask
    both matter)."""
    from mlx.utils import tree_flatten
    from mlx_lm.models.cache import KVCache as MKV
    tq, mq5, _ = _qwen()
    targs = dict(model_type="qwen3_5_moe_text", hidden_size=64, num_attention_heads=4,
                 num_key_value_heads=2, head_dim=32, rms_norm_eps=1e-6,
                 rope_parameters={"rope_type": "default", "rope_theta": 1e7,
                                  "partial_rotary_factor": 0.25})
    ma = mq5.Attention(mq5.TextModelArgs.from_dict(dict(targs,
                       rope_parameters=dict(targs["rope_parameters"]))))
    ta = tq.Attention(tq.TextModelArgs.from_dict(dict(targs,
                      rope_parameters=dict(targs["rope_parameters"]))))
    rng = np.random.default_rng(34)
    params = {k: (rng.standard_normal(v.shape) * 0.2).astype(np.float32)
              for k, v in tree_flatten(ma.parameters())}
    ma.load_weights([(k, mx.array(v)) for k, v in params.items()])
    ta.load_state_dict({k: torch.from_numpy(v) for k, v in params.items()})
    mc, tc = MKV(), tq.KVCache()
    for step, L in enumerate((5, 3, 1, 1)):
        x = _f32((1, L, 64), 40 + step)
        with mx.stream(mx.cpu):
            mask = "causal" if L > 1 else None
            ref = ma(mx.array(x), mask, mc)
            mx.eval(ref)
        with torch.no_grad():
            got = ta(torch.from_numpy(x), "causal" if L > 1 else None, tc)
        np.testing.assert_allclose(got.numpy(), np.array(ref), rtol=1e-4,
                                   atol=1e-5, err_msg=f"step {step} (L={L})")


def test_swiglu_matches_mlx():
    import mlx.nn as mnn
    up, gate = mx.random.normal((4, 32)), mx.random.normal((4, 32))
    ref = np.array(mnn.silu(gate) * up)
    got = cq.swiglu(_t(up), _t(gate)).numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)
