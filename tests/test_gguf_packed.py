"""GPU checks against fixtures emitted by the pinned C dequantizers."""
from pathlib import Path
import numpy as np
import pytest
from edge0.checkpoints.gguf import QUANT_SIZES


@pytest.mark.parametrize('kind', list(QUANT_SIZES))
def test_packed_gpu_reference(kind):
    import mlx.core as mx
    from edge0.backends.mlx.gguf_packed import matmul, decode_rows
    data = np.load(Path(__file__).parent/'fixtures/gguf_quant_reference.npz')
    block, size = QUANT_SIZES[kind]
    raw = data[f'raw_{kind}']
    ref = data[f'decoded_{kind}'].reshape(4, -1)
    w = mx.array(raw.reshape(-1))
    decoded = decode_rows(w, kind, *ref.shape)
    np.testing.assert_allclose(np.array(decoded), ref, rtol=2e-6, atol=1e-7)
    rng = np.random.default_rng(73)
    base = rng.normal(size=(3, ref.shape[1]*2)).astype(np.float32)
    x = mx.array(base)[:, ::2]
    actual = matmul([w, w], kind, x, mx.array([0, 1, 0], dtype=mx.uint32), 4)
    np.testing.assert_allclose(np.array(actual), base[:, ::2]@ref.T, rtol=3e-5, atol=2e-4)


@pytest.mark.parametrize('kind', list(QUANT_SIZES))
@pytest.mark.parametrize('budget', [4096, 16384])
def test_projections_never_cpu_decode(tmp_path, monkeypatch, kind, budget):
    import mlx.core as mx
    from gguf_fixture import write_gguf
    from edge0.checkpoints.gguf import GGUFSource
    from edge0.streaming.gguf import GGUFWeights, GGUFExperts
    import edge0.checkpoints.quant as quant
    data = np.load(Path(__file__).parent/'fixtures/gguf_quant_reference.npz')
    raw = data[f'raw_{kind}'].tobytes()
    width = 8*QUANT_SIZES[kind][0]
    ref = data[f'decoded_{kind}'].reshape(4, width)
    p = write_gguf(tmp_path/'packed.gguf', [('w', (4, width), kind, raw)])
    weights = GGUFWeights(GGUFSource(p), budget, 4096)
    monkeypatch.setattr(quant, 'decode', lambda *a: pytest.fail('CPU decoder invoked'))
    from edge0.backends.mlx import gguf_packed
    monkeypatch.setattr(gguf_packed, 'decode_rows', lambda *a: pytest.fail('decoded projection matrix'))
    x = mx.ones((2, width))
    np.testing.assert_allclose(np.array(weights.linear('w', x)), np.ones((2, width))@ref.T, rtol=2e-5, atol=1e-4)
    reads = weights.source.bytes_read
    weights.linear('w', x)
    if budget == 16384:
        assert weights.source.bytes_read == reads
    assert weights.cache.peak_bytes <= weights.cache.budget
    assert all(v.dtype == mx.uint8 for v in weights.cache._cache.values())
    weights.close()


def test_oversized_rows(tmp_path):
    import mlx.core as mx
    from gguf_fixture import write_gguf
    from edge0.checkpoints.gguf import GGUFSource
    from edge0.streaming.gguf import GGUFWeights
    a = np.random.default_rng(1).normal(size=(3, 2048)).astype(np.float32)
    p = write_gguf(tmp_path/'wide.gguf', [('w', a.shape, 0, a.tobytes())])
    weights = GGUFWeights(GGUFSource(p), 4096, 4096)
    np.testing.assert_allclose(np.array(weights.linear('w', mx.ones((2, 2048)))), np.ones((2, 2048))@a.T, atol=3e-5)
    np.testing.assert_array_equal(np.array(weights.rows('w', [2, 0])), a[[2, 0]])
    assert weights.cache.peak_bytes<=4096
    weights.close()


@pytest.mark.parametrize('budget', [4096, 1024**2])
def test_mixed_routed_batches_and_repeated_experts(tmp_path, monkeypatch, budget):
    import mlx.core as mx
    from gguf_fixture import write_gguf
    from edge0.checkpoints.gguf import GGUFSource
    from edge0.streaming.gguf import GGUFWeights, GGUFExperts
    from edge0.checkpoints.quant import decode
    rng = np.random.default_rng(18)
    n, hidden, intermediate = 35, 32, 7
    gate = (rng.normal(size=(n, intermediate, hidden))*.01).astype(np.float32)
    down = (rng.normal(size=(n, hidden, intermediate))*.01).astype(np.float32)
    data = np.load(Path(__file__).parent/'fixtures/gguf_quant_reference.npz')
    raw = np.resize(data['raw_8'].reshape(-1, 34), (n*intermediate, 34)).tobytes()
    up = decode(raw, 8).reshape(n, intermediate, hidden)
    prefix = 'blk.0.ffn_'
    tensors = [(prefix+'gate_exps.weight', gate.shape, 0, gate.tobytes()),
               (prefix+'up_exps.weight', up.shape, 8, raw),
               (prefix+'down_exps.weight', down.shape, 0, down.tobytes())]
    weights = GGUFWeights(GGUFSource(write_gguf(tmp_path/'experts.gguf', tensors)), budget, 4096)
    import edge0.checkpoints.quant as quant
    monkeypatch.setattr(quant, 'decode', lambda *a: pytest.fail('CPU decoder invoked'))
    selection = np.stack([np.arange(n), np.arange(n)], axis=1)
    x = rng.normal(size=(n, hidden)).astype(np.float32)
    scores = rng.uniform(size=(n, 2)).astype(np.float32)
    expected = np.empty((n, 2, hidden), np.float32)
    for row in range(n):
        g = x[row]@gate[row].T
        expected[row, :] = ((g/(1+np.exp(-g)))*(x[row]@up[row].T))@down[row].T
    experts = GGUFExperts(weights, 0, n)
    actual = experts(mx.array(x), mx.array(selection), mx.array(scores))
    np.testing.assert_allclose(np.array(actual), (expected*scores[..., None]).sum(1), rtol=2e-5, atol=2e-4)
    # 35 experts in three batches, three projections each, no per-expert dispatch.
    if budget == 1024**2:
        assert weights.cache.dispatches == 9
        assert weights.cache.retirements == 9
    assert experts.last_used == list(range(n))
    assert weights.cache.peak_bytes <= weights.cache.budget
    np.testing.assert_allclose(np.array(experts(mx.array(x), mx.array(selection))), expected, rtol=2e-5, atol=2e-4)
    weights.close()


def test_pending_cancel_and_payload_fidelity(tmp_path):
    import mlx.core as mx
    from gguf_fixture import write_gguf
    from edge0.checkpoints.gguf import GGUFSource
    from edge0.streaming.gguf import PackedCache
    raw = np.arange(8192, dtype=np.uint8)
    source = GGUFSource(write_gguf(tmp_path/'payload.gguf', [('w', (64, 32), 0, raw.tobytes())]))
    cache = PackedCache(source, 16384, 4096)
    cache.prefetch([('w', 0, 8), ('w', 8, 8)])
    w = cache.get(('w', 0, 8))
    np.testing.assert_array_equal(np.array(w), raw[:1024])
    output = w.astype(mx.float32)*2
    cache.track(output)
    mx.async_eval(output)
    for row in range(16, 64, 8):
        cache.get(('w', row, 8))
    cache.clear()
    np.testing.assert_array_equal(np.array(output), raw[:1024].astype(np.float32)*2)
    assert cache.resident_bytes == cache.pending_bytes == cache.inflight_bytes == cache.staging_bytes == 0
    cache.close()
    cache.close()
    with pytest.raises(RuntimeError, match='closed'):
        cache.get(('w', 0, 8))
    source.close()


def test_config_aliases_and_conflicts(tmp_path):
    from edge0 import AutoConfig
    from gguf_fixture import tiny_qwen4exp
    # Configuration validation does not read checkpoint payloads.
    path = tiny_qwen4exp(tmp_path/'config.gguf')
    with pytest.warns(DeprecationWarning):
        cfg = AutoConfig.from_pretrained(model_path=path, decoded_cache_bytes=16384, decode_chunk_bytes=4096)
    assert cfg.weight_cache_bytes == 16384 and cfg.weight_chunk_bytes == 4096
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match='conflicting'):
        AutoConfig.from_pretrained(model_path=path, decoded_cache_bytes=16384, weight_cache_bytes=32768)


def test_packed_split_checkpoint(tmp_path):
    import mlx.core as mx
    from gguf_fixture import write_gguf
    from edge0.checkpoints.gguf import GGUFSource
    from edge0.streaming.gguf import GGUFWeights
    data = np.load(Path(__file__).parent/'fixtures/gguf_quant_reference.npz')
    paths = [tmp_path/f'split-{i:05}-of-00003.gguf' for i in range(1, 4)]
    for i, p in enumerate(paths):
        kind = (0, 8, 19)[i]
        tensors = [] if i==0 else [(f'w{i}', (4, 8*QUANT_SIZES[kind][0]), kind, data[f'raw_{kind}'].tobytes())]
        write_gguf(p, tensors, {'split.no': i, 'split.count': 3, 'split.tensors.count': 2})
    weights = GGUFWeights(GGUFSource(paths[-1]), 16384, 4096)
    assert weights.source.bytes_read == 0
    for i, kind in ((1, 8), (2, 19)):
        ref = data[f'decoded_{kind}'].reshape(4, -1)
        actual = weights.linear(f'w{i}', mx.ones((2, ref.shape[1])))
        np.testing.assert_allclose(np.array(actual), np.ones((2, ref.shape[1]))@ref.T, rtol=2e-5, atol=1e-4)
    assert weights.source.bytes_read == sum(data[f'raw_{kind}'].nbytes for kind in (8, 19))
    weights.close()


def test_close_drains_failed_prefetch(tmp_path, monkeypatch):
    from gguf_fixture import write_gguf
    from edge0.checkpoints.gguf import GGUFSource
    from edge0.streaming.gguf import PackedCache
    source = GGUFSource(write_gguf(tmp_path/'badread.gguf', [('w', (32, 32), 0, bytes(4096))]))
    cache = PackedCache(source, 16384, 4096)
    import threading
    started = threading.Event()
    def fail(*args):
        started.set()
        raise OSError('injected read failure')
    monkeypatch.setattr(source, 'read_bytes', fail)
    cache.prefetch([('w', 0, 8), ('w', 8, 8)])
    assert started.wait(2)
    with pytest.raises(OSError, match='injected'):
        cache.close()
    assert cache._closed and not cache._pending and not cache._cache
    assert cache.pending_bytes == cache.staging_bytes == cache.resident_bytes == 0
    cache.close()
    source.close()
