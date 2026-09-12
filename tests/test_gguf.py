"""Small local fixtures; no model downloads or Metal needed."""
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from edge0.checkpoints.gguf import GGUFSource, QUANT_SIZES
from edge0.checkpoints.quant import decode


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def value(v):
    if isinstance(v, bool):
        return 7, struct.pack("<?", v)
    if isinstance(v, str):
        return 8, string(v)
    if isinstance(v, int):
        return (4, struct.pack("<I", v)) if 0 <= v < 2**32 else (10, struct.pack("<Q", v))
    if isinstance(v, float):
        return 6, struct.pack("<f", v)
    if isinstance(v, list):
        values = [value(x) for x in v]
        kind = values[0][0] if values else 11
        assert all(k == kind for k, _ in values)
        return 9, struct.pack("<IQ", kind, len(v)) + b"".join(b for _, b in values)
    raise TypeError(v)


def write_gguf(path, tensors=(), metadata=None):
    m = {"general.architecture": "qwen4exp", **(metadata or {})}
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(m))
    for k, v in m.items():
        kind, data = value(v)
        if k.endswith(("ple.head_offsets", "ple.head_vocab_sizes")):
            kind = 9
            data = struct.pack("<IQ", 10, len(v)) + struct.pack("<" + "Q" * len(v), *v)
        if k == "tokenizer.ggml.merges" and not v:
            kind, data = 9, struct.pack("<IQ", 8, 0)
        if k == "tokenizer.ggml.token_type":
            kind = 9
            data = struct.pack("<IQ", 5, len(v)) + struct.pack("<" + "i" * len(v), *v)
        header += string(k) + struct.pack("<I", kind) + data
    payload = b""
    for name, shape, kind, data in tensors:
        offset = len(payload)
        header += string(name) + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *reversed(shape))
        header += struct.pack("<IQ", kind, offset)
        payload += data
        payload += bytes((-len(payload)) % 32)
    if tensors:
        header += bytes((-len(header)) % 32)
    path.write_bytes(header + payload)
    return path


def test_single_file_rows_and_close(tmp_path):
    data = np.arange(48, dtype=np.float32).reshape(2, 3, 8)
    p = write_gguf(tmp_path / "a.gguf", [("experts", data.shape, 0, data.tobytes())])
    s = GGUFSource(p)
    assert s.bytes_read == 0
    np.testing.assert_array_equal(s.read_rows("experts", 3, 3), data[1])
    assert s.bytes_read == data[1].nbytes
    for start, count in [(-1, 1), (6, 1), (1, -1)]:
        with pytest.raises(ValueError, match="range"):
            s.read_rows("experts", start, count)
    s.close()
    s.close()
    with pytest.raises(RuntimeError, match="closed"):
        s.read_rows("experts", 0, 1)


def split_fixture(tmp_path):
    paths = [tmp_path / f"model-{i:05}-of-00003.gguf" for i in range(1, 4)]
    for i, p in enumerate(paths):
        tensors = [] if i == 0 else [(f"tensor{i}", (1,), 0, struct.pack("<f", i))]
        write_gguf(p, tensors, {"split.no": i, "split.count": 3, "split.tensors.count": 2})
    return paths


def test_split_metadata_only_any_shard(tmp_path):
    for p in split_fixture(tmp_path):
        with GGUFSource(p) as source:
            assert len(source.tensors) == 2
            assert source.read_rows("tensor2", 0, 1).item() == 2


def test_missing_shard(tmp_path):
    paths = split_fixture(tmp_path)
    paths[1].unlink()
    with pytest.raises(FileNotFoundError, match="missing GGUF shard"):
        GGUFSource(paths[-1])


@pytest.mark.parametrize("change,match", [
    ({"split.no": 0}, "split metadata"),
    ({"general.architecture": "qwen35moe"}, "inconsistent"),
    ({"split.tensors.count": 3}, "inconsistent"),
])
def test_split_inconsistency(tmp_path, change, match):
    paths = split_fixture(tmp_path)
    write_gguf(paths[2], [("tensor2", (1,), 0, bytes(4))],
        {"split.no": 2, "split.count": 3, "split.tensors.count": 2, **change})
    with pytest.raises(ValueError, match=match):
        GGUFSource(paths[0])


def test_duplicate_tensors(tmp_path):
    paths = split_fixture(tmp_path)
    write_gguf(paths[2], [("tensor1", (1,), 0, bytes(4))],
        {"split.no": 2, "split.count": 3, "split.tensors.count": 2})
    with pytest.raises(ValueError, match="duplicate"):
        GGUFSource(paths[0])


@pytest.mark.parametrize("payload,match", [(b"nope" + bytes(20), "magic"),
    (b"GGUF" + struct.pack("<I", 99), "version"), (b"GGUF", "truncated"),
    (b"GGUF" + struct.pack("<IQQ", 3, 1 << 63, 0), "counts")])
def test_malformed(tmp_path, payload, match):
    p = tmp_path / "bad.gguf"
    p.write_bytes(payload)
    with pytest.raises(ValueError, match=match):
        GGUFSource(p)


def test_unsupported_and_bounds(tmp_path):
    p = write_gguf(tmp_path / "bad.gguf", metadata={"general.architecture": "llama"})
    with pytest.raises(ValueError, match="architecture"):
        GGUFSource(p)
    write_gguf(p, [("bad", (1,), 99, bytes(4))])
    with pytest.raises(ValueError, match="encoding 99"):
        GGUFSource(p)
    write_gguf(p, [("bad", (4096,), 0, bytes(4))])
    with pytest.raises(ValueError, match="out-of-bounds"):
        GGUFSource(p)


@pytest.mark.parametrize("kind", QUANT_SIZES)
def test_decoders_zero_and_slice_boundaries(tmp_path, kind):
    block, size = QUANT_SIZES[kind]
    data = bytes(size * 6)
    p = write_gguf(tmp_path / "q.gguf", [("q", (2, 3, block), kind, data)])
    with GGUFSource(p) as s:
        np.testing.assert_array_equal(s.read_rows("q", 3, 3), np.zeros((3, block)))
        assert s.bytes_read == size * 3
        assert s.read_rows("q", 6, 0).shape == (0, block)
    with pytest.raises(ValueError, match="incomplete"):
        decode(bytes(size - 1), kind)


@pytest.mark.parametrize("kind", QUANT_SIZES)
def test_reference_expert_and_embedding_boundaries(tmp_path, kind):
    fixture = np.load(Path(__file__).parent / "fixtures" / "gguf_quant_reference.npz")
    block, size = QUANT_SIZES[kind]
    raw = fixture[f"raw_{kind}"]
    expected = fixture[f"decoded_{kind}"].reshape(32, block)
    p = write_gguf(tmp_path / "slices.gguf", [
        ("expert", (4, 8, block), kind, raw.tobytes()),
        ("embedding", (32, block), kind, raw.tobytes()),
    ])
    with GGUFSource(p) as source:
        for start, count in [(8, 8), (7, 2), (31, 1), (0, 1)]:
            for name in ("expert", "embedding"):
                np.testing.assert_allclose(source.read_rows(name, start, count),
                    expected[start:start + count], rtol=1e-6, atol=1e-6)
        assert source.bytes_read == 2 * 12 * size


def test_quant_reference_vectors():
    """Committed fixtures generated by pinned upstream C dequantizers."""
    fixture = Path(__file__).parent / "fixtures" / "gguf_quant_reference.npz"
    assert fixture.exists(), "generate the upstream C reference fixtures first"
    with np.load(fixture) as data:
        for kind in QUANT_SIZES:
            actual = decode(data[f"raw_{kind}"].tobytes(), kind)
            np.testing.assert_allclose(actual, data[f"decoded_{kind}"], rtol=2e-6, atol=1e-7)


def test_model_path_conflicts():
    from edge0 import AutoConfig, AutoModel, AutoEngine
    for api in (AutoConfig, AutoModel, AutoEngine):
        for other in ({"name": "edge0-35b"}, {"model_dir": "x"}):
            with pytest.raises(ValueError, match="cannot be combined"):
                api.from_pretrained(model_path="x", **other)


def test_custom_safetensors_config(tmp_path):
    from edge0 import AutoConfig
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe",
        "num_experts": 16, "num_experts_per_tok": 6, "moe_intermediate_size": 128,
        "eos_token_id": [4, 5], "quantization": {"bits": 4, "group_size": 64}}))
    cfg = AutoConfig.from_pretrained(model_path=tmp_path)
    assert cfg.prerouter is None and cfg.lora == ""
    assert cfg.moe_spec.top_k == 6 and cfg.options.top_k is None
    assert not cfg.options.staged
    assert cfg.gen.eos_ids == (4, 5) and not cfg.gen.first_token_greedy


def test_tokenizer_missing_requirements():
    from edge0.checkpoints.tokenizer import load_tokenizer
    with pytest.raises(ValueError, match="tokenizer-path"):
        load_tokenizer({})


def test_packed_cache_bound_and_shutdown(tmp_path):
    from edge0.streaming.gguf import PackedCache
    p = write_gguf(tmp_path / "q.gguf", [("q", (32, 1024), 0, bytes(32 * 4096))])
    with GGUFSource(p) as s:
        c = PackedCache(s, 16384, 4096)
        for i in range(32):
            c.prefetch([("q", min(i + 1, 31), 1)])
            c.get(("q", i, 1))
            c.retire()
        assert c.peak_bytes <= c.budget
        assert c.resident_bytes + c.pending_bytes <= c.budget
        c.close()
        c.close()
        assert not c._pending and not c._cache
