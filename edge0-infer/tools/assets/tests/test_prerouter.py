import json
from pathlib import Path
import struct

import numpy as np
import pytest

from asset_common import sha256
from format_gguf import F16, Reader
from edge0_35b import prerouter

pytestmark = pytest.mark.l1


def safetensors(path: Path, owners, n_in=12, hidden=4, n_out=6, dtype='F16', metadata=None, drop=None):
    rng = np.random.default_rng(1)
    header, body = {}, b''
    for owner in owners:
        for part, shape in [('fc1', [hidden, n_in]), ('fc2', [n_out, hidden]), ('linear_init', [n_out, n_in])]:
            name = f'layers.{owner}.{part}.weight'
            if name == drop:
                continue
            data = rng.normal(size=shape).astype('<f2').tobytes()
            header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [len(body), len(body) + len(data)]}
            body += data
    header['__metadata__'] = metadata or {'kind': 'prerouter', 'format_version': '1', 'K': '4', 'r': '16',
                                          'alpha': '32', 'owners': json.dumps(list(owners))}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + body)
    return header


def test_payload_is_copied_unchanged(tmp_path):
    source = tmp_path / prerouter.SOURCE
    header = safetensors(source, [2, 3])
    prerouter.pack(source, tmp_path / 'prerouter.gguf')
    reader = Reader(tmp_path / 'prerouter.gguf')
    _, tensors = prerouter.read_safetensors(source)
    assert len(reader.tensors) == 6
    for owner in [2, 3]:
        for part in prerouter.PARTS:
            shape, payload = tensors[f'layers.{owner}.{part}.weight']
            t = reader.tensors[f'blk.{owner}.prerouter_{part}.weight']
            assert t.type == F16 and t.shape == tuple(reversed(shape))
            assert reader.data(t.name) == payload
    m = reader.metadata
    assert m['general.type'].data == 'prerouter' and m['prerouter.owners'].data == [2, 3]
    assert m['prerouter.top_k'].data == 4 and m['prerouter.feed_forward_length'].data == 4
    assert m['prerouter.lora_alpha'].data == 32.0
    assert header


@pytest.mark.parametrize('kwargs, message', [
    ({'dtype': 'F32'}, 'expected F16'),
    ({'drop': 'layers.3.fc2.weight'}, 'does not match owners'),
    ({'metadata': {'kind': 'prerouter', 'format_version': '1', 'K': '4', 'r': '16', 'alpha': '32', 'owners': '[2]'}}, 'does not match owners'),
    ({'metadata': {'kind': 'lora', 'format_version': '1'}}, 'not a format_version 1'),
])
def test_rejects_inconsistent_source(tmp_path, kwargs, message):
    source = tmp_path / prerouter.SOURCE
    safetensors(source, [2, 3], **kwargs)
    with pytest.raises(ValueError, match=message):
        prerouter.pack(source, tmp_path / 'prerouter.gguf')
    assert not (tmp_path / 'prerouter.gguf').exists()


def assets_dir(tmp_path):
    source, assets = tmp_path / 'source', tmp_path / 'assets'
    source.mkdir()
    assets.mkdir()
    safetensors(source / prerouter.SOURCE, [2, 3])
    for name in ['base.gguf', 'assets.tsv']:
        (assets / name).write_text(name)
    (assets / 'provenance.json').write_text('{}')
    lock = {'files': {prerouter.SOURCE: sha256(source / prerouter.SOURCE)}}
    return source, assets, lock


def test_add_failure_restores_directory(tmp_path, monkeypatch):
    source, assets, lock = assets_dir(tmp_path)
    before = {p.name: p.read_bytes() for p in assets.iterdir()}
    monkeypatch.setattr(prerouter, 'write_catalog', lambda root: (root / 'assets.tsv').write_text('rewritten'))
    monkeypatch.setattr(prerouter, 'check', lambda root: {'status': 'failed'})
    with pytest.raises(ValueError, match='asset check failed'):
        prerouter.add(assets, source, lock)
    assert {p.name: p.read_bytes() for p in assets.iterdir()} == before


def test_add_keeps_other_payloads_and_refuses_twice(tmp_path, monkeypatch):
    source, assets, lock = assets_dir(tmp_path)
    monkeypatch.setattr(prerouter, 'write_catalog', lambda root: None)
    monkeypatch.setattr(prerouter, 'check', lambda root: {'status': 'passed'})
    prerouter.add(assets, source, lock)
    assert (assets / 'base.gguf').read_text() == 'base.gguf'
    record = json.loads((assets / 'provenance.json').read_text())['prerouter']
    assert record['source_sha256'] == lock['files'][prerouter.SOURCE] and record['tensors'] == 6
    with pytest.raises(ValueError, match='already present'):
        prerouter.add(assets, source, lock)


def test_add_rejects_unregistered_source(tmp_path):
    source, assets, lock = assets_dir(tmp_path)
    lock['files'][prerouter.SOURCE] = '0' * 64
    with pytest.raises(ValueError, match='digest mismatch'):
        prerouter.add(assets, source, lock)
    assert not (assets / 'prerouter.gguf').exists()


def test_accepts_nested_published_metadata(tmp_path):
    inner = {'kind': 'prerouter', 'format_version': '1', 'K': '4', 'r': '16', 'alpha': '32', 'owners': '[2, 3]'}
    source = tmp_path / prerouter.SOURCE
    safetensors(source, [2, 3], metadata={'__metadata__': json.dumps(inner)})
    record = prerouter.pack(source, tmp_path / 'prerouter.gguf')
    assert record['source_metadata'] == inner
    assert Reader(tmp_path / 'prerouter.gguf').metadata['prerouter.owners'].data == [2, 3]


def test_reader_keeps_one_copy_of_large_tensor_payloads(tmp_path):
    import tracemalloc

    size, count = 2 * 1024 * 1024, 8
    source = tmp_path / 'large.safetensors'
    # Reverse header order verifies that streaming follows validated offsets.
    header = {f'tensor.{i}': {'dtype': 'F16', 'shape': [size // 2],
                              'data_offsets': [i * size, (i + 1) * size]}
              for i in reversed(range(count))}
    header['__metadata__'] = {'source': 'synthetic'}
    encoded = json.dumps(header).encode()
    with source.open('wb') as file:
        file.write(struct.pack('<Q', len(encoded)))
        file.write(encoded)
        for i in range(count):
            file.write(bytes([i]) * size)
    tracemalloc.start()
    try:
        metadata, tensors = prerouter.read_safetensors(source)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert metadata == {'source': 'synthetic'}
    assert len(tensors) == count
    for i in range(count):
        assert tensors[f'tensor.{i}'] == ((size // 2,), bytes([i]) * size)
    # Returned payloads alone require count * size bytes. The reader must not
    # also retain both a complete-file buffer and a copied payload section.
    assert peak < count * size * 2, f'peak allocation: {peak}'


@pytest.mark.parametrize('offsets, shape, payload', [
    ([0, 2**61], [2**60], b''),
    ([1, 3], [1], b'abc'),
    ([0, 4], [2], b'ab'),
    ([0, 2], [1], b'abc'),
])
def test_reader_rejects_inconsistent_payload_extent(tmp_path, offsets, shape, payload):
    source = tmp_path / 'invalid.safetensors'
    header = json.dumps({'t': {'dtype': 'F16', 'shape': shape, 'data_offsets': offsets}}).encode()
    source.write_bytes(struct.pack('<Q', len(header)) + header + payload)
    with pytest.raises(ValueError):
        prerouter.read_safetensors(source)
