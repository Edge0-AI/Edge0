"""Persistence/index tests run without loading model weights."""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from edge0.conversation import CacheConfig, CheckpointStore
from edge0.conversation.store import Radix


def writer(*payloads):
    def write(put):
        return [put(lambda path, data=data: path.write_bytes(data)) for data in payloads]
    return write


def reader(manifest, paths):
    return [paths[key].read_bytes() for key in manifest]


def test_radix_branches_and_exact():
    tree = Radix()
    for tokens in ([1, 2, 3], [1, 2, 4], [1], [1, 2, 3, 5], [8, 9]):
        tree.insert(tokens, tokens)
    assert tree.longest([1, 2]) == [1]
    assert tree.longest([1, 2, 3]) == [1, 2, 3]
    assert tree.longest([1, 2, 3, 5, 6]) == [1, 2, 3, 5]
    assert tree.longest([8]) is None
    assert tree.longest([2]) is None


def test_restart_namespace_corruption_and_orphans(tmp_path):
    cfg = CacheConfig(str(tmp_path))
    store = CheckpointStore(cfg, 'a')
    store.publish([1, 2], writer(b'first'))
    (tmp_path / ('a' * 32 + '.tmp')).write_bytes(b'partial')
    (tmp_path / ('f' * 64 + '.safetensors')).write_bytes(b'orphan')
    store = CheckpointStore(cfg, 'a')
    assert not list(tmp_path.glob('*.tmp'))
    assert store.restore([1, 2, 3], reader) == ([1, 2], [b'first'])
    assert CheckpointStore(cfg, 'b').restore([1, 2], reader) is None
    next(tmp_path.glob('*.safetensors')).write_bytes(b'corrupt')
    assert store.restore([1, 2], reader) is None
    assert store.inspect()['checkpoints'] == 0


def test_shared_eviction_and_oversized(tmp_path):
    store = CheckpointStore(CacheConfig(str(tmp_path), budget_bytes=10), 'a')
    store.publish([1], writer(b'abc', b'def'))
    store.publish([2], writer(b'abc', b'ghi'))
    assert store.inspect()['stored_bytes'] == 9
    store.publish([3], writer(b'abc', b'jkl'))
    assert store.restore([1], reader) is None
    assert store.restore([2], reader)[1] == [b'abc', b'ghi']
    store.publish([4], writer(b'x' * 11))
    assert store.restore([4], reader) is None
    assert store.inspect()['stored_bytes'] <= 10
    assert sum(p.stat().st_size for p in tmp_path.glob('*.safetensors')) <= 10


def test_failed_writer_never_published(tmp_path):
    store = CheckpointStore(CacheConfig(str(tmp_path)), 'a')
    def fail(put):
        put(lambda p: p.write_bytes(b'first'))
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):
        store.publish([1], fail)
    assert store.restore([1], reader) is None
    assert not list(tmp_path.glob('*.safetensors'))


def test_concurrent_instances_refresh(tmp_path):
    cfg = CacheConfig(str(tmp_path))
    stores = [CheckpointStore(cfg, 'a') for _ in range(4)]
    def worker(i):
        stores[i % 4].publish([i, i + 1], writer(str(i).encode()))
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(worker, range(40)))
    assert stores[0].inspect()['checkpoints'] == 40
    for i in range(40):
        assert stores[0].restore([i, i + 1, 99], reader)[1] == [str(i).encode()]


def test_tokenization_bounded_and_exact(tmp_path):
    store = CheckpointStore(CacheConfig(str(tmp_path), token_budget_bytes=10), 'a')
    calls = []
    def encode():
        calls.append(1)
        return [1, 2]
    assert store.tokenize({'thinking': False}, encode) == [1, 2]
    store.tokenize({'thinking': False}, encode)
    assert len(calls) == 1
    store.tokenize({'thinking': True}, encode)
    store.tokenize({'messages': ['different']}, encode)
    assert len(calls) == 3
    assert store.inspect()['tokenization_bytes'] <= 10


def test_artifact_identity_validation(tmp_path, monkeypatch):
    from dataclasses import dataclass
    from types import SimpleNamespace
    from edge0.conversation import store as module
    @dataclass
    class Config:
        lora: str = ''
        prerouter: object = None
    model = tmp_path / 'model'
    model.mkdir()
    artifact = model / 'model.safetensors'
    artifact.write_bytes(b'weights')
    engine = SimpleNamespace(dir=str(model), cfg=Config(), _tok=None)
    store = CheckpointStore(CacheConfig(str(tmp_path / 'cache')), '')
    calls = []
    original = module.file_hash
    def track(path):
        if path == artifact:
            calls.append(path)
        return original(path)
    monkeypatch.setattr(module, 'file_hash', track)
    first = module.engine_namespace(engine, store)
    assert module.engine_namespace(engine, store) == first
    assert len(calls) == 1
    artifact.write_bytes(b'changed')
    assert module.engine_namespace(engine, store) != first
    assert len(calls) == 2


def test_missing_payload(tmp_path):
    store = CheckpointStore(CacheConfig(str(tmp_path)), 'a')
    store.publish([1], writer(b'one'))
    next(tmp_path.glob('*.safetensors')).unlink()
    assert store.restore([1], reader) is None


def test_multiprocess_access(tmp_path):
    import subprocess
    import sys
    program = '''
import sys
from edge0.conversation import CacheConfig, CheckpointStore
s = CheckpointStore(CacheConfig(sys.argv[1]), 'a')
for i in range(12):
    s.publish([int(sys.argv[2]), i], lambda put: [put(lambda p: p.write_bytes(b'shared'))])
'''
    workers = [subprocess.Popen([sys.executable, '-c', program, str(tmp_path), str(i)]) for i in range(3)]
    for worker in workers:
        assert worker.wait(timeout=30) == 0
    store = CheckpointStore(CacheConfig(str(tmp_path)), 'a')
    assert store.inspect()['checkpoints'] == 36
    assert store.inspect()['stored_bytes'] == len(b'shared')


def test_metadata_corruption_and_shared_repair(tmp_path):
    store = CheckpointStore(CacheConfig(str(tmp_path)), 'a')
    store.publish([1], writer(b'shared'))
    store.publish([2], writer(b'shared'))
    next(tmp_path.glob('*.safetensors')).write_bytes(b'broken')
    assert store.restore([1], reader) is None
    store.publish([1], writer(b'shared'))
    assert store.restore([2], reader)[1] == [b'shared']
    with store.locked() as db:
        db.execute("UPDATE checkpoints SET manifest=?", (b'[]',))
    assert store.restore([1], reader) is None


def test_cleanup_preserves_unmanaged_files(tmp_path):
    (tmp_path / 'model.safetensors').write_bytes(b'not a cache payload')
    (tmp_path / 'user.tmp').write_bytes(b'not a cache temporary')
    store = CheckpointStore(CacheConfig(str(tmp_path)), 'a')
    store.clear()
    assert (tmp_path / 'model.safetensors').exists()
    assert (tmp_path / 'user.tmp').exists()


def test_local_writes_update_index_without_rebuild(tmp_path, monkeypatch):
    store = CheckpointStore(CacheConfig(str(tmp_path)), 'a')
    store.publish([1], writer(b'one'))
    index = store.index
    store.publish([1, 2], writer(b'two'))
    assert store.restore([1, 2, 3], reader)[0] == [1, 2]
    assert store.index is index


def test_physical_budget_with_metadata(tmp_path):
    cfg = CacheConfig(str(tmp_path), budget_bytes=2048, token_budget_bytes=1024)
    store = CheckpointStore(cfg, 'a')
    for i in range(20):
        store.publish([i] * 300, writer(bytes([i]) * 1000))
        store.tokenize(str(i), lambda: list(range(100)))
    physical = sum(p.stat().st_size for p in tmp_path.iterdir() if p.is_file())
    assert physical <= cfg.budget_bytes + cfg.token_budget_bytes + 65536


def test_process_exit_before_metadata_publication(tmp_path):
    import subprocess
    import sys
    program = '''
import os, sys
from edge0.conversation import CacheConfig, CheckpointStore
s = CheckpointStore(CacheConfig(sys.argv[1]), 'a')
def write(put):
    put(lambda p: p.write_bytes(b'unpublished'))
    os._exit(9)
s.publish([1, 2], write)
'''
    result = subprocess.run([sys.executable, '-c', program, str(tmp_path)], timeout=30)
    assert result.returncode == 9
    store = CheckpointStore(CacheConfig(str(tmp_path)), 'a')
    assert store.restore([1, 2], reader) is None
    assert not list(tmp_path.glob('*.safetensors'))
