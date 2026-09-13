"""Page-warming contracts, runnable with NumPy + pytest and no MLX.

Load the backend-free source directly: edge0.streaming.__init__ imports
its MLX layer, which is intentionally unnecessary for these I/O tests.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest

_SOURCE = Path(__file__).resolve().parents[1] / "src/edge0/streaming/mmap.py"
_SPEC = importlib.util.spec_from_file_location("edge0_mmap_under_test", _SOURCE)
mmap_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mmap_module)
SafetensorsMmap = mmap_module.SafetensorsMmap


def write_shard(path, size=100003):
    payload = np.random.default_rng(73).integers(0, 256, size, dtype=np.uint8)
    header = json.dumps({
        "w": {"dtype": "U8", "shape": [size], "data_offsets": [0, size]},
        "empty": {"dtype": "U8", "shape": [0], "data_offsets": [size, size]},
    }).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload.tobytes())
    return payload


@pytest.fixture
def shard(tmp_path):
    path = tmp_path / "weights.safetensors"
    payload = write_shard(path)
    mapped = SafetensorsMmap(str(path))
    yield mapped, path, payload
    mapped.close()


def test_warming_preserves_bytes_and_raw_readonly_contract(shard):
    mapped, path, payload = shard
    before = path.read_bytes()
    assert mapped.touch() is None
    assert mapped.touch(len(before), 0) is None
    mapped.seq_read()
    mapped.seq_read(7919)
    raw = mapped.raw("w")
    np.testing.assert_array_equal(raw, payload)
    assert not raw.flags.writeable
    assert not raw.flags.owndata
    assert mapped.raw("empty").size == 0
    assert path.read_bytes() == before
    del raw  # no outstanding buffer when the fixture closes the mmap


@pytest.mark.parametrize("offset,length", [(-1, 0), (10**9, 0), (0, -1),
                                           (0, 10**9), (100000, 100000)])
def test_out_of_range_rejected(shard, offset, length):
    with pytest.raises(ValueError):
        shard[0].touch(offset, length)


@pytest.mark.parametrize("offset,length", [(0.5, 1), (0, 1.5)])
def test_noninteger_ranges_rejected(shard, offset, length):
    with pytest.raises(TypeError):
        shard[0].touch(offset, length)


@pytest.mark.parametrize("chunk", [0, -1, -4096])
def test_nonpositive_chunks_rejected(shard, chunk):
    with pytest.raises(ValueError):
        shard[0].seq_read(chunk)


def test_numpy_integer_arguments(shard):
    shard[0].touch(np.int64(3), np.int64(71))
    shard[0].seq_read(np.int64(7919))


class ReadTrace:
    """Instrument both scalar mmap reads and strided NumPy consumption."""

    def __init__(self, length):
        self.length = length
        self.reads = []

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # A slice here would be a payload-sized bytes copy.
        assert isinstance(index, int)
        assert 0 <= index < self.length
        self.reads.append(index)
        return 0

    def frombuffer(self, buffer, *, dtype, count, offset):
        assert buffer is self
        assert dtype == np.uint8
        assert 0 <= offset <= offset + count <= self.length
        owner = self

        class View:
            def __init__(self, positions):
                self.positions = positions

            def __getitem__(self, key):
                return View(self.positions[key])

            def sum(self, dtype):
                assert dtype == np.uint64
                owner.reads.extend(self.positions)
                return 0

        return View(range(offset, offset + count))


def traced_shard(monkeypatch, page_size, length):
    trace = ReadTrace(length)
    mapped = object.__new__(SafetensorsMmap)
    mapped._mm = trace
    # Replace only the module-local namespaces, not global NumPy/mmap.
    monkeypatch.setattr(mmap_module, "mmap", SimpleNamespace(PAGESIZE=page_size))
    monkeypatch.setattr(mmap_module, "np", SimpleNamespace(
        frombuffer=trace.frombuffer, uint8=np.uint8, uint64=np.uint64))
    return mapped, trace


@pytest.mark.parametrize("page_size", [4096, 16384])
def test_exact_page_coverage_including_unaligned_tail(monkeypatch, page_size):
    total = 11 * page_size + 37
    mapped, trace = traced_shard(monkeypatch, page_size, total)
    cases = [(0, 0), (total, 0), (0, total), (page_size - 1, 2),
             (page_size - 1, page_size + 2), (3, page_size),
             (page_size, page_size), (total - 1, 1)]
    rng = np.random.default_rng(91)
    for _ in range(1000):
        start = int(rng.integers(total + 1))
        cases.append((start, int(rng.integers(total - start + 1))))
    for start, length in cases:
        trace.reads.clear()
        mapped.touch(start, length)
        expected = (set(range(start // page_size,
                              (start + length - 1) // page_size + 1))
                    if length else set())
        actual = [address // page_size for address in trace.reads]
        assert set(actual) == expected
        assert len(actual) == len(expected), "must read exactly once per page"
        assert all(start <= address < start + length for address in trace.reads)


@pytest.mark.parametrize("page_size", [4096, 16384])
@pytest.mark.parametrize("chunk_kind", ["small", "unaligned", "page", "large"])
def test_seq_read_covers_all_pages_without_copy(monkeypatch, page_size, chunk_kind):
    chunks = {"small": 97, "unaligned": page_size - 1,
              "page": page_size, "large": 1 << 24}
    total = 5 * page_size + 13
    mapped, trace = traced_shard(monkeypatch, page_size, total)
    mapped.seq_read(chunks[chunk_kind])
    assert {a // page_size for a in trace.reads} == set(range(6))
    assert all(0 <= a < total for a in trace.reads)


def test_close_has_no_retained_numpy_views(tmp_path):
    path = tmp_path / "weights.safetensors"
    write_shard(path)
    mapped = SafetensorsMmap(str(path))
    for _ in range(10):
        mapped.touch(3, 99997)
        mapped.seq_read(8191)
    mapped.close()  # would raise BufferError if a warming view escaped
    assert mapped._file.closed
