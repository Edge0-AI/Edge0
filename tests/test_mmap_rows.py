"""Exact typed row reads, including buffer lifetime and concurrent callers."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from safetensors.numpy import save_file

from edge0.streaming.mmap import SafetensorsMmap


@pytest.fixture
def shard(tmp_path):
    path = tmp_path / "rows.safetensors"
    save_file({
        "prefix": np.arange(17, dtype=np.uint8),
        "u16": np.arange(9 * 37, dtype=np.uint16).reshape(9, 37),
        "u32": np.arange(9 * 5 * 7, dtype=np.uint32).reshape(9, 5, 7),
        "f32": np.linspace(-2, 2, 9 * 19, dtype=np.float32).reshape(9, 19),
        "scalar": np.array(7, dtype=np.uint32),
        "empty": np.empty((0, 7), dtype=np.uint32),
        "empty_rows": np.empty((9, 0), dtype=np.uint32),
    }, str(path))
    mm = SafetensorsMmap(str(path))
    yield mm
    if not mm._mm.closed:
        mm.close()


@pytest.mark.parametrize("name,dtype", [
    ("u16", "<u2"), ("u32", "<u4"), ("f32", "<f4"),
    ("u32", "u1"), ("u16", "u1"),
])
def test_reader_is_exact_flat_readonly_view(shard, name, dtype):
    read = shard.row_reader(name, dtype)
    raw = shard.raw(name)
    per = raw.size // read.rows
    for row in range(read.rows):
        actual = read(row)
        expected = raw[row * per:(row + 1) * per].view(dtype)
        assert actual.ndim == 1
        assert actual.dtype == np.dtype(dtype)
        assert actual.tobytes() == expected.tobytes()
        assert not actual.flags.writeable
        assert np.shares_memory(actual, raw)
    del actual, expected, raw


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16,
                                   np.int32, np.uint32, np.int64, np.uint64])
def test_reader_accepts_numpy_integer(shard, dtype):
    read = shard.row_reader("u32", "<u4")
    assert read(dtype(8)).tobytes() == read(8).tobytes()


@pytest.mark.parametrize("row", [-1, 9, 1000])
def test_reader_rejects_out_of_range_rows(shard, row):
    with pytest.raises(IndexError):
        shard.row_reader("u16", "<u2")(row)


def test_reader_rejects_fractional_offset(shard):
    with pytest.raises(TypeError):
        shard.row_reader("u32", "<u4")(1.5)


@pytest.mark.parametrize("dtype", [object, "V0", np.dtype([("item", object)])])
def test_reader_rejects_unsafe_dtypes(shard, dtype):
    with pytest.raises(ValueError, match="fixed, non-object"):
        shard.row_reader("u32", dtype)


@pytest.mark.parametrize("name", ["scalar", "empty"])
def test_reader_requires_leading_rows(shard, name):
    with pytest.raises(ValueError, match="leading dimension"):
        shard.row_reader(name, "<u4")


def test_reader_allows_empty_rows(shard):
    assert shard.row_reader("empty_rows", "<u4")(8).size == 0


def test_reader_rejects_nondivisible_row_size(shard):
    with pytest.raises(ValueError, match="divisible"):
        shard.row_reader("u16", "<u4")


def test_reader_construction_reads_no_payload(shard, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("row plan construction read the payload")
    monkeypatch.setattr(np, "frombuffer", forbidden)
    read = shard.row_reader("u32", "<u4")
    assert read.rows == 9
    assert read.count == 35


def test_descriptor_does_not_keep_exported_buffer(shard):
    read = shard.row_reader("u32", "<u4")
    shard.close()
    with pytest.raises(ValueError):
        read(0)


def test_live_view_prevents_close_until_released(shard):
    read = shard.row_reader("u32", "<u4")
    view = read(0)
    with pytest.raises(BufferError):
        shard.close()
    del view
    shard.close()


def test_concurrent_reads_have_no_shared_cursor(shard):
    read = shard.row_reader("u32", "<u4")
    expected = [read(i).tobytes() for i in range(9)]
    indices = np.random.default_rng(4444).integers(0, 9, size=1024)
    with ThreadPoolExecutor(max_workers=8) as pool:
        actual = list(pool.map(lambda i: read(int(i)).tobytes(), indices))
    assert actual == [expected[i] for i in indices]


def test_multiple_shards_have_independent_readers(tmp_path):
    shards = []
    try:
        for i in range(3):
            path = tmp_path / f"{i}.safetensors"
            save_file({"w": np.full((9, 7), i, dtype=np.uint32)}, str(path))
            shards.append(SafetensorsMmap(str(path)))
        readers = [shard.row_reader("w", "<u4") for shard in shards]
        for i, read in enumerate(readers):
            assert np.all(read(4) == i)
    finally:
        for shard in shards:
            shard.close()
