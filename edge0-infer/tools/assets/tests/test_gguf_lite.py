import struct

import numpy as np
import pytest

from format_gguf import F16, F32, Q4_1, Reader, Value, pack_q4_1, write

pytestmark = pytest.mark.l1


def test_metadata_and_tensor_roundtrip(tmp_path):
    metadata = {str(t): Value(t, v) for t, v in [
        (0, 255), (1, -128), (2, 65535), (3, -32768),
        (4, 2**32 - 1), (5, -2**31), (6, 0.5), (7, True),
        (8, 'test\U0001f600'), (10, 2**64 - 1), (11, -2**63), (12, 0.125)]}
    metadata['array'] = Value(9, ['one', 'two'], 8)
    metadata['empty'] = Value(9, [], 4)
    for t in [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12]:
        metadata[f'array-{t}'] = Value(9, [metadata[str(t)].data], t)
    tensors = [('f32', (2, 3), F32, bytes(24)),
               ('f16', (7,), F16, bytes(14)),
               ('q41', (32, 2), Q4_1, bytes(40))]
    path = tmp_path / 'model.gguf'
    write(path, metadata, tensors)
    r = Reader(path)
    assert r.metadata == metadata
    assert r.data_offset % 32 == 0
    for name, shape, type, data in tensors:
        assert r.tensors[name].shape == shape
        assert r.tensors[name].type == type
        assert r.tensors[name].offset % 32 == 0
        assert r.data(name) == data


def test_q41_hand_computed():
    # d=1, m=-8; both halves contain quantized nibbles 0..15.
    x = np.tile(np.arange(-8, 8, dtype=np.float32), 2)
    expected = struct.pack('<ee', 1, -8) + bytes(i | (i << 4) for i in range(16))
    assert pack_q4_1(x) == expected
    assert pack_q4_1(np.full(32, 2, dtype=np.float32)) == struct.pack('<ee', 0, 2) + bytes(16)


def test_truncation(tmp_path):
    path = tmp_path / 'valid.gguf'
    write(path, {'name': Value(8, 'hello')}, [('t', (32,), Q4_1, bytes(20))])
    complete = path.read_bytes()
    for length in [0, 3, 7, 20, 30, len(complete) - 1]:
        bad = tmp_path / str(length)
        bad.write_bytes(complete[:length])
        with pytest.raises(ValueError):
            Reader(bad)


def test_invalid_payloads(tmp_path):
    with pytest.raises(ValueError):
        write(tmp_path / 'bad.gguf', {}, [('t', (3,), Q4_1, bytes(20))])
    with pytest.raises(ValueError):
        pack_q4_1(np.full(32, np.nan))
    with pytest.raises(ValueError):
        pack_q4_1(np.zeros(31))


def test_numeric_metadata_arrays_use_bounded_block_reads(tmp_path, monkeypatch):
    import format_gguf

    path = tmp_path / 'scores.gguf'
    values = list(range(25000))
    write(path, {'scores': Value(9, values, 4)}, [])
    original = format_gguf._read
    requests = []

    def counted(stream, size):
        requests.append(size)
        return original(stream, size)

    monkeypatch.setattr(format_gguf, '_read', counted)
    reader = Reader(path)
    assert reader.metadata['scores'].data == values
    print(f'read_calls={len(requests)}, max_read={max(requests)}')
    assert len(requests) < 32
    assert max(requests) <= 65536


@pytest.mark.parametrize('type,value', [(0, 255), (1, -128), (2, 65535), (3, -32768),
                                      (4, 2**32-1), (5, -2**31), (6, 0.5), (7, True),
                                      (10, 2**64-1), (11, -2**63), (12, 0.125)])
def test_numeric_array_block_boundaries(tmp_path, type, value):
    path = tmp_path / 'array.gguf'
    values = [value] * 8193
    write(path, {'values': Value(9, values, type), 'empty': Value(9, [], type)}, [])
    reader = Reader(path)
    assert reader.metadata['values'].data == values
    assert reader.metadata['empty'].data == []
    # Cut inside the array, not in the writer's final padding.
    size = struct.calcsize('<' + __import__('format_gguf').SCALARS[type])
    path.write_bytes(path.read_bytes()[:24 + 8 + len('values') + 4 + 4 + 8 + len(values)*size - 1])
    with pytest.raises(ValueError, match='truncated'):
        Reader(path)
