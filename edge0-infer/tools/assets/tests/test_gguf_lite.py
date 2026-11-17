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


@pytest.mark.parametrize('values', [
    np.full(32, 65520, dtype=np.float32),
    np.full(32, -65520, dtype=np.float32),
    np.tile(np.array([0, 2_000_000], dtype=np.float32), 16),
    np.tile(np.array([-np.finfo(np.float32).max, np.finfo(np.float32).max]), 16),
])
def test_q41_rejects_unrepresentable_block_metadata(values):
    assert np.isfinite(values).all()
    with pytest.raises(ValueError, match='representable in F16'):
        pack_q4_1(values)


def test_q41_accepts_finite_f16_boundary():
    for value in (65504, -65504):
        payload = pack_q4_1(np.full(32, value, dtype=np.float32))
        scale, minimum = struct.unpack('<ee', payload[:4])
        assert scale == 0 and minimum == value
        assert payload[4:] == bytes(16)
