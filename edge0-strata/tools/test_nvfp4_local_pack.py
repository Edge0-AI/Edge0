"""Tests for tools/nvfp4_local_pack.py: the CSF scale decode, the MXFP8 dequantization, the PLE rows."""
import json
import pathlib
import struct
import sys

import numpy as np
import unittest

try:
    import pytest
except ImportError:                                  # not in requirements.txt: python -m unittest skips this module instead of failing to import it (#3325)
    raise unittest.SkipTest("pytest is not installed")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import nvfp4_local_pack as P  # noqa: E402


def _scales(rows, cols, seed, outliers=0.01):
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 100, size=(rows, 1))
    s = base + rng.integers(0, 16, size=(rows, cols))
    m = rng.random((rows, cols)) < outliers
    s[m] = rng.integers(0, 127, size=int(m.sum()))
    return s.astype(np.uint8)


@pytest.mark.parametrize("rows,cols", [(32, 8), (640, 160), (2560, 40)])
def test_csf_roundtrip_is_exact(rows, cols):
    s = _scales(rows, cols, rows + cols)
    fixed, exc = P.csf_encode(s)
    assert fixed.shape == (rows // 16, 16 * (1 + cols // 2))     # the shapes of the real tensors
    assert np.array_equal(P.csf_decode(fixed, exc), s)


def test_csf_matches_the_spec_by_hand():
    # one 16-row slab, 4 columns: base bytes 10..25, rows' codes = col index, one exception at row 3 col 2
    rows, cols = 16, 4
    fixed = np.zeros((1, 16 * (1 + cols // 2)), np.uint8)
    fixed[0, :16] = np.arange(10, 26)
    for r in range(rows):
        fixed[0, 16 + r * 2] = 0 | (1 << 4)       # cols 0 (low nibble), 1 (high)
        fixed[0, 16 + r * 2 + 1] = 2 | (3 << 4)   # cols 2, 3
    exc = np.array([(3 * cols + 2) | (99 << 24)], np.uint32)
    got = P.csf_decode(fixed, exc)
    assert got[0].tolist() == [10, 11, 12, 13]
    assert got[5].tolist() == [15, 16, 17, 18]
    assert got[3].tolist() == [13, 14, 99, 16]


def test_csf_rejects_bad_streams():
    with pytest.raises(ValueError):
        P.csf_decode(np.zeros((1, 33), np.uint8), np.zeros(0, np.uint32))
    fixed = np.zeros((1, 32), np.uint8)
    fixed[0, 0] = 250
    with pytest.raises(ValueError):
        P.csf_decode(fixed, np.zeros(0, np.uint32))
    with pytest.raises(ValueError):
        P.csf_decode(np.zeros((1, 32), np.uint8), np.array([16 * 4], np.uint32))


def test_mxfp8_dequant_exact():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    vals = rng.integers(0, 126, size=(4, 64)).astype(np.uint8) | (rng.integers(0, 2, size=(4, 64)).astype(np.uint8) << 7)
    sc = rng.integers(100, 130, size=(4, 2)).astype(np.uint8)
    got = P.mxfp8_dequant(vals, sc).numpy()
    ref = torch.from_numpy(vals).view(torch.float8_e4m3fn).to(torch.float32).numpy().reshape(4, 2, 32)
    ref = ref * np.exp2(sc.astype(np.float64) - 127)[:, :, None]
    assert np.array_equal(got.reshape(4, 2, 32), ref.astype(np.float32))
    # exactly representable in BF16 (what the converter hands on)
    bf = torch.from_numpy(got).to(torch.bfloat16).to(torch.float32).numpy()
    assert np.array_equal(bf, got)


def test_mxfp8_rejects_nan_scale():
    pytest.importorskip("torch")
    with pytest.raises(ValueError):
        P.mxfp8_dequant(np.zeros((1, 32), np.uint8), np.full((1, 1), 255, np.uint8))


def test_ple_rows_are_codes_then_scales():
    c = np.arange(160, dtype=np.uint8).reshape(2, 80)
    s = np.arange(20, dtype=np.uint8).reshape(2, 10) + 200
    r = P.interleave_rows(c, s)
    assert r.shape == (2, P.PLE_ROW_BYTES)
    assert np.array_equal(r[1, :80], c[1]) and np.array_equal(r[1, 80:], s[1])


def test_nvfp4_dequant_by_hand():
    # element 2i in the low nibble: byte 0x21 = codes 1 (0.5) and 2 (1.0); 0xF7 = 7 (6.0) and 15 (-6.0)
    codes = np.zeros((1, 8), np.uint8)
    codes[0, 0], codes[0, 1] = 0x21, 0xF7
    sc = np.array([[0x38]], np.uint8)            # E4M3 0x38 = 1.0
    got = P.nvfp4_dequant(codes, sc, 2.0)
    assert got[0, :4].tolist() == [1.0, 2.0, 12.0, -12.0]


def test_e4m3_table_and_bf16_rounding():
    t = P.e4m3_table()
    assert t[0x38] == 1.0 and t[0x7E] == 448.0 and np.isnan(t[0x7F]) and t[0x01] == 2.0 ** -9
    x = np.array([1.0, 1.0 + 2.0 ** -8, 1.0 + 3 * 2.0 ** -9, -2.5], np.float32)
    b = P.f32_to_bf16_bits(x)
    back = (b.astype(np.uint32) << 16).view(np.float32)
    assert back.tolist() == [1.0, 1.0, 1.0078125, -2.5]    # ties to even; 1+3*2^-9 rounds up


def test_nvfp4_blocks_layout_by_hand():
    # one row of 64 values: codes 0..15 repeated; scales 0x38 0x39 0x3A 0x3B with a sign bit set on the first (dropped)
    codes = np.arange(64) % 16
    packed = (codes[0::2] | (codes[1::2] << 4)).astype(np.uint8).reshape(1, 32)
    sc = np.array([[0xB8, 0x39, 0x3A, 0x3B]], np.uint8)
    out = P.nvfp4_blocks(packed, sc)
    assert out.shape == (1, 36)
    assert out[0, :4].tolist() == [0x38, 0x39, 0x3A, 0x3B]
    # within a 16-block: element j in the low nibble, element j+8 in the high one
    blk0 = out[0, 4:12]
    assert [int(b & 15) for b in blk0] == list(range(8)) and [int(b >> 4) for b in blk0] == list(range(8, 16))


def _write_shard(path, tensors):
    """tensors: name -> (dtype, shape, uint8 bytes), written safetensors-style (data after the header)."""
    hdr, data, off = {}, bytearray(), 0
    for name, (dt, shape, buf) in tensors.items():
        hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(buf)]}
        data += buf
        off += len(buf)
    hj = json.dumps(hdr).encode("utf-8")
    hj += b" " * (-len(hj) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)) + hj + data)


def test_checkpoint_reads_the_shard_bytes_exactly(tmp_path):
    a = np.arange(200, dtype=np.uint8)
    _write_shard(tmp_path / "s1.safetensors", {"t.a": ("U8", [100, 2], a.tobytes())})
    d = np.arange(4, dtype=np.uint32)
    c = np.arange(32, dtype=np.uint8)
    _write_shard(tmp_path / "s2.safetensors", {"t.d": ("U32", [4], d.tobytes()), "t.c": ("U8", [8, 4], c.tobytes())})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "t.a": "s1.safetensors", "t.c": "s2.safetensors", "t.d": "s2.safetensors"}}), encoding="utf-8")
    ck = P.Checkpoint(tmp_path)
    assert "t.a" in ck and ck.shape("t.a") == [100, 2] and ck.dtype("t.a") == "U8"
    assert np.array_equal(ck.raw("t.a"), a)                       # the bytes at the tensor's exact range
    assert np.array_equal(ck.array("t.c", np.uint8), c.reshape(8, 4))
    assert np.array_equal(ck.array("t.d", np.uint32), d)          # .view through a fresh buffer
    assert np.array_equal(ck.raw("t.a"), ck.raw("t.a"))           # a second read is still correct


def test_checkpoint_names_a_truncated_shard(tmp_path):
    c = np.arange(32, dtype=np.uint8)
    _write_shard(tmp_path / "s1.safetensors", {"t.c": ("U8", [8, 4], c.tobytes())})
    p = tmp_path / "s1.safetensors"
    p.write_bytes(p.read_bytes()[:-8])                            # the last tensor's range now ends past EOF
    ck = P.Checkpoint(tmp_path)
    with pytest.raises(EOFError):
        ck.raw("t.c")
