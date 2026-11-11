"""setup.check_shards: a metadata-only GGUF shard that ends at its header (no alignment padding, #2346) is whole;
a shard whose tensor data runs past its end is still refused, including inside an integer tensor."""
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

# ggml type id -> bytes per value, for the unquantised types these tests write
VALUE_BYTES = {0: 4, 27: 8}                     # F32, I64


def _str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def gguf(path, tensors=(), pad_data=0):
    """A GGUF v3 file with one string key and the given (name, n_values, type_id) tensors."""
    kv = _str("general.name") + struct.pack("<I", 8) + _str("t")
    tinfo = b""
    off = 0
    for name, n, type_id in tensors:
        tinfo += _str(name) + struct.pack("<I", 1) + struct.pack("<Q", n)
        tinfo += struct.pack("<I", type_id) + struct.pack("<Q", off)
        off += ((n * VALUE_BYTES[type_id] + 31) // 32) * 32
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1) + kv + tinfo
    body = head
    if tensors:
        body += b"\0" * ((32 - len(head) % 32) % 32)
        body += b"\0" * pad_data
    path.write_bytes(body)
    return len(head)


class ShardCheck(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def test_metadata_only_shard_without_padding_is_whole(self):
        p = self.d / "m-00001-of-00006.gguf"
        n = gguf(p)
        self.assertNotEqual(n % 32, 0, "the test file must end unaligned, like the real shard")
        setup.check_shards([p])          # must not stop setup

    def test_truncated_tensor_data_is_still_refused(self):
        p = self.d / "t-00002-of-00006.gguf"
        gguf(p, tensors=[("w", 64, 0)], pad_data=100)       # needs 256 bytes of data, has 100
        with self.assertRaises(SystemExit):
            setup.check_shards([p])

    def test_a_complete_gguf_with_integer_tensors_is_whole(self):
        # The IST-DAS DeepSeek-V4.1 first shard keeps the Engram's hash parameters as I32/I64 tensors
        # (tools/dsv41/pack_gguf.py), so a complete file of that set must not be turned away as cut short.
        p = self.d / "i-00001-of-00004.gguf"
        gguf(p, tensors=[("engram_primes", 3, 27)], pad_data=24)       # exactly its 24 bytes
        self.assertTrue(setup.own_gguf_is_whole(p))
        setup.check_shards([p])          # must not stop setup

    def test_a_shard_cut_inside_an_integer_tensor_is_still_refused(self):
        p = self.d / "j-00001-of-00004.gguf"
        gguf(p, tensors=[("engram_primes", 3, 27)], pad_data=16)       # needs 24 bytes of data, has 16
        with self.assertRaises(SystemExit):
            setup.check_shards([p])


if __name__ == "__main__":
    unittest.main()
