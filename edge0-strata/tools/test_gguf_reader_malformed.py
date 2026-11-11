"""A malformed GGUF header is refused with a ValueError that names the file, never a KeyError, OverflowError
or struct.error (#1919). Tensor-size support and setup compatibility policy are unchanged.
Over minimal GGUF v3 files written here (no download, no model, no GPU).

    python -m unittest tools.test_gguf_reader_malformed
"""
from __future__ import annotations

import contextlib
import io
import struct
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import setup as S  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402


class Stop(Exception):
    """setup.fail(), caught instead of exiting."""


def gstr(s: bytes) -> bytes:
    return struct.pack("<Q", len(s)) + s


def header(n_tensors: int, kv: bytes = b"", n_kv: int = 0) -> bytes:
    return struct.pack("<IIQQ", 0x46554747, 3, n_tensors, n_kv) + kv


def tensor(name: bytes, n: int, type_id: int, offset: int) -> bytes:
    return gstr(name) + struct.pack("<IQIQ", 1, n, type_id, offset)


def pad(b: bytes, data: int) -> bytes:
    return b + bytes(-len(b) % 32 + data)


class MalformedHeader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "m-00001-of-00002.gguf"

    def tearDown(self):
        self.tmp.cleanup()

    def refused(self, raw: bytes) -> str:
        self.path.write_bytes(raw)
        with self.assertRaises(ValueError) as cm:
            GGUFFile(self.path)
        self.assertIn(self.path.name, str(cm.exception))
        return str(cm.exception)

    def test_unknown_metadata_value_type(self):
        self.refused(pad(header(0, gstr(b"k") + struct.pack("<I", 99), 1), 0))

    def test_unknown_array_element_type(self):
        self.refused(pad(header(0, gstr(b"k") + struct.pack("<IIQ", 9, 99, 1), 1), 0))

    def test_string_length_past_the_end(self):
        self.refused(pad(header(0, struct.pack("<Q", 2 ** 63), 1), 0))

    def test_array_count_past_the_end(self):
        self.refused(pad(header(0, gstr(b"k") + struct.pack("<IIQ", 9, 4, 2 ** 40), 1), 0))

    def test_string_array_count_past_the_end(self):
        self.refused(pad(header(0, gstr(b"k") + struct.pack("<IIQ", 9, 8, 2 ** 62), 1), 0))

    def test_tensor_count_past_the_end(self):
        self.refused(header(2 ** 40))

    def test_header_cut_mid_tensor_directory(self):
        self.refused(header(1) + tensor(b"blk.0.w", 8, 0, 0)[:-5])

    def test_header_cut_inside_the_magic(self):
        self.refused(b"GG")


class Alignment(unittest.TestCase):
    setUp = MalformedHeader.setUp
    tearDown = MalformedHeader.tearDown
    refused = MalformedHeader.refused
    def test_invalid_alignment(self):
        for alignment in (0, -8, 3):
            with self.subTest(alignment=alignment):
                kv = gstr(b"general.alignment") + struct.pack("<Ii", 5, alignment)
                self.assertIn("alignment", self.refused(pad(header(0, kv, 1), 0)))


class UnknownTensorCompatibility(unittest.TestCase):
    def test_readable_unknown_tensor_keeps_setup_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "future.gguf"
            raw = pad(header(2) + tensor(b"known", 8, 0, 0) + tensor(b"future", 64, 999, 32), 256)
            p.write_bytes(raw)
            g = GGUFFile(p)
            self.assertIsNone(g.tensors[1].expected_bytes())
            self.assertTrue(S.whole_shard(p))
            S.check_shards([p])
            self.assertEqual(p.read_bytes(), raw)


class ReaderCompatibility(unittest.TestCase):
    setUp = MalformedHeader.setUp
    tearDown = MalformedHeader.tearDown
    refused = MalformedHeader.refused

    def test_supported_scalar_and_array_values(self):
        values = [(0, 'B', 255), (1, 'b', -7), (2, 'H', 65535), (3, 'h', -300),
                  (4, 'I', 2**31), (5, 'i', -70000), (6, 'f', 1.25), (7, '?', True),
                  (10, 'Q', 2**63), (11, 'q', -2**40), (12, 'd', 1.125)]
        kv, expected = b'', {}
        for type_id, fmt, value in values:
            for array in (False, True):
                key = f'{type_id}_{array}'
                data = struct.pack('<' + fmt, value)
                kv += gstr(key.encode()) + (struct.pack('<IIQ', 9, type_id, 1) if array
                                            else struct.pack('<I', type_id)) + data
                expected[key] = [value] if array else value
        for key, data, value in [('text', struct.pack('<I', 8) + gstr('测试'.encode()), '测试'),
                                 ('strings', struct.pack('<IIQ', 9, 8, 2) + gstr(b'') + gstr(b'x'), ['', 'x']),
                                 ('empty', struct.pack('<IIQ', 9, 4, 0), [])]:
            kv += gstr(key.encode()) + data
            expected[key] = value
        self.path.write_bytes(pad(header(1, kv, len(expected)) + tensor(b'w', 8, 0, 0), 32))
        g = GGUFFile(self.path)
        self.assertEqual(g.metadata, expected)
        self.assertEqual((g.tensors[0].name, g.tensors[0].shape, g.tensors[0].expected_bytes()), ('w', [8], 32))
        S.check_shards([self.path])

    def test_metadata_only_file_needs_no_alignment_padding(self):
        raw = header(0, gstr(b'general.name') + struct.pack('<I', 8) + gstr(b't'), 1)
        self.path.write_bytes(raw)
        g = GGUFFile(self.path)
        self.assertEqual(g.metadata, {'general.name': 't'})
        self.assertEqual(g.tensors, [])
        self.assertGreater(g.data_start, len(raw))
        S.check_shards([self.path])
        self.assertEqual(self.path.read_bytes(), raw)

    def test_explicit_alignment_keeps_tensor_offsets(self):
        for alignment in (24, 64):
            with self.subTest(alignment=alignment):
                kv = gstr(b'general.alignment') + struct.pack('<II', 4, alignment)
                raw = header(1, kv, 1) + tensor(b'w', 8, 0, 0)
                start = (len(raw) + alignment - 1) // alignment * alignment
                self.path.write_bytes(raw + bytes(start - len(raw) + 32))
                g = GGUFFile(self.path)
                self.assertEqual((g.alignment, g.data_start), (alignment, start))
                S.check_shards([self.path])

    def test_setup_reports_malformed_file_without_changing_it(self):
        for raw in (header(0, gstr(b'k') + struct.pack('<I', 99), 1),
                    header(0, struct.pack('<Q', 2**63) + bytes(4), 1)):
            with self.subTest(raw=raw.hex()):
                self.path.write_bytes(raw)
                self.assertFalse(S.whole_shard(self.path))
                out = io.StringIO()
                with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
                    S.check_shards([self.path])
                self.assertIn(self.path.name + ' is not a whole GGUF shard', out.getvalue())
                self.assertEqual(self.path.read_bytes(), raw)

    def test_unsupported_array_and_impossible_dimension_count_are_named(self):
        self.refused(header(0, gstr(b'k') + struct.pack('<IIQ', 9, 9, 0), 1))
        self.refused(header(1) + gstr(b'w') + struct.pack('<I', 2**32 - 1) + bytes(12))


if __name__ == "__main__":
    unittest.main()
