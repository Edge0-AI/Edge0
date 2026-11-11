"""The pair-trellis ggml types (147-149: TQP5 / TQP6 / TQP7, Gyro-L): the Python reader and the C++ reader agree on
their names and block sizes, and the sizes follow the layout (fp16 scale + 8 * K bytes per 128 values)."""

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import gguf_reader  # noqa: E402


class PairTrellisTypes(unittest.TestCase):
    def test_python_reader_names_and_sizes(self):
        for t, name, k in ((147, "TQP5", 5), (148, "TQP6", 6), (149, "TQP7", 7)):
            self.assertEqual(gguf_reader.GGML_TYPES[t], name)
            self.assertEqual(gguf_reader.BLOCK_GEOMETRY[name], (128, 2 + 8 * k))

    def test_older_trellis_types_unchanged(self):
        self.assertEqual(gguf_reader.BLOCK_GEOMETRY["TQ2_T"], (128, 34))
        self.assertEqual(gguf_reader.BLOCK_GEOMETRY["TQK6"], (128, 26))
        self.assertEqual(gguf_reader.BLOCK_GEOMETRY["TQK7"], (128, 30))

    def test_cpp_reader_has_the_same_numbers(self):
        text = (ROOT / "include" / "strata" / "artifact" / "gguf_reader.hpp").read_text(encoding="utf-8")
        for t, name, k in ((147, "TQP5", 5), (148, "TQP6", 6), (149, "TQP7", 7)):
            self.assertRegex(text, r"case %d:\s*return \"%s\";" % (t, name))
            self.assertRegex(text, r"case %d:\s*elems = 128;\s*bytes = %d;" % (t, 2 + 8 * k))


if __name__ == "__main__":
    unittest.main()
