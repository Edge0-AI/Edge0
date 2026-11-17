"""A four-part version (0.1.40.2) cannot break setup: versions compare all four numbers (0.1.40 < 0.1.40.2, so a
0.1.40 engine is updated), the engine zips are looked for under the checkout's own v<project version> tag first and
the latest release second, and a binary that says `engine=0.1.40.2` is read as (0, 1, 40, 2).

    python -m unittest tools.test_setup_hotfix_tag
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


class HotfixTag(unittest.TestCase):
    def test_a_four_part_version_compares_all_four_numbers(self):
        ver = tuple(int(x) for x in "0.1.40.3".split("."))   # a hotfix version has four numbers; MIN_ENGINE may have three
        self.assertEqual(len(ver), 4)
        self.assertLess(ver, setup.MIN_ENGINE)                 # an installed hotfix of the older release is replaced
        self.assertGreaterEqual((0, 1, 46), setup.MIN_ENGINE)   # the minimum itself (or a later hotfix) is accepted
        self.assertLess((0, 1, 45), setup.MIN_ENGINE)       # 0.1.45 is replaced (#3469)
        self.assertLess((0, 1, 44, 1), setup.MIN_ENGINE)    # the 0.1.44.1 hotfix is replaced
        self.assertLess((0, 1, 44), setup.MIN_ENGINE)       # the release before is replaced
        self.assertLess((0, 1, 43, 1), setup.MIN_ENGINE)    # the 0.1.43.1 hotfix is replaced
        self.assertLess((0, 1, 42, 1), setup.MIN_ENGINE)    # 0.1.42.1 is replaced
        self.assertLess((0, 1, 40), setup.MIN_ENGINE)       # an installed 0.1.40 engine is replaced

    def test_the_engine_zips_are_found_for_a_hotfix_tag(self):
        # a hotfix that keeps CMakeLists.txt at the engine's version: the v0.1.40 release still holds the zips
        bases = setup.prebuilt_bases(setup.PREBUILT_URL)
        self.assertEqual(bases[0], setup.PREBUILT_TAG_URL.format(version=setup.source_version()))
        self.assertEqual(bases[-1], setup.PREBUILT_URL)
        # and if a four-part project version ever exists, its tag is tried first, the latest release is the fallback
        with mock.patch.object(setup, "source_version", return_value="0.1.40.1"):
            self.assertEqual(setup.prebuilt_bases(setup.PREBUILT_URL),
                             ["https://github.com/Niko1221/Strata/releases/download/v0.1.40.1/", setup.PREBUILT_URL])

    def test_an_engine_that_says_four_parts_is_read_as_four(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "strata.exe"
            exe.write_bytes(b"MZ\x00engine=0.1.40.2\n\x00")
            self.assertEqual(setup.engine_version(exe), (0, 1, 40, 2))
            exe.write_bytes(b"MZ\x00engine=0.1.40\n\x00")
            self.assertEqual(setup.engine_version(exe), (0, 1, 40))


if __name__ == "__main__":
    unittest.main()
