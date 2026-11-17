"""#670: an engine update keeps the engine it replaces in engine/.previous (one generation), the swap is all or
nothing, and `setup.py --rollback-engine` puts the kept one back.

    python -m unittest tools.test_setup_engine_keep
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


def engine(folder: Path, version: str, exe: bytes) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "BUILD.json").write_text(json.dumps({"version": version}), encoding="utf-8")
    (folder / "strata.exe").write_bytes(exe)
    (folder / "lib").mkdir(exist_ok=True)
    (folder / "lib" / "x.dll").write_bytes(exe)


class KeepPrevious(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.eng = self.root / "engine"
        self.new = self.eng / "_unpack"

    def test_the_replaced_engine_is_kept_and_rolled_back(self):
        engine(self.eng, "0.1.39", b"old")
        engine(self.new, "0.1.40", b"new")
        setup.install_unpacked(self.new, self.eng)
        self.assertEqual((self.eng / "strata.exe").read_bytes(), b"new")
        self.assertEqual(setup.engine_version_of(self.eng), "0.1.40")
        prev = self.eng / ".previous"
        self.assertEqual((prev / "strata.exe").read_bytes(), b"old")
        self.assertEqual((prev / "lib" / "x.dll").read_bytes(), b"old")
        self.assertEqual(setup.engine_version_of(prev), "0.1.39")
        with mock.patch.object(setup, "ROOT", self.root):
            self.assertEqual(setup.rollback_engine(), 0)
            self.assertEqual((self.eng / "strata.exe").read_bytes(), b"old")
            self.assertEqual(setup.engine_version_of(self.eng), "0.1.39")
            self.assertEqual(setup.engine_version_of(prev), "0.1.40")     # forward again
            self.assertEqual(setup.rollback_engine(), 0)
            self.assertEqual((self.eng / "strata.exe").read_bytes(), b"new")

    def test_previous_keeps_its_build_json_after_get_prebuilt_moved_it_aside(self):
        """get_prebuilt takes the old BUILD.json out of the way (BUILD.json.prev) before the unpack, so the
        engine kept in .previous still says its version instead of '?'."""
        engine(self.eng, "0.1.39", b"old")
        (self.eng / "BUILD.json").replace(self.eng / "BUILD.json.prev")
        engine(self.new, "0.1.40", b"new")
        setup.install_unpacked(self.new, self.eng)
        self.assertEqual(setup.engine_version_of(self.eng / ".previous"), "0.1.39")
        self.assertFalse((self.eng / "BUILD.json.prev").exists())
        self.assertEqual(setup.engine_version_of(self.eng), "0.1.40")

    def test_one_generation_only(self):
        engine(self.eng, "0.1.38", b"a")
        for version, exe in (("0.1.39", b"b"), ("0.1.40", b"c")):
            engine(self.new, version, exe)
            setup.install_unpacked(self.new, self.eng)
            for p in self.new.iterdir():          # (get_prebuilt removes what is left of the unpack)
                p.unlink() if p.is_file() else None
        self.assertEqual(setup.engine_version_of(self.eng / ".previous"), "0.1.39")

    def test_a_first_install_keeps_nothing(self):
        self.eng.mkdir()
        engine(self.new, "0.1.40", b"new")
        setup.install_unpacked(self.new, self.eng)
        self.assertEqual((self.eng / "strata.exe").read_bytes(), b"new")
        self.assertFalse((self.eng / ".previous").exists())
        with mock.patch.object(setup, "ROOT", self.root):
            self.assertEqual(setup.rollback_engine(), 1)

    def test_a_failure_part_way_leaves_the_old_engine_whole(self):
        engine(self.eng, "0.1.39", b"old")
        engine(self.new, "0.1.40", b"new")
        real = Path.replace
        calls = []

        def flaky(self_, target):
            calls.append(self_.name)
            if len(calls) == 2:
                raise PermissionError("held open")
            return real(self_, target)

        with mock.patch.object(Path, "replace", flaky), self.assertRaises(OSError):
            setup.install_unpacked(self.new, self.eng)
        self.assertEqual((self.eng / "strata.exe").read_bytes(), b"old")
        self.assertEqual((self.eng / "lib" / "x.dll").read_bytes(), b"old")
        self.assertEqual(setup.engine_version_of(self.eng), "0.1.39")
        self.assertFalse((self.eng / ".previous").exists())


if __name__ == "__main__":
    unittest.main()
