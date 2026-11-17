"""A build folder made by the manual build in docs/INSTALL.md (CMake's default generator) does not stop setup --build (#4038).

CMake refuses a folder configured with another generator ("Does not match the generator used previously: Unix Makefiles"), and setup
always asks for Ninja. The folder is configured afresh, with one plain line; a folder made with Ninja, or never configured, is left alone.

    python -m unittest tools.test_setup_build_dir
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def configured(folder: Path, generator: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "CMakeCache.txt").write_text(
        f"# This is the CMakeCache file.\nCMAKE_BUILD_TYPE:STRING=Release\nCMAKE_GENERATOR:INTERNAL={generator}\nCMAKE_HOME_DIRECTORY:INTERNAL=/x\n", encoding="utf-8")
    (folder / "CMakeFiles").mkdir(exist_ok=True)
    (folder / "CMakeFiles" / "marker.txt").write_text("x", encoding="utf-8")
    (folder / "keep.o").write_text("object", encoding="utf-8")
    return folder


class OtherGenerator(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def call(self, folder):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            return setup.fresh_if_other_generator(folder), out.getvalue()

    def test_the_generator_is_read_from_the_cache(self):
        self.assertEqual(setup.configured_generator(configured(self.root / "a", "Unix Makefiles")), "Unix Makefiles")
        self.assertEqual(setup.configured_generator(configured(self.root / "b", "Ninja")), "Ninja")
        self.assertIsNone(setup.configured_generator(self.root / "never"))

    def test_a_folder_made_with_makefiles_is_configured_again(self):
        folder = configured(self.root / "build", "Unix Makefiles")
        again, said = self.call(folder)
        self.assertTrue(again)
        self.assertIn("Unix Makefiles", said)
        self.assertIn("Ninja", said)
        self.assertFalse((folder / "CMakeCache.txt").exists())
        self.assertFalse((folder / "CMakeFiles").exists())
        self.assertTrue((folder / "keep.o").exists(), "only the CMake cache goes, no other file of the folder")

    def test_a_ninja_folder_and_a_new_folder_are_left_alone(self):
        folder = configured(self.root / "ninja", "Ninja")
        again, said = self.call(folder)
        self.assertFalse(again)
        self.assertEqual(said, "")
        self.assertTrue((folder / "CMakeCache.txt").exists() and (folder / "CMakeFiles" / "marker.txt").exists())
        self.assertEqual(self.call(self.root / "new"), (False, ""))
        self.assertFalse((self.root / "new").exists(), "a folder that does not exist is not created by the check")

    def test_cmake_build_does_it_before_it_configures(self):
        folder = configured(self.root / "build", "Unix Makefiles")
        seen = []

        def fake_run(cmd, **kw):
            seen.append((list(map(str, cmd)), (folder / "CMakeCache.txt").exists()))
            return mock.Mock(returncode=0)

        with mock.patch.object(setup, "ensure_build_tools", return_value=("cmake", "ninja")), \
                mock.patch.object(setup.acquisition, "require"), mock.patch.object(setup, "WIN", False), \
                mock.patch.object(setup, "run", fake_run), contextlib.redirect_stdout(io.StringIO()):
            setup.cmake_build(self.root, folder, "strata", [], None, "")
        configure = seen[0]
        self.assertIn("Ninja", configure[0])
        self.assertFalse(configure[1], "the old cache was gone when cmake was asked to configure")


if __name__ == "__main__":
    unittest.main()
