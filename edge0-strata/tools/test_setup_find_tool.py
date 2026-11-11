"""setup.find_tool takes the virtualenv's own cmake before a PATH one and skips a cmake older than CMakeLists.txt needs (#3822).

    python -m unittest tools.test_setup_find_tool
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class FindTool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.bin = Path(self.tmp.name) / "venv" / "bin"
        self.bin.mkdir(parents=True)
        self.venv_python = self.bin / "python"

    def versions(self, table):
        return lambda path, name: table.get(str(path), (False, ()))

    def venv_tool(self, name="cmake"):
        return str(self.bin / (name + (".exe" if setup.WIN else "")))

    def find(self, table, on_path, name="cmake", venv_copy=True):
        if venv_copy:
            Path(self.venv_tool(name)).write_text("x")
        with mock.patch.object(setup.sys, "executable", str(self.venv_python)), \
                mock.patch.object(setup.shutil, "which", return_value=on_path), \
                mock.patch.object(setup, "_tool_version", side_effect=self.versions(table)), \
                mock.patch.object(setup.Path, "home", return_value=Path(self.tmp.name) / "home"):
            return setup.find_tool(name), setup.tool_usable(name)

    def test_the_virtualenv_copy_wins_over_an_old_system_cmake(self):
        got, usable = self.find({"/usr/bin/cmake": (True, (3, 22, 1)), self.venv_tool(): (True, (4, 1, 0))}, "/usr/bin/cmake")
        self.assertEqual(got, self.venv_tool())
        self.assertTrue(usable)

    def test_an_old_cmake_on_its_own_is_returned_but_not_usable(self):
        got, usable = self.find({"/usr/bin/cmake": (True, (3, 22, 1))}, "/usr/bin/cmake", venv_copy=False)
        self.assertEqual(got, "/usr/bin/cmake")                # the build would say "CMake 3.24 or higher is required"
        self.assertFalse(usable)                               # so setup installs the pip cmake first

    def test_a_new_enough_system_cmake_is_fine_without_a_venv_one(self):
        got, usable = self.find({"/usr/bin/cmake": (True, (3, 28, 3))}, "/usr/bin/cmake", venv_copy=False)
        self.assertEqual((got, usable), ("/usr/bin/cmake", True))

    def test_ninja_has_no_version_floor(self):
        got, usable = self.find({"/usr/bin/ninja": (True, (1, 10))}, "/usr/bin/ninja", "ninja", venv_copy=False)
        self.assertTrue(usable)

    def test_a_launcher_that_does_not_run_is_skipped(self):
        got, _ = self.find({self.venv_tool(): (False, ()), "/usr/bin/cmake": (True, (3, 30, 0))}, "/usr/bin/cmake")
        self.assertEqual(got, "/usr/bin/cmake")


if __name__ == "__main__":
    unittest.main()
