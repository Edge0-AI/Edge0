"""The engine test start with no model installed: an engine without a --version flag (it prints its options and exits 2) still counts as running."""

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from strata_control import updater


def make_engine(folder: Path, body_posix: str, body_nt: str) -> Path:
    if os.name == "nt":
        path = folder / "strata.cmd"
        path.write_text("@echo off\r\n" + body_nt, encoding="utf-8")
    else:
        path = folder / "strata"
        path.write_text("#!/bin/sh\n" + body_posix, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


USAGE_POSIX = "echo 'strata generate [options]' >&2\nfor f in a b c d e f; do echo \"  --opt-$f N   something\" >&2; done\nexit 2\n"
USAGE_NT = "echo strata generate [options] 1>&2\r\n" + "".join(f"echo   --opt-{c} N   something 1>&2\r\n" for c in "abcdef") + "exit /b 2\r\n"


class BinarySmoke(unittest.TestCase):
    def component(self, folder: Path, exe: Path):
        return SimpleNamespace(exe=exe, root=folder, version="0.1.45")

    def test_an_engine_that_lists_its_options_instead_of_answering_version_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            res = updater.binary_smoke(self.component(folder, make_engine(folder, USAGE_POSIX, USAGE_NT)))
            self.assertEqual(res["mode"], "binary-only")
            self.assertIn("does not answer --version", res["detail"])
            self.assertIn("listed its options", res["detail"])

    def test_an_engine_that_answers_version_passes_as_before(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            res = updater.binary_smoke(self.component(folder, make_engine(folder, "echo strata 0.1.45\nexit 0\n", "echo strata 0.1.45\r\nexit /b 0\r\n")))
            self.assertIn("--version only", res["detail"])

    def test_a_real_failure_still_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            exe = make_engine(folder, "echo 'No device of requested type available' >&2\nexit 250\n", "echo No device of requested type available 1>&2\r\nexit /b 250\r\n")
            with self.assertRaises(updater.TestStartFailed) as e:
                updater.binary_smoke(self.component(folder, exe))
            self.assertIn("No device of requested type", e.exception.reason)

    def test_exit_2_without_a_list_of_options_still_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            exe = make_engine(folder, "echo 'bad usage' >&2\nexit 2\n", "echo bad usage 1>&2\r\nexit /b 2\r\n")
            with self.assertRaises(updater.TestStartFailed):
                updater.binary_smoke(self.component(folder, exe))


if __name__ == "__main__":
    unittest.main()
