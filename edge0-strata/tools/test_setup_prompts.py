"""Setup prompts require input or --yes; EOF must not authorize system installs (#393).
No GPU, downloads or installers are used.

    python -m unittest tools.test_setup_prompts
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class Prompts(unittest.TestCase):
    def test_eof_stops_with_guidance(self):
        out = io.StringIO()
        with mock.patch("sys.stdin", io.StringIO()), contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as cm:
                setup.ask("Install them now?", ["y", "n"], "y", False)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("Install them now?", out.getvalue())
        self.assertIn("--yes", out.getvalue())
        self.assertIn("Setup stopped", out.getvalue())

    def test_yes_accepts_the_default_without_reading_input(self):
        with mock.patch("builtins.input", side_effect=AssertionError("unexpected prompt")):
            for default in ("y", "n"):
                with self.subTest(default=default):
                    self.assertEqual(setup.ask("Install?", ["y", "n"], default, True), default)

    def test_enter_and_explicit_answers(self):
        for answer, expected in (("\n", "y"), ("  \n", "y"), ("y\n", "y"), (" N \n", "n")):
            with self.subTest(answer=answer), mock.patch("sys.stdin", io.StringIO(answer)), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(setup.ask("Install?", ["y", "n"], "y", False), expected)

    def test_invalid_answer_reprompts_and_preserves_choice_case(self):
        out = io.StringIO()
        with mock.patch("sys.stdin", io.StringIO("invalid\nyArN\n")), contextlib.redirect_stdout(out):
            self.assertEqual(setup.ask("Method?", ["Yarn", "Linear"], "Linear", False), "Yarn")
        self.assertIn("please answer one of: Yarn, Linear", out.getvalue())

    def test_eof_after_invalid_answer_stops(self):
        with mock.patch("sys.stdin", io.StringIO("invalid\n")), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                setup.ask("Install?", ["y", "n"], "y", False)
        self.assertEqual(cm.exception.code, 1)


class BuildTools(unittest.TestCase):
    def test_eof_never_starts_an_installer(self):
        for win in (True, False):
            with self.subTest(win=win), mock.patch.object(setup, "WIN", win), \
                    mock.patch.object(setup, "find_nvcc", return_value=(None, (0, 0))), \
                    mock.patch.object(setup, "find_vcvars", return_value=None), \
                    mock.patch.object(setup.shutil, "which", side_effect=lambda name: None if name == "g++" else name), \
                    mock.patch.object(setup, "run", side_effect=AssertionError("installer started")) as run, \
                    mock.patch.object(setup, "download", side_effect=AssertionError("download started")) as download, \
                    mock.patch("sys.stdin", io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    setup.install_build_tools({"arch": "120"}, False)
                self.assertEqual(cm.exception.code, 1)
                run.assert_not_called()
                download.assert_not_called()

    def test_yes_still_installs_missing_windows_tools(self):
        nvcc, vcvars = Path("nvcc"), Path("vcvars64.bat")
        with mock.patch.object(setup, "WIN", True), \
                mock.patch.object(setup, "find_nvcc", side_effect=[(None, (0, 0)), (nvcc, (13, 0))]), \
                mock.patch.object(setup, "find_vcvars", side_effect=[None, vcvars, vcvars, vcvars]), \
                mock.patch.object(setup.shutil, "which", return_value="winget"), \
                mock.patch.object(setup, "run") as run, \
                mock.patch("builtins.input", side_effect=AssertionError("unexpected prompt")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(setup.install_build_tools({"arch": "120"}, True), (nvcc, vcvars))
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands), 2)
        self.assertIn("Microsoft.VisualStudio.2022.BuildTools", commands[0])
        self.assertIn("Nvidia.CUDA", commands[1])


if __name__ == "__main__":
    unittest.main()
