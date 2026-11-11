"""The packaged Terminal (#4188 parts 2 and 3): setup.py without a CMakeLists.txt, and `strata install` handing setup its options.
Nothing here starts an engine, downloads anything or reads the real profile."""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import setup as S
from strata_cli import cli


class PackagedSetupWithoutCMake(unittest.TestCase):
    """#4188: the packaged Terminal ships VERSION and BUNDLE.json but no CMakeLists.txt."""

    def test_source_version_is_zero_without_a_source_tree(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(S, "ROOT", Path(tmp)):
            self.assertEqual(S.source_version(), "0")
            (Path(tmp) / "CMakeLists.txt").write_text("project(strata VERSION 0.1.52 LANGUAGES C)\n", encoding="utf-8")
            self.assertEqual(S.source_version(), "0.1.52")
            (Path(tmp) / "CMakeLists.txt").write_bytes(b"\xff\xfe\x00 not text")
            self.assertEqual(S.source_version(), "0")

    def test_prebuilt_bases_names_the_latest_release_when_there_is_no_source_tree(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(S, "ROOT", Path(tmp)):
            self.assertEqual(S.prebuilt_bases(S.PREBUILT_URL), [S.PREBUILT_URL])
            (Path(tmp) / "CMakeLists.txt").write_text("project(strata VERSION 0.1.52)\n", encoding="utf-8")
            bases = S.prebuilt_bases(S.PREBUILT_URL)
            self.assertEqual(len(bases), 2)
            self.assertIn("v0.1.52", bases[0])
            self.assertEqual(S.prebuilt_bases("https://example.org/engine"), ["https://example.org/engine/"])

    def test_an_installed_engine_is_not_called_stale_just_because_there_is_no_source_to_compare(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(S, "ROOT", Path(tmp)), \
                mock.patch.object(S, "binary_engine_version", return_value="0.1.50"):
            self.assertFalse(S.stale_binary("x"))
            (Path(tmp) / "CMakeLists.txt").write_text("project(strata VERSION 0.1.52)\n", encoding="utf-8")
            self.assertTrue(S.stale_binary("x"))


class InstallForwardsSetupOptions(unittest.TestCase):
    """#4188: `strata install --source modelscope --family qwen ...` was refused by strata's own parser."""

    def test_the_options_of_setup_go_behind_a_double_dash_and_update_stays_strata_s(self):
        self.assertEqual(cli._install_argv(["install", "--source", "modelscope", "--yes"]), ["install", "--", "--source", "modelscope", "--yes"])
        self.assertEqual(cli._install_argv(["install", "--family", "qwen", "--update"]), ["install", "--update", "--", "--family", "qwen"])
        self.assertEqual(cli._install_argv(["install"]), ["install", "--"])
        self.assertEqual(cli._install_argv(["install", "-h"]), ["install", "-h"])
        self.assertEqual(cli._install_argv(["status", "--source", "x"]), ["status", "--source", "x"])
        self.assertEqual(cli._install_argv([]), [])

    def test_the_parser_hands_all_of_it_to_setup_in_order(self):
        ap = cli.parser()
        for argv in (["install", "--source", "modelscope", "--family", "qwen", "--model", "IQ3_XXS", "--context", "32k", "--yes"],
                     ["install", "--family", "qwen", "--yes"], ["install", "--yes"]):
            a, extras = ap.parse_known_args(cli._install_argv(argv))
            self.assertEqual(extras, [], argv)
            self.assertEqual([x for x in a.setup_args if x != "--"], argv[1:], argv)

    def test_main_runs_setup_with_every_option_and_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "setup.py").write_text("# stand-in\n", encoding="utf-8")
            calls = []
            with mock.patch.dict(os.environ, {"STRATA_INSTALL_DIR": tmp}), \
                    mock.patch.object(subprocess, "call", side_effect=lambda argv, **kw: calls.append(argv) or 0):
                code = cli.main(["install", "--source", "modelscope", "--family", "qwen", "--model", "IQ3_XXS", "--context", "32k", "--yes"])
                self.assertEqual(code, 0)
                code = cli.main(["install", "--update", "--yes"])
                self.assertEqual(code, 0)
        self.assertEqual(calls[0][2:], ["--setup", "--source", "modelscope", "--family", "qwen", "--model", "IQ3_XXS", "--context", "32k", "--yes"])
        self.assertEqual(calls[1][2:], ["--update", "--yes"])


if __name__ == "__main__":
    unittest.main()
