"""`--data-dir` names a new folder: the model library is never moved without a choice (0.2.1 R).

Everything lives in temporary folders: settings.json (APPDATA / XDG_CONFIG_HOME), the Strata folder (setup.ROOT) and the data folders.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import setup
from strata_cli.launch import cmd, library
from strata_cli.launch import settings as S
from strata_cli.launch.runner import Runner

from tests.terminal.test_launch import HW_12, Home, catalog


def put(path: Path, size: int = 64) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


class DataFolderChoice(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "Strata"
        self.root.mkdir()
        self.old = self.base / "old-data"
        self.new = self.base / "new-data"
        env = mock.patch.dict(os.environ, {"APPDATA": str(self.base / "appdata"), "XDG_CONFIG_HOME": str(self.base / "appdata"),
                                           "STRATA_SETTINGS_FILE": ""})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("STRATA_SETTINGS_FILE", None)
        for patcher in (mock.patch.object(setup, "ROOT", self.root), mock.patch.object(setup, "_CONTROL_HOME", None)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.assertTrue(str(setup.settings_path()).startswith(str(self.base)), setup.settings_path())
        self.weights = put(self.old / "models" / "Q2_0" / "w-00001-of-00001.gguf", 200)
        self.pack = put(self.old / "packs" / "q2_0" / "tokenizer" / "vocab.json", 20)
        self.mtp = put(self.old / "mtp" / "rt" / "experts.bin", 30)
        setup.settings_path().parent.mkdir(parents=True, exist_ok=True)
        self.write_settings({"data_dir": str(self.old), "installs": []})

    def write_settings(self, data: dict) -> None:
        setup.settings_path().write_text(json.dumps(data), encoding="utf-8")

    def saved(self) -> dict:
        return json.loads(setup.settings_path().read_text(encoding="utf-8"))

    def call(self, *args, **kw):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            res = setup.data_folder(*args, **kw)
        return res, out.getvalue()

    def assertLibraryUntouched(self):
        for f in (self.weights, self.pack, self.mtp):
            self.assertTrue(f.is_file(), f)

    def test_a_new_data_dir_with_yes_keeps_the_library_and_the_saved_default(self):
        (dest, elsewhere), _ = self.call(str(self.new), yes=True)
        self.assertEqual(dest.resolve(), self.new.resolve())
        self.assertTrue(dest.is_dir())
        self.assertEqual([Path(e).resolve() for e in elsewhere], [self.old.resolve()])     # still found: the old folder is a root
        self.assertLibraryUntouched()
        self.assertFalse((self.new / "models").exists())
        self.assertEqual(self.saved()["data_dir"], str(self.old))                          # the saved default is not rewritten

    def test_the_default_answer_with_no_keyboard_keeps_everything(self):
        with mock.patch("builtins.input", side_effect=EOFError):
            with self.assertRaises(SystemExit):                                            # ask() ends a setup with nobody to answer
                self.call(str(self.new), yes=False)
        self.assertLibraryUntouched()
        self.assertEqual(self.saved()["data_dir"], str(self.old))

    def test_asked_the_answer_keep_moves_nothing(self):
        with mock.patch("builtins.input", return_value=""):
            (dest, elsewhere), said = self.call(str(self.new), yes=False)
        self.assertIn("keep)", said)
        self.assertLibraryUntouched()
        self.assertEqual(self.saved()["data_dir"], str(self.old))
        self.assertEqual([Path(e).resolve() for e in elsewhere], [self.old.resolve()])

    def test_asked_the_answer_move_moves_and_remembers_the_new_folder(self):
        with mock.patch("builtins.input", return_value="move"):
            (dest, elsewhere), _ = self.call(str(self.new), yes=False)
        self.assertEqual(elsewhere, [])
        self.assertTrue((self.new / "models" / "Q2_0" / "w-00001-of-00001.gguf").is_file())
        self.assertTrue((self.new / "packs" / "q2_0" / "tokenizer" / "vocab.json").is_file())
        self.assertFalse(self.weights.exists())
        self.assertEqual(Path(self.saved()["data_dir"]).resolve(), self.new.resolve())

    def test_move_data_moves_without_asking(self):
        with mock.patch("builtins.input", side_effect=AssertionError("must not ask")):
            (dest, elsewhere), _ = self.call(str(self.new), move=True, yes=True)
        self.assertTrue((self.new / "mtp" / "rt" / "experts.bin").is_file())
        self.assertEqual(Path(self.saved()["data_dir"]).resolve(), self.new.resolve())

    def test_set_default_data_dir_remembers_the_folder_and_moves_nothing(self):
        (dest, elsewhere), _ = self.call(str(self.new), yes=True, set_default=True)
        self.assertLibraryUntouched()
        self.assertEqual(Path(self.saved()["data_dir"]).resolve(), self.new.resolve())
        self.assertEqual([Path(e).resolve() for e in elsewhere], [self.old.resolve()])

    def test_a_pc_with_no_models_anywhere_remembers_the_new_folder(self):
        for d in ("models", "packs", "mtp"):
            shutil.rmtree(self.old / d)
        (dest, elsewhere), _ = self.call(str(self.new), yes=True)
        self.assertEqual(elsewhere, [])
        self.assertEqual(Path(self.saved()["data_dir"]).resolve(), self.new.resolve())

    def test_the_library_in_the_strata_folder_is_asked_about_too(self):
        # no remembered data folder: the models sit next to the checkout (an old install)
        self.write_settings({})
        for d in ("models", "packs", "mtp"):
            shutil.rmtree(self.old / d)
        mine = put(self.root / "Strata-data" / "models" / "Q2_0" / "w-00001-of-00001.gguf")
        (dest, elsewhere), _ = self.call(str(self.new), yes=True)
        self.assertTrue(mine.is_file())
        self.assertEqual([Path(e).resolve() for e in elsewhere], [(self.root / "Strata-data").resolve()])
        self.assertNotIn("data_dir", self.saved())

    def test_no_data_dir_still_gathers_into_the_remembered_folder(self):
        # the data folder's own job (an earlier Strata folder's models) is not part of the --data-dir question
        other = self.base / "Strata-old"
        other.mkdir()
        (other / "setup.py").write_text("# old\n")
        put(other / "models" / "Q2_0" / "extra.gguf")
        self.write_settings({"data_dir": str(self.old), "installs": [str(other)]})
        (dest, _), _ = self.call(None, yes=True)
        self.assertEqual(dest.resolve(), self.old.resolve())
        self.assertTrue((self.old / "models" / "Q2_0" / "extra.gguf").is_file())

    def test_the_same_folder_again_is_not_a_change(self):
        (dest, elsewhere), _ = self.call(str(self.old), yes=False)
        self.assertEqual(elsewhere, [])
        self.assertLibraryUntouched()


class TerminalQuestion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.old = self.base / "old"
        self.new = self.base / "new"
        put(self.old / "models" / "Q2_0" / "w.gguf", 3_000_000)
        self.cat = mock.Mock()
        self.cat.data_dir.return_value = self.old

    def args(self, **kw):
        ns = dict(data_dir=str(self.new), move_data=False, yes=False, plan=False)
        ns.update(kw)
        return argparse.Namespace(**ns)

    def test_no_data_dir_asks_nothing(self):
        self.assertFalse(cmd._move_data_choice(self.args(data_dir=None), S.LaunchSettings(), self.cat, False))

    def test_move_data_says_yes_without_asking(self):
        with mock.patch("builtins.input", side_effect=AssertionError("must not ask")):
            self.assertTrue(cmd._move_data_choice(self.args(move_data=True), S.LaunchSettings(), self.cat, False))

    def test_yes_machine_and_plan_keep_the_library(self):
        with mock.patch("builtins.input", side_effect=AssertionError("must not ask")):
            for kw, machine in (({"yes": True}, False), ({}, True), ({"plan": True}, False)):
                self.assertFalse(cmd._move_data_choice(self.args(**kw), S.LaunchSettings(), self.cat, machine), kw)

    def test_no_keyboard_keeps_the_library(self):
        with mock.patch.object(sys.stdin, "isatty", return_value=False), mock.patch("builtins.input", side_effect=AssertionError):
            self.assertFalse(cmd._move_data_choice(self.args(), S.LaunchSettings(), self.cat, False))

    def test_asked_on_a_terminal(self):
        out = io.StringIO()
        with mock.patch.object(sys.stdin, "isatty", return_value=True), contextlib.redirect_stdout(out):
            with mock.patch("builtins.input", return_value="m"):
                self.assertTrue(cmd._move_data_choice(self.args(), S.LaunchSettings(), self.cat, False))
            with mock.patch("builtins.input", return_value=""):
                self.assertFalse(cmd._move_data_choice(self.args(), S.LaunchSettings(), self.cat, False))
        self.assertIn("use the new folder only for new downloads", out.getvalue())

    def test_an_empty_or_same_library_asks_nothing(self):
        self.cat.data_dir.return_value = self.new
        with mock.patch.object(sys.stdin, "isatty", return_value=True), mock.patch("builtins.input", side_effect=AssertionError):
            self.assertFalse(cmd._move_data_choice(self.args(), S.LaunchSettings(), self.cat, False))
        self.assertEqual(library.movable_library(self.base / "nothing", self.new), 0.0)

    def test_setup_gets_move_data_only_when_chosen_and_a_folder_is_named(self):
        with Home():
            cat = catalog(HW_12)
            s = S.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768, vision="off", data_dir="/d")
            self.assertNotIn("--move-data", Runner(s, catalog=cat).setup_args())
            self.assertIn("--move-data", Runner(s, catalog=cat, move_data=True).setup_args())
            s2 = S.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768, vision="off")
            self.assertNotIn("--move-data", Runner(s2, catalog=cat, move_data=True).setup_args())
            with mock.patch.object(sys, "platform", "darwin"):
                mac = Runner(s, catalog=cat, move_data=True).setup_args()
            self.assertIn("--move-data", mac)
            self.assertEqual(mac[mac.index("--data-dir") + 1], "/d")


if __name__ == "__main__":
    unittest.main()
