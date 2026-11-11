"""TUI stress test fixes on the screens: the Folder step (B05), the Save summary (B12), ctrl+c and Escape (B13), PermissionError words, setup guard."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import setup
from strata_control import folders
from strata_cli.launch import settings as S
from strata_cli.launch.runner import setup_failure
from tests.terminal.launch_support import run_cli
from tests.terminal.test_launch_tui import Base, FakeRunner, flat, press, text_of, typed


class FolderRules(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def test_usable_folders_pass_and_nothing_is_left_behind(self):
        self.assertIsNone(folders.problem(""))
        self.assertIsNone(folders.problem(self.base))
        self.assertIsNone(folders.problem(self.base / "new" / "deeper"))          # the parent exists and can be written: it can be created
        self.assertEqual(os.listdir(self.base), [])                               # nothing was created, the test file is gone

    def test_a_file_is_not_a_folder(self):
        f = self.base / "a.txt"
        f.write_text("x")
        what, nxt = folders.problem(f)
        self.assertIn("is a file", what)
        self.assertTrue(nxt)

    def test_a_missing_parent_that_cannot_be_created_is_refused(self):
        f = self.base / "a.txt"
        f.write_text("x")
        self.assertIn("is a file", folders.problem(f / "inside")[0])
        with mock.patch.object(folders, "_can_write", return_value=False):
            what, nxt = folders.problem(self.base / "x" / "y")
        self.assertIn("cannot be created", what)

    def test_a_folder_that_cannot_be_written_is_refused(self):
        with mock.patch.object(folders, "_can_write", return_value=False):
            what, nxt = folders.problem(self.base)
        self.assertIn("cannot write", what)
        self.assertIn("Pick another folder", nxt)

    def test_the_write_test_creates_and_deletes_a_file(self):
        seen = []
        real = open

        def spy(path, *a, **k):
            seen.append(str(path))
            return real(path, *a, **k)

        with mock.patch("builtins.open", spy):
            self.assertTrue(folders._can_write(self.base))
        self.assertTrue(any(".strata-write-test-" in s for s in seen))
        self.assertEqual(os.listdir(self.base), [])

    @unittest.skipUnless(os.name == "nt", "Windows system folders")
    def test_system_folders_are_refused(self):
        for p in (r"C:\Windows\System32", r"C:\Windows", r"C:\Program Files\Strata", r"C:\Program Files (x86)\x"):
            with self.subTest(p=p):
                what, nxt = folders.problem(p)
                self.assertIn("system folder", what)

    @unittest.skipUnless(os.name == "nt", "Windows path rules")
    def test_malformed_and_relative_paths_are_refused(self):
        for p in (r"C:\a C:\b", r"C:\models|x", r"D:\what?", "models", r"..\x"):
            with self.subTest(p=p):
                self.assertIsNotNone(folders.problem(p), p)

    def test_the_plain_words_carry_a_next_step(self):
        text = folders.words(("This is bad.", "Do that."))
        self.assertEqual(text, "This is bad. Do that.")


class FolderScreen(Base):
    def test_a_bad_folder_stays_on_the_screen_with_one_plain_line_and_is_not_kept(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "enter")
            f = app.screen.query_one("#folderpath")
            f.value = ""
            await typed(pilot, str(self.home.base / "afile.txt"))
            (self.home.base / "afile.txt").write_text("x")
            await press(pilot, "enter")
            t = flat(app)
            self.assertIn("Where should the model files be stored?", t)
            self.assertIn("is a file, not a folder", t)
            self.assertIn("Pick another folder", t)
            self.assertIsNone(ctx.settings.data_dir)
            f.value = ""
            await typed(pilot, str(self.home.base / "good"))
            await press(pilot, "enter")
            self.assertIn("How much text should it remember at once?", text_of(app))
            self.assertEqual(ctx.settings.data_dir, str(self.home.base / "good"))
        self.run_app(case)

    def test_the_save_summary_shows_the_chosen_storage_path(self):
        async def case(app, pilot, ctx):
            await press(pilot, *("enter",) * 7, "n", "enter")
            t = flat(app)
            self.assertIn("Storage", t)
            self.assertIn(str(ctx.default_folder()), t)
        self.run_app(case)

    def test_the_save_summary_shows_a_typed_folder(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "enter")
            f = app.screen.query_one("#folderpath")
            f.value = ""
            await typed(pilot, str(self.home.base / "mine"))
            await press(pilot, *("enter",) * 5, "n", "enter")
            self.assertIn("Step 9 of 9", flat(app))
            self.assertIn("mine", flat(app))
        self.run_app(case)


class QuitKeys(Base):
    def test_ctrl_c_quits_the_setup_screens_like_ctrl_q(self):
        for key in ("ctrl+c", "ctrl+q"):
            with self.subTest(key=key):
                async def case(app, pilot, ctx):
                    await press(pilot, "enter")                               # the Model screen
                    await press(pilot, key)
                    await pilot.pause(0.3)
                    self.assertTrue(app._exit)                                 # the app is closing
                self.run_app(case)

    def test_ctrl_c_in_the_folder_field_still_quits(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "enter")
            await press(pilot, "ctrl+c")
            await pilot.pause(0.3)
            self.assertTrue(app._exit)
        self.run_app(case)

    def test_escape_in_the_stop_dialog_only_closes_it(self):
        from strata_cli.launch import store
        async def case(app, pilot, ctx):
            store.save(S.LaunchSettings(name="T", family="qwen", model="IQ3_XXS", context=32768))
            FakeRunner.behaviour = "wait"
            app.start_preset("T")
            await pilot.pause(0.4)
            await press(pilot, "ctrl+c")                                       # work would be lost: the dialog stays
            self.assertIn("Stop now?", text_of(app))
            await press(pilot, "escape")
            self.assertNotIn("Stop now?", text_of(app))
            self.assertTrue(app.running)                                       # still running, and no Back happened
            self.assertEqual(app.current, "run")
            app.cancel.set()
            await pilot.pause(0.5)
        self.run_app(case)


class PermissionWords(unittest.TestCase):
    def test_a_permission_error_names_the_folder_and_what_to_do(self):
        lines = ["Traceback (most recent call last):", "  f",
                 r"PermissionError: [WinError 5] Access is denied: 'C:\Windows\System32\models'"]
        e = setup_failure(lines, "", "", r"C:\x\a.setup.log", False)
        self.assertIn(r"C:\Windows\System32", e.text)
        self.assertIn("access denied", e.text)
        self.assertNotIn("WinError", e.text + e.next)
        self.assertNotIn("Run it again", e.next)
        self.assertIn("--data-dir", e.next)

    def test_other_errors_keep_their_words(self):
        e = setup_failure(["Traceback (most recent call last):", "  f", "KeyError: 'cuda'"], "", "", None, True)
        self.assertIn("KeyError", e.text)


class SetupGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "Strata"
        self.root.mkdir()
        env = mock.patch.dict(os.environ, {"APPDATA": str(self.base / "appdata"), "XDG_CONFIG_HOME": str(self.base / "appdata")})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("STRATA_SETTINGS_FILE", None)
        for patcher in (mock.patch.object(setup, "ROOT", self.root), mock.patch.object(setup, "_CONTROL_HOME", None)):
            patcher.start()
            self.addCleanup(patcher.stop)
        setup.settings_path().parent.mkdir(parents=True, exist_ok=True)

    def test_a_bad_requested_folder_stops_with_plain_words_and_is_never_saved(self):
        setup.settings_path().write_text(json.dumps({"installs": []}), encoding="utf-8")
        afile = self.base / "afile"
        afile.write_text("x")
        with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
            setup.data_folder(str(afile), yes=True)
        self.assertIn("is a file, not a folder", out.getvalue())
        self.assertNotIn("data_dir", json.loads(setup.settings_path().read_text(encoding="utf-8")))

    def test_a_bad_saved_folder_is_dropped_not_kept(self):
        afile = self.base / "afile"
        afile.write_text("x")
        setup.settings_path().write_text(json.dumps({"data_dir": str(afile), "installs": []}), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            dest, _ = setup.data_folder(None, yes=True)
        self.assertNotEqual(str(dest.resolve()), str(afile.resolve()))
        self.assertNotEqual(json.loads(setup.settings_path().read_text(encoding="utf-8")).get("data_dir"), str(afile))


class DataDirFlag(unittest.TestCase):
    def test_launch_data_dir_with_a_bad_path_refuses_before_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            afile = Path(tmp) / "afile"
            afile.write_text("x")
            code, out, err = run_cli(["launch", "--plan", "--model", "q2_0", "--data-dir", str(afile)])
        self.assertEqual(code, 2)
        self.assertIn("is a file, not a folder", err)


if __name__ == "__main__":
    unittest.main()
