"""Fixes after the TUI stress re-run on 67a48427b: a failed start cleans up and quotes this attempt's logs, the Folder text, the save/restart order,
the saved config without --mtp when drafting is off, and the reused-pack line."""

import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch import runner as R
from strata_cli.launch.catalog import Catalog
from strata_cli.launch.runner import LaunchError, Runner
from tests.terminal.launch_support import HW_12, Home, catalog


class FailedStart(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
        p.start()
        self.addCleanup(p.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def runner(self, port=8080):
        s = S.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768, name="gyro", port=port)
        r = Runner(s, catalog=catalog())
        r.said = []
        r.emit_cb = lambda e: r.said.append(e.text)
        return r

    def test_a_busy_port_is_said_before_anything_starts(self):
        with socket.socket() as srv:
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            r = self.runner(port)
            with mock.patch("strata_control.runtime.status", return_value={"state": "stopped"}), \
                 mock.patch("strata_control.runtime.start", side_effect=AssertionError("must not start")):
                with self.assertRaises(LaunchError) as cm:
                    r.start()
        self.assertIn(f"Port {port} is already in use by another program.", cm.exception.text)
        self.assertIn("--port", cm.exception.next)

    def test_the_cause_comes_from_this_attempts_server_log_not_the_old_engine_log(self):
        engine = self.base / "engine.log"
        server = self.base / "srv.log"
        engine.write_text("strata verify: captured the 2-token window\nlong ago error: old cause\n")
        server.write_text("")
        cfg = self.base / "cfg.json"
        cfg.write_text(json.dumps({"log": str(engine)}))
        marks = {str(engine): engine.stat().st_size, str(server): 0}
        with open(server, "a") as f:
            f.write("server.py: error: port 8190 is already in use - is Strata (or another server) already running?\n")
        text, nxt = R.explain_engine_exit({"state": "failed", "error": "server_exited", "engine_exit_code": 2}, cfg, marks=marks, server_log=server)
        self.assertIn("Port 8190 is already in use by another program.", text)
        self.assertIn("--port 8191", nxt)
        self.assertNotIn("old cause", text + nxt)
        self.assertNotIn("captured the 2-token", text + nxt)

    def test_an_old_engine_log_is_never_quoted_when_this_attempt_wrote_nothing(self):
        engine = self.base / "engine.log"
        engine.write_text("error: from the earlier successful run\n")
        cfg = self.base / "cfg.json"
        cfg.write_text(json.dumps({"log": str(engine)}))
        marks = {str(engine): engine.stat().st_size}
        text, _ = R.explain_engine_exit({"state": "failed", "error": "server_exited"}, cfg, marks=marks, server_log=None)
        self.assertNotIn("earlier successful run", text)

    def test_a_failed_start_ends_its_own_host(self):
        r = self.runner()
        r.config_path = None
        calls = []
        with mock.patch("strata_control.runtime.status", return_value={"state": "stopped"}), \
             mock.patch.object(Runner, "check_port_free"), \
             mock.patch("strata_control.runtime.start", return_value={"ok": False, "state": "failed", "error": "server_exited", "engine_exit_code": 2}), \
             mock.patch("strata_control.runtime.stop", side_effect=lambda n, timeout=0: calls.append(("stop", n))), \
             mock.patch("strata_control.runtime.cleanup", side_effect=lambda n=None: calls.append(("cleanup", n))):
            with self.assertRaises(LaunchError):
                r.start()
        self.assertEqual(calls, [("stop", "launch-gyro"), ("cleanup", "launch-gyro")])

    def test_a_start_that_raises_also_ends_its_host(self):
        r = self.runner()
        calls = []
        with mock.patch("strata_control.runtime.status", return_value={"state": "stopped"}), \
             mock.patch.object(Runner, "check_port_free"), \
             mock.patch("strata_control.runtime.start", side_effect=RuntimeError("runtime host failed to start")), \
             mock.patch("strata_control.runtime.stop", side_effect=lambda n, timeout=0: calls.append("stop")), \
             mock.patch("strata_control.runtime.cleanup", side_effect=lambda n=None: calls.append("cleanup")):
            with self.assertRaises(LaunchError):
                r.start()
        self.assertEqual(calls, ["stop", "cleanup"])


class FolderChangeMovesNothing(unittest.TestCase):
    """The Folder step text: changing the folder never moves the models Strata already keeps (only --move-data / strata models move do)."""

    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
        p.start()
        self.addCleanup(p.stop)

    def test_start_passes_no_move_flag_unless_asked(self):
        s = S.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768, data_dir=str(self.home.base / "new"))
        self.assertNotIn("--move-data", Runner(s, catalog=catalog()).setup_args())
        self.assertIn("--move-data", Runner(s, catalog=catalog(), move_data=True).setup_args())

    def test_setup_keeps_the_existing_models_where_they_are(self):
        import contextlib
        import io
        import setup
        base = self.home.base
        old, new, root = base / "old-data", base / "new-data", base / "Strata"
        root.mkdir()
        weights = old / "models" / "Q2_0" / "w.gguf"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"x" * 100)
        env = {"APPDATA": str(base / "appdata"), "XDG_CONFIG_HOME": str(base / "appdata")}
        with mock.patch.dict(os.environ, env), mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "_CONTROL_HOME", None):
            os.environ.pop("STRATA_SETTINGS_FILE", None)
            setup.settings_path().parent.mkdir(parents=True, exist_ok=True)
            setup.settings_path().write_text(json.dumps({"data_dir": str(old), "installs": []}), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                setup.data_folder(str(new), move=False, yes=True, set_default=True)
        self.assertTrue(weights.is_file())
        self.assertFalse((new / "models").exists())

    def test_the_screen_text_says_they_stay(self):
        src = open(os.path.join(os.path.dirname(R.__file__), "screens.py"), encoding="utf-8").read()
        self.assertIn("stay there and keep working", src)
        self.assertNotIn("are moved into this folder when you start", src)


class SaveWaitsForRestartAnswer(unittest.TestCase):
    """Save as a preset and launch: nothing is saved before the restart question is answered Restart."""

    def _runner(self, **kw):
        s = S.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768, name="gyro")
        return Runner(s, catalog=catalog(), **kw)

    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
        p.start()
        self.addCleanup(p.stop)

    def rows(self, r):
        return [{"profile": "launch-gyro", "name": "gyro", "port": 8080,
                 "choices": {**r.choices(), "context": 65536, "family": "qwen", "model": "IQ3_XXS"}}]

    def test_cancel_saves_nothing_and_the_question_names_the_running_model(self):
        saved, asked = [], []
        r = self._runner(confirm_restart=lambda q, d: asked.append((q, d)) or False, after_check=lambda: saved.append(1))
        with mock.patch.object(Runner, "serving_launches", lambda self: self.rows_):
            Runner.rows_ = self.rows(r)
            with self.assertRaises(LaunchError):
                r.run()
        self.assertEqual(saved, [])
        self.assertIn("is running on port 8080 with 64K. Restart it with these settings?", asked[0][0])
        self.assertIn("IQ3_XXS", asked[0][0])
        self.assertEqual(asked[0][1], "Answers in progress stop.")

    def test_restart_saves_after_the_answer(self):
        order = []
        r = self._runner(confirm_restart=lambda q, d: order.append("asked") or True, after_check=lambda: order.append("saved"))
        Runner.rows_ = self.rows(r)
        with mock.patch.object(Runner, "serving_launches", lambda self: self.rows_),              mock.patch.object(Runner, "ensure_engine", side_effect=RuntimeError("went on")):
            with self.assertRaises(RuntimeError):
                r.run()
        self.assertEqual(order, ["asked", "saved"])

    def test_nothing_running_saves_at_once(self):
        saved = []
        r = self._runner(after_check=lambda: saved.append(1))
        with mock.patch.object(Runner, "serving_launches", lambda self: []),              mock.patch.object(Runner, "ensure_engine", side_effect=RuntimeError("went on")):
            with self.assertRaises(RuntimeError):
                r.run()
        self.assertEqual(saved, [1])

    def test_the_save_screen_no_longer_writes_the_preset_itself(self):
        import asyncio
        from strata_cli.launch import store
        from tests.terminal.test_launch_tui import Base, FakeRunner, press, make_ctx
        from strata_cli.launch.app import LaunchApp

        class Case(Base):
            def runTest(self):
                pass

        case = Case()
        case.setUp()
        self.addCleanup(case.tearDown)
        seen = {}

        class Holding(FakeRunner):
            def run(self, start=True):
                seen["presets_during_run"] = [p.name for p in store.list_presets()]
                return super().run(start)

        with mock.patch("strata_cli.launch.app.Runner", Holding):
            async def go():
                app = LaunchApp(make_ctx())
                async with app.run_test(size=(110, 34)) as pilot:
                    await pilot.pause(0.2)
                    await press(pilot, *("enter",) * 7, "n", "enter")
                    self.assertEqual(store.list_presets(), [])           # still nothing on the Save screen
                    await press(pilot, "enter")
                    await pilot.pause(0.8)
                    return [p.name for p in store.list_presets()]
            after = asyncio.run(go())
        self.assertEqual(seen["presets_during_run"], [])               # not saved before the runner's check
        self.assertEqual(after, ["IQ3_S 32K"])                          # saved once the run went on


class DraftingOffConfig(unittest.TestCase):
    """B10: the saved launch config has no --mtp in args when "mtp" is false, whatever made it false."""

    BASE = {"args": ["--model", "m.gguf", "--mtp", "C:/x/mtp", "--spec-min-p", "0.5", "--batch-mtp", "--max-context", "32768"], "exe": "x"}

    def derive(self, base, **kw):
        from strata_control import tune_knobs
        s = S.LaunchSettings(family="gyro", model="Gyro-S", context=32768, **kw)
        return R.derive_config(base, s, bool_flags=tune_knobs.BOOL_FLAGS)

    def test_drafting_off_and_mtp_off(self):
        for kw in ({"drafting": "off"}, {"mtp": "off"}, {"drafting": "off", "extra": {"prefill": "4096"}}):
            with self.subTest(kw=kw):
                cfg = self.derive(self.BASE, **kw)
                self.assertIs(cfg["mtp"], False)
                self.assertNotIn("--mtp", cfg["args"])
                self.assertIn("--max-context", cfg["args"])

    def test_an_installed_config_that_already_says_mtp_false(self):
        cfg = self.derive({**self.BASE, "mtp": False})
        self.assertIs(cfg["mtp"], False)
        self.assertNotIn("--mtp", cfg["args"])
        self.assertNotIn("--spec-min-p", cfg["args"])

    def test_mtp_stays_when_it_is_on(self):
        cfg = self.derive(self.BASE)
        self.assertIn("--mtp", cfg["args"])


class ReusedDataLine(unittest.TestCase):
    """C08 note: one line says where the pack and the draft layer are read from."""

    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
        p.start()
        self.addCleanup(p.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def runner(self, cfg_args, **fields):
        cfg = self.base / "strata-x.json"
        cfg.write_text(json.dumps({"args": cfg_args}))
        s = S.LaunchSettings(family="qwen", model="Q2_0", context=32768, **fields)
        r = Runner(s, catalog=catalog())
        r.said = []
        r.emit_cb = lambda e: r.said.append(e.text)
        return r, cfg

    def test_pack_and_draft_layer_from_the_old_folder_are_named(self):
        old = self.base / "dl-data"
        r, cfg = self.runner(["--pack", str(old / "packs" / "x"), "--mtp", str(old / "mtp" / "rt")],
                             model_path=str(self.base / "docs"), data_dir=str(self.base / "c08-data"))
        r.say_reused_data(cfg)
        self.assertEqual(len(r.said), 1)
        self.assertIn("the prepared pack and the draft layer are read from", r.said[0].lower())
        self.assertIn("dl-data", r.said[0])
        self.assertIn("nothing was copied", r.said[0])

    def test_no_line_when_they_are_in_the_given_folder_or_nothing_was_given(self):
        new = self.base / "c08-data"
        r, cfg = self.runner(["--pack", str(new / "packs" / "x")], model_path=str(self.base / "docs"), data_dir=str(new))
        r.say_reused_data(cfg)
        self.assertEqual(r.said, [])
        r, cfg = self.runner(["--pack", str(self.base / "o" / "packs" / "x")], model_path=str(self.base / "docs"))
        r.say_reused_data(cfg)
        self.assertEqual(r.said, [])


if __name__ == "__main__":
    unittest.main()
