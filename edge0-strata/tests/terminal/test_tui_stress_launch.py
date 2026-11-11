"""G04/E01 from the TUI stress test: a second launch while a model serves (same choices, other choices, no screens, --restart,
another context on the same port)."""

import os
import unittest
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.catalog import Catalog
from strata_cli.launch.runner import LaunchError, Runner
from tests.terminal.launch_support import HW_12, Home, catalog, run_cli


class SecondLaunchTests(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
        p.start()
        self.addCleanup(p.stop)

    def runner(self, **kw):
        s = S.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768, name="gyro")
        r = Runner(s, catalog=catalog(), **kw)
        r.said = []
        r.emit_cb = lambda e: r.said.append(e.text)
        return r

    @staticmethod
    def row(r, port=8080, name=None, choices=None):
        return [{"profile": "launch-" + (name or r.name), "name": name or r.name, "port": port,
                 "choices": r.choices() if choices is None else choices}]

    def test_same_choices_reuse_the_running_server(self):
        r = self.runner()
        rows = self.row(r)
        with mock.patch.object(Runner, "serving_launches", lambda self: rows), \
             mock.patch.object(Runner, "ensure_engine", side_effect=AssertionError("must not run")), \
             mock.patch("strata_control.runtime.stop", side_effect=AssertionError("must not stop")):
            res = r.run()
        self.assertTrue(res.reused)
        self.assertTrue(res.urls)
        self.assertTrue(any("nothing was restarted" in x for x in r.said))

    def test_other_choices_without_screens_refuse_in_one_line(self):
        r = self.runner()
        rows = self.row(r, choices={**r.choices(), "vision": "cpu", "context": 65536})
        with mock.patch.object(Runner, "serving_launches", lambda self: rows):
            with self.assertRaises(LaunchError) as cm:
                r.run()
        self.assertEqual(cm.exception.code, 6)
        self.assertIn("port 8080", cm.exception.text)
        self.assertIn("context 64K", cm.exception.text)
        self.assertIn("--restart", cm.exception.next)
        self.assertIn("strata launch --stop", cm.exception.next)

    def test_yes_alone_does_not_restart(self):
        r = self.runner(assume_yes=True)
        rows = self.row(r, choices={**r.choices(), "context": 65536})
        with mock.patch.object(Runner, "serving_launches", lambda self: rows):
            with self.assertRaises(LaunchError):
                r.run()

    def test_screens_ask_once_and_cancel_leaves_it_running(self):
        asked = []
        r = self.runner(confirm_restart=lambda q, d: asked.append((q, d)) or False)
        rows = self.row(r, choices={**r.choices(), "context": 65536})
        with mock.patch.object(Runner, "serving_launches", lambda self: rows), \
             mock.patch("strata_control.runtime.stop") as stop:
            with self.assertRaises(LaunchError) as cm:
                r.run()
        self.assertEqual(len(asked), 1)
        self.assertIn("Restart it with these settings?", asked[0][0])
        self.assertEqual(asked[0][1], "Answers in progress stop.")
        self.assertEqual(cm.exception.code, 130)
        stop.assert_not_called()

    def test_screens_restart_goes_on(self):
        r = self.runner(confirm_restart=lambda q, d: True)
        rows = self.row(r, choices={**r.choices(), "context": 65536})
        with mock.patch.object(Runner, "serving_launches", lambda self: rows), \
             mock.patch.object(Runner, "ensure_engine", side_effect=RuntimeError("went on")):
            with self.assertRaises(RuntimeError):
                r.run()

    def test_restart_flag_goes_on_without_asking(self):
        r = self.runner(restart=True)
        rows = self.row(r, choices={**r.choices(), "context": 65536})
        with mock.patch.object(Runner, "serving_launches", lambda self: rows), \
             mock.patch.object(Runner, "ensure_engine", side_effect=RuntimeError("went on")):
            with self.assertRaises(RuntimeError):
                r.run()

    def test_a_changed_context_on_the_same_port_stops_the_old_profile_first(self):
        r = self.runner(restart=True)
        rows = self.row(r, name="gyro-s-32k", choices={"context": 32768})      # another profile name, same port 8080
        order = []
        with mock.patch.object(Runner, "serving_launches", lambda self: rows):
            self.assertIsNone(r.check_serving())
        self.assertEqual(r._stop_first, ["launch-gyro-s-32k"])
        with mock.patch("strata_control.runtime.stop", side_effect=lambda n, timeout=0: order.append(n)), \
             mock.patch("strata_cli.launch.access.stop_tunnel"):
            r.stop_others()
        self.assertEqual(order, ["launch-gyro-s-32k"])
        self.assertEqual(r._stop_first, [])

    def test_a_changed_context_without_restart_names_the_running_one(self):
        r = self.runner()
        rows = self.row(r, name="gyro-s-32k", choices={"context": 32768})
        with mock.patch.object(Runner, "serving_launches", lambda self: rows):
            with self.assertRaises(LaunchError) as cm:
                r.check_serving()
        self.assertIn("gyro-s-32k", cm.exception.text)

    def test_a_different_port_runs_beside_with_a_memory_warning(self):
        r = self.runner()
        r.s.port = 8081
        rows = self.row(r, name="other", port=8080, choices={})
        with mock.patch.object(Runner, "serving_launches", lambda self: rows):
            self.assertIsNone(r.check_serving())
        self.assertEqual(r._stop_first, [])
        self.assertTrue(any("graphics memory" in n for n in r.notices))

    def test_command_line_refuses_restarts_and_yes_alone_does_not(self):
        rows = [{"profile": "launch-q2-0-32k", "name": "q2-0-32k", "port": 8080, "choices": {"context": 1}}]
        with mock.patch.object(Runner, "serving_launches", lambda self: rows):
            code, out, err = run_cli(["launch", "--no-tui", "--model", "q2_0", "--context", "32k"])
            self.assertEqual(code, 6, err)
            self.assertIn("--restart", err)
            self.assertIn("q2-0-32k", err)
            code, out, err = run_cli(["launch", "--no-tui", "--yes", "--model", "q2_0", "--context", "32k"])
            self.assertEqual(code, 6)
            seen = {}

            def go(self, *a, **k):
                seen["restart"] = self.restart
                from strata_cli.launch.runner import Result
                return Result("launch-x", ["http://127.0.0.1:8080"], None, None, [], "/x")

            with mock.patch.object(Runner, "run", go):
                code, out, err = run_cli(["launch", "--no-tui", "--yes", "--restart", "--model", "q2_0", "--context", "32k"])
            self.assertEqual(code, 0, err)
            self.assertTrue(seen["restart"])

    def test_restart_is_in_help_and_docs(self):
        code, out, _ = run_cli(["launch", "--help"])
        self.assertIn("--restart", out)
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(root, "docs", "TERMINAL_LAUNCH.md"), encoding="utf-8") as f:
            self.assertIn("--restart", f.read())


class ModelPathWins(unittest.TestCase):
    """C08: --model-path and --data-dir are honoured even when an installed copy with the same model and context exists."""

    def setUp(self):
        import json
        import tempfile
        from pathlib import Path
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
        p.start()
        self.addCleanup(p.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.installed = self.base / "dl-data"
        (self.installed / "models" / "Q2_0").mkdir(parents=True)
        self.gguf = self.installed / "models" / "Q2_0" / "m.gguf"
        self.gguf.write_bytes(b"x")
        self.pack = self.installed / "packs" / "q2_0"
        self.pack.mkdir(parents=True)
        self.cfg_path = self.base / "strata-q2_0.json"
        self.cfg_path.write_text(json.dumps({"args": ["--model", str(self.gguf), "--pack", str(self.pack), "--max-context", "32768"]}), encoding="utf-8")
        self.mine = self.base / "documents"
        self.mine.mkdir()
        (self.mine / "mine.gguf").write_bytes(b"y")

    def runner(self, **fields):
        s = S.LaunchSettings(family="qwen", model="Q2_0", context=32768, **fields)
        return Runner(s, catalog=catalog())

    def test_installed_copy_serves_when_nothing_was_given(self):
        self.assertTrue(self.runner().installed_serves_request(self.cfg_path))

    def test_a_model_path_elsewhere_is_not_served_by_the_installed_copy(self):
        self.assertFalse(self.runner(model_path=str(self.mine)).installed_serves_request(self.cfg_path))
        self.assertFalse(self.runner(model_path=str(self.mine / "mine.gguf")).installed_serves_request(self.cfg_path))

    def test_a_model_path_that_is_the_installed_folder_is_the_installed_copy(self):
        self.assertTrue(self.runner(model_path=str(self.installed / "models" / "Q2_0")).installed_serves_request(self.cfg_path))

    def test_a_data_dir_elsewhere_is_not_served_by_the_installed_copy(self):
        self.assertFalse(self.runner(data_dir=str(self.base / "c08-data")).installed_serves_request(self.cfg_path))
        self.assertTrue(self.runner(data_dir=str(self.installed)).installed_serves_request(self.cfg_path))

    def test_ensure_model_runs_setup_with_the_given_path_instead_of_reusing_the_profile(self):
        r = self.runner(model_path=str(self.mine))
        said = []
        r.emit_cb = lambda e: said.append(e.text)
        desc = {"context": 32768, "kv": "int8", "images": "off"}
        with mock.patch.object(Runner, "check_own_files"), mock.patch.object(Runner, "resolved_vision"),              mock.patch.object(Runner, "installed_state", return_value=(self.cfg_path, desc)),              mock.patch.object(Runner, "_spawn", side_effect=RuntimeError("setup would run")):
            with self.assertRaises(RuntimeError):
                r.ensure_model()
        self.assertFalse(any("already installed with these choices" in x for x in said))
        self.assertTrue(any("uses the folder you gave" in x for x in said))
        self.assertIn(str(self.mine), r.setup_args())

    def test_without_a_path_the_installed_copy_is_still_reused(self):
        r = self.runner()
        said = []
        r.emit_cb = lambda e: said.append(e.text)
        desc = {"context": 32768, "kv": "int8", "images": "off"}
        with mock.patch.object(Runner, "check_own_files"), mock.patch.object(Runner, "resolved_vision"),              mock.patch.object(Runner, "installed_state", return_value=(self.cfg_path, desc)):
            self.assertEqual(r.ensure_model(), self.cfg_path)
        self.assertTrue(any("already installed with these choices" in x for x in said))


if __name__ == "__main__":
    unittest.main()
