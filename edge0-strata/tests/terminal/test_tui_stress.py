"""Fixes from the owner's TUI stress test of the 0.2.3 RC (status/logs discovery, key reuse, retune, folder, quit keys, --model-path)."""

import io
import json
import os
import unittest
from contextlib import redirect_stdout
from unittest import mock

from strata_cli.cli import main


class StatusLogsDiscoveryTests(unittest.TestCase):
    """#7 (F02): status and logs find the running launch without --url."""

    def _run(self, *argv):
        info = {"running": True, "url": "http://127.0.0.1:8080", "key": "k-launch"}
        env = {k: v for k, v in os.environ.items() if k not in ("STRATA_URL", "STRATA_CONNECTION")}
        out = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("strata_cli.launch.cmd.launch_info", return_value=info), \
             mock.patch("strata_cli.monitor_commands.Endpoint") as endpoint, redirect_stdout(out):
            endpoint.return_value.snapshot.return_value = {"schema_version": 1}
            code = main(list(argv))
        return code, endpoint

    def test_status_uses_the_running_launch(self):
        code, endpoint = self._run("status", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(endpoint.call_args[0][0], "http://127.0.0.1:8080")
        self.assertEqual(endpoint.call_args[0][1], "k-launch")

    def test_logs_use_the_running_launch(self):
        with mock.patch("strata_cli.logs.run", return_value=0) as run:
            code, endpoint = self._run("logs")
        self.assertEqual(code, 0)
        self.assertEqual(endpoint.call_args[0][0], "http://127.0.0.1:8080")

    def test_explicit_url_wins(self):
        code, endpoint = self._run("status", "--json", "--url", "http://127.0.0.1:9999")
        self.assertEqual(endpoint.call_args[0][0], "http://127.0.0.1:9999")


class KeptKeyTests(unittest.TestCase):
    """#8 (E05/G01): a key made for a launch survives a restart."""

    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = os.path.join(self.temp.name, "launch.json")

    def _write(self, key):
        with open(self.cfg, "w", encoding="utf-8") as f:
            json.dump({"api_key": key}, f)

    def test_key_of_the_launch_config_is_kept(self):
        from strata_cli.launch import access
        self._write("sk-strata-AAA")
        with mock.patch("strata_control.profiles.load", return_value=({"source_config": self.cfg}, "r")):
            self.assertEqual(access.kept_key("gyro-s 32K"), "sk-strata-AAA")

    def test_no_key_and_none_are_not_kept(self):
        from strata_cli.launch import access
        self._write("none")
        with mock.patch("strata_control.profiles.load", return_value=({"source_config": self.cfg}, "r")),              mock.patch("strata_control.profiles.list_profiles", return_value=[]):
            self.assertIsNone(access.kept_key("gyro-s 32K"))

    def test_newest_launch_config_is_the_fallback(self):
        from strata_cli.launch import access
        self._write("sk-strata-BBB")
        rows = [{"id": "launch-other", "source_config": self.cfg}, {"id": "other", "source_config": "x"}]
        with mock.patch("strata_control.profiles.load", side_effect=Exception("none")),              mock.patch("strata_cli.launch.store.load", return_value=None),              mock.patch("strata_control.profiles.list_profiles", return_value=rows):
            self.assertEqual(access.kept_key("gyro-s 64K"), "sk-strata-BBB")

    def test_apply_defaults_reuses_the_key_and_new_key_is_a_different_step(self):
        from strata_cli.launch import cmd, settings as S
        from tests.terminal.launch_support import HW_12, Home, catalog
        s = S.LaunchSettings(family="qwen", model="IQ3_XXS")
        s.access = "lan"
        with Home(), mock.patch("strata_cli.launch.access.kept_key", return_value="sk-strata-KEPT"):
            s2, notes = cmd.apply_defaults(s, catalog(HW_12))
        self.assertEqual(s2.api_key, "sk-strata-KEPT")
        self.assertFalse(any("A new API key was made" in n.line() for n in notes))


from tests.terminal import test_updater, update_fixture as fx   # noqa: E402  (shared update fixtures)
from strata_control import updater                                  # noqa: E402


class OwnLaunchHoldsTheCardTests(test_updater.UpdaterCase):
    """#10 (H04/H06): the engine update says one next step when Strata's own launch holds the card."""

    def test_test_start_failure_with_own_launch_serving_gives_the_next_step(self):
        self.install_current()
        self.publish(("0.1.47", fx.GOOD_ENGINE))
        flag = {"held": False}

        def runner(stage_root, staged, ctx):
            flag["held"] = True
            raise updater.TestStartFailed(staged.version, "CUDA error: out of memory (raw engine text)")

        serving = lambda: [{"profile": "launch-gyro-s-32k", "state": "ready"}] if flag["held"] else []
        out = updater.run_update(self.ctx(test_start=runner, serving=serving), control=False)
        result = out["results"][0]
        self.assertEqual((out["exit_code"], result["status"]), (7, "failed"))
        self.assertIn("strata launch --stop", result["error"])
        self.assertIn("run `strata update` again", result["error"])
        self.assertNotIn("raw engine text", result["error"])

    def test_own_launch_serving_before_the_test_is_also_one_plain_step(self):
        self.install_current()
        self.publish(("0.1.47", fx.GOOD_ENGINE))
        out = updater.run_update(self.ctx(serving=lambda: [{"profile": "launch-gyro-s-32k", "state": "ready"}]), control=False)
        self.assertEqual(out["exit_code"], 7)
        self.assertIn("Stop the model first (`strata launch --stop`)", out["results"][0]["error"])


class RetuneStopsOwnServerTests(unittest.TestCase):
    """#10 (E07): measuring while the same launch serves stops it first, and says so."""

    def _runner(self, state):
        from strata_cli.launch import settings as S
        from strata_cli.launch.runner import Runner
        class Bare(Runner):
            profile = "launch-x"
        r = Bare.__new__(Bare)
        r.s = S.LaunchSettings(name="x")
        r.restart_after = True
        r._stop_first = []
        r.said = []
        r.emit = lambda *a, **k: r.said.append(a[1])
        return r

    def test_serving_profile_is_stopped_and_the_line_is_plain(self):
        r = self._runner(None)
        with mock.patch("strata_control.runtime.status", return_value={"state": "ready"}),              mock.patch("strata_control.runtime.stop") as stop:
            r.stop_own_server_for_tuning()
        stop.assert_called_once()
        self.assertEqual(r.said, ["Stopping the model to measure; it starts again afterwards."])

    def test_nothing_is_stopped_when_nothing_serves(self):
        r = self._runner(None)
        with mock.patch("strata_control.runtime.status", return_value={"state": "stopped"}),              mock.patch("strata_control.runtime.stop") as stop:
            r.stop_own_server_for_tuning()
        stop.assert_not_called()
        self.assertEqual(r.said, [])


if __name__ == "__main__":
    unittest.main()
