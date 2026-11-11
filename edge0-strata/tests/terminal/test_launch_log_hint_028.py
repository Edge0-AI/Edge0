"""#3714: the launch hint named an engine log that was never written (the start stopped before the engine existed). Nothing here starts
an engine or reads the real profile."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import runner as R, settings as LS
from strata_control.profiles import ProfileError
from tests.terminal.launch_support import HW_12, Home, catalog


class LogHint(unittest.TestCase):
    """#3714: 'The engine's log has the reason: <path>' for a log that was never written (the start stopped before the engine existed)."""

    def config(self, home: Home, log: Path) -> Path:
        path = home.base / "run.json"
        path.write_text('{"exe": "e", "args": [], "log": %s}' % json.dumps(str(log)), encoding="utf-8")
        return path

    def test_a_busy_profile_gets_its_own_sentence_and_no_log_path(self):
        with Home() as h:
            log = h.base / "missing.log"
            text = R._log_hint(self.config(h, log), ProfileError("this profile already has a live managed runtime", 6))
            self.assertNotIn(str(log), text)
            self.assertIn("still running or starting", text)
            self.assertIn("strata launch --stop", text)

    def test_a_log_that_does_not_exist_is_not_pointed_at(self):
        with Home() as h:
            log = h.base / "missing.log"
            text = R._log_hint(self.config(h, log), RuntimeError("something else"))
            self.assertNotIn(str(log), text)
            self.assertIn("did not get as far as writing its log", text)

    def test_a_log_that_exists_is_still_pointed_at(self):
        with Home() as h:
            log = h.base / "engine.log"
            log.write_text("line\n", encoding="utf-8")
            self.assertEqual(R._log_hint(self.config(h, log), RuntimeError("x")), f"The engine's log has the reason: {log}")

    def test_without_a_config_the_general_sentence_stays(self):
        self.assertIn("strata logs", R._log_hint(None))

    def test_explain_engine_exit_does_not_name_a_missing_log_either(self):
        with Home() as h:
            log = h.base / "missing.log"
            text, nxt = R.explain_engine_exit({"state": "failed", "error": "server_exited"}, self.config(h, log))
            self.assertNotIn(str(log), nxt)
            self.assertIn("did not get as far as writing its log", nxt)
            log.write_text("[strata] out of memory\n", encoding="utf-8")
            text, nxt = R.explain_engine_exit({"state": "failed", "error": "server_exited"}, self.config(h, log))
            self.assertIn(f"The engine's log: {log}", nxt)

    def test_start_reports_a_busy_profile_without_a_log_path(self):
        with Home() as h:
            log = h.base / "never-written.log"
            runner = R.Runner(LS.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768, vision="off"), catalog=catalog(HW_12))
            runner.config_path = self.config(h, log)
            busy = ProfileError("this profile already has a live managed runtime", 6)
            with mock.patch.object(R.Runner, "stop_others"), mock.patch.object(R.Runner, "check_port_free"), \
                    mock.patch.object(R.Runner, "log_marks", return_value={}), mock.patch.object(R.Runner, "end_failed_start"), \
                    mock.patch.object(R.Runner, "start_wait", return_value=5), mock.patch.object(R.Runner, "emit"), \
                    mock.patch("strata_control.runtime.status", return_value={"state": "stopped"}), \
                    mock.patch("strata_control.runtime.start", side_effect=busy):
                with self.assertRaises(R.LaunchError) as raised:
                    runner.start()
        self.assertIn("already has a live managed runtime", raised.exception.text)
        self.assertNotIn(str(log), raised.exception.next)
        self.assertIn("strata launch --stop", raised.exception.next)


if __name__ == "__main__":
    unittest.main()
