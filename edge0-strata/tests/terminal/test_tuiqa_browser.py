"""Owner rule 2026-10-18: the Terminal never opens a browser by itself; only an explicit --open does (the monitor's b key is also a request). The generic
start message promises a browser only when one will open."""

from __future__ import annotations

import contextlib
import io
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import cmd, settings as S
from strata_cli.launch.runner import Result, Runner, derive_config
from strata_control import tune_knobs
from tests.terminal.launch_support import Home, run_cli


def served() -> Result:
    return Result("launch-x", ["http://127.0.0.1:8081"], None, None, [], "/x")


class NoBrowserByItself(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)

    def test_the_launch_config_forbids_a_browser_and_drops_open_from_the_args(self):
        base = {"exe": "e", "args": ["--native", "x.gguf", "--open", "--max-context", "32768"], "open_browser": True}
        cfg = derive_config(base, S.LaunchSettings(), bool_flags=tune_knobs.BOOL_FLAGS)
        self.assertIs(cfg["open_browser"], False)
        self.assertNotIn("--open", cfg["args"])
        self.assertEqual(cfg["args"][:2], ["--native", "x.gguf"])

    def test_a_plain_launch_opens_no_browser(self):
        with mock.patch.object(Runner, "run", return_value=served()), mock.patch("webbrowser.open") as opened:
            code, out, err = run_cli(["launch", "--no-tui", "--model", "q2_0"])
        self.assertEqual(code, 0, err)
        opened.assert_not_called()

    def test_open_opens_the_local_page_once(self):
        with mock.patch.object(Runner, "run", return_value=served()), mock.patch("webbrowser.open", return_value=True) as opened:
            code, out, err = run_cli(["launch", "--no-tui", "--model", "q2_0", "--open"])
        self.assertEqual(code, 0, err)
        opened.assert_called_once_with("http://127.0.0.1:8081")

    def test_the_screens_open_a_browser_only_for_open(self):
        from strata_cli.launch import app as launch_app
        from tests.terminal.launch_support import HW_12, catalog
        for flag, calls in ((False, 0), (True, 1)):
            with mock.patch.object(launch_app.LaunchApp, "run", lambda self, **kw: served()), \
                    mock.patch.object(launch_app, "run_monitor", lambda info, **kw: 0), mock.patch("webbrowser.open", return_value=True) as opened:
                with contextlib.redirect_stdout(io.StringIO()):
                    launch_app.run_tui(S.LaunchSettings(family="qwen", model="Q2_0", context=32768, vision="off"), catalog=catalog(HW_12),
                                       auto_run=True, open_page=flag)
            self.assertEqual(opened.call_count, calls, flag)

    def test_open_is_in_the_help(self):
        code, out, err = run_cli(["launch", "--help"])
        self.assertIn("--open", out)
        self.assertIn("nothing opens a browser without this", out.replace("\n", " ").replace("  ", " "))


class StartMessage(unittest.TestCase):
    def narrate(self, opens: bool) -> str:
        from serve import server
        d = tempfile.mkdtemp()
        log = Path(d) / "engine.log"
        log.write_text("strata generate: expert arena: resident, 31.64 GiB\n", encoding="utf-8")
        done, out = threading.Event(), io.StringIO()
        with mock.patch.object(server, "OPENS_BROWSER", opens), contextlib.redirect_stdout(out):
            t = threading.Thread(target=server.narrate_start, args=(str(log), 0, [], done))
            t.start()
            time.sleep(0.8)
            done.set()
            t.join(5)
        return out.getvalue()

    def test_the_browser_is_promised_only_with_open(self):
        self.assertNotIn("browser", self.narrate(False))
        self.assertIn("the server says when it is ready", self.narrate(False))
        self.assertIn("the browser opens when it is ready", self.narrate(True))


if __name__ == "__main__":
    unittest.main()
