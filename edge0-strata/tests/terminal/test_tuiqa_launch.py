"""Tester report 2026-10-18 (Terminal 0.1.45), T02: `strata launch` with flags opened the screens straight into the run, which crashed
(IndexError: pop from empty list in switch_screen) and exited 0. The run now opens on its own screen; an error of the screens ends non-zero."""

from __future__ import annotations

import asyncio
import contextlib
import json
import io
import unittest
from unittest import mock

from textual.app import App

from strata_cli.launch import app as launch_app
from strata_cli.launch import settings as S
from strata_cli.launch.app import Ctx, LaunchApp
from strata_cli.launch.widgets import Glyphs
from tests.terminal.launch_support import HW_12, Home, catalog, text_of
from tests.terminal.test_launch_tui import FakeRunner, SIZE, make_ctx


def flags_settings() -> S.LaunchSettings:
    """What `strata launch --model q2_0 --family qwen --context 128k --kv int8 --vision cpu --parallel 1 --tune skip --access local --port 8081` gives."""
    return S.LaunchSettings(name="QA flags", family="qwen", model="Q2_0", context=131072, kv="int8", vision="cpu", parallel=1, tune="skip",
                            access="local", port=8081)


class FlagsOpenTheRun(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        FakeRunner.instances.clear()
        FakeRunner.behaviour = "ok"
        FakeRunner.download = 0.0
        p = mock.patch("strata_cli.launch.app.Runner", FakeRunner)
        p.start()
        self.addCleanup(p.stop)

    def test_auto_run_shows_the_run_screen_and_finishes(self):
        app = LaunchApp(make_ctx(settings=flags_settings()), auto_run=True)

        async def go():
            async with app.run_test(size=SIZE) as pilot:
                for _ in range(60):
                    await pilot.pause(0.05)
                    if app.result is not None:
                        break
                return app.result

        result = asyncio.run(go())
        self.assertIsNotNone(result, "the run did not finish: the screens crashed instead of opening the run")
        self.assertEqual(result.urls, ["http://127.0.0.1:8081"])

    def test_auto_run_with_a_big_download_asks_first_and_then_runs(self):
        FakeRunner.download = 66.0
        app = LaunchApp(make_ctx(settings=flags_settings()), auto_run=True)

        async def go():
            async with app.run_test(size=SIZE) as pilot:
                await pilot.pause(0.3)
                self.assertIn("Download about 66 GB?", text_of(app))
                await pilot.press("enter")                      # Download and start
                for _ in range(60):
                    await pilot.pause(0.05)
                    if app.result is not None:
                        break
                return app.result

        self.assertIsNotNone(asyncio.run(go()))

    def headless_run(self, app):
        return App.run(app, headless=True)

    def test_run_tui_with_flags_returns_zero_and_opens_the_monitor(self):
        calls = []
        with mock.patch.object(LaunchApp, "run", lambda self, **kw: App.run(self, headless=True)), \
                mock.patch.object(launch_app, "run_monitor", lambda info, **kw: calls.append(info) or 0), \
                mock.patch.object(launch_app, "Catalog", create=True):
            cat = catalog(HW_12)
            code = launch_app.run_tui(flags_settings(), catalog=cat, auto_run=True)
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)

    def test_an_error_of_the_screens_ends_non_zero_with_a_plain_sentence(self):
        class Broken(FakeRunner):
            def __init__(self, *a, **kw):
                raise RuntimeError("boom")

        err = io.StringIO()
        with mock.patch.object(LaunchApp, "run", lambda self, **kw: App.run(self, headless=True)), \
                mock.patch.object(launch_app, "Runner", Broken), \
                contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = launch_app.run_tui(flags_settings(), catalog=catalog(HW_12), auto_run=True)
        self.assertNotEqual(code, 0)
        self.assertIn("nothing was started", err.getvalue())
        self.assertIn("--no-tui", err.getvalue())

    def test_the_command_turns_a_crash_into_exit_one(self):
        from strata_cli.launch import cmd
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = cmd._tui(lambda *a, **k: (_ for _ in ()).throw(IndexError("pop from empty list")))
        self.assertEqual(code, 1)
        self.assertIn("unexpected problem", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())


if __name__ == "__main__":
    unittest.main()


class NoEngineFlow(unittest.TestCase):
    """Owner decision 2026-10-18: no usable engine is a warning, not a refusal; the model downloads and the end message names the problem."""

    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)

    def served_nothing(self):
        from strata_cli.launch.runner import Result, no_engine_final
        text = no_engine_final("The feed has no engine package for this PC.")
        return Result("launch-x", [], None, None, [text], "/x", serving=False, engine_problem="The feed has no engine package for this PC.",
                      final_text=text)

    def test_the_command_ends_with_exit_zero_and_the_engine_message(self):
        from strata_cli.launch.runner import Runner
        from tests.terminal.launch_support import run_cli
        with mock.patch.object(Runner, "run", return_value=self.served_nothing()):
            code, out, err = run_cli(["launch", "--no-tui", "--model", "q2_0", "--yes"])
        self.assertEqual(code, 0, err)
        self.assertIn("The model is ready, but Strata can't start it yet", out)
        self.assertIn("The feed has no engine package for this PC", out)
        self.assertIn("Next step:", out)
        self.assertNotIn("Strata is serving", out)

    def test_json_says_not_serving(self):
        from strata_cli.launch.runner import Runner
        from tests.terminal.launch_support import run_cli
        with mock.patch.object(Runner, "run", return_value=self.served_nothing()):
            code, out, err = run_cli(["launch", "--no-tui", "--model", "q2_0", "--output", "json"])
        data = json.loads(out)
        self.assertEqual(code, 0)
        self.assertTrue(data["ok"])
        self.assertFalse(data["serving"])
        self.assertIn("engine", data["engine_problem"])

    def test_yes_is_passed_to_the_runner_and_json_never_asks(self):
        from strata_cli.launch.runner import Runner
        from tests.terminal.launch_support import run_cli
        seen = []
        real = Runner.__init__

        def spy(self, *a, **kw):
            seen.append(kw.get("assume_yes"))
            real(self, *a, **kw)

        with mock.patch.object(Runner, "__init__", spy), mock.patch.object(Runner, "run", return_value=self.served_nothing()):
            run_cli(["launch", "--no-tui", "--model", "q2_0", "--yes"])
            run_cli(["launch", "--no-tui", "--model", "q2_0", "--output", "json"])
            run_cli(["launch", "--no-tui", "--model", "q2_0"])
        self.assertEqual(seen, [True, True, False])

    def test_the_question_default_is_yes(self):
        from strata_cli.launch import cmd
        with mock.patch("builtins.input", return_value=""), mock.patch.object(cmd.sys, "stdin") as sin, \
                mock.patch.object(cmd.sys, "stdout") as sout:
            sin.isatty.return_value = True
            sout.isatty.return_value = True
            self.assertTrue(cmd._ask_yes("t", "text"))
        with mock.patch("builtins.input", return_value="n"), mock.patch.object(cmd.sys, "stdin") as sin, \
                mock.patch.object(cmd.sys, "stdout") as sout:
            sin.isatty.return_value = True
            sout.isatty.return_value = True
            self.assertFalse(cmd._ask_yes("t", "text"))
        self.assertTrue(cmd._ask_yes("t", "text"))               # no keyboard: the person already named the model

    def test_the_screens_ask_with_a_dialog_and_enter_means_yes(self):
        FakeRunner.instances.clear()
        FakeRunner.behaviour = "ok"
        FakeRunner.download = 0.0
        answers = []

        class Asking(FakeRunner):
            def run(self, start=True):
                answers.append(self.confirm("No engine", "Strata couldn't find an engine for this computer (x). Download?"))
                return super().run(start)

            def __init__(self, *a, confirm=None, **kw):
                super().__init__(*a, **kw)
                self.confirm = confirm

        app = LaunchApp(make_ctx(settings=flags_settings()), auto_run=True)

        async def go():
            with mock.patch("strata_cli.launch.app.Runner", Asking):
                async with app.run_test(size=SIZE) as pilot:
                    await pilot.pause(0.4)
                    self.assertIn("Strata couldn't find an engine for this computer", text_of(app))
                    await pilot.press("enter")                  # Download anyway
                    for _ in range(60):
                        await pilot.pause(0.05)
                        if app.result is not None:
                            break
                    return app.result

        self.assertIsNotNone(asyncio.run(go()))
        self.assertEqual(answers, [True])

    def test_the_screens_show_the_end_message_and_exit_zero(self):
        from strata_cli.launch.app import LaunchApp as LA
        res = self.served_nothing()
        with mock.patch.object(LA, "run", lambda self, **kw: res), mock.patch.object(launch_app, "run_monitor", side_effect=AssertionError("no monitor")):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = launch_app.run_tui(flags_settings(), catalog=catalog(HW_12), auto_run=True)
        self.assertEqual(code, 0)
        self.assertIn("The model is ready, but Strata can't start it yet", out.getvalue())
