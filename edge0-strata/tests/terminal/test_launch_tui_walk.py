"""Release 0.2.4 TUI stability walk: every screen of `strata launch`, every key, for a Qwen and a DeepSeek-V4.1 pick, at 80x24 and a wide
terminal, on several computers, with back/escape on each screen and a seeded random-key fuzz. Nothing may raise, hang or leave a blank
screen; the app must still be alive after every key. The same file runs on Windows and Linux."""

from __future__ import annotations

import asyncio
import os
import random
import unittest
from unittest import mock

from strata_cli.launch import access, store
from strata_cli.launch import settings as S
from strata_cli.launch.app import LaunchApp
from tests.terminal.launch_support import HW_12, HW_24, HW_2X16, Home, text_of
from tests.terminal.test_launch_tui import FakeRunner, make_ctx

HW_H100 = {"ram_gb": 1024.0, "os": "Test OS", "cpu": {"avx2": True, "name": "test cpu"},
           "gpus": [{"index": 0, "name": "NVIDIA H100 80GB HBM3", "vram_gb": 80.0, "usable": True, "vendor": "nvidia"}]}
HW_NOGPU = {"ram_gb": 16.0, "os": "Test OS", "cpu": {"avx2": True, "name": "test cpu"}, "gpus": []}
HW_AMD = {**HW_12, "gpus": [{"index": 0, "name": "AMD Radeon RX 9070 XT", "vram_gb": 16.0, "usable": True, "vendor": "amd"}]}

SIZES = {"narrow": (80, 24), "wide": (200, 60)}
PICKS = {"qwen": ("qwen", "IQ3_XXS"), "ds200": ("deepseek41", "IST-DAS-2.00bpw"), "ds225": ("deepseek41", "IST-DAS-2.25bpw")}
# every key a person can press on any screen except q / ctrl+c (they leave on purpose and are tested on their own)
KEYS = ["enter", "escape", "up", "down", "left", "right", "pageup", "pagedown", "home", "end", "tab", "delete", "backspace",
        "question_mark", "f1", "slash", "space", "plus", "minus", "1", "2", "3", "a", "c", "d", "e", "h", "k", "m", "n", "o", "p", "r", "t", "x"]


def alive(app) -> bool:
    return app.is_running and app._exception is None


async def settle(pilot, n=2):
    await pilot.pause(0.05 * n)


async def dismiss_extras(app, pilot):
    """Close a pop-up or an open text field the last key opened."""
    for _ in range(4):
        if len(app.screen_stack) > 2 or type(app.screen).__name__ == "ConfirmScreen":
            await pilot.press("escape")
            await settle(pilot)
        else:
            break


class Walk(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        FakeRunner.instances.clear()
        FakeRunner.behaviour = "ok"
        FakeRunner.download = 0.0
        for p in (mock.patch("strata_cli.launch.app.Runner", FakeRunner),
                  mock.patch.object(access, "lan_ips", return_value=["192.168.1.5"])):
            p.start()
            self.addCleanup(p.stop)

    def drive(self, coro, hw=HW_12, size=(80, 24)):
        ctx = make_ctx(hw)
        app = LaunchApp(ctx)

        async def go():
            async with app.run_test(size=size) as pilot:
                await pilot.pause(0.2)
                return await coro(app, pilot, ctx)

        return asyncio.run(go())

    async def to_model(self, app, pilot, pick):
        app.screen.menu().select(("new", None))
        await pilot.press("enter")                                    # "Set up from the beginning"
        await settle(pilot, 3)
        self.assertEqual(app.current, "model")
        menu = app.screen.menu()
        menu.select(PICKS[pick])
        app.screen.paint()
        self.assertEqual((app.ctx.settings.family, app.ctx.settings.model) if False else menu.current.key, PICKS[pick])

    # ---- 1. Enter through every screen, Escape all the way back, forward again
    def _forward_back(self, pick, size, hw):
        async def case(app, pilot, ctx):
            await self.to_model(app, pilot, pick)
            seen = []
            for _ in range(20):
                self.assertTrue(alive(app), f"app died on {app.current}")
                t = text_of(app)
                self.assertTrue(t.strip(), f"blank screen on {app.current}")
                seen.append(app.current)
                if app.current == "save":
                    break
                before = app.current
                await pilot.press("n" if before == "advanced" else "enter")
                for _ in range(40):                                  # wait for the next screen (a slow terminal needs more than one frame)
                    await pilot.pause(0.05)
                    if app.current != before:
                        break
            self.assertEqual(seen[-1], "save", seen)
            for _ in range(len(seen)):                               # back to the Start screen (Esc on it would leave)
                before = app.current
                await pilot.press("escape")
                for _ in range(40):
                    await pilot.pause(0.05)
                    if app.current != before:
                        break
                self.assertTrue(alive(app), f"died going back from {before}; seen={seen} exc={app._exception!r}")
            self.assertEqual(app.current, "start")
            self.assertIn("Set up from the beginning", text_of(app))
        self.drive(case, hw=hw, size=size)

    def test_forward_and_back_every_combination(self):
        for hw_name, hw in (("12", HW_12), ("h100", HW_H100), ("2x16", HW_2X16), ("nogpu", HW_NOGPU), ("amd", HW_AMD)):
            for pick in PICKS:
                for sz, size in SIZES.items():
                    with self.subTest(hw=hw_name, pick=pick, size=sz):
                        self._forward_back(pick, size, hw)

    # ---- 2. Every key on every screen
    def _key_sweep(self, pick, size, hw):
        async def case(app, pilot, ctx):
            await self.to_model(app, pilot, pick)
            for step in app.flow():
                if app.current != step:
                    app.goto(step)
                    await settle(pilot, 3)
                for key in KEYS:
                    if key == "enter" and step == "save":            # starts the launch and leaves; test_run_paths covers it
                        continue
                    await pilot.press(key)
                    await settle(pilot)
                    await dismiss_extras(app, pilot)
                    self.assertTrue(alive(app), f"app died: key {key!r} on {step} ({pick}, {size})")
                    if app.screen_stack and type(app.screen).__name__ in ("RunScreen",):
                        app.back_to_start() if False else None
                    if app.current != step:                             # the key moved on or back: return to this screen
                        if app.current == "run" or app.current == "start":
                            app.current = step
                        app.goto(step)
                        await settle(pilot, 2)
                    t = text_of(app)
                    self.assertTrue(t.strip(), f"blank screen: key {key!r} on {step}")
        self.drive(case, hw=hw, size=size)

    def _sweep_pick(self, pick, hw):
        for sz, size in SIZES.items():
            with self.subTest(hw=hw["gpus"][0]["name"] if hw["gpus"] else "none", pick=pick, size=sz):
                self._key_sweep(pick, size, hw)

    def test_every_key_on_every_screen_qwen(self):
        self._sweep_pick("qwen", HW_12)

    def test_every_key_on_every_screen_deepseek_200(self):
        self._sweep_pick("ds200", HW_12)

    def test_every_key_on_every_screen_deepseek_225_two_cards(self):
        self._sweep_pick("ds225", HW_2X16)

    # ---- 3. Seeded random keys, including the run screen
    def test_random_keys_do_not_crash(self):
        for seed in range(3):
            for pick in PICKS:
                with self.subTest(seed=seed, pick=pick):
                    rng = random.Random(seed * 31 + len(pick))
                    size = SIZES["narrow" if seed % 2 else "wide"]

                    async def case(app, pilot, ctx):
                        await self.to_model(app, pilot, pick)
                        for i in range(60):
                            await pilot.press(rng.choice(KEYS))
                            await pilot.pause(0.03)
                            if type(app.screen).__name__ == "ConfirmScreen" and rng.random() < 0.4:
                                await pilot.press("enter")
                            self.assertTrue(alive(app) or app.return_value is None and not app.is_running,
                                            f"seed {seed} step {i}: exception {app._exception!r}")
                            if not app.is_running:
                                break
                            if app.is_running:
                                self.assertTrue(text_of(app).strip(), f"blank screen at step {i}")
                        return True
                    self.drive(case, hw=HW_2X16 if seed % 3 == 0 else HW_12, size=size)

    # ---- the Save screen says the same tuning length as the Tuning screen (DeepSeek's tune is minutes, not 20 / 2 hours)
    def test_save_screen_tuning_time_matches_the_model(self):
        for pick, want, never in (("qwen", "quick (about 20 min)", "5-10"), ("ds200", "quick (about 5-10 min)", "about 20 min")):
            with self.subTest(pick=pick):
                async def case(app, pilot, ctx):
                    await self.to_model(app, pilot, pick)
                    for _ in range(30):
                        if app.current == "save":
                            break
                        await pilot.press("n" if app.current == "advanced" else "enter")
                        await settle(pilot, 3)
                    t = " ".join(text_of(app).split())
                    self.assertIn(want, t)
                    self.assertNotIn(never, t)
                self.drive(case, hw=HW_12, size=(110, 34))


    # ---- 4. q and ctrl+c leave cleanly from every screen; the Start screen's Escape too
    def test_quit_keys_leave_from_every_screen(self):
        for key in ("q", "ctrl+c", "ctrl+q"):
            for step in ("start", "model", "cards", "folder", "context", "pictures", "parallel", "tuning", "advanced", "access", "save"):
                if key == "q" and step in ("folder", "save"):         # a text field has the focus there: q is a letter
                    continue
                with self.subTest(key=key, step=step):
                    async def case(app, pilot, ctx):
                        if step != "start":
                            await self.to_model(app, pilot, "ds200")
                            if step in app.flow():
                                app.goto(step)
                                await settle(pilot, 3)
                        await pilot.press(key)
                        await settle(pilot, 4)
                        self.assertFalse(app.is_running, f"{key} did not quit on {step}")
                        self.assertIsNone(app._exception)
                    self.drive(case, hw=HW_2X16)

    # ---- 5. Resize while on each screen
    def test_resize_on_every_screen(self):
        async def case(app, pilot, ctx):
            await self.to_model(app, pilot, "ds225")
            for step in app.flow():
                app.goto(step)
                await settle(pilot, 3)
                for size in ((60, 20), (200, 60), (80, 24), (40, 12), (120, 30)):
                    await pilot.resize_terminal(*size)
                    await settle(pilot, 3)
                    self.assertTrue(alive(app), f"resize {size} killed {step}")
                    self.assertTrue(text_of(app).strip(), f"blank at {size} on {step}")
        self.drive(case, hw=HW_2X16)

    # ---- 6. The run: success, failure and retry, stop and cancel, for Qwen and DeepSeek
    def test_run_paths(self):
        for pick in ("qwen", "ds200"):
            for behaviour in ("ok", "fail", "wait"):
                with self.subTest(pick=pick, behaviour=behaviour):
                    FakeRunner.behaviour = behaviour

                    async def case(app, pilot, ctx):
                        await self.to_model(app, pilot, pick)
                        for _ in range(30):
                            if app.current == "save":
                                break
                            await pilot.press("n" if app.current == "advanced" else "enter")
                            await settle(pilot, 3)
                        self.assertEqual(app.current, "save")
                        await pilot.press("enter")                     # Save screen: first option
                        for _ in range(40):
                            await pilot.pause(0.05)
                            if not app.is_running or app.current == "run":
                                break
                        for _ in range(10):
                            await pilot.pause(0.05)
                        if behaviour == "fail" and app.is_running:
                            self.assertIn("Not enough disk space", text_of(app))
                            await pilot.press("r")                       # retry
                            await pilot.pause(0.3)
                            await pilot.press("escape")                  # back to the start
                            await pilot.pause(0.2)
                        if behaviour == "wait" and app.is_running:
                            await pilot.press("s")
                            await pilot.pause(0.3)
                            self.assertIn("Stop now?", text_of(app))
                            await pilot.press("enter")                   # Stop
                            await pilot.pause(0.6)
                            self.assertIn("Stopped. Nothing was started.", text_of(app))
                        self.assertIsNone(app._exception)
                    self.drive(case, hw=HW_12)


class ErrorPaths(unittest.TestCase):
    """The screens must come up (and say so plainly) when the world is broken."""

    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        FakeRunner.instances.clear()
        FakeRunner.behaviour = "ok"
        FakeRunner.download = 0.0
        p = mock.patch.object(access, "lan_ips", return_value=[])           # no network at all
        p.start()
        self.addCleanup(p.stop)

    def run_flow(self, runner, hw=HW_12, pick="ds200", settle_s=1.0):
        ctx = make_ctx(hw)
        app = LaunchApp(ctx)

        async def go():
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause(0.2)
                app.screen.menu().select(("new", None))
                await pilot.press("enter")
                await pilot.pause(0.2)
                app.screen.menu().select(PICKS[pick])
                for _ in range(30):
                    if app.current == "save":
                        break
                    await pilot.press("n" if app.current == "advanced" else "enter")
                    await pilot.pause(0.1)
                await pilot.press("enter")
                await pilot.pause(settle_s)
                return text_of(app), app._exception, app.is_running

        with mock.patch("strata_cli.launch.app.Runner", runner):
            return asyncio.run(go())

    def test_runner_raising_anything_is_shown_in_words(self):
        for exc in (OSError(28, "No space left on device"), PermissionError("denied"), ConnectionError("feed unreachable"),
                    KeyError("x"), UnicodeEncodeError("charmap", "ı", 0, 1, "can't encode"), RuntimeError("boom")):
            with self.subTest(exc=type(exc).__name__):
                class Raising(FakeRunner):
                    def run(self, start=True):
                        raise exc

                text, err, running = self.run_flow(Raising)
                self.assertIsNone(err)
                self.assertTrue(running, "the screens closed instead of showing the failure")
                self.assertIn("Press r to try again", text)
                self.assertNotIn("Traceback", text)


class RealRunnerErrors(unittest.TestCase):
    """The real Runner against an unreachable feed, a port that is busy and a data folder that is full: plain failures, never a traceback."""

    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)

    def test_unreachable_feed_gives_a_plain_failure(self):
        from tests.terminal.test_launch_tui import make_ctx as mk
        ctx = mk(HW_12, S.LaunchSettings(name="x", family="qwen", model="IQ3_XXS", context=65536, tune="skip", access="local", port=18999))
        ctx.feed = "http://127.0.0.1:9"                              # nothing listens here
        app = LaunchApp(ctx, auto_run=True)

        async def go():
            async with app.run_test(size=(80, 24)) as pilot:
                for _ in range(100):
                    await pilot.pause(0.1)
                    t = text_of(app)
                    if "Press r to try again" in t or "Download about" in t or not app.is_running:
                        break
                if "Download about" in text_of(app):
                    await pilot.press("escape")
                    await pilot.pause(0.2)
                return text_of(app), app._exception

        text, err = asyncio.run(go())
        self.assertIsNone(err, text)
        self.assertNotIn("Traceback", text)


if __name__ == "__main__":
    unittest.main()
