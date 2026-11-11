"""The monitor as `strata launch` opens it: the Connect view (key 7), its keys, the check, and the quit question."""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import check
from strata_cli.launch.monitor_ext import LaunchMonitor
from tests.terminal.launch_support import Home, text_of
from tests.terminal.test_overview import Source, metrics

INFO = {"url": "http://127.0.0.1:8080", "urls": ["http://127.0.0.1:8080", "http://192.168.1.5:8080"], "key": "sk-strata-abcdefghijklmnop",
        "model": "qwen3.8-flash-next-iq3_xxs", "context": 32768, "profile": "launch-x", "name": "Demo"}


def flat(app) -> str:
    return " ".join(text_of(app).split())


class Connect(unittest.TestCase):
    def run_app(self, coro, info=None):
        async def go():
            app = LaunchMonitor(Source(metrics("idle")), None, info=info or dict(INFO), interval=30)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.5)
                return await coro(app, pilot)
        return asyncio.run(go())

    def test_overview_first_and_connect_on_7(self):
        async def case(app, pilot):
            t = text_of(app)
            self.assertIn("Model state", t)
            self.assertIn("7 Connect", t)
            self.assertIn("q asks: keep serving or stop", t)
            await pilot.press("7")
            await pilot.pause(0.2)
            t = text_of(app)
            self.assertIn("Connect an app", t)
            self.assertIn("URL http://127.0.0.1:8080/v1", t)
            self.assertIn("also reachable at http://192.168.1.5:8080/v1", t)
            self.assertIn("key sk-strata-", t)
            self.assertNotIn("abcdefghijklmnop", t)                      # masked until k
            for name in ("OpenCode", "Codex CLI", "Claude Code", "oh-my-pi", "pi (pi.dev)", "Hermes Agent", "OpenClaw", "DeepSeek Harness"):
                self.assertIn(name, t)
            self.assertIn("Vendor tested", t)
            self.assertIn("This server", t)
            self.assertIn("Not checked", t)
            self.assertIn("opencode 1.19.3", t)
            self.assertIn("OpenCode setup", t)
            self.assertIn("~/.config/opencode/opencode.json", t)
            await pilot.press("k")
            await pilot.pause(0.2)
            self.assertIn("abcdefghijklmnop", flat(app).replace("▒", ""))
            await pilot.press("7")
            await pilot.pause(0.2)
            self.assertIn("Connect an app", text_of(app))                 # 7 is a view like the others: pressing it again stays
            await pilot.press("escape")
            await pilot.pause(0.2)
            self.assertIn("Model state", text_of(app))                    # Esc goes back to the view you came from
        self.run_app(case)

    def test_views_one_to_six_leave_connect(self):
        async def case(app, pilot):
            await pilot.press("7")
            await pilot.press("2")
            await pilot.pause(0.2)
            self.assertNotIn("Connect an app", text_of(app))
        self.run_app(case)

    def test_choosing_an_app_shows_its_limits_in_words(self):
        async def case(app, pilot):
            await pilot.press("7")
            for _ in range(5):
                await pilot.press("down")
            await pilot.pause(0.2)
            t = flat(app)
            self.assertIn("Hermes Agent setup", t)
            self.assertIn("provider: lmstudio", t)
            self.assertIn("WARNING: Hermes is built for 64K or more", t)
            await pilot.press("down")
            await pilot.pause(0.2)
            self.assertIn("Node 24.16", flat(app))
            await pilot.press("down")
            await pilot.pause(0.2)
            self.assertIn("built into the Strata app", flat(app))
            await pilot.press("up", "up", "up", "up", "up", "up", "up")
            await pilot.pause(0.2)
            self.assertIn("OpenCode setup", flat(app))
        self.run_app(case)

    def test_check_reports_each_step_and_the_row(self):
        results = [check.Check("prompt", True, "It answered: 'pong'.", 1.0), check.Check("tool call", False, "The model answered in words and did not call the tool.", 2.0)]
        with mock.patch.object(check, "run_checks", return_value=results):
            async def case(app, pilot):
                await pilot.press("7", "h")
                for _ in range(30):
                    await pilot.pause(0.1)
                    if "Failed (1/2)" in text_of(app):
                        break
                t = flat(app)
                self.assertIn("Failed (1/2)", t)
                self.assertIn("prompt: It answered: 'pong'.", t)
                self.assertIn("did not call the tool", t)
            self.run_app(case)

    def test_write_asks_then_writes_with_a_backup(self):
        with Home() as home:
            target = Path(os.environ["HOME"]) / ".config" / "opencode" / "opencode.json"
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps({"theme": "dark"}))

            async def case(app, pilot):
                await pilot.press("7", "w")
                await pilot.pause(0.2)
                self.assertIn("Write OpenCode's settings?", flat(app))
                self.assertIn("copied to a backup first", flat(app))
                await pilot.press("enter")
                await pilot.pause(0.3)
            self.run_app(case)
            data = json.loads(target.read_text())
            self.assertEqual(data["theme"], "dark")
            self.assertIn("strata", data["provider"])
            self.assertEqual(len(list(target.parent.glob("opencode.json.strata-bak-*"))), 1)

    def test_write_can_be_declined(self):
        with Home():
            target = Path(os.environ["HOME"]) / ".config" / "opencode" / "opencode.json"

            async def case(app, pilot):
                await pilot.press("7", "w", "down", "enter")
                await pilot.pause(0.2)
            self.run_app(case)
            self.assertFalse(target.exists())

    def test_copy_does_not_crash(self):
        async def case(app, pilot):
            await pilot.press("7", "c")
            await pilot.pause(0.2)
            self.assertTrue("Copied" in flat(app) or "does not allow copying" in flat(app))
        self.run_app(case)


class Quit(unittest.TestCase):
    def go(self, keys):
        holder = {}

        async def go():
            app = LaunchMonitor(Source(metrics("idle")), None, info=dict(INFO), interval=30)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.4)
                for k in keys:
                    await pilot.press(k)
                    await pilot.pause(0.2)
                holder["app"] = app
                holder["running"] = app.is_running
                holder["stop"] = app.stop_requested
        asyncio.run(go())
        return holder

    def test_q_asks_with_three_choices(self):
        async def go():
            app = LaunchMonitor(Source(metrics("idle")), None, info=dict(INFO), interval=30)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.4)
                await pilot.press("q")
                await pilot.pause(0.2)
                t = flat(app)
                self.assertIn("Leave the monitor?", t)
                self.assertIn("Keep serving in the background", t)
                self.assertIn("Stop serving and free the memory", t)
                self.assertIn("Stay in the monitor", t)
                await pilot.press("down", "down", "enter")                # Stay
                await pilot.pause(0.2)
                self.assertTrue(app.is_running)
                self.assertFalse(app.stop_requested)
        asyncio.run(go())

    def test_keep_serving_leaves_without_stopping(self):
        h = self.go(["q", "enter"])
        self.assertFalse(h["stop"])

    def test_stop_asks_the_caller_to_stop(self):
        h = self.go(["q", "down", "enter"])
        self.assertTrue(h["stop"])

    def test_escape_cancels_the_question(self):
        async def go():
            app = LaunchMonitor(Source(metrics("idle")), None, info=dict(INFO), interval=30)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.4)
                await pilot.press("q", "escape")
                await pilot.pause(0.2)
                self.assertTrue(app.is_running)
        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
