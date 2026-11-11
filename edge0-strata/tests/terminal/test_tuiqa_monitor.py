"""Tester report 2026-10-18, T06 / T08 / T10 and the smaller monitor items: one seven-view model in every view and at every width,
Connect that says what is vendor history and what is this server's live result, and a light theme the Connect view and its pop-ups follow."""

from __future__ import annotations

import asyncio
import re
import unittest
from unittest import mock

from strata_cli import nav
from strata_cli.launch import check, harness
from strata_cli.launch.monitor_ext import LaunchMonitor
from strata_cli.launch.widgets import DARK, LIGHT
from strata_cli.monitor import MonitorApp
from tests.terminal.launch_support import Home, text_of
from tests.terminal.test_launch_monitor import INFO
from tests.terminal.test_overview import Source, metrics

SIZES = ((120, 40), (80, 24), (60, 20), (40, 15))
NAMES = ["Overview", "Serving", "Requests", "Engine", "Hardware", "Logs", "Connect"]
SHORT = {"Overview": "Over", "Serving": "Serve", "Requests": "Reqs", "Engine": "Engine", "Hardware": "GPU", "Logs": "Logs", "Connect": "Conn"}


def tab_line(shown: str) -> str:
    """The row with the tab names: the one that has the marked tab."""
    for line in shown.splitlines():
        if re.search(r"\[\d [A-Za-z]+\]", line) and re.search(r"\b\d [A-Za-z]+", line.replace("[", " ")):
            return line
    return ""


def marked(line: str) -> str | None:
    m = re.search(r"\[(\d) ([A-Za-z]+)\]", line)
    return None if m is None else m.group(2)


class TabModel(unittest.TestCase):
    def test_all_names_when_there_is_room(self):
        self.assertEqual(nav.plain(1, 120, True), "1 Overview  [2 Serving]  3 Requests  4 Engine  5 Hardware  6 Logs  7 Connect")
        self.assertEqual(nav.plain(1, 120, False), "1 Overview  [2 Serving]  3 Requests  4 Engine  5 Hardware  6 Logs")

    def test_short_names_then_a_window_around_the_selected_tab_and_it_is_always_marked(self):
        for connect in (False, True):
            for current in range(nav.count(connect)):
                for width in (120, 90, 80, 60, 50, 40, 30, 24):
                    line = nav.plain(current, width, connect)
                    self.assertLessEqual(len(line), width, (current, width, connect))
                    name = NAMES[current]
                    self.assertTrue(f"[{current + 1} {name}]" in line or f"[{current + 1} {SHORT[name]}]" in line, (line, current, width))
                    if width >= 60 + (10 if connect else 0):
                        self.assertNotIn("<", line)                       # nothing hidden when the short names fit
        self.assertEqual(nav.plain(3, 40, True), "< 3 Reqs [4 Engine] 5 GPU 6 Logs 7 Conn")
        self.assertEqual(nav.plain(3, 34, True), "< 3 Reqs [4 Engine] 5 GPU 6 Logs >")

    def test_the_hint_matches_the_number_of_views(self):
        self.assertEqual((nav.keys(False), nav.keys(True)), ("1-6", "1-7"))

    def test_styled_marks_the_selected_tab_bold(self):
        t = nav.styled(2, 120, True, accent="#34d399")
        spans = [(t.plain[s.start:s.end], s.style) for s in t.spans if "bold" in str(s.style)]
        self.assertEqual([x for x, _ in spans], ["[3 Requests]"])


class MonitorViews(unittest.TestCase):
    def run_app(self, cls, size, coro, **kw):
        async def go():
            if cls is LaunchMonitor:
                app = LaunchMonitor(Source(metrics("idle")), None, info=dict(INFO), interval=30, **kw)
            else:
                app = MonitorApp(Source(metrics("idle")), None, interval=30, **kw)
            async with app.run_test(size=size) as pilot:
                await pilot.pause(0.4)
                return await coro(app, pilot)
        with Home():
            return asyncio.run(go())

    def test_arrows_visit_all_seven_views_and_wrap_both_ways(self):
        async def case(app, pilot):
            seen = [app.view]
            for _ in range(7):
                await pilot.press("right")
                await pilot.pause(0.05)
                seen.append(app.view)
            self.assertEqual(seen, [0, 1, 2, 3, 4, 5, 6, 0])           # Logs -> Connect -> Overview
            await pilot.press("left")
            await pilot.pause(0.05)
            self.assertEqual(app.view, 6)                               # Overview <- Connect
            await pilot.press("left")
            await pilot.pause(0.05)
            self.assertEqual(app.view, 5)                               # Connect <- Logs: the previous tab, not a relative of the last ordinary view
        self.run_app(LaunchMonitor, (120, 40), case)

    def test_number_keys_and_arrows_agree(self):
        async def case(app, pilot):
            for n in range(1, 8):
                await pilot.press(str(n))
                await pilot.pause(0.05)
                self.assertEqual(app.view, n - 1)
        self.run_app(LaunchMonitor, (120, 40), case)

    def test_every_view_at_every_size_shows_the_tabs_the_hint_and_the_selection(self):
        async def case(app, pilot):
            for size in SIZES:
                await pilot.resize_terminal(*size)
                await pilot.pause(0.15)
                for index in range(7):
                    await pilot.press(str(index + 1))
                    await pilot.pause(0.08)
                    shown = text_of(app)
                    line = tab_line(shown)
                    name = marked(line)
                    self.assertIsNotNone(name, (size, index, shown[:400]))
                    self.assertIn(name, (NAMES[index], SHORT[NAMES[index]]), (size, index, line))
                    self.assertNotIn("1-6 views", shown, (size, index))           # the hint never disagrees with the tab line
                    self.assertIn("Overview" if index == 0 and size[0] >= 78 else NAMES[index][:3], shown.replace("[", " "), (size, index))
        self.run_app(LaunchMonitor, (120, 40), case)

    def test_the_plain_monitor_has_six_views_and_says_1_6(self):
        async def case(app, pilot):
            for size in SIZES:
                await pilot.resize_terminal(*size)
                await pilot.pause(0.15)
                shown = text_of(app)
                self.assertNotIn("Connect", shown)
                self.assertNotIn("1-7", shown)
            await pilot.press("7")
            await pilot.pause(0.05)
            self.assertEqual(app.view, 0)                                      # no seventh view here
            await pilot.press("left")
            await pilot.pause(0.05)
            self.assertEqual(app.view, 5)
        self.run_app(MonitorApp, (120, 40), case)

    def test_strata_monitor_is_the_same_seven_view_monitor_as_launchs(self):
        """`strata monitor --url ...` builds the launch monitor without the quit question: seven views, key 7, Connect points at that server."""
        from strata_cli import monitor_commands
        seen = {}

        class Fake:
            def __init__(self, endpoint, snapshot, **kw):
                seen.update(kw, endpoint=endpoint)

            def run(self):
                seen["ran"] = True

        args = monitor_commands.parser().parse_args(["monitor", "--url", "http://127.0.0.1:18765"])
        with mock.patch("strata_cli.launch.monitor_ext.LaunchMonitor", Fake), mock.patch("sys.stdout.isatty", return_value=True), Home():
            self.assertEqual(monitor_commands.run(args), 0)
        self.assertTrue(seen.get("ran"))
        self.assertFalse(seen["ask_quit"])                                   # q detaches: this command started nothing
        self.assertEqual(seen["info"]["url"], "http://127.0.0.1:18765")      # the Connect view's address is the --url server
        self.assertIn("seven views", " ".join(monitor_commands.parser().format_help().split()))

    def test_the_monitor_strata_monitor_opens_has_seven_views_and_connect_shows_the_server(self):
        async def case(app, pilot):
            shown = text_of(app)
            self.assertIn("7 Connect", shown)
            self.assertNotIn("1-6", shown)
            await pilot.press("7")
            await pilot.pause(0.2)
            self.assertEqual(app.view, 6)
            self.assertIn(INFO["url"], text_of(app))
            await pilot.press("escape")
            await pilot.pause(0.1)
            self.assertEqual(app.view, 0)
        async def go():
            app = LaunchMonitor(Source(metrics("idle")), None, info=dict(INFO), interval=30, ask_quit=False)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.4)
                return await case(app, pilot)
        with Home():
            asyncio.run(go())

    def test_the_overview_is_titled_overview(self):
        async def case(app, pilot):
            self.assertIn("STRATA  Overview", text_of(app))
            self.assertNotIn("STRATA  Serving", text_of(app))
            await pilot.press("2")
            await pilot.pause(0.1)
            self.assertIn("Serving", text_of(app))
        self.run_app(LaunchMonitor, (120, 40), case)


class ConnectStatus(unittest.TestCase):
    def run_app(self, coro, size=(120, 40), **kw):
        async def go():
            app = LaunchMonitor(Source(metrics("idle")), None, info=dict(INFO), interval=30, **kw)
            async with app.run_test(size=size) as pilot:
                await pilot.pause(0.4)
                return await coro(app, pilot)
        with Home():
            return asyncio.run(go())

    def test_vendor_history_and_this_servers_result_are_separate_columns(self):
        async def case(app, pilot):
            await pilot.press("7")
            await pilot.pause(0.2)
            shown = text_of(app)
            header = next(ln for ln in shown.splitlines() if "Vendor tested" in ln)
            self.assertIn("This server", header)
            for h in harness.HARNESSES:
                said = "built into Strata" if h.built_in else h.tested
                row = next(ln for ln in shown.splitlines() if h.title[:16] in ln and said[:23] in ln)
                self.assertIn("Not checked", row)                          # nothing has been tried against THIS server yet
                self.assertNotIn("tested ok", row)
                self.assertIn(said[:23], row)
            self.assertNotIn("Passed", shown)
        self.run_app(case)

    def test_a_check_turns_not_checked_into_passed_or_failed_for_that_app_only(self):
        good = [check.Check("prompt", True, "ok", 1.0), check.Check("tool call", True, "ok", 1.0)]
        bad = [check.Check("prompt", True, "ok", 1.0), check.Check("tool call", False, "no tool call", 1.0)]

        async def case(app, pilot):
            await pilot.press("7")
            for results, want in ((good, "Passed (2/2)"), (bad, "Failed (1/2)")):
                with mock.patch.object(check, "run_checks", return_value=results):
                    await pilot.press("h")
                    for _ in range(30):
                        await pilot.pause(0.1)
                        if want in text_of(app):
                            break
                self.assertIn(want, text_of(app))
                await pilot.press("down")
                await pilot.pause(0.1)
            shown = text_of(app)
            self.assertEqual(shown.count("Not checked"), 6)                # 8 apps, 2 checked
        self.run_app(case)

    def test_narrow_widths_keep_the_this_server_column(self):
        async def case(app, pilot):
            for width in (80, 60, 40):
                await pilot.resize_terminal(width, 24)
                await pilot.pause(0.15)
                await pilot.press("7")
                await pilot.pause(0.1)
                shown = text_of(app)
                self.assertIn("This server", shown, width)
                self.assertIn("Not checked", shown, width)
        self.run_app(case)


def luminance(hex_color: str) -> float:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4          # noqa: E731
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


class LightTheme(unittest.TestCase):
    def test_the_light_palette_is_readable_on_white_and_on_its_selection(self):
        for name in ("fg", "mute", "accent", "warn", "bad"):
            self.assertGreaterEqual(contrast(getattr(LIGHT, name), LIGHT.panel), 4.5, name)
        self.assertGreaterEqual(contrast(LIGHT.accent, LIGHT.sel), 4.5)
        self.assertLess(contrast(DARK.accent, "#ffffff"), 3.0)                  # the old dark colours are what the report measured (1.92:1)

    def run_app(self, coro, size=(120, 40)):
        async def go():
            app = LaunchMonitor(Source(metrics("idle")), None, info=dict(INFO), interval=30, theme="light", color="always")
            async with app.run_test(size=size) as pilot:
                await pilot.pause(0.4)
                return await coro(app, pilot)
        with Home():
            return asyncio.run(go())

    def test_connect_uses_the_light_colours_only(self):
        async def case(app, pilot):
            await pilot.press("7")
            await pilot.pause(0.2)
            t = app.connect_text(118)
            used = {str(s.style) for s in t.spans}
            joined = " ".join(used)
            for dark_only in ("#34d399", "#8e979f", "#f28b82", "#f5c56b", "#1f2a27"):
                self.assertNotIn(dark_only, joined)
            self.assertIn(LIGHT.accent, joined)
            self.assertIn(LIGHT.mute, joined)
        self.run_app(case)

    def test_the_dialog_opened_from_light_mode_is_light(self):
        async def case(app, pilot):
            self.assertTrue(app.has_class("theme-light"))
            await pilot.press("7")
            await pilot.pause(0.1)
            await pilot.press("w")
            await pilot.pause(0.3)
            dialog = app.screen.query_one("#dialog")
            self.assertEqual(str(dialog.styles.background.hex).lower()[:7], "#ffffff")
            await pilot.press("escape")
            await pilot.pause(0.1)
        self.run_app(case)

    def test_theme_cycling_moves_the_class_and_the_dialog_back_to_dark(self):
        async def case(app, pilot):
            self.assertTrue(app.has_class("theme-light"))
            await pilot.press("t")                                           # light -> terminal
            await pilot.pause(0.05)
            self.assertFalse(app.has_class("theme-light"))
            await pilot.press("t")                                           # terminal -> dark
            await pilot.press("7", "w")
            await pilot.pause(0.3)
            self.assertEqual(str(app.screen.query_one("#dialog").styles.background.hex).lower()[:7], "#161a1d")
        self.run_app(case)


if __name__ == "__main__":
    unittest.main()
