"""The monitor in the classic Windows console (Cascadia Mono switch, the solid look, the tip) and with several cards / CPUs."""

from __future__ import annotations

import io
import os
import time
import unittest
from unittest import mock

from rich.console import Console

from strata_cli import consoleinfo, monitor, overview
from tests.terminal.test_overview import GIB, metrics

TIP = "For the full look, use Windows Terminal (built into Windows 11)"


def draw(model, width=110, height=40, **kw) -> str:
    c = Console(width=width + 2, record=True, file=io.StringIO(), force_terminal=False, legacy_windows=False)
    c.print(overview.render(model, width=width, height=height, connection="http://127.0.0.1:8080", **kw))
    return c.export_text()


def three_cards(sockets=2):
    m = metrics("generating")
    cards = [{"index": i, "name": "NVIDIA GeForce RTX 3090", "util": u, "mem_used": used * GIB, "mem_total": 24 * GIB, "temp": t, "power": p, "power_limit": 350}
             for i, (u, used, t, p) in enumerate(((88, 22.1, 61, 236), (71, 21.4, 58, 201), (64, 20.8, 55, 188)))]
    m["hardware"]["gpus"] = cards
    m["hardware_static"].update({"gpu_name": "NVIDIA GeForce RTX 3090", "gpu_count": 3, "cores": 64, "threads": 128, "sockets": sockets, "numa_nodes": 2})
    return m


class WhichLook(unittest.TestCase):
    def setUp(self):
        consoleinfo.forget_style()
        self.addCleanup(consoleinfo.forget_style)

    def test_decide_style_by_the_facts(self):
        d = consoleinfo.decide_style(legacy=False, cascadia_ok=False, font=None)
        self.assertEqual((d["mode"], d["tip"]), ("none", None))                       # Windows Terminal, Linux, macOS: nothing changes
        d = consoleinfo.decide_style(legacy=True, cascadia_ok=True, font="Cascadia Mono")
        self.assertEqual((d["mode"], d["tip"]), ("full", TIP))                        # the font was switched: the full Unicode look, the tip anyway
        d = consoleinfo.decide_style(legacy=True, cascadia_ok=False, font="Consolas")
        self.assertEqual((d["mode"], d["bullet"], d["tip"]), ("solid", "●", TIP))
        d = consoleinfo.decide_style(legacy=True, cascadia_ok=False, font="Lucida Console")
        self.assertEqual((d["mode"], d["bullet"]), ("solid", "•"))                    # Lucida has no ●
        self.assertEqual(consoleinfo.decide_style(legacy=True, cascadia_ok=False, font=None)["bullet"], "•")

    def test_classic_style_switches_the_font_once_when_cascadia_is_installed(self):
        with mock.patch.object(consoleinfo, "legacy_console", return_value=True), mock.patch.object(consoleinfo.os, "name", "nt"), \
                mock.patch.dict(os.environ, {"STRATA_GLYPHS": ""}), mock.patch.object(consoleinfo, "cascadia_installed", return_value=True), \
                mock.patch.object(consoleinfo, "console_font", side_effect=["Consolas", "Cascadia Mono"]), \
                mock.patch.object(consoleinfo, "switch_font", return_value=True) as sw:
            s = consoleinfo.classic_style()
            self.assertEqual(s["mode"], "full")
            sw.assert_called_once_with("Cascadia Mono")
            consoleinfo.classic_style()
            sw.assert_called_once()                                                   # cached: the font API is not called again

    def test_without_cascadia_or_when_the_switch_fails_the_solid_look_with_the_font_it_has(self):
        for installed, switched in ((False, False), (True, False)):
            consoleinfo.forget_style()
            with mock.patch.object(consoleinfo, "legacy_console", return_value=True), mock.patch.object(consoleinfo.os, "name", "nt"), \
                    mock.patch.dict(os.environ, {"STRATA_GLYPHS": ""}), mock.patch.object(consoleinfo, "cascadia_installed", return_value=installed), \
                    mock.patch.object(consoleinfo, "console_font", return_value="Lucida Console"), \
                    mock.patch.object(consoleinfo, "switch_font", return_value=switched):
                s = consoleinfo.classic_style()
                self.assertEqual((s["mode"], s["bullet"]), ("solid", "•"))

    def test_never_in_windows_terminal_linux_or_mac(self):
        with mock.patch.object(consoleinfo, "legacy_console", return_value=False), mock.patch.object(consoleinfo, "switch_font") as sw:
            self.assertEqual(consoleinfo.classic_style()["mode"], "none")
            sw.assert_not_called()
        env = {k: v for k, v in os.environ.items() if k not in ("WT_SESSION", "TERM_PROGRAM", "STRATA_GLYPHS")}
        env["WT_SESSION"] = "x"
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(consoleinfo.os, "name", "nt"), mock.patch.object(consoleinfo.sys, "platform", "win32"):
            self.assertFalse(consoleinfo.legacy_console())                            # Windows Terminal
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(consoleinfo.os, "name", "posix"), mock.patch.object(consoleinfo.sys, "platform", "linux"):
            self.assertFalse(consoleinfo.legacy_console())
        with mock.patch.object(consoleinfo.os, "name", "posix"):
            self.assertIsNone(consoleinfo.console_font())
            self.assertFalse(consoleinfo.switch_font())
            self.assertFalse(consoleinfo.cascadia_installed())

    def test_choose_glyphs(self):
        with mock.patch.object(consoleinfo, "legacy_console", return_value=True), mock.patch.object(consoleinfo, "classic_style", return_value={"mode": "full"}):
            self.assertEqual(monitor.choose_glyphs("auto", "utf-8", "xterm"), "unicode")
        with mock.patch.object(consoleinfo, "legacy_console", return_value=True), mock.patch.object(consoleinfo, "classic_style", return_value={"mode": "solid"}):
            self.assertEqual(monitor.choose_glyphs("auto", "utf-8", None), "solid")
            with mock.patch.dict(os.environ, {"STRATA_GLYPHS": "ascii"}):
                self.assertEqual(monitor.choose_glyphs("auto", "utf-8", None), "ascii")      # forced ASCII stays ASCII
            self.assertEqual(monitor.choose_glyphs("unicode", "utf-8", None), "unicode")      # an explicit --glyphs wins
        with mock.patch.object(consoleinfo, "legacy_console", return_value=False):
            self.assertEqual(monitor.choose_glyphs("auto", "utf-8", "xterm"), "unicode")


class SolidLook(unittest.TestCase):
    def test_solid_blocks_only_no_dithered_shades_and_no_eighth_blocks(self):
        text = draw(overview.build(metrics("generating")), glyphs="solid")
        for ch in "░▒▓▁▂▃▅▆▇╭╮╰╯":
            self.assertNotIn(ch, text, ch)
        for ch in "█▄┌┐└┘":
            self.assertIn(ch, text, ch)

    def test_bars_use_a_thin_track_and_boxes_are_square(self):
        bar = overview.bar(0.25, 20, "solid", "bright_green")
        self.assertEqual(bar.plain, "█" * 5 + "▄" * 15)
        self.assertEqual(overview.bar(None, 4, "solid", "x").plain, "▄▄▄▄")
        self.assertEqual(overview.spark([0, 50, 100], 3, "solid", lo=0, hi=100).plain, "▄▄█")

    def test_the_16_palette_names_not_hex_colours(self):
        group = overview.render(overview.build(metrics("generating")), width=110, height=40, connection="x", glyphs="solid")
        c = Console(width=112, record=True, file=io.StringIO(), force_terminal=True, color_system="standard", legacy_windows=False)
        c.print(group)
        out = c.export_text(styles=True)
        self.assertNotIn("38;2;", out)                                          # no 24-bit colours: conhost maps them to cyan or black

    def test_the_bullet_follows_the_font(self):
        m = overview.build(metrics("generating"))
        self.assertIn("● Generating", draw(m, glyphs="solid", bullet="●"))
        self.assertIn("• Generating", draw(m, glyphs="solid", bullet="•"))

    def test_the_other_looks_are_unchanged(self):
        m = overview.build(metrics("generating"))
        self.assertIn("╭", draw(m, glyphs="unicode"))
        self.assertNotIn("╭", draw(m, glyphs="ascii"))


class SeveralCards(unittest.TestCase):
    def test_one_card_looks_exactly_as_before(self):
        base = metrics("generating")
        one = metrics("generating")
        one["hardware"]["gpus"] = [{"index": 0, "name": "NVIDIA GeForce RTX 5070", "util": 88, "mem_used": 11.2 * GIB, "mem_total": 12 * GIB, "temp": 54, "power": 114}]
        for glyphs in ("unicode", "ascii", "solid"):
            self.assertEqual(draw(overview.build(base), glyphs=glyphs, now=1.0), draw(overview.build(one), glyphs=glyphs, now=1.0), glyphs)
        self.assertNotIn("Cards", draw(overview.build(base)))

    def test_the_tiles_say_it_for_the_whole_computer(self):
        m = overview.build(three_cards())
        g = m["gpu"]
        self.assertEqual((g["util"], g["temp"], g["count"]), (88.0, 61.0, 3.0))                  # the busiest, the hottest
        self.assertAlmostEqual(g["vram_used"], 64.3, places=1)
        self.assertEqual(g["vram_total"], 72.0)                                                    # all cards
        self.assertEqual((g["power"], g["power_limit"]), (625.0, 1050.0))
        text = draw(m)
        for want in ("busiest of 3 cards", "all 3 cards", "hottest card", "64.3 / 72 GB", "625 W", "of 1,050 W limit", "61 °C"):
            self.assertIn(want, text, want)
        self.assertNotIn("+2", text)                                                              # no more "RTX 3090 +2"

    def test_the_cards_strip_one_line_per_card(self):
        text = draw(overview.build(three_cards()))
        lines = [ln for ln in text.splitlines() if ln.startswith("GPU ")]
        self.assertEqual(len(lines), 3)
        self.assertIn("Cards", text)
        for want, line in zip(("88 %", "71 %", "64 %"), lines):
            self.assertIn(want, line)
        self.assertIn("22.1 / 24 GB", lines[0])
        self.assertIn("61 °C", lines[0])
        self.assertIn("236 W", lines[0])
        self.assertIn("RTX 3090", lines[0])

    def test_up_to_four_cards_then_a_pointer_to_view_5(self):
        m = three_cards()
        m["hardware"]["gpus"] += [dict(m["hardware"]["gpus"][0], index=i) for i in (3, 4, 5)]
        text = draw(overview.build(m), height=44)
        self.assertEqual(len([ln for ln in text.splitlines() if ln.startswith("GPU ")]), 4)
        self.assertIn("+ 2 more cards: press 5", text)

    def test_the_cpu_tile_says_how_many_cpus(self):
        self.assertIn("2 CPUs · 64 cores", draw(overview.build(three_cards(sockets=2))))
        one = draw(overview.build(three_cards(sockets=1)))
        self.assertIn("64 cores · 128 threads", one)
        self.assertNotIn("CPUs", one)
        self.assertIn("6 cores · 12 threads", draw(overview.build(metrics("generating"))))     # no socket count reported: as before

    def test_a_missing_reading_stays_missing(self):
        m = three_cards()
        for c in m["hardware"]["gpus"]:
            c["temp"] = None
            c["power"] = None
        g = overview.build(m)["gpu"]
        self.assertIsNone(g["temp"])
        self.assertIsNone(g["power"])
        self.assertIn("-- °C", draw(overview.build(m)))

    def test_it_fits_the_width_and_the_classic_looks_too(self):
        for glyphs in ("unicode", "ascii", "solid"):
            for ln in draw(overview.build(three_cards()), glyphs=glyphs, width=100).splitlines():
                self.assertLessEqual(len(ln.rstrip()), 102, (glyphs, ln))


class FooterAndViews(unittest.TestCase):
    def screen_text(self, classic):
        import asyncio
        from tests.terminal.test_overview import Source

        async def go():
            consoleinfo.forget_style()
            style = consoleinfo.decide_style(legacy=bool(classic), cascadia_ok=classic == "full", font="Consolas")
            with mock.patch.object(consoleinfo, "legacy_console", return_value=bool(classic)), mock.patch.object(consoleinfo, "classic_style", return_value=style):
                app = monitor.MonitorApp(Source(three_cards()), None, interval=30)
                async with app.run_test(size=(120, 40)) as pilot:
                    await pilot.pause(0.5)
                    return "\n".join(s.text for s in app.screen._compositor.render_strips())
        try:
            return asyncio.run(go())
        finally:
            consoleinfo.forget_style()

    def test_the_tip_is_in_the_classic_console_only(self):
        self.assertIn(TIP, self.screen_text("solid"))
        self.assertIn(TIP, self.screen_text("full"))                  # also when the font was switched: one line, always
        self.assertNotIn(TIP, self.screen_text(None))                  # Windows Terminal, Linux, macOS

    def test_the_monitor_draws_the_several_cards_in_every_look(self):
        for classic in ("solid", "full", None):
            shown = self.screen_text(classic)
            self.assertIn("busiest of 3 cards", shown, classic)
            self.assertIn("Cards", shown, classic)
            self.assertIn("2 CPUs", shown, classic)

    def test_view_5_shows_the_whole_card_name(self):
        from strata_cli.render import _device
        row = {"index": 1, "name": "NVIDIA GeForce RTX 3090", "mem_used": 21.4 * GIB, "mem_total": 24 * GIB, "util": 71, "temp": 58, "power": 201}
        line = _device(row, 100)
        self.assertIn("RTX 3090", line)
        self.assertNotIn("NVIDIA", line)


if __name__ == "__main__":
    unittest.main()
