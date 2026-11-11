"""The monitor's Overview (view 1): numbers from /metrics, the drawing, glyph fallbacks, and the monitor showing it."""

from __future__ import annotations

import asyncio
import io
import time
import unittest

from rich.console import Console

from strata_cli import overview
from strata_cli.monitor import MonitorApp

GIB = 2 ** 30


def metrics(state="reading", **over):
    now = time.time()
    hist = lambda base, amp: [base + amp * ((i * 7) % 5) / 4 for i in range(60)]     # noqa: E731
    m = {"time": now,
         "engine": {"model": "IQ3_S", "max_context": 131072, "expert_slots": 1644, "experts": 5120, "expert_cache_mib": 3280},
         "live": {"state": state, "queued": 0, "prompt_read": 35610 if state == "reading" else None, "prompt_total": 36742 if state == "reading" else None,
                  "prompt_tokens": 36742, "generated": 120 if state == "generating" else 0, "max_tokens": 4096,
                  "tok_s": 50.4 if state == "generating" else None, "prefill_tok_s_mean": 820.0},
         "hardware": {"gpu_util": 88, "gpu_mem_used": 11.2 * GIB, "gpu_mem_total": 12 * GIB, "gpu_temp": 54, "gpu_power": 114, "gpu_power_limit": 250,
                      "gpu_pcie_rx_mb": 18237, "gpu_pcie_gen": 4, "gpu_pcie_width": 16, "cpu": 14, "ram_used": 58.5 * GIB, "ram_total": 63 * GIB,
                      "disk_read_mb": 0.1, "disk_write_mb": 0.0},
         "hardware_static": {"gpu_name": "RTX 5070", "gpu_count": 1, "cores": 6, "threads": 12},
         "history": {"tok_s": hist(40, 10), "gpu_util": hist(80, 10), "gpu_mem_used": [11.2 * GIB] * 60, "gpu_temp": [54] * 60,
                     "gpu_power": hist(110, 10), "gpu_pcie_rx_mb": hist(10000, 9000), "cpu": hist(10, 8), "disk_read_mb": hist(0, 3)},
         "requests": [{"id": "a", "time": now - 5, "duration_s": 429.8, "finish_reason": "stop", "prompt_tokens": 17183, "reused": 0,
                       "output_tokens": 19511, "decode_ms": 386000},
                      {"id": "b", "time": now - 9, "duration_s": 4.0, "finish_reason": "stop", "prompt_tokens": 626, "reused": 0,
                       "output_tokens": 11, "decode_ms": 410}]}
    m.update(over)
    return m


def draw(model, width=110, height=30, **kw) -> str:
    c = Console(width=width + 2, record=True, file=io.StringIO(), force_terminal=False, legacy_windows=False)
    c.print(overview.render(model, width=width, height=height, connection="http://127.0.0.1:8080", **kw))
    return c.export_text()


class Build(unittest.TestCase):
    def test_reading_state_and_progress(self):
        m = overview.build(metrics("reading"))
        self.assertEqual(m["state"], "Reading")
        self.assertEqual(m["progress"], (35610.0, 36742.0))
        self.assertEqual(m["context"], (36742.0, 131072.0))
        self.assertEqual(m["speed"][:2], (820.0, "reading the prompt"))

    def test_generating_uses_the_live_rate_and_prompt_plus_generated_for_context(self):
        m = overview.build(metrics("generating"))
        self.assertEqual(m["state"], "Generating")
        self.assertEqual(m["speed"][:2], (50.4, "now"))
        self.assertEqual(m["context"][0], 36742 + 120)
        self.assertEqual(m["progress"], (120.0, 4096.0))

    def test_idle_shows_the_last_request(self):
        m = overview.build(metrics("idle"))
        self.assertEqual(m["state"], "Idle")
        self.assertEqual(m["speed"][1], "last request")
        self.assertAlmostEqual(m["speed"][0], 50.5, places=1)               # 19511 tokens in 386 s
        self.assertEqual(m["context"][0], 17183 + 19511)
        self.assertEqual(m["progress"], (None, None))

    def test_queued_and_error(self):
        self.assertEqual(overview.build(metrics("idle", live={"state": "idle", "queued": 3}))["state"], "Queued")
        self.assertEqual(overview.build(metrics("idle", engine={"model": "x", "error": "boom"}))["state"], "Error")

    def test_hardware_in_gb(self):
        m = overview.build(metrics())
        self.assertAlmostEqual(m["gpu"]["vram_used"], 11.2)
        self.assertEqual(m["gpu"]["vram_total"], 12.0)
        self.assertAlmostEqual(m["ram"][0], 58.5)
        self.assertEqual(m["experts"], {"slots": 1644.0, "total": 5120.0, "gb": 3280 / 1024.0})

    def test_missing_numbers_stay_missing(self):
        m = overview.build({"live": {"state": "idle"}, "engine": {}, "hardware": {}, "requests": []})
        self.assertIsNone(m["gpu"]["util"])
        self.assertIsNone(m["speed"][0])
        text = draw(m)
        self.assertIn("-- tok/s", text)
        self.assertIn("No requests yet", text)
        self.assertNotIn("0 %", text.replace("-- %", ""))                    # no invented zero

    def test_a_server_without_metrics_gives_none(self):
        self.assertIsNone(overview.build(None))
        self.assertIsNone(overview.build({"engine": {}}))


class Draw(unittest.TestCase):
    def test_the_approved_layout(self):
        text = draw(overview.build(metrics("reading")), connect=True)
        for part in ("STRATA  Overview", "IQ3_S 128K", "Reading prompt · 97%", "[LIVE]", "[1 Overview]", "7 Connect",
                     "Model state", "[Reading]", "35,610 / 36,742 tokens · 97%",
                     "Speed", "GPU load", "VRAM", "GPU temp", "Power", "PCIe", "CPU", "Disk read",
                     "88 %", "RTX 5070", "11.2 / 12 GB", "1,644 experts cached", "54 °C", "114 W", "of 250 W limit", "Gen4 x16",
                     "to GPU 18,237 MB/s", "6 cores · 12 threads", "0.1 MB/s", "write 0.0 MB/s",
                     "Context fill", "36K / 128K", "Experts in VRAM", "1,644 · 3.2 GB", "System RAM", "58.5 / 63 GB",
                     "Recent requests", "TIME", "STATUS", "PROMPT", "REUSED", "OUTPUT", "TOK/S", "DURATION", "17,183", "19,511", "429.8 s",
                     "1-7 views", "b open in browser", "q quit"):
            self.assertIn(part, text, part)

    def test_without_connect_the_hint_says_six_views(self):
        text = draw(overview.build(metrics()), connect=False)
        self.assertIn("1-6 views", text)
        self.assertNotIn("7 Connect", text)

    def test_ascii_mode_has_no_glyphs_the_classic_windows_console_font_lacks(self):
        text = draw(overview.build(metrics()), glyphs="ascii", color=False)
        for bad in "▁▂▃▄▅▆▇█░╭╮╰╯✓●":
            self.assertNotIn(bad, text, bad)
        self.assertIn("#", text)
        self.assertIn("54 C", text)

    def test_fits_the_width_it_is_given(self):
        for w in (80, 100, 120):
            text = draw(overview.build(metrics()), width=w - 2)
            self.assertTrue(all(len(line) <= w for line in text.splitlines()), w)

    def test_spark(self):
        self.assertEqual(len(overview.spark([1, 2, 3], 8, "unicode")), 8)
        self.assertEqual(overview.spark([None, None], 4, "unicode"), "    ")
        self.assertEqual(overview.spark([0, 100], 2, "unicode", lo=0, hi=100), "▁█")
        self.assertEqual(overview.spark([0, 100], 2, "ascii", lo=0, hi=100), ".@")


class Source:
    root = "http://127.0.0.1:8095"

    def __init__(self, raw):
        self.raw = raw
        self.base_url = self.root

    def get(self, path):
        if path == "/metrics":
            return self.raw
        raise AssertionError(path)

    def snapshot(self):
        return {"source": "fixture", "sampled_at": time.time(), "schema_version": 1, "instance": {"id": "x"},
                "model": {"id": "IQ3_S", "loaded": True}, "work": {}, "rates": {}, "devices": [], "recent": [], "totals": {},
                "topology": {}, "quality": {}}


class InTheMonitor(unittest.TestCase):
    def test_view_one_is_the_new_overview_and_the_others_stay(self):
        async def go():
            app = MonitorApp(Source(metrics("generating")), None, interval=30)
            async with app.run_test(size=(120, 36)) as pilot:
                await pilot.pause(0.5)
                shown = "\n".join(s.text for s in app.screen._compositor.render_strips())
                self.assertIn("Model state", shown)
                self.assertIn("Generating", shown)
                await pilot.press("2")
                await pilot.pause(0.2)
                shown = "\n".join(s.text for s in app.screen._compositor.render_strips())
                self.assertNotIn("Model state", shown)
                self.assertIn("Serving", shown)
        asyncio.run(go())

    def test_an_older_server_without_metrics_keeps_the_earlier_overview(self):
        class Old(Source):
            def get(self, path):
                from strata_cli.client import ClientError
                raise ClientError("not found", http_status=404)

        async def go():
            app = MonitorApp(Old(None), None, interval=30)
            async with app.run_test(size=(120, 36)) as pilot:
                await pilot.pause(0.5)
                shown = "\n".join(s.text for s in app.screen._compositor.render_strips())
                self.assertNotIn("Model state", shown)
                self.assertIn("MONITOR", shown)
        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
