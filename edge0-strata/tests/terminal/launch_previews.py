"""Writes the text previews of every `strata launch` screen to docs/terminal-previews/launch-*.txt (fake computer: RTX 5070 12 GB, 64 GB RAM).

    python tests/terminal/launch_previews.py

They are what the screens draw at 110 x 34 (the monitor views at 120 x 36), as text, so the design can be read without running anything.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rich.console import Console  # noqa: E402

from strata_cli.launch import access, settings as S, store  # noqa: E402
from strata_cli.launch.app import Ctx, LaunchApp  # noqa: E402
from strata_cli.launch.widgets import Glyphs  # noqa: E402
from tests.terminal import test_overview as ov  # noqa: E402
from tests.terminal.launch_support import HW_12, HW_2X16, Home, catalog, text_of  # noqa: E402

OUT = ROOT / "docs" / "terminal-previews"
SIZE = (110, 34)


def ctx_for(hw):
    cat = catalog(hw)
    comp = cat.computer()
    return Ctx(catalog=cat, computer=comp, rows=cat.rows(comp.raw), settings=S.LaunchSettings(), glyphs=Glyphs(True))


def write(name: str, text: str, note: str) -> None:
    body = f"# strata launch preview: {note}\n# (text of the screen as drawn at the size named in tests/terminal/launch_previews.py)\n\n" + text.rstrip() + "\n"
    (OUT / f"launch-{name}.txt").write_text(body, encoding="utf-8")
    print("wrote", f"launch-{name}.txt")


async def keys(pilot, *ks):
    for k in ks:
        await pilot.press(k)
        await pilot.pause(0.15)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with Home():
        store.save(S.LaunchSettings(name="IQ3_XXS 128K - Full speed", family="qwen", model="IQ3_XXS", context=131072, parallel=1,
                                    tuned_preset="x", tuned_at="2026-10-16", last_used="2026-10-16 09:00:00"))
        store.save(S.LaunchSettings(name="Coder 64K - agents", family="coder", model="IQ1_M", context=65536, parallel=2, last_used="2026-10-14 09:00:00"))
        store.save(S.LaunchSettings(name="Network serving", family="qwen", model="Q2_0", context=32768, parallel=1, tuned_preset="y",
                                    tuned_at="2026-10-15", last_used="2026-10-15 09:00:00"))
        base = {"exe": "component:strata-engine", "args": ["--max-context", "131072", "--kv", "int8", "--prefill", "auto", "--expert-cache", "auto",
                                                          "--spec", "4"]}
        help_text = ("  --max-context N      memory for the conversation (default 4096)\n  --kv fp16|int8       how the conversation memory is stored\n"
                     "  --prefill N          how many prompt tokens are read at once\n  --expert-cache N     experts kept in graphics memory\n"
                     "  --spec N             draft tokens per step (faster replies)\n  --native-rope        experimental pinned rotary arithmetic\n")
        patches = [mock.patch.object(Ctx, "base_config", lambda self: base), mock.patch.object(Ctx, "engine_exe", lambda self: "/engine/strata"),
                   mock.patch("strata_cli.launch.params.engine_help_text", return_value=help_text),
                   mock.patch.object(access, "lan_ips", return_value=["192.168.88.21"]),
                   mock.patch.object(access, "find_cloudflared", return_value="/usr/bin/cloudflared"),
                   mock.patch.object(access, "tailscale_status", return_value={"installed": False, "running": False, "ip": None, "name": None})]
        for p in patches:
            p.start()

        async def tour(hw, prefix, steps_after_start):
            ctx = ctx_for(hw)
            app = LaunchApp(ctx)
            async with app.run_test(size=SIZE) as pilot:
                await pilot.pause(0.3)
                write(f"{prefix}start", text_of(app), "start screen with saved presets")
                for name, ks in steps_after_start:
                    await keys(pilot, *ks)
                    write(f"{prefix}{name}", text_of(app), name)

        asyncio.run(tour(HW_12, "", [
            ("step1-model", ["down", "down", "down", "enter"]),
            ("step2-folder", ["enter"]),
            ("step3-context", ["enter", "down", "down"]),
            ("step4-pictures", ["enter"]),
            ("step5-parallel", ["enter"]),
            ("step6-tuning", ["enter"]),
            ("step7-advanced", ["enter"]),
            ("step8-access", ["n", "down"]),
            ("step9-save", ["enter"]),
        ]))
        asyncio.run(tour(HW_2X16, "twocards-", [("step2-cards", ["down", "down", "down", "enter", "enter"])]))

        class Runner:
            def __init__(self, s, **kw):
                self.s, self.emit = s, kw["emit"]

            def run(self, start=True):
                from strata_cli.launch.runner import Event, Result
                self.emit(Event("engine", "Engine 0.1.45 (cuda) is installed.", 100))
                self.emit(Event("model", "Downloading 41.0 of 75.8 GB", 54.0))
                time.sleep(1.2)
                self.emit(Event("model", "Model is ready.", 100))
                self.emit(Event("config", "Settings are saved.", 100))
                self.emit(Event("tune", "Tuning", 62.0, data={"round": (5, 8), "minutes_left": 7,
                                                              "speeds": [{"label": "Default settings", "decode_tok_s": 83.0},
                                                                         {"label": "Keep more context on GPU", "decode_tok_s": 78.5},
                                                                         {"label": "Longer prompt chunks", "decode_tok_s": 85.1}]}))
                time.sleep(3)
                return Result("launch-x", ["http://127.0.0.1:8080"], None, "faster", [], "/x")

        async def run_screen():
            ctx = ctx_for(HW_12)
            ctx.settings = S.LaunchSettings(name="Demo", family="qwen", model="IQ3_XXS", context=131072, tune="quick")
            with mock.patch("strata_cli.launch.app.Runner", Runner):
                app = LaunchApp(ctx)
                async with app.run_test(size=SIZE) as pilot:
                    await pilot.pause(0.2)
                    app.start_preset("Demo") if False else app.run_launch()
                    await pilot.pause(2.0)
                    write("run-progress", text_of(app), "the progress screen while it downloads and tunes")

        asyncio.run(run_screen())

        # the monitor: Overview, Connect, quit question
        from strata_cli.launch.monitor_ext import LaunchMonitor

        async def monitor():
            src = ov.Source(ov.metrics("reading"))
            info = {"url": "http://127.0.0.1:8080", "urls": ["http://127.0.0.1:8080", "http://192.168.88.21:8080"], "key": "sk-strata-abcdefghijklmn",
                    "model": "qwen3.8-flash-next-iq3_xxs", "context": 32768, "profile": "launch-x", "name": "Demo"}
            app = LaunchMonitor(src, None, info=info, interval=30)
            async with app.run_test(size=(120, 36)) as pilot:
                await pilot.pause(0.6)
                write("monitor-overview", text_of(app), "the monitor's new Overview (view 1)")
                await keys(pilot, "7")
                write("monitor-connect-opencode", text_of(app), "the monitor's Connect view (key 7), OpenCode selected")
                await keys(pilot, "down", "down", "down", "down", "down")
                write("monitor-connect-hermes", text_of(app), "the Connect view, Hermes selected (its limits in plain words)")
                await keys(pilot, "7", "q")
                write("monitor-quit", text_of(app), "q asks: keep serving or stop")

        asyncio.run(monitor())
        for p in patches:
            p.stop()
    Path(OUT / "launch-overview-ascii.txt").write_text(
        "# strata monitor Overview in the classic Windows console mode (--glyphs ascii, or detected): no characters Consolas lacks\n\n"
        + ov.draw(ov.overview.build(ov.metrics("generating")), glyphs="ascii", color=False, connect=True), encoding="utf-8")


if __name__ == "__main__":
    run()
