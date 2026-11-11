"""Tester report T05, T09, T11 and the smaller wizard items, with Textual's pilot at 120x40, 80x24, 60x20 and 40x15: the Folder step knows an installed
model, the Advanced command follows the choices, the final choices stay on screen, the key hints wrap instead of being cut off."""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.app import Ctx, LaunchApp
from strata_cli.launch import params as P
from tests.terminal.launch_support import HW_12, Home, text_of
from tests.terminal.test_launch_tui import FakeRunner, make_ctx, press, flat, typed

SIZES = ((120, 40), (80, 24), (60, 20), (40, 15))


class Base(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        FakeRunner.instances.clear()
        FakeRunner.behaviour = "ok"
        FakeRunner.download = 0.0
        p = mock.patch("strata_cli.launch.app.Runner", FakeRunner)
        p.start()
        self.addCleanup(p.stop)

    def run_app(self, coro, ctx=None, size=(120, 40)):
        ctx = ctx or make_ctx()
        app = LaunchApp(ctx)

        async def go():
            async with app.run_test(size=size) as pilot:
                await pilot.pause(0.2)
                return await coro(app, pilot, ctx)
        return asyncio.run(go())


class SaveChoices(Base):
    def test_the_three_choices_and_the_name_are_on_screen_at_every_size(self):
        async def case(app, pilot, ctx):
            await press(pilot, *("enter",) * 7, "n", "enter")
            for size in SIZES:
                await pilot.resize_terminal(*size)
                await pilot.pause(0.3)
                shown = text_of(app)
                for choice in ("Save as a preset and launch", "Launch without saving", "Save the preset and stop here"):
                    self.assertIn(choice, shown, (size, shown))
                body = app.screen.query_one("#body")
                menu = app.screen.menu()
                self.assertLessEqual(menu.region.bottom, body.region.bottom, size)               # the choices are inside the body, not below it
                self.assertGreaterEqual(menu.region.y, body.region.y, size)
                name = app.screen.query_one("#name")
                self.assertLessEqual(name.region.bottom, body.region.bottom, size)
                self.assertIn("Enter:", flat(app), size)                                           # the footer says which action Enter does
        self.run_app(case)

    def test_the_footer_names_the_action_enter_will_do(self):
        async def case(app, pilot, ctx):
            await press(pilot, *("enter",) * 7, "n", "enter")
            self.assertIn("Enter: Save as a preset and launch", flat(app))
            await press(pilot, "down")
            self.assertIn("Enter: Launch without saving", flat(app))
            await press(pilot, "down")
            self.assertIn("Enter: Save the preset and stop here", flat(app))
        self.run_app(case)

    def test_the_full_layout_is_kept_when_it_fits(self):
        async def case(app, pilot, ctx):
            await press(pilot, *("enter",) * 7, "n", "enter")
            shown = text_of(app)
            for part in ("Save as preset, then launch", "Context", "Pictures", "Parallel", "Tuning", "Access", "Storage"):
                self.assertIn(part, shown)
        self.run_app(case)


class AdvancedLayout(Base):
    HELP = "  --prefill N          prompt chunk size (default 2048)\n  --kv fp16|int8       KV storage\n"

    def ctx_with_engine(self, **choices):
        ctx = make_ctx()
        ctx.settings.family, ctx.settings.model = "qwen", "IQ3_XXS"
        for k, v in choices.items():
            setattr(ctx.settings, k, v)
        base = {"exe": "component:strata-engine", "vision": {"mmproj": "/x/mmproj.gguf", "gpu": False},
                "args": ["--native", "/very/long/path/to/the/models/folder/Qwen3.8-Flash-Next-Something-IQ3_XXS-00001-of-00002.gguf",
                         "--pack", "/very/long/path/to/the/data/packs/iq3_xxs", "--max-context", "131072", "--kv", "int8", "--prefill", "auto",
                         "--vision"]}
        for p in (mock.patch.object(Ctx, "base_config", lambda self: base), mock.patch.object(Ctx, "engine_exe", lambda self: "/engine/strata"),
                  mock.patch("strata_cli.launch.params.engine_help_text", return_value=self.HELP)):
            p.start()
            self.addCleanup(p.stop)
        return ctx

    def test_the_command_follows_the_wizard_choices(self):
        ctx = self.ctx_with_engine(context=102400, kv="q4_0", vision="off")
        cmd = ctx.command_text()
        self.assertIn("--max-context 102400", cmd)
        self.assertIn("--kv q4_0", cmd)
        self.assertNotIn("--vision", cmd)
        self.assertNotIn("131072", cmd)
        ctx2 = self.ctx_with_engine(context=65536, kv="int8", vision="cpu")
        cmd2 = ctx2.command_text()
        self.assertIn("--max-context 65536", cmd2)
        self.assertIn("--vision", cmd2)
        # and the "Same as" line agrees with it
        self.assertIn("--context 100k", S.flags_text(ctx.settings))
        self.assertIn("--kv q4_0", S.flags_text(ctx.settings))
        self.assertIn("--vision off", S.flags_text(ctx.settings))

    def test_the_preview_is_inside_its_own_scroll_area_at_every_size_and_the_next_key_is_visible(self):
        ctx = self.ctx_with_engine(context=102400, kv="q4_0", vision="off")

        async def case(app, pilot, ctx):
            await press(pilot, *("enter",) * 7)
            self.assertEqual(app.current, "advanced")
            for size in SIZES:
                await pilot.resize_terminal(*size)
                await pilot.pause(0.3)
                body = app.screen.query_one("#body")
                box = app.screen.query_one("#cmdbox")
                self.assertGreaterEqual(box.region.y, body.region.y, size)
                self.assertLessEqual(box.region.bottom, body.region.bottom, size)                # the preview never leaves the body
                self.assertGreaterEqual(box.region.height, 3, size)
                self.assertIn("n next", flat(app), size)                                           # the key hints wrap instead of being cut off
                self.assertIn("Command that will run", text_of(app), size)
        self.run_app(case, ctx)

    def test_tab_scrolls_the_preview_and_esc_comes_back(self):
        ctx = self.ctx_with_engine(context=102400, kv="q4_0", vision="off")

        async def case(app, pilot, ctx):
            await press(pilot, *("enter",) * 7)
            await pilot.resize_terminal(60, 20)
            await pilot.pause(0.3)
            await press(pilot, "tab")
            self.assertTrue(app.screen.query_one("#cmdbox").has_focus)
            await press(pilot, "down", "down")
            await press(pilot, "escape")
            self.assertFalse(app.screen.query_one("#cmdbox").has_focus)
            self.assertEqual(app.current, "advanced")                      # Esc only gave the focus back; it did not leave the step
            await press(pilot, "n")
            self.assertEqual(app.current, "access")
        self.run_app(case, ctx)

    def test_descriptions_use_plain_words(self):
        text = "  --expert-cache N   R4: experts held in VRAM (default auto)\n  --mystery N   R7 internal thing T2\n"
        rows = {p.name: p for p in P.parse_help(text)}
        self.assertNotIn("R4", rows["--expert-cache"].short)
        self.assertIn("graphics memory", rows["--expert-cache"].short)
        self.assertNotRegex(rows["--mystery"].short, r"\bR\d|\bT\d")


class HintsWrap(Base):
    def test_long_key_hints_wrap_instead_of_being_cut(self):
        async def case(app, pilot, ctx):
            await press(pilot, *("enter",) * 7)
            for width in (120, 80, 60, 40):
                await pilot.resize_terminal(width, 24)
                await pilot.pause(0.25)
                hints = app.screen.query_one("#hints")
                self.assertGreaterEqual(hints.region.height, 1)
                self.assertIn("n next", flat(app), width)
        ctx = make_ctx()
        ctx.settings.family, ctx.settings.model = "qwen", "IQ3_XXS"
        self.run_app(case, ctx)


class InstalledModelFolder(Base):
    """The GUI's installation: weights, pack, picture reader there; the draft layer gone."""

    def build(self):
        base = Path(os.environ["LOCALAPPDATA"]).parent / "gui"
        docs = base / "Documents" / "Strata"
        gguf = docs / "Models" / "qwen-IQ3_XXS" / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
        gguf.parent.mkdir(parents=True)
        gguf.write_bytes(b"x")
        pack = docs / "packs" / "iq3_xxs"
        pack.mkdir(parents=True)
        mm = docs / "Models" / "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
        mm.write_bytes(b"x")
        install = Path(os.environ["STRATA_INSTALL_DIR"])
        install.mkdir(parents=True, exist_ok=True)
        cfg = {"exe": "component:strata-engine", "tokenizer": str(pack), "args": ["--pack", str(pack), "--native", str(gguf), "--mtp", str(docs / "mtp" / "rt"),
                                                                                 "--max-context", "32768", "--kv", "int8"],
               "vision": {"mmproj": str(mm), "gpu": False}}
        (install / "strata-iq3_xxs.json").write_text(json.dumps(cfg))
        return docs

    def test_the_folder_starts_where_the_model_is_and_says_what_is_missing(self):
        docs = self.build()
        ctx = make_ctx()
        ctx.settings.family, ctx.settings.model, ctx.settings.vision = "qwen", "IQ3_XXS", "cpu"

        async def case(app, pilot, ctx):
            await press(pilot, "enter", "up", "enter")                 # Start -> Model (up: IQ3_XXS, the 64 GB tier pre-selects IQ3_S) -> Folder
            self.assertEqual(app.current, "folder")
            field = app.screen.query_one("#folderpath")
            self.assertEqual(Path(field.value).resolve(), docs.resolve())
            shown = flat(app)
            self.assertIn("The model is already downloaded", shown)
            self.assertIn("Only the draft layer is missing (5.2 GB)", shown)
            self.assertNotIn("Needs about", shown)
            self.assertNotIn("download the model", shown.lower())
        self.run_app(case, ctx)

    def test_everything_there_says_already_downloaded(self):
        docs = self.build()
        (docs / "mtp" / "rt").mkdir(parents=True)
        (docs / "mtp" / "rt" / "experts.bin").write_bytes(b"x")
        ctx = make_ctx()
        ctx.settings.family, ctx.settings.model, ctx.settings.vision = "qwen", "IQ3_XXS", "cpu"

        async def case(app, pilot, ctx):
            await press(pilot, "enter", "up", "enter")                 # up: IQ3_XXS (IQ3_S is pre-selected)
            shown = flat(app)
            self.assertIn("Already downloaded", shown)
            self.assertIn("Nothing more to download", shown)
            # and the final screen does not warn about a download
            await press(pilot, *("enter",) * 5, "n", "enter")
            self.assertIn("Nothing needs to be downloaded", flat(app))
        self.run_app(case, ctx)


if __name__ == "__main__":
    unittest.main()
