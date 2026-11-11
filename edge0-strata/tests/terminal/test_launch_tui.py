"""The `strata launch` screens, driven with Textual's pilot: every screen, its keys and what it writes into the settings."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import access, settings as S, store
from strata_cli.launch.app import Ctx, LaunchApp, run_monitor
from strata_cli.launch.runner import Cancelled, Event, LaunchError, Result
from strata_cli.launch.widgets import Glyphs
from tests.terminal.launch_support import HW_12, HW_24, HW_2X16, Home, catalog, text_of

SIZE = (110, 34)


def flat(app) -> str:
    return " ".join(text_of(app).split())


def make_ctx(hw=None, settings=None):
    cat = catalog(hw or HW_12)
    comp = cat.computer()
    return Ctx(catalog=cat, computer=comp, rows=cat.rows(comp.raw), settings=settings or S.LaunchSettings(), glyphs=Glyphs(True))


class FakeRunner:
    """Stands in for the real Runner: emits the same events, installs nothing, starts nothing."""

    instances: list = []
    behaviour = "ok"
    hold = 0
    download = 0.0

    def __init__(self, s, *, catalog=None, emit=None, cancel=None, feed=None, **kw):
        self.s, self.emit_cb, self.cancel, self.feed = s, emit, cancel, feed
        FakeRunner.instances.append(self)

    def download_gb(self):
        return FakeRunner.download

    def run(self, start=True):
        e = self.emit_cb
        e(Event("engine", "Engine 0.1.45 (cuda) is installed.", 100))
        e(Event("model", "Downloading 12.0 of 75.8 GB", 15.8))
        if FakeRunner.behaviour == "wait":
            while not self.cancel.is_set():
                time.sleep(0.02)
            raise Cancelled()
        if FakeRunner.behaviour == "fail":
            raise LaunchError("The model could not be installed. Not enough disk space.", "Free some space and run it again.", 6)
        if getattr(FakeRunner, "hold", 0):
            time.sleep(FakeRunner.hold)
        e(Event("model", "Model is ready.", 100))
        e(Event("config", "Settings are saved.", 100))
        if self.s.tune != "skip":
            e(Event("tune", "Tuning", 40.0, data={"round": (2, 5), "minutes_left": 12,
                                                  "speeds": [{"label": "Default settings", "decode_tok_s": 83.0},
                                                             {"label": "Longer prompt chunks", "decode_tok_s": 85.1}]}))
            if getattr(FakeRunner, "hold", 0):
                time.sleep(FakeRunner.hold)
            e(Event("tune", "Done: answers are about 3% faster with the tuned settings.", 100))
        e(Event("start", "The model is loaded and answering.", 100))
        urls = [f"http://127.0.0.1:{self.s.port}"]
        return Result("launch-x", urls, self.s.api_key, "faster", [], "/nonexistent.json")


async def press(pilot, *keys, pause=0.12):
    for k in keys:
        await pilot.press(k)
        await pilot.pause(pause)


async def typed(pilot, text):
    for ch in text:
        await pilot.press(ch)
    await pilot.pause(0.1)


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
        p = mock.patch.object(access, "lan_ips", return_value=["192.168.1.5"])
        p.start()
        self.addCleanup(p.stop)

    def run_app(self, coro, ctx=None, auto_run=False):
        ctx = ctx or make_ctx()
        app = LaunchApp(ctx, auto_run=auto_run)

        async def go():
            async with app.run_test(size=SIZE) as pilot:
                await pilot.pause(0.2)
                return await coro(app, pilot, ctx)

        return asyncio.run(go())


class StartScreenTests(Base):
    def test_no_presets_offers_the_two_ways_in(self):
        async def case(app, pilot, ctx):
            t = text_of(app)
            self.assertIn("No saved presets yet", t)
            self.assertIn("Set up from the beginning", t)
            self.assertIn("Use model files I already have", t)
            self.assertIn("RTX 5070 12 GB", t)                       # the computer, top right
            self.assertIn("q quit", t)
        self.run_app(case)

    def test_presets_are_listed_with_their_columns(self):
        store.save(S.LaunchSettings(name="IQ3_XXS 128K", family="qwen", model="IQ3_XXS", context=131072, parallel=1, tuned_preset="x", tuned_at="2026-10-16"))
        store.save(S.LaunchSettings(name="Coder 64K", family="coder", model="IQ1_M", context=65536, parallel=2))

        async def case(app, pilot, ctx):
            t = text_of(app)
            self.assertIn("Start from a saved preset", t)
            self.assertIn("IQ3_XXS 128K", t)
            self.assertIn("128K", t)
            self.assertIn("1 request", t)
            self.assertIn("2 requests", t)
            self.assertIn("tuned 10-16", t)
            self.assertIn("not tuned", t)
            self.assertIn("e edit preset", t)
        self.run_app(case)

    def test_enter_on_a_preset_launches_it_and_opens_the_run_screen(self):
        store.save(S.LaunchSettings(name="Office", family="qwen", model="IQ3_XXS", context=65536, tune="quick"))

        async def case(app, pilot, ctx):
            await press(pilot, "enter")
            await pilot.pause(0.6)
            self.assertEqual(FakeRunner.instances[0].s.name, "Office")
            self.assertEqual(app.result.profile, "launch-x")
        self.run_app(case)

    def test_a_model_that_is_not_here_asks_before_downloading(self):
        store.save(S.LaunchSettings(name="Big", family="qwen", model="IQ3_XXS", context=65536))
        FakeRunner.download = 75.8

        async def case(app, pilot, ctx):
            await press(pilot, "enter")
            t = flat(app)
            self.assertIn("Download about 76 GB?", t)
            self.assertIn("not on this computer yet", t)
            self.assertIn("Download and start", t)
            self.assertEqual(FakeRunner.instances[-1].behaviour, "ok")
            await press(pilot, "down", "enter")                      # Not now
            await pilot.pause(0.3)
            self.assertIsNone(app.result)
            await press(pilot, "enter")
            await press(pilot, "enter")                              # Download and start
            await pilot.pause(0.8)
            self.assertIsNotNone(app.result)
        self.run_app(case)

    def test_delete_asks_then_removes_only_the_preset(self):
        store.save(S.LaunchSettings(name="Old", family="qwen", model="Q2_0", context=32768))

        async def case(app, pilot, ctx):
            await press(pilot, "d")
            t = text_of(app)
            self.assertIn("Delete the preset 'Old'?", t)
            self.assertIn("Only the saved choices go", t)
            await press(pilot, "enter")                              # first option: Delete it
            self.assertEqual(store.list_presets(), [])
            self.assertIn("No saved presets yet", text_of(app))
        self.run_app(case)

    def test_delete_can_be_cancelled(self):
        store.save(S.LaunchSettings(name="Keep", family="qwen", model="Q2_0", context=32768))

        async def case(app, pilot, ctx):
            await press(pilot, "d", "escape")
            self.assertEqual([p.name for p in store.list_presets()], ["Keep"])
        self.run_app(case)

    def test_edit_opens_setup_with_the_preset_loaded(self):
        store.save(S.LaunchSettings(name="E", family="swift", model="IQ2_XS", context=65536, parallel=2))

        async def case(app, pilot, ctx):
            await press(pilot, "e")
            t = text_of(app)
            self.assertIn("Which model?", t)
            self.assertEqual((ctx.settings.family, ctx.settings.model), ("swift", "IQ2_XS"))
        self.run_app(case)

    def test_q_leaves(self):
        async def case(app, pilot, ctx):
            await press(pilot, "q")
            await pilot.pause(0.2)
            self.assertFalse(app.is_running)
        self.run_app(case)

    def test_help_is_in_words(self):
        async def case(app, pilot, ctx):
            await press(pilot, "question_mark")
            self.assertIn("A preset remembers", text_of(app))
        self.run_app(case)


class ModelStep(Base):
    def go(self):
        async def start(app, pilot, ctx):
            await press(pilot, "enter")
        return start

    def test_the_list_shows_real_figures_and_the_recommendation(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter")
            t = text_of(app)
            self.assertIn("Which model?", t)
            for col in ("Model", "Download", "Needs (RAM + VRAM)", "On this computer"):
                self.assertIn(col, t)
            self.assertIn("IQ3_XXS", t)
            line = next(l for l in t.splitlines() if "Qwen3.8-Flash-Next IQ3_S" in l)     # 64 GB + 12 GB: the 64 GB tier
            self.assertIn("84 GB", line)                              # setup.py's download_gb 83.6
            self.assertIn("51 GB", line)                              # the app's figure: experts 50.3 + 1 GB, RAM + VRAM together
            self.assertIn("(recommended)", line)
            self.assertIn("› ", line)                                 # the cursor starts on the recommendation
            self.assertIn("too big (can still try)", t)          # NVFP4: a warning, not a block
            big = next(l for l in t.splitlines() if "UD-Q5_K_XL" in l)
            self.assertIn("SSD ", big)                            # a model bigger than RAM says how much is read from the SSD, as the app does
            self.assertIn("The n-gram table stays on your SSD", t)
        self.run_app(case)

    def test_the_verdict_column_is_not_cut_off_at_80_columns(self):                       # 0.2.5 list
        ctx = make_ctx()
        app = LaunchApp(ctx)

        async def go():
            async with app.run_test(size=(80, 34)) as pilot:
                await pilot.pause(0.2)
                await press(pilot, "enter")
                t = text_of(app)
                self.assertIn("Fits?", t)
                rec = next(l for l in t.splitlines() if "Qwen3.8-Flash-Next IQ3_S " in l)
                self.assertIn("fits (recommended)", rec)
                for model in ("Q2_0", "NVFP4 ", "UD-Q5_K_XL"):
                    row = next(l for l in t.splitlines() if model in l and "GB" in l)
                    self.assertTrue(any(w in row for w in ("fits", "tight", "too big")), row)
                self.assertTrue(all(len(l) <= 80 for l in t.splitlines()))
        asyncio.run(go())

    def test_choosing_a_model_and_going_on(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "up", "up", "up", "enter")
            self.assertEqual((ctx.settings.family, ctx.settings.model), ("qwen", "Q2_0"))
            self.assertIn("Where should the model files be stored?", text_of(app))
        self.run_app(case)

    def test_a_model_that_does_not_fit_can_still_be_chosen(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "end", "enter")
            last = ctx.rows[-1]
            self.assertEqual(last.verdict, "no")                      # the last row does not fit a 12 GB card
            self.assertEqual((ctx.settings.family, ctx.settings.model), (last.family, last.model))
            self.assertIn("Where should the model files be stored?", text_of(app))
            self.assertTrue(any("does not fit" in n for n in ctx.notes))
        self.run_app(case)

    def test_own_files_are_checked_with_the_fingerprint_table(self):
        plan = {"candidate_selection": {"family": "qwen", "model": "IQ3_XXS"}, "issues": [], "file_count": 2, "total_bytes": 44_000_000_000}

        async def case(app, pilot, ctx):
            await press(pilot, "enter", "o")
            self.assertIn("Type or paste the folder", text_of(app))
            with mock.patch("strata_control.models.plan_import", return_value=plan) as pi, mock.patch("strata_control.models.own_path_problem", return_value=None):
                await typed(pilot, "/data/models/IQ3_XXS")
                await press(pilot, "enter")
            self.assertEqual(ctx.settings.model_path, os.path.abspath("/data/models/IQ3_XXS"))
            self.assertEqual((ctx.settings.family, ctx.settings.model), ("qwen", "IQ3_XXS"))
            self.assertIn("These files are qwen IQ3_XXS (2 file(s), 44 GB)", text_of(app))
            await press(pilot, "enter")
            self.assertIn("Your model files stay where they are", text_of(app))
        self.run_app(case)

    def test_own_files_that_are_not_a_known_model_say_why(self):
        plan = {"candidate_selection": None, "issues": ["incomplete_or_invalid_shard_set"], "file_count": 1, "total_bytes": 1}

        async def case(app, pilot, ctx):
            await press(pilot, "enter", "o")
            with mock.patch("strata_control.models.plan_import", return_value=plan), mock.patch("strata_control.models.own_path_problem", return_value=None):
                await typed(pilot, "/x")
                await press(pilot, "enter")
            t = flat(app)
            self.assertIn("some parts of the model are missing", t)
            self.assertIn("pick a model from the list", t)
            self.assertIsNone(ctx.settings.model_path)
        self.run_app(case)

    def test_own_files_folder_that_is_not_there_is_named(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "o")
            await typed(pilot, "/no/such/folder")
            await press(pilot, "enter")
            t = flat(app)
            self.assertIn("does not exist", t)
            self.assertIn("Nothing was downloaded", t)
            self.assertIsNone(ctx.settings.model_path)
        self.run_app(case)

    def test_own_files_path_from_the_start_screen(self):
        async def case(app, pilot, ctx):
            await press(pilot, "down", "enter")
            self.assertIn("Type or paste the folder", text_of(app))
            await press(pilot, "escape")
            self.assertIn("Which model?", text_of(app))
        self.run_app(case)

    def test_escape_goes_back_to_the_start(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "escape")
            self.assertIn("Set up from the beginning", text_of(app))
        self.run_app(case)


class FolderStep(Base):
    def test_default_folder_and_free_space_and_change(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "enter")
            t = text_of(app)
            self.assertIn("Where should the model files be stored?", t)
            self.assertIn("Not downloaded yet: about 90 GB", t)
            self.assertIn("Free space in that folder's drive", t)
            f = app.screen.query_one("#folderpath")
            f.value = ""
            await typed(pilot, str(self.home.base / "Strata-data"))
            self.assertEqual(f.value, str(self.home.base / "Strata-data"))
            await press(pilot, "enter")
            self.assertEqual(ctx.settings.data_dir, str(self.home.base / "Strata-data"))
            self.assertIn("How much text should it remember at once?", text_of(app))
        self.run_app(case)

    def test_a_new_folder_says_that_the_models_already_stored_will_move(self):
        async def case(app, pilot, ctx):
            old = self.home.base / "old-data"
            (old / "models" / "Q2_0").mkdir(parents=True)
            (old / "models" / "Q2_0" / "x.gguf").write_text("x")
            with mock.patch.object(type(ctx.catalog), "data_dir", return_value=old):
                await press(pilot, "enter", "enter")
                self.assertNotIn("stay there and keep working", text_of(app))
                f = app.screen.query_one("#folderpath")
                f.value = ""
                await typed(pilot, str(self.home.base / "new-data"))
                t = " ".join(text_of(app).replace("\n", " ").split())
            self.assertIn("Model files Strata already keeps in", t)
            self.assertIn("stay there and keep working", t)
            self.assertNotIn("are moved into this folder", t)
        self.run_app(case)

    def test_keeping_the_default_stores_nothing(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "enter", "enter")
            self.assertIsNone(ctx.settings.data_dir)
        self.run_app(case)

    def test_not_enough_space_is_a_warning_you_can_go_past(self):
        async def case(app, pilot, ctx):
            with mock.patch.object(type(ctx.catalog), "disk_free", return_value=10.0):
                await press(pilot, "enter", "enter")
                t = text_of(app)
            self.assertIn("Not enough for", t)
            await press(pilot, "enter")
            self.assertIn("How much text", text_of(app))
        self.run_app(case)


class ContextStep(Base):
    def reach(self):
        async def f(app, pilot, ctx):
            await press(pilot, "enter", "enter", "enter")
        return f

    def test_sizes_words_memory_and_recommendation(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "enter", "enter")
            t = text_of(app)
            for size in ("8K", "32K", "64K", "128K", "256K", "512K", "Custom"):
                self.assertIn(size, t)
            self.assertIn("about 98,000 words", t)
            self.assertIn("recommended", t)
            self.assertIn("+1.8 GB", t)                               # 128K at 8-bit: 131072 x 13 x 1056 bytes
            self.assertIn("experimental: rope scaling", t)
            self.assertIn("Conversation memory: 8-bit", t)
        self.run_app(case)

    def test_choose_128k(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "enter", "enter", "down", "down", "enter")
            self.assertEqual(ctx.settings.context, 131072)
            self.assertIn("Should the model read pictures?", text_of(app))
        self.run_app(case)

    def test_memory_format_toggle_changes_the_figures(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "enter", "enter", "m")
            self.assertEqual(ctx.settings.kv, "q4_0")
            t = text_of(app)
            self.assertIn("4-bit", t)
            self.assertIn("+0.2 GB", t)                               # 32K at 4-bit
            await press(pilot, "m")
            self.assertEqual(ctx.settings.kv, "int8")
        self.run_app(case)

    def test_custom_size(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "enter", "enter", "c")
            await typed(pilot, "100k")
            await press(pilot, "enter")
            self.assertEqual(ctx.settings.context, 102400)
            self.assertIn("100K", text_of(app))
            await press(pilot, "enter")
            self.assertEqual(ctx.settings.context, 102400)
        self.run_app(case)

    def test_bad_custom_size_is_explained(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "enter", "enter", "c")
            await typed(pilot, "lots")
            await press(pilot, "enter")
            self.assertIn("is not a context size", text_of(app))
        self.run_app(case)

    def test_bigger_card_recommends_more(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "enter", "enter")
            t = text_of(app)
            line = next(l for l in t.splitlines() if "128K" in l)
            self.assertIn("recommended", line)
        self.run_app(case, ctx=make_ctx(HW_24))


class PicturesParallelTuning(Base):
    def reach_pictures(self):
        return ("enter", "enter", "enter", "enter")

    def test_pictures_default_off_and_on(self):
        async def case(app, pilot, ctx):
            await press(pilot, *self.reach_pictures())
            t = text_of(app)
            self.assertIn("No pictures", t)
            self.assertIn("Pictures are on by default", flat(app))
            self.assertIn("0.9 GB download", t)
            self.assertIn("runs on the CPU", flat(app))               # a 12 GB card: the CPU reader (the GPU one from 24 GB)
            self.assertEqual(app.screen.menu().index, 1)              # "Yes" is preselected
            await press(pilot, "enter")
            self.assertEqual(ctx.settings.vision, "cpu")
            self.assertIn("How many requests at the same time?", text_of(app))
        self.run_app(case)

    def test_pictures_no_is_one_key_away(self):
        async def case(app, pilot, ctx):
            await press(pilot, *self.reach_pictures())
            await press(pilot, "up", "enter")
            self.assertEqual(ctx.settings.vision, "off")
        self.run_app(case)

    def test_pictures_not_offered_for_a_model_without_them(self):
        store.save(S.LaunchSettings(name="G", family="gyro", model="Gyro-S", context=32768))

        async def case(app, pilot, ctx):
            await press(pilot, "e", "enter", "enter", "enter")
            self.assertEqual((ctx.settings.family, ctx.settings.model), ("gyro", "Gyro-S"))
            t = flat(app)
            self.assertIn("not available for this model yet", t)
            self.assertIn("This model cannot read pictures yet", t)
            await press(pilot, "down", "enter")                      # the greyed row is skipped
            self.assertEqual(ctx.settings.vision, "off")
        # every catalog model reads pictures now: a row that does not stands in for a future one
        import dataclasses
        from strata_cli.launch.catalog import Catalog
        real_rows = Catalog.rows

        def rows(self, *args, **kwargs):
            return [dataclasses.replace(r, images=False) if r.model == "Gyro-S" else r for r in real_rows(self, *args, **kwargs)]

        with mock.patch.object(Catalog, "rows", rows):
            self.run_app(case)

    def test_parallel_fit_to_card_and_warnings(self):
        async def case(app, pilot, ctx):
            await press(pilot, *self.reach_pictures(), "enter")
            t = text_of(app)
            self.assertIn("Fit to my graphics card (1)", t)
            self.assertIn("may not fit: slower answers", t)
            self.assertIn("12 GB: 1, 20 GB: 2, 24 GB: 3, 32 GB or more: 4", " ".join(t.split()))
            await press(pilot, "down", "down", "down", "enter")
            self.assertEqual(ctx.settings.parallel, 3)
            self.assertIn("Tune for the best speed", text_of(app))
        self.run_app(case)

    def test_parallel_default_is_fit_to_card(self):
        async def case(app, pilot, ctx):
            await press(pilot, *self.reach_pictures(), "enter", "enter")
            self.assertIsNone(ctx.settings.parallel)
        self.run_app(case)

    def test_bigger_card_fits_more(self):
        async def case(app, pilot, ctx):
            await press(pilot, *self.reach_pictures(), "enter")
            self.assertIn("Fit to my graphics card (3)", text_of(app))
        self.run_app(case, ctx=make_ctx(HW_24))

    def test_tuning_choices_and_the_answer_changing_tick(self):
        async def case(app, pilot, ctx):
            await press(pilot, *self.reach_pictures(), "enter", "enter")
            t = text_of(app)
            self.assertIn("Quick", t)
            self.assertIn("about 20 minutes", t)
            self.assertIn("Thorough", t)
            self.assertIn("up to 2 hours", t)
            self.assertIn("Skip", t)
            self.assertIn("[ ] Allow settings that change answers very slightly", t)
            line = next(l for l in t.splitlines() if "›" in l)
            self.assertIn("Quick", line)                              # a first setup recommends tuning
            await press(pilot, "a")
            self.assertIn("[x] Allow settings", text_of(app))
            self.assertTrue(ctx.settings.tune_allow_small)
            await press(pilot, "down", "enter")
            self.assertEqual(ctx.settings.tune, "thorough")
        self.run_app(case)

    def test_tuning_skip(self):
        async def case(app, pilot, ctx):
            await press(pilot, *self.reach_pictures(), "enter", "enter", "down", "down", "enter")
            self.assertEqual(ctx.settings.tune, "skip")
        self.run_app(case)


class AdvancedStep(Base):
    HELP = "  --prefill N          prompt chunk size (default 2048)\n  --kv fp16|int8       KV storage\n  --native-rope        experimental pinned rotary\n"

    def reach(self):
        return ("enter", "enter", "enter", "enter", "enter", "enter", "enter")

    def ctx_with_engine(self):
        ctx = make_ctx()
        ctx.settings.family, ctx.settings.model = "qwen", "IQ3_XXS"
        base = {"exe": "component:strata-engine", "args": ["--max-context", "32768", "--kv", "int8", "--prefill", "auto"]}
        p1 = mock.patch.object(Ctx, "base_config", lambda self: base)
        p2 = mock.patch.object(Ctx, "engine_exe", lambda self: "/engine/strata")
        p3 = mock.patch("strata_cli.launch.params.engine_help_text", return_value=self.HELP)
        for p in (p1, p2, p3):
            p.start()
            self.addCleanup(p.stop)
        return ctx

    def test_table_columns_command_and_same_as_line(self):
        ctx = self.ctx_with_engine()

        async def case(app, pilot, ctx):
            await press(pilot, *self.reach())
            t = text_of(app)
            for col in ("Setting", "Default", "Tuned", "Yours", "What it does"):
                self.assertIn(col, t)
            self.assertIn("--max-context", t)
            self.assertIn("--prefill", t)
            self.assertIn("--native-rope", t)
            self.assertIn("[experimental]", t)
            self.assertIn("+ Add a setting by name", t)
            self.assertIn("Command that will run", t)
            self.assertIn("Same as: strata launch --model iq3_s", t)
        self.run_app(case, ctx=ctx)

    def test_edit_a_value_then_reset_it(self):
        ctx = self.ctx_with_engine()

        async def case(app, pilot, ctx):
            await press(pilot, *self.reach())
            names = [it.key for it in app.screen.menu().items]
            idx = names.index("--prefill")
            for _ in range(idx):
                await press(pilot, "down")
            await press(pilot, "enter")
            self.assertIn("Type the value for --prefill", text_of(app))
            await typed(pilot, "65536")
            await press(pilot, "enter")
            self.assertEqual(ctx.settings.extra, {"--prefill": "65536"})
            self.assertIn("65536", text_of(app))
            self.assertIn("--set prefill=65536", text_of(app))
            await press(pilot, "delete")
            self.assertEqual(ctx.settings.extra, {})
            self.assertIn("back to the tuned value", text_of(app))
        self.run_app(case, ctx=ctx)

    def test_add_by_name_warns_about_an_unknown_flag_but_keeps_it(self):
        ctx = self.ctx_with_engine()

        async def case(app, pilot, ctx):
            await press(pilot, *self.reach())
            await press(pilot, "a")
            await typed(pilot, "prefil=1")
            await press(pilot, "enter")
            self.assertEqual(ctx.settings.extra, {"--prefil": "1"})
            t = text_of(app)
            self.assertIn("does not list --prefil", t)
            self.assertIn("Did you mean --prefill", t)
            self.assertIn("--prefil", t)
        self.run_app(case, ctx=ctx)

    def test_add_an_environment_setting(self):
        ctx = self.ctx_with_engine()

        async def case(app, pilot, ctx):
            await press(pilot, *self.reach(), "a")
            await typed(pilot, "STRATA_PIN_BY_NEED=1")
            await press(pilot, "enter")
            self.assertEqual(ctx.settings.extra, {"STRATA_PIN_BY_NEED": "1"})
        self.run_app(case, ctx=ctx)

    def test_experimental_filter_and_text_filter(self):
        ctx = self.ctx_with_engine()

        async def case(app, pilot, ctx):
            await press(pilot, *self.reach(), "x")
            keys = [it.key for it in app.screen.menu().items]
            self.assertIn("--native-rope", keys)
            self.assertNotIn("--max-context", keys)
            await press(pilot, "x", "slash")
            await typed(pilot, "kv")
            await press(pilot, "enter")
            keys = [it.key for it in app.screen.menu().items]
            self.assertIn("--kv", keys)
            self.assertNotIn("--native-rope", keys)
        self.run_app(case, ctx=ctx)

    def test_next_and_back(self):
        async def case(app, pilot, ctx):
            await press(pilot, *self.reach(), "n")
            self.assertIn("Who can use it?", text_of(app))
            await press(pilot, "escape")
            self.assertIn("All settings", text_of(app))
        self.run_app(case)

    def test_works_before_the_engine_is_installed(self):
        async def case(app, pilot, ctx):
            await press(pilot, *self.reach())
            t = text_of(app)
            self.assertIn("The engine is not installed yet", t)
            self.assertIn("Command that will run", t)
        self.run_app(case)

    def test_copy_does_not_crash(self):
        async def case(app, pilot, ctx):
            await press(pilot, *self.reach(), "c")
            self.assertTrue(any(w in text_of(app) for w in ("Copied", "does not allow copying")))
        self.run_app(case)


class AccessStep(Base):
    REACH = ("enter",) * 8

    def test_four_ways_and_what_each_needs(self):
        async def case(app, pilot, ctx):
            with mock.patch.object(access, "find_cloudflared", return_value=None), \
                    mock.patch.object(access, "tailscale_status", return_value={"installed": False, "running": False, "ip": None, "name": None}):
                await press(pilot, *self.REACH[:7], "n")
                t = text_of(app)
            self.assertIn("This computer only", t)
            self.assertIn("http://127.0.0.1:8080", t)
            self.assertIn("http://192.168.1.5:8080", t)
            self.assertIn("cloudflared is not installed", t)
            self.assertIn("Tailscale is not installed", t)
            self.assertIn("Port      [ 8080 ]", t)
            self.assertIn("API key   [ off ]", t)
        self.run_app(case)

    def reach(self):
        return self.REACH[:7] + ("n",)

    def test_lan_makes_a_key_and_keeps_the_port(self):
        async def case(app, pilot, ctx):
            await press(pilot, *self.reach(), "down")
            t = text_of(app)
            self.assertTrue(ctx.settings.api_key.startswith("sk-strata-"))
            self.assertIn("API key   [ on ]", t)
            self.assertIn("An API key was made for you", t)
            await press(pilot, "enter")
            self.assertEqual(ctx.settings.access, "lan")
            self.assertIn("Save as preset, then launch", text_of(app))
        self.run_app(case)

    def test_cloudflare_without_the_helper_says_how_to_install_it_and_does_not_block(self):
        async def case(app, pilot, ctx):
            with mock.patch.object(access, "find_cloudflared", return_value=None):
                await press(pilot, *self.reach(), "down", "down")
                t = text_of(app)
                self.assertIn("cloudflared is not installed on this computer", t)
                self.assertIn("quick tunnel", flat(app))
                await press(pilot, "enter")
            self.assertEqual(ctx.settings.access, "cloudflare")
            self.assertIn("Save as preset, then launch", text_of(app))
            self.assertTrue(any("cloudflared" in w for w in ctx.warnings()))
        self.run_app(case)

    def test_tailscale_found(self):
        async def case(app, pilot, ctx):
            with mock.patch.object(access, "tailscale_status", return_value={"installed": True, "running": True, "ip": "100.64.0.1", "name": "pc.tail.ts.net"}):
                await press(pilot, *self.reach(), "down", "down", "down")
                self.assertIn("http://pc.tail.ts.net:8080", text_of(app))
        self.run_app(case)

    def test_port_change_and_bad_port(self):
        async def case(app, pilot, ctx):
            await press(pilot, *self.reach(), "p")
            app.screen.query_one("#portfield").value = ""
            await typed(pilot, "abc")
            await press(pilot, "enter")
            self.assertIn("is not a port number", text_of(app))
            app.screen.query_one("#portfield").value = ""
            await typed(pilot, "9090")
            await press(pilot, "enter")
            self.assertEqual(ctx.settings.port, 9090)
            self.assertIn("http://127.0.0.1:9090", text_of(app))
        self.run_app(case)

    def test_windows_firewall_line_shows_on_lan_only(self):
        async def case(app, pilot, ctx):
            with mock.patch.object(access, "WINDOWS", True):
                await press(pilot, *self.reach())
                self.assertNotIn("Windows may ask", flat(app))
                await press(pilot, "down")
                self.assertIn("Windows may ask whether to allow Strata", flat(app))
                self.assertIn("choose Allow for private networks", flat(app))
        self.run_app(case)

    def test_key_toggle_and_new_key(self):
        async def case(app, pilot, ctx):
            await press(pilot, *self.reach(), "down")
            first = ctx.settings.api_key
            await press(pilot, "k")
            self.assertNotEqual(ctx.settings.api_key, first)
            await press(pilot, "t")
            self.assertIsNone(ctx.settings.api_key)
            self.assertTrue(ctx.settings.no_key)
            self.assertIn("anyone who can reach this port can use the model and its tools", text_of(app))
            await press(pilot, "t")
            self.assertTrue(ctx.settings.api_key)
            self.assertFalse(ctx.settings.no_key)
        self.run_app(case)

    def test_default_is_a_key_and_no_key_survives_moving_on(self):
        async def case(app, pilot, ctx):
            await press(pilot, *self.reach(), "down")
            self.assertTrue(ctx.settings.api_key)                       # default: Strata makes a key
            await press(pilot, "t")                                     # one keypress: no key
            await press(pilot, "down")                                  # moving the cursor does not bring the key back
            await press(pilot, "up")
            self.assertIsNone(ctx.settings.api_key)
            self.assertTrue(ctx.settings.no_key)
        self.run_app(case)

    def test_local_never_asks_for_a_key(self):
        async def case(app, pilot, ctx):
            await press(pilot, *self.reach())
            self.assertEqual(ctx.settings.access, "local")
            self.assertIsNone(ctx.settings.api_key)
            self.assertNotIn("An API key was made", text_of(app))
        self.run_app(case)


class SaveAndRun(Base):
    def reach(self):
        return ("enter",) * 8 + ("n", "enter")

    def test_summary_and_save_then_launch(self):
        async def case(app, pilot, ctx):
            await press(pilot, *("enter",) * 7, "n", "enter")
            t = text_of(app)
            self.assertIn("Save as preset, then launch", t)
            self.assertIn("Qwen3.8-Flash-Next IQ3_S", t)
            self.assertIn("32K, memory int8", t)
            self.assertIn("quick (about 20 min)", t)
            self.assertIn("This computer only, port 8080", t)
            await press(pilot, "enter")                               # Save as a preset and launch
            await pilot.pause(0.8)
            self.assertEqual([p.name for p in store.list_presets()], ["IQ3_S 32K"])
            self.assertEqual(FakeRunner.instances[0].s.name, "IQ3_S 32K")
            self.assertIsNotNone(app.result)
        self.run_app(case)

    def test_launch_without_saving(self):
        async def case(app, pilot, ctx):
            await press(pilot, *("enter",) * 7, "n", "enter", "down", "enter")
            await pilot.pause(0.8)
            self.assertEqual(store.list_presets(), [])
            self.assertIsNotNone(app.result)
        self.run_app(case)

    def test_save_and_stop_here(self):
        async def case(app, pilot, ctx):
            await press(pilot, *("enter",) * 7, "n", "enter", "down", "down", "enter")
            await pilot.pause(0.3)
            self.assertEqual(len(store.list_presets()), 1)
            self.assertEqual(FakeRunner.instances, [])
            self.assertTrue(app.stopped_after_save)
        self.run_app(case)

    def test_the_name_can_be_typed(self):
        async def case(app, pilot, ctx):
            await press(pilot, *("enter",) * 7, "n", "enter")
            f = app.screen.query_one("#name")
            f.value = ""
            f.focus()
            await typed(pilot, "My preset")
            app.screen.set_focus(None)
            await press(pilot, "enter")
            await pilot.pause(0.5)
            self.assertEqual([p.name for p in store.list_presets()], ["My preset"])
        self.run_app(case)

    def test_run_screen_shows_the_stages_and_the_tuning_table(self):
        shown = {}

        async def case(app, pilot, ctx):
            store.save(S.LaunchSettings(name="T", family="qwen", model="IQ3_XXS", context=32768, tune="quick"))
            FakeRunner.behaviour = "ok"
            FakeRunner.hold = 1.5
            self.addCleanup(setattr, FakeRunner, "hold", 0)
            app.start_preset("T")
            await pilot.pause(0.05)
            for _ in range(100):
                t = text_of(app)
                if "Default settings" in t:
                    shown["t"] = t
                    break
                await pilot.pause(0.02)
            await pilot.pause(0.5)
        self.run_app(case)
        t = shown.get("t", "")
        self.assertIn("Tuning", t)
        self.assertIn("measurement 2 of 5", t)
        self.assertIn("Default settings", t)
        self.assertIn("83.0 tokens/s", t)
        self.assertIn("Close other heavy programs while it measures", t)

    def test_failure_says_what_it_means_and_retry_works(self):
        async def case(app, pilot, ctx):
            store.save(S.LaunchSettings(name="T", family="qwen", model="IQ3_XXS", context=32768))
            FakeRunner.behaviour = "fail"
            app.start_preset("T")
            await pilot.pause(0.6)
            t = flat(app)
            self.assertIn("Not enough disk space", t)
            self.assertIn("Press r to try again", t)
            FakeRunner.behaviour = "ok"
            await press(pilot, "r")
            await pilot.pause(0.8)
            self.assertIsNotNone(app.result)
        self.run_app(case)

    def test_stop_asks_and_cancels(self):
        async def case(app, pilot, ctx):
            store.save(S.LaunchSettings(name="T", family="qwen", model="IQ3_XXS", context=32768))
            FakeRunner.behaviour = "wait"
            app.start_preset("T")
            await pilot.pause(0.4)
            await press(pilot, "s")
            self.assertIn("Stop now?", text_of(app))
            self.assertIn("Files already downloaded are kept", flat(app))
            await press(pilot, "enter")                               # Stop
            await pilot.pause(0.6)
            self.assertIn("Stopped. Nothing was started.", text_of(app))
        self.run_app(case)


class CardsStep(Base):
    def test_cards_step_shows_a_role_for_every_card_and_the_recommendation_is_selected(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "enter")
            t = text_of(app)
            self.assertIn("Which graphics cards should it use?", t)
            self.assertIn("Step 2 of 10", t)
            flat = " ".join(t.split())
            self.assertIn("[ Main ]", flat)                          # equal cards: the highest number is Main (the engine order stays [0, 1])
            self.assertIn("[ Helper ]", flat)
            self.assertIn("recommended", flat)
            self.assertIn("M Main", flat)
            self.assertNotIn("slower", flat)                         # no explaining warning for a recommendation
            await press(pilot, "enter")                              # keeping the recommendation stores "not set": a card added later is re-recommended
            self.assertIsNone(ctx.settings.gpu_roles)
            self.assertIsNone(ctx.settings.gpus)
            self.assertIn("Where should the model files be stored?", text_of(app))
        self.run_app(case, ctx=make_ctx(HW_2X16))

    def test_m_h_o_set_the_role_of_the_card_under_the_cursor_and_r_goes_back(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "enter")
            await press(pilot, "o")                                  # card 1 (a Helper) Off
            flat = " ".join(text_of(app).split())
            self.assertIn("recommended: helper", flat)               # the muted tag says what was recommended
            await press(pilot, "m")                                  # Main on card 1: card 2 becomes a Helper, one quiet line says so
            flat = " ".join(text_of(app).split())
            self.assertIn("is now a Helper.", flat)
            await press(pilot, "enter")
            self.assertEqual(ctx.settings.gpu_roles, {"0": "main", "1": "helper"})
            self.assertIsNone(ctx.settings.gpus)
        self.run_app(case, ctx=make_ctx(HW_2X16))

    def test_r_restores_the_recommendation(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "enter", "down", "o", "r", "enter")
            self.assertIsNone(ctx.settings.gpu_roles)
        self.run_app(case, ctx=make_ctx(HW_2X16))

    def test_every_card_off_is_allowed_with_one_short_line(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "enter", "down", "o", "up", "o")
            flat = " ".join(text_of(app).split())
            self.assertIn("! This engine needs a graphics card to start. Turn one on.", flat)
            await press(pilot, "enter")
            self.assertEqual(ctx.settings.gpu_roles, {"0": "off", "1": "off"})      # allowed; the server says plainly what the engine can do
        self.run_app(case, ctx=make_ctx(HW_2X16))

    def test_d_shows_what_the_roles_mean(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "enter", "d")
            self.assertIn("Helper: holds a share of the model", " ".join(text_of(app).split()))
        self.run_app(case, ctx=make_ctx(HW_2X16))

    def test_an_integrated_chip_beside_one_card_gives_no_cards_step(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter")
            self.assertIn("Step 1 of 9", text_of(app))
        hw = dict(HW_2X16)
        hw["gpus"] = [dict(HW_2X16["gpus"][0]), {"index": 1, "name": "AMD Radeon Graphics", "vram_gb": 16.0, "usable": True, "integrated": True}]
        self.run_app(case, ctx=make_ctx(hw))

    def test_one_card_has_nine_steps(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter")
            self.assertIn("Step 1 of 9", text_of(app))
        self.run_app(case)


class NarrowTerminal(Base):
    def test_80_columns_still_readable(self):
        ctx = make_ctx()
        app = LaunchApp(ctx)

        async def go():
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause(0.2)
                await press(pilot, "enter")
                t = text_of(app)
                self.assertIn("Which model?", t)
                self.assertIn("Step 1 of 9", t)
                self.assertIn("Enter next", t)
                await press(pilot, "enter", "enter")
                self.assertIn("How much text", text_of(app))
        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
