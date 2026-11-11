"""DeepSeek's speed choice (0.2.4.3): Faster (--experts 5, the default on a PC with 16 GB of graphics memory or less) or Full quality (--experts 6)."""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.app import LaunchApp
from strata_cli.launch.runner import (Runner, apply_experts, derive_config, dsv41_engine_version, engine_takes_experts, experts_from_log,
                                      faster_by_default, used_vram_gb, uses_integrated_gpu)
from strata_cli.launch.screens import SpeedScreen, flow_for
from tests.terminal import test_launch as TL
from tests.terminal.launch_support import Home, text_of
from tests.terminal.test_launch_tui import SIZE, make_ctx, press

run_cli = TL.run_cli


class Words(unittest.TestCase):
    def test_the_default_is_faster_up_to_16_gb_of_graphics_memory(self):
        for gb in (8, 12, 15.9, 16.0):                       # the engine's rule: 16 GiB or less (17,179,869,184 bytes)
            self.assertTrue(S.faster_default(gb), gb)
        for gb in (16.01, 16.4, 20, 24, 48, 0, None):
            self.assertFalse(S.faster_default(gb), gb)

    def test_plain_words_and_the_quality_note(self):
        self.assertEqual(S.speed_words("faster", True), "Faster (default on this PC)")
        self.assertEqual(S.speed_words("faster", False), "Faster")
        self.assertEqual(S.speed_words("full", False), "Full quality: all 6 experts per word")
        self.assertEqual(S.FASTER_NOTE, "Faster uses 5 of 6 experts per word: a small quality change, usually not noticeable.")

    def test_the_flags_text_says_the_same(self):
        self.assertNotIn("--speed", S.flags_text(S.LaunchSettings(family="deepseek41", model="IST-DAS-2.00bpw")))
        self.assertIn("--full-quality", S.flags_text(S.LaunchSettings(family="deepseek41", model="IST-DAS-2.00bpw", speed="full")))
        self.assertIn("--speed faster", S.flags_text(S.LaunchSettings(family="deepseek41", model="IST-DAS-2.00bpw", speed="faster")))


class Units(unittest.TestCase):
    """The catalog's vram_gb is GiB, the engine's unit: checked through the code that reads the cards (#3843-style: exact MiB values)."""

    def nvidia(self, *mib):
        text = "\n".join(f"{i}, NVIDIA GeForce RTX {i}, {m}, 8.9, 580.97" for i, m in enumerate(mib))
        import setup
        from tools import strata_mcp
        with mock.patch.object(setup, "out", return_value=text), mock.patch.object(strata_mcp, "run_quiet", return_value=text):
            return setup.gpus(), strata_mcp.own_nvidia_gpus()

    def total(self, cards):
        return sum(round(c["vram_gb"], 1) for c in cards)             # the catalog rounds each card to 0.1 GiB

    def test_nvidia_mib_is_divided_by_1024(self):
        for cards in self.nvidia(16376):
            self.assertAlmostEqual(cards[0]["vram_gb"], 16376 / 1024)

    def test_16_gb_cards_count_as_16_gib_or_less(self):
        for mib in (16376, 16384, 16380):                              # RTX 4080 / 4060 Ti 16 GB, an exact 16 GiB card, in between
            for cards in self.nvidia(mib):
                self.assertTrue(S.faster_default(self.total(cards)), mib)
        for mib in (12288, 12282, 8188):                               # RTX 3060, 12 GiB, 8 GB
            for cards in self.nvidia(mib):
                self.assertTrue(S.faster_default(self.total(cards)), mib)

    def test_bigger_cards_and_sums_over_16_gib_do_not(self):
        for mib in (24564, 24576, 20475, 32607):
            for cards in self.nvidia(mib):
                self.assertFalse(S.faster_default(self.total(cards)), mib)
        for cards in self.nvidia(8188, 8188):                          # two 8 GB cards: 15.99 GiB together
            self.assertTrue(S.faster_default(self.total(cards)))
        for cards in self.nvidia(12288, 8188):                         # 12 + 8: over
            self.assertFalse(S.faster_default(self.total(cards)))

    def test_an_amd_16_gb_card_counts(self):
        for size in (17179869184, 17163091968):                        # 16 GiB exactly, and a 16 GB card's 16,368 MiB
            self.assertTrue(S.faster_default(round(size / 2 ** 30, 1)), size)
        self.assertFalse(S.faster_default(round(34342961152 / 2 ** 30, 1)))     # a 32 GB card


class EngineFlag(unittest.TestCase):
    def test_experts_replaces_the_saved_value_and_goes_last(self):
        cfg = {"args": ["--pack", "p", "--experts", "6", "--dspark", "auto"]}
        apply_experts(cfg, 5)
        self.assertEqual(cfg["args"], ["--pack", "p", "--dspark", "auto", "--experts", "5"])
        cfg = {"args": ["--experts=5", "--pack", "p"]}
        apply_experts(cfg, 6)
        self.assertEqual(cfg["args"], ["--pack", "p", "--experts", "6"])
        cfg = {"args": ["--pack", "p"]}
        apply_experts(cfg, None)
        self.assertEqual(cfg["args"], ["--pack", "p"])

    def test_derive_config_passes_it(self):
        base = {"exe": "x", "args": ["--pack", "p"]}
        cfg = derive_config(base, S.LaunchSettings(), bool_flags=frozenset(), experts=5)
        self.assertEqual(cfg["args"][-2:], ["--experts", "5"])
        self.assertEqual(derive_config(base, S.LaunchSettings(), bool_flags=frozenset())["args"], ["--pack", "p"])


class Resolve(unittest.TestCase):
    def runner(self, *cards, **kw):
        cat = mock.Mock()
        cat.strata.multi_family.return_value = True
        gpus = [{"index": i, "vram_gb": gb} for i, gb in enumerate(cards)]
        cat.computer.return_value = mock.Mock(usable_gpus=gpus)
        r = Runner(S.LaunchSettings(family="deepseek41", model="IST-DAS-2.00bpw", **kw), catalog=cat, emit=lambda e: None)
        r.dsv41_engine_found = lambda: (Path(self.engine), True)
        return r

    engine = "/x/engine-dsv41/0.1.50-linux-x86_64-cuda-cuda13/strata-dsv41"

    def test_a_12_gb_pc_gets_faster_and_a_big_one_full_quality_in_the_words(self):
        self.assertEqual(self.runner(12).resolved_speed(), ("faster", True))
        self.assertEqual(self.runner(96).resolved_speed(), ("full", True))
        self.assertEqual(self.runner(8, 8).resolved_speed(), ("faster", True))         # summed over the cards, as the engine sums them
        self.assertEqual(self.runner(12, 12).resolved_speed(), ("full", True))

    def test_nothing_is_passed_to_the_engine_unless_the_person_chose(self):
        for cards in ((12,), (96,), (8, 8)):
            self.assertIsNone(self.runner(*cards).experts_flag(), cards)                # the engine's --experts auto decides: one rule
        self.assertEqual(self.runner(12, speed="full").experts_flag(), 6)
        self.assertEqual(self.runner(96, speed="faster").experts_flag(), 5)

    def test_a_choice_wins_over_the_default_in_the_words(self):
        self.assertEqual(self.runner(12, speed="full").resolved_speed(), ("full", False))

    def test_only_the_cards_the_launch_uses_are_summed(self):
        comp = mock.Mock(usable_gpus=[{"index": 0, "vram_gb": 12}, {"index": 1, "vram_gb": 24}])
        self.assertEqual(used_vram_gb(comp, S.LaunchSettings()), 36)
        self.assertEqual(used_vram_gb(comp, S.LaunchSettings(gpus=[0])), 12)
        self.assertEqual(used_vram_gb(comp, S.LaunchSettings(gpu_roles={"0": "main", "1": "off"})), 12)

    def test_an_integrated_gpu_is_faster_by_default_whatever_its_memory(self):          # Strix Halo: 96 GB of shared memory, the engine says 5
        halo = {"index": 0, "vram_gb": 96, "vendor": "amd", "uma": True, "name": "AMD Radeon 8060S (gfx1151)"}
        comp = mock.Mock(usable_gpus=[halo])
        s = S.LaunchSettings()
        self.assertTrue(uses_integrated_gpu(comp, s))
        self.assertTrue(faster_by_default(comp, s))
        for key in ("integrated", "unified_memory"):
            other = {"index": 0, "vram_gb": 64, "vendor": "amd", key: True}
            self.assertTrue(faster_by_default(mock.Mock(usable_gpus=[other]), s), key)
        big = {"index": 0, "vram_gb": 96, "vendor": "amd"}                               # a 96 GB discrete card is still Full quality
        self.assertFalse(faster_by_default(mock.Mock(usable_gpus=[big]), s))
        mac = {"index": 0, "vram_gb": 0, "vendor": "apple", "unified_gb": 64}
        self.assertFalse(uses_integrated_gpu(mock.Mock(usable_gpus=[mac]), s))           # DeepSeek is not on the Mac
        two = mock.Mock(usable_gpus=[{"index": 0, "vram_gb": 24, "vendor": "nvidia"}, {"index": 1, "vram_gb": 32, "vendor": "amd", "uma": True}])
        self.assertFalse(uses_integrated_gpu(two, S.LaunchSettings(gpus=[0])))            # only the cards this launch uses count
        self.assertTrue(uses_integrated_gpu(two, S.LaunchSettings(gpus=[1])))

    def test_the_label_and_the_plan_say_faster_on_strix_halo_and_the_guard_still_holds(self):
        cat = mock.Mock()
        cat.strata.multi_family.return_value = True
        cat.computer.return_value = mock.Mock(usable_gpus=[{"index": 0, "vram_gb": 96, "vendor": "amd", "uma": True}])
        r = Runner(S.LaunchSettings(family="deepseek41", model="IST-DAS-2.00bpw"), catalog=cat, emit=lambda e: None)
        r.dsv41_engine_found = lambda: (Path(self.engine), True)
        self.assertEqual(r.resolved_speed(), ("faster", True))
        self.assertIn("Faster (default on this PC)", r.speed_line())
        self.assertIsNone(r.experts_flag())                                              # not chosen: the engine's auto rule decides
        r.dsv41_engine_found = lambda: (Path("/x/engine-dsv41/0.1.49.2-a/strata-dsv41"), True)
        self.assertIsNone(r.resolved_speed())                                            # engine below 0.1.50: no line, no flag
        self.assertEqual(r.speed_line(), "")

    def test_the_engines_own_line_wins_in_the_summary(self):
        r = self.runner(12)
        self.assertIn("Faster (default on this PC)", r.speed_line(5))
        self.assertIn("Full quality (default on this PC)", r.speed_line(6))             # the engine said 6 although the rule here said Faster
        self.assertIn("Full quality: all 6 experts per word", self.runner(12, speed="full").speed_line(5))   # a choice stays what was chosen

    def test_the_number_is_read_back_from_the_engine_log(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "e.log"
            log.write_text("ds4: loading\nds4: experts per word: 6\nlater\nds4: experts per word: 5\n", encoding="utf-8")
            self.assertEqual(experts_from_log(log), 5)
            log.write_text("nothing here\n", encoding="utf-8")
            self.assertIsNone(experts_from_log(log))
            self.assertIsNone(experts_from_log(Path(d) / "missing.log"))

    def test_the_line_has_the_note_only_for_faster(self):
        self.assertIn(S.FASTER_NOTE, self.runner(12).speed_line())
        self.assertIn("Faster (default on this PC)", self.runner(12).speed_line())
        self.assertEqual(self.runner(12, speed="full").speed_line(), "Speed: Full quality: all 6 experts per word")

    def test_an_engine_older_than_0_1_50_gets_no_experts_flag_and_no_speed(self):
        old = "/x/engine-dsv41/0.1.49.2-linux-x86_64-cuda-cuda13/strata-dsv41"
        r = self.runner(12, speed="full")
        r.dsv41_engine_found = lambda: (Path(old), True)
        self.assertFalse(r.speed_engine_ok())
        self.assertIsNone(r.experts_flag())                    # an explicit choice too: this engine would not understand the flag
        self.assertIsNone(r.resolved_speed())
        self.assertEqual(r.speed_line(), "")                    # no line in --plan, no error
        r.dsv41_engine_found = lambda: (None, None)            # no engine yet: nothing either
        self.assertIsNone(r.experts_flag())
        self.assertEqual(self.runner(12, speed="full").experts_flag(), 6)
        self.assertEqual(r.experts_flag(self.engine), 6)       # the config's engine decides in build_config (after the self-upgrade)

    def test_the_version_comes_from_the_package_folder(self):
        for name, want in (("0.1.49.2", False), ("0.1.49.9", False), ("0.1.50", True), ("0.1.50.1", True), ("0.2.0", True), ("0.1.49", False)):
            exe = f"/x/engine-dsv41/{name}-windows-x86_64-cuda-cuda13/strata-dsv41.exe"
            self.assertEqual(engine_takes_experts(exe), want, name)
        self.assertEqual(dsv41_engine_version("/x/engine-dsv41/0.1.50-a/strata-dsv41"), (0, 1, 50))
        self.assertFalse(engine_takes_experts(None))
        self.assertFalse(engine_takes_experts("/nowhere/strata-dsv41"))        # a name that says nothing and a program that is not there

    def test_a_build_of_your_own_is_asked_for_its_version(self):
        done = mock.Mock(stdout="strata-dsv41 0.1.50.1 (abc)\n", stderr="")
        with mock.patch("strata_cli.launch.runner.subprocess.run", return_value=done):
            self.assertTrue(engine_takes_experts("/home/me/build/strata-dsv41"))
        done = mock.Mock(stdout="strata-dsv41 0.1.49.2\n", stderr="")
        with mock.patch("strata_cli.launch.runner.subprocess.run", return_value=done):
            self.assertFalse(engine_takes_experts("/home/me/build/strata-dsv41"))

    def test_another_model_has_no_speed(self):
        cat = mock.Mock()
        cat.strata.multi_family.return_value = False
        r = Runner(S.LaunchSettings(family="qwen", model="IQ3_S"), catalog=cat, emit=lambda e: None)
        self.assertIsNone(r.resolved_speed())
        self.assertIsNone(r.experts_flag())
        self.assertEqual(r.speed_line(), "")


class Flags(unittest.TestCase):
    def plan(self, *args):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", "--model", "ist-das-2.00bpw", "--family", "deepseek41", *args])
        self.assertEqual(code, 0, err + out)
        return json.loads(out)["plan"]

    def test_plan_names_the_speed_in_plain_words(self):
        with mock.patch.object(Runner, "resolved_speed", return_value=("faster", True)):
            p = self.plan()
        self.assertIn("Faster (default on this PC)", p["speed"])
        self.assertIn(S.FASTER_NOTE, p["speed"])

    def test_full_quality_flag_and_speed_flag(self):
        self.assertEqual(self.plan("--full-quality")["settings"]["speed"], "full")
        self.assertEqual(self.plan("--speed", "full")["settings"]["speed"], "full")
        self.assertEqual(self.plan("--speed", "faster")["settings"]["speed"], "faster")
        self.assertIn("--full-quality", self.plan("--full-quality")["flags"])

    def test_both_flags_say_so_and_full_quality_wins(self):
        p = self.plan("--full-quality", "--speed", "faster")
        self.assertEqual(p["settings"]["speed"], "full")
        self.assertTrue(any("full quality is used" in w for w in p["warnings"]))

    def test_another_model_warns_and_drops_it(self):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", "--model", "q2_0", "--full-quality"])
        p = json.loads(out)["plan"]
        self.assertIsNone(p["settings"]["speed"])
        self.assertTrue(any("for DeepSeek" in w for w in p["warnings"]))

    def test_human_plan_prints_a_speed_line(self):
        with mock.patch.object(Runner, "resolved_speed", return_value=("faster", True)):
            code, out, err = run_cli(["launch", "--plan", "--model", "ist-das-2.00bpw", "--family", "deepseek41"])
        self.assertIn("  speed     Faster (default on this PC)", out)


class Screen(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)

    def test_the_step_exists_only_for_deepseek(self):
        ctx = make_ctx()
        self.assertNotIn("speed", flow_for(ctx))
        ctx.settings.family = "deepseek41"
        with mock.patch.object(Runner, "speed_engine_ok", return_value=False):          # engine 0.1.49.2 or none: the step is not there
            self.assertNotIn("speed", flow_for(ctx))
        ctx._speed_ok = None
        with mock.patch.object(Runner, "speed_engine_ok", return_value=True):
            flow = flow_for(ctx)
        self.assertEqual(flow[flow.index("context") + 1], "speed")

    def test_the_screen_offers_two_choices_and_remembers_only_a_change(self):
        ctx = make_ctx()
        ctx.settings.family, ctx.settings.model = "deepseek41", "IST-DAS-2.00bpw"
        faster_here = faster_by_default(ctx.computer, ctx.settings)
        ctx._speed_ok = True
        app = LaunchApp(ctx)

        async def go():
            async with app.run_test(size=SIZE) as pilot:
                await pilot.pause(0.2)
                app.goto("speed")
                await pilot.pause(0.3)
                t = text_of(app)
                self.assertIn("Faster", t)
                self.assertIn("Full quality", t)
                self.assertIn(S.FASTER_NOTE, t)
                screen = app.screen
                self.assertIsInstance(screen, SpeedScreen)
                screen.commit()
                self.assertEqual(ctx.settings.speed, None if faster_here else "faster")
                await press(pilot, "down")
                screen.commit()
                self.assertIn(ctx.settings.speed, (None, "full"))
        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
