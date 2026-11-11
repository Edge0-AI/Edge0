"""DeepSeek on several AMD cards: which cards a run uses when the person named none, and who chose the cards in a saved config.

A PC with four AMD cards used one: setup picked the first card itself, counted that as "asked", wrote "gpus_asked", and every later launch handed
the "choice" back.  The cases below are the rules the fix keeps, with fake cards (nothing is read from the box the test runs on).

    python -m unittest tools.test_setup_amd_cards
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import test_setup_dsv41 as D  # noqa: E402
import serve.server as server  # noqa: E402
from strata_control import dsv41_fit, gpu_pick  # noqa: E402


def amd(i, arch="gfx1201", vram=32.0, **kw):
    return {"index": i, "name": f"AMD card {i} ({arch})", "vram_gb": vram, "arch": arch, "driver": "amdgpu", "vendor": "amd", **kw}


def apu(i, vram=96.0):
    return amd(i, "gfx1151", vram, uma=True)


FOUR = [amd(i) for i in range(4)]
INTEL2 = [{"index": i, "name": "Intel Arc B580", "vram_gb": 12.0, "arch": "xe", "driver": "xe", "devid": "e20b"} for i in range(2)]


def ids(cards):
    return [c["index"] for c in cards]


class Rule(unittest.TestCase):
    """dsv41_fit.run_cards: the one rule `strata launch --plan`, setup and the app share."""

    def test_four_same_cards_all_run(self):
        keep, left = dsv41_fit.run_cards(FOUR, "amd", windows=False)
        self.assertEqual(ids(keep), [0, 1, 2, 3])
        self.assertEqual(left, [])

    def test_an_apu_beside_a_discrete_card_is_left_out(self):
        keep, left = dsv41_fit.run_cards([apu(0), amd(1)], "amd", windows=False)
        self.assertEqual(ids(keep), [1])
        self.assertIn("shares the system memory", left[0]["why_ignored"])

    def test_an_apu_alone_is_the_one_that_counts(self):
        self.assertEqual(ids(dsv41_fit.run_cards([apu(0)], "amd", windows=False)[0]), [0])

    def test_a_card_the_engine_package_has_no_code_for_is_left_out_and_said(self):
        keep, left = dsv41_fit.run_cards([amd(0, "gfx1201"), amd(1, "gfx1100", 24.0)], "amd", windows=False)
        self.assertEqual(ids(keep), [0])
        self.assertIn("gfx1100", left[0]["why_ignored"])
        self.assertIn("gfx1200, gfx1201", left[0]["why_ignored"])

    def test_the_installed_package_decides_what_it_covers(self):
        keep, _ = dsv41_fit.run_cards([amd(0, "gfx1201"), amd(1, "gfx1100", 24.0)], "amd", windows=False, covered=("gfx1100", "gfx1201"))
        self.assertEqual(ids(keep), [0, 1])

    def test_windows_runs_on_one_card(self):
        keep, left = dsv41_fit.run_cards([amd(0), amd(1)], "amd", windows=True)
        self.assertEqual(ids(keep), [0])
        self.assertIn("Windows", left[0]["why_ignored"])

    def test_named_cards_are_used_as_named(self):
        keep, left = dsv41_fit.run_cards([amd(0, "gfx1201"), amd(1, "gfx1100", 24.0)], "amd", windows=True, named=True)
        self.assertEqual(ids(keep), [0, 1])
        self.assertEqual(left, [])

    def test_windows_rx_7900_and_rx_9070_run_on_the_card_setup_takes_and_nothing_is_dropped_by_arch(self):
        cards = [amd(0, "gfx1201", 16.0), amd(1, "gfx1100", 24.0)]
        keep, left = dsv41_fit.run_cards(cards, "amd", windows=True)
        self.assertEqual(ids(keep), [1])                              # discrete, then the most memory: setup's amd_rank
        self.assertEqual([c["index"] for c in left], [0])
        self.assertIn("Windows", left[0]["why_ignored"])
        self.assertNotIn("no code", left[0]["why_ignored"])           # Windows ships another package: the Linux list is not applied
        with mock.patch.object(setup, "amd_problem", lambda g: None), mock.patch.object(setup, "WIN", True):
            self.assertEqual(ids(setup.dsv41_auto_cards(cards, Path("nowhere"))), [1])
            self.assertEqual(min(cards, key=setup.amd_rank)["index"], 1)      # what setup pins as a.gpu

    def test_windows_prefers_a_discrete_card_to_an_apu(self):
        keep, _ = dsv41_fit.run_cards([apu(0), amd(1, "gfx1100", 24.0), amd(2, "gfx1100", 24.0)], "amd", windows=True)
        self.assertEqual(ids(keep), [1])

    def test_no_manifest_list_means_no_code_warning_for_a_named_card(self):
        self.assertIsNone(setup.dsv41_covered_archs(Path("nowhere")))

    def test_nvidia_small_cards_are_said_as_before_and_options_single_is_the_largest(self):
        nv = [{"index": 0, "name": "RTX 3060", "vram_gb": 12.0, "vendor": "nvidia", "arch": 86},
              {"index": 1, "name": "RTX 3090", "vram_gb": 24.0, "vendor": "nvidia", "arch": 86},
              {"index": 2, "name": "GT 1030", "vram_gb": 2.0, "vendor": "nvidia", "arch": 61}]
        keep, left = dsv41_fit.run_cards(nv, "nvidia", windows=True)
        self.assertEqual(ids(keep), [0, 1])
        self.assertEqual([c["index"] for c in left], [2])             # only the truth: 2 GB (usable_cards says it already)
        opts = dsv41_fit.options(nv, 128.0, vendor="nvidia", model="IST-DAS-2.25bpw")
        single = next(o for o in opts["options"] if o["key"] == "single")
        self.assertEqual(single["cards"], [1])

    def test_nvidia_cards_are_not_touched(self):
        nv = [{"index": i, "name": "RTX 3090", "vram_gb": 24.0, "vendor": "nvidia", "arch": 86} for i in range(2)]
        self.assertEqual(ids(dsv41_fit.run_cards(nv, "nvidia", windows=True)[0]), [0, 1])

    def test_the_plan_and_the_options_say_what_the_run_does(self):
        cases = {"four": (FOUR, False, [0, 1, 2, 3]), "apu": ([apu(0), amd(1)], False, [1]),
                 "mixed": ([amd(0), amd(1, "gfx1100", 24.0)], False, [0]), "windows": ([amd(0), amd(1)], True, [0])}
        for name, (cards, windows, want) in cases.items():
            with mock.patch.object(dsv41_fit, "_on_windows", lambda w=windows: w):
                plan = dsv41_fit.plan(cards, 251.0, None, vendor="amd", model="IST-DAS-2.25bpw")
                opts = dsv41_fit.options(cards, 251.0, vendor="amd", model="IST-DAS-2.25bpw")
            self.assertEqual([c["index"] for c in plan["cards"]], want, name)
            self.assertEqual(opts["cards"], len(want), name)


class Pick(unittest.TestCase):
    """strata_control/gpu_pick.py: who chose the cards in a saved config."""

    def cfg(self, **kw):
        return {"arch": "deepseek41", "backend": "hip", "gpu": 0, "gpus_asked": True, **kw}

    def test_a_config_of_setup_0_2_8_that_holds_setups_own_pick_is_not_a_choice(self):
        self.assertFalse(gpu_pick.chosen_by_person(self.cfg(), FOUR))

    def test_another_card_than_the_one_setup_would_take_was_the_persons(self):
        self.assertTrue(gpu_pick.chosen_by_person(self.cfg(gpu=2), FOUR))

    def test_a_new_config_says_who_chose(self):
        self.assertTrue(gpu_pick.chosen_by_person(self.cfg(gpus_by="person"), FOUR))
        self.assertFalse(gpu_pick.chosen_by_person(self.cfg(gpus_by="setup"), FOUR))

    def test_a_person_who_chose_card_0_after_the_fix_keeps_it(self):
        self.assertTrue(gpu_pick.chosen_by_person(self.cfg(gpus_by="person", gpu=0), FOUR))

    def test_one_usable_card_is_the_only_choice(self):
        self.assertTrue(gpu_pick.chosen_by_person(self.cfg(), [amd(0)]))

    def test_card_lists_and_roles_are_the_persons(self):
        self.assertTrue(gpu_pick.chosen_by_person(self.cfg(gpu=[0, 1]), FOUR))
        self.assertTrue(gpu_pick.chosen_by_person({"gpu_roles": {"0": "main"}, "gpu": 0}, FOUR))

    def test_other_models_and_makes_keep_what_they_had(self):
        self.assertTrue(gpu_pick.chosen_by_person(self.cfg(arch="qwen4exp"), FOUR))
        self.assertTrue(gpu_pick.chosen_by_person(self.cfg(backend="cuda"), FOUR))

    def test_no_choice_was_made_when_none_is_marked(self):
        self.assertFalse(gpu_pick.chosen_by_person({"arch": "deepseek41", "backend": "hip", "gpu": 0}, FOUR))

    def test_the_apu_is_not_a_card_for_the_pick(self):
        usable = [c for c in FOUR[:1]]                          # an APU beside one discrete card: one card counts
        self.assertTrue(gpu_pick.chosen_by_person(self.cfg(), usable))


class Intel(unittest.TestCase):
    """sycl/setup_intel.py runs setup's AMD branch with its Intel cards in the AMD list (arch "xe")."""

    def test_two_intel_cards_keep_the_one_card_setup_picked_for_deepseek(self):
        with mock.patch.object(setup, "amd_problem", lambda g: None), mock.patch.object(setup, "WIN", False):
            self.assertEqual(setup.dsv41_auto_cards(INTEL2, Path("nowhere")), [])      # [] -> dsv41_cards falls back to the one card

    def test_a_qwen_config_of_the_intel_path_is_not_widened(self):
        cfg = {"arch": "qwen4exp", "backend": "sycl", "gpu": 0, "gpus_asked": True}
        self.assertTrue(gpu_pick.chosen_by_person(cfg, INTEL2))


class Install(unittest.TestCase):
    """setup.install_dsv41 with the Install test harness: what the config says."""
    setUp = D.Install.setUp
    install = D.Install.install

    def go(self, all_cards, cards, hip=True, pick=None, **kw):
        named = kw.pop("named", False)
        if pick is not None:                                   # setup's own pick, written into a.gpu by the AMD branch (more than one device)
            kw["gpu"] = pick
        gpu = {"index": cards[0]["index"], "count": len(all_cards or cards), "vram_gb": 32.0, "name": "R9700", "cards": cards, "arch": "gfx1201"}
        self.exe.parent.mkdir(parents=True, exist_ok=True)
        self.exe.write_text("x")
        a = D.args(models_dir=str(self.root / "models"), model="UD-Q8_K_XL", vision="no", **kw)
        a._cards_named = named
        rc, out = D.run(setup.install_dsv41, a, "deepseek41", gpu, hip, [], 13, [], 8080, 251.0, None, all_cards)
        self.assertEqual(rc, 0, out)
        return out

    def test_nothing_asked_uses_every_card_and_says_setup_chose_them(self):
        out = self.go(FOUR, [FOUR[0]])
        self.assertEqual(self.cfg["gpu"], [0, 1, 2, 3])
        self.assertTrue(self.cfg["gpus_asked"])                       # as setup always wrote it for a card list; gpus_by says who chose
        self.assertEqual(self.cfg["gpus_by"], "setup")
        self.assertIn("4 cards", out)
        self.assertEqual(server.child_env({**self.cfg})["HIP_VISIBLE_DEVICES"], "0,1,2,3")

    def test_gpu_0_is_one_card_and_remembered_as_the_persons(self):
        self.go(None, [FOUR[0]], gpu=0, named=True)
        self.assertEqual(self.cfg["gpu"], 0)
        self.assertTrue(self.cfg["gpus_asked"])
        self.assertEqual(self.cfg["gpus_by"], "person")

    def test_gpus_0_1_are_those_two_cards(self):
        self.go(None, FOUR[:2], gpus="0,1", named=True)
        self.assertEqual(self.cfg["gpu"], [0, 1])
        self.assertTrue(self.cfg["gpus_asked"])
        self.assertEqual(self.cfg["gpus_by"], "person")

    def test_windows_with_two_cards_pins_the_one_card_and_the_engine_sees_only_it(self):
        self.go(None, [FOUR[0]], pick=0)
        self.assertEqual(self.cfg["gpu"], 0)
        self.assertTrue(self.cfg["gpus_asked"])
        self.assertEqual(self.cfg["gpus_by"], "setup")
        self.assertEqual(server.child_env({**self.cfg})["HIP_VISIBLE_DEVICES"], "0")

    def test_an_apu_beside_a_discrete_card_not_at_index_0_pins_the_discrete_card(self):
        discrete = amd(1)
        self.go([discrete], [discrete], pick=1)
        self.assertEqual(self.cfg["gpu"], 1)
        self.assertTrue(self.cfg["gpus_asked"])
        self.assertEqual(self.cfg["gpus_by"], "setup")
        self.assertEqual(server.child_env({**self.cfg})["HIP_VISIBLE_DEVICES"], "1")

    def test_two_mixed_cards_pin_the_one_the_package_runs_even_when_setup_picked_the_other(self):
        covered = amd(0, "gfx1201", 16.0)
        self.go([covered], [covered], pick=1)                            # setup's own pick (more memory) was the gfx1100 card
        self.assertEqual(self.cfg["gpu"], 0)
        self.assertEqual(self.cfg["gpus_by"], "setup")
        self.assertEqual(server.child_env({**self.cfg})["HIP_VISIBLE_DEVICES"], "0")

    def test_one_card_in_the_pc_writes_no_card_choice(self):
        self.go(None, [FOUR[0]])
        self.assertNotIn("gpu", self.cfg)

    def test_nvidia_is_written_as_before(self):
        nv = [{"index": i, "name": "RTX 3090", "vram_gb": 24.0, "vendor": "nvidia", "arch": 86} for i in range(2)]
        with mock.patch.object(setup, "dsv41_engine_pending", lambda *a, **k: False):
            self.go(nv, nv[:1], hip=False)
        self.assertEqual(self.cfg["gpu"], [0, 1])
        self.assertTrue(self.cfg["gpus_asked"])
        self.assertNotIn("gpus_by", self.cfg)


class Widen(unittest.TestCase):
    """An installed config of setup 0.2.8 and earlier, at the start."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "strata-deepseek-v4.1-flash-ist-das-2.25bpw.json"
        self.args = ["--pack", "/p/pack", "--eos", "1"]
        for p in (mock.patch.object(setup, "WIN", False), mock.patch.object(setup, "amd_problem", lambda g: None),
                  mock.patch.object(setup, "ok", lambda *a, **k: None), mock.patch.object(setup, "warn", lambda *a, **k: None),
                  mock.patch.object(setup, "say", lambda *a, **k: None), mock.patch.object(setup, "ROOT", Path(self.tmp.name))):
            p.start()
            self.addCleanup(p.stop)

    def widen(self, cfg, cards=FOUR):
        full = {"exe": "/e/strata-dsv41", "args": self.args, "arch": "deepseek41", "backend": "hip", **cfg}
        self.path.write_text(json.dumps(full))
        return setup.widen_setups_one_card(self.path, full, cards), json.loads(self.path.read_text())

    def test_an_old_config_by_accident_moves_to_every_card_and_keeps_its_pack(self):
        new, saved = self.widen({"gpu": 0, "gpus_asked": True})
        self.assertEqual(saved["gpu"], [0, 1, 2, 3])
        self.assertEqual(new["gpu"], [0, 1, 2, 3])
        self.assertTrue(saved["gpus_asked"])                          # as for every card list; "gpus_by" says setup chose them
        self.assertEqual(saved["gpus_by"], "setup")
        self.assertEqual(saved["args"], self.args)                   # the pack and the files are not touched

    def test_a_second_start_stays_on_every_card(self):
        _, saved = self.widen({"gpu": 0, "gpus_asked": True})
        again, saved2 = self.widen(saved)
        self.assertEqual(saved2["gpu"], [0, 1, 2, 3])

    def test_an_old_config_by_choice_of_another_card_stays(self):
        new, saved = self.widen({"gpu": 2, "gpus_asked": True})
        self.assertEqual(saved["gpu"], 2)

    def test_a_choice_made_after_the_fix_stays_even_on_card_0(self):
        _, saved = self.widen({"gpu": 0, "gpus_asked": True, "gpus_by": "person"})
        self.assertEqual(saved["gpu"], 0)

    def test_windows_stays_on_one_card(self):
        with mock.patch.object(setup, "WIN", True):
            _, saved = self.widen({"gpu": 0, "gpus_asked": True})
        self.assertEqual(saved["gpu"], 0)

    def test_one_usable_card_has_nothing_to_widen(self):
        _, saved = self.widen({"gpu": 0, "gpus_asked": True}, [amd(0)])
        self.assertEqual(saved["gpu"], 0)

    def test_an_apu_beside_a_discrete_card_is_not_added(self):
        _, saved = self.widen({"gpu": 1, "gpus_asked": True}, [apu(0), amd(1)])
        self.assertEqual(saved["gpu"], 1)

    def test_a_qwen_config_is_not_widened(self):
        _, saved = self.widen({"gpu": 0, "gpus_asked": True, "arch": "qwen4exp"})
        self.assertEqual(saved["gpu"], 0)

    def test_mixed_architectures_widen_only_to_the_cards_the_package_runs(self):
        _, saved = self.widen({"gpu": 0, "gpus_asked": True}, [amd(0), amd(1, "gfx1100", 24.0), amd(2)])
        self.assertEqual(saved["gpu"], [0, 2])


class Adoption(unittest.TestCase):
    """A new copy set up like an earlier install: only a DeepSeek-on-AMD config of setup's own pick loses its card."""

    def kept(self, cfg, listing=None, win=False):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "strata-x.json"
            path.write_text(json.dumps(cfg))
            with mock.patch.object(setup, "WIN", win), mock.patch.object(setup, "amd_problem", lambda g: None), \
                    mock.patch.object(setup, "amd_gpus", listing or (lambda: FOUR)):
                return setup.earlier_card_kept(path)

    def test_nvidia_and_qwen_configs_keep_their_card_whatever_marks_they_have(self):
        def boom():
            raise AssertionError("the AMD listing must not be read for another model or make")
        self.assertTrue(self.kept({"arch": "qwen4exp", "backend": "cuda", "gpu": 1}, boom))
        self.assertTrue(self.kept({"arch": "qwen4exp", "backend": "hip", "gpu": 1, "gpus_asked": True}, boom))
        self.assertTrue(self.kept({"arch": "deepseek41", "backend": "cuda", "gpu": 1}, boom))

    def test_deepseek_on_amd_by_setups_own_pick_is_not_carried_over(self):
        self.assertFalse(self.kept({"arch": "deepseek41", "backend": "hip", "gpu": 0, "gpus_asked": True}))

    def test_deepseek_on_amd_by_the_person_is_carried_over(self):
        self.assertTrue(self.kept({"arch": "deepseek41", "backend": "hip", "gpu": 2, "gpus_asked": True}))
        self.assertTrue(self.kept({"arch": "deepseek41", "backend": "hip", "gpu": 0, "gpus_asked": True, "gpus_by": "person"}))

    def test_a_listing_that_fails_or_windows_cannot_stop_a_setup(self):
        def boom():
            raise OSError("no kfd")
        self.assertTrue(self.kept({"arch": "deepseek41", "backend": "hip", "gpu": 0, "gpus_asked": True}, boom))
        self.assertTrue(self.kept({"arch": "deepseek41", "backend": "hip", "gpu": 0, "gpus_asked": True}, win=True))
        self.assertTrue(setup.earlier_card_kept(Path("does-not-exist.json")))


class Remember(unittest.TestCase):
    """`strata launch --gpu N` / `setup --gpu N` on a DeepSeek AMD model is the person's choice from then on."""

    def test_the_choice_is_saved_and_then_kept_by_every_later_launch(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "strata-ds.json"
            old = {"arch": "deepseek41", "backend": "hip", "gpu": 0, "gpus_asked": True, "args": ["--pack", "/p"]}
            path.write_text(json.dumps(old))
            self.assertIsNotNone(gpu_pick.remember(path, [0]))
            saved = json.loads(path.read_text())
            self.assertEqual((saved["gpu"], saved["gpus_by"], saved["args"]), (0, "person", ["--pack", "/p"]))
            from strata_cli.launch.runner import installed_cards
            self.assertEqual(installed_cards(saved, FOUR), [0])                # a person's card 0 is not widened again
            self.assertFalse(Path(str(path) + ".tmp").exists())

    def test_other_configs_are_left_alone(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "strata-q.json"
            path.write_text(json.dumps({"arch": "qwen4exp", "backend": "hip", "gpu": 0}))
            self.assertIsNone(gpu_pick.remember(path, [1]))
            self.assertEqual(json.loads(path.read_text())["gpu"], 0)
            self.assertIsNone(gpu_pick.remember(Path(d) / "missing.json", [1]))


class LauncherCards(unittest.TestCase):
    """cards_for_launch: what the launcher puts on the engine matches what setup and the plan say."""

    class Comp:
        def __init__(self, gpus):
            self.gpus, self.engine_vendor = gpus, "amd"

        @property
        def usable_gpus(self):
            from strata_control import gpu_fit
            return gpu_fit.usable_cards([{**g, "usable": True} for g in self.gpus])[0]

    class Cat:
        def __init__(self, gpus):
            self.comp = LauncherCards.Comp(gpus)

        def computer(self):
            return self.comp

    def cards(self, gpus, base, windows=False, gpus_named=None):
        from strata_cli.launch import runner
        from strata_cli.launch import settings as S
        s = S.LaunchSettings(family="deepseek41", gpus=gpus_named)
        with mock.patch.object(dsv41_fit, "_on_windows", lambda: windows):
            return runner.cards_for_launch(s, self.Cat(gpus), base=base)

    def base(self, **kw):
        return {"arch": "deepseek41", "backend": "hip", **kw}

    def test_nothing_named_on_linux_uses_every_card_in_roles_order(self):
        self.assertEqual(sorted(self.cards(FOUR, self.base())), [0, 1, 2, 3])

    def test_an_old_config_of_setups_own_pick_is_widened_for_this_launch(self):
        self.assertEqual(sorted(self.cards(FOUR, self.base(gpu=0, gpus_asked=True))), [0, 1, 2, 3])

    def test_a_card_the_person_named_is_kept(self):
        self.assertIsNone(self.cards(FOUR, self.base(gpu=2, gpus_asked=True)))
        self.assertIsNone(self.cards(FOUR, self.base(gpu=0, gpus_asked=True, gpus_by="person")))

    def test_gpu_0_named_on_this_launch_is_one_card(self):
        self.assertEqual(self.cards(FOUR, self.base(), gpus_named=[0]), [0])

    def test_windows_pins_the_one_card_setup_takes(self):
        self.assertEqual(self.cards([amd(0), amd(1)], self.base(gpus_by="setup", gpu=0, gpus_asked=True), windows=True), [0])

    def test_an_apu_beside_a_discrete_card_leaves_nothing_to_split(self):
        self.assertIsNone(self.cards([apu(0), amd(1)], self.base(gpus_by="setup", gpu=1, gpus_asked=True)))

    def test_mixed_cards_run_on_the_ones_the_package_has_code_for(self):
        self.assertEqual(self.cards([amd(0), amd(1, "gfx1100", 24.0)], self.base(gpus_by="setup", gpu=0, gpus_asked=True)), [0])


class Launcher(unittest.TestCase):
    def test_installed_cards_skips_setups_own_pick_and_keeps_a_choice(self):
        from strata_cli.launch.runner import installed_cards
        base = {"arch": "deepseek41", "backend": "hip", "gpu": 0, "gpus_asked": True}
        usable = FOUR
        self.assertIsNone(installed_cards(base, usable))
        self.assertEqual(installed_cards({**base, "gpu": 2}, usable), [2])
        self.assertEqual(installed_cards({**base, "gpus_by": "person"}, usable), [0])
        self.assertEqual(installed_cards({**base, "gpu": [0, 1]}, usable), [0, 1])
        self.assertEqual(installed_cards(base, None), [0])                # no card list: as before


if __name__ == "__main__":
    unittest.main()
