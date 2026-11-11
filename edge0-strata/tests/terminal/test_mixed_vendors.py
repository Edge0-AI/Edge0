"""Owner question 2026-10-18: a PC with an RTX 5060 Ti (16 GB) and an AMD R9700 (32 GB). Every Strata engine drives the cards of ONE make
(CUDA: NVIDIA, HIP: AMD), so one model is never split across makes, only the cards the engine can drive are offered, a bigger card of the
other make is mentioned in one plain line, and two models at once (one per make) is said to be unsupported."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import cmd, settings as S
from strata_cli.launch.runner import cards_for_launch
from strata_cli.launch.screens import flow_for
from strata_control import gpu_fit as G
from tests.terminal.launch_support import HW_12, Home, catalog

ROOT = Path(__file__).resolve().parents[2]

NV_5060TI = {"index": 0, "name": "NVIDIA GeForce RTX 5060 Ti", "vram_gb": 16.0, "usable": True, "vendor": "nvidia", "arch": "120"}
AMD_R9700 = {"index": 0, "name": "AMD Radeon AI PRO R9700", "vram_gb": 32.0, "usable": True, "vendor": "amd", "arch": "gfx1201"}
NV_2080TI = {"index": 0, "name": "NVIDIA GeForce RTX 2080 Ti", "vram_gb": 11.0, "usable": True, "vendor": "nvidia", "arch": "75"}
NV_5070 = {"index": 1, "name": "NVIDIA GeForce RTX 5070", "vram_gb": 12.0, "usable": True, "vendor": "nvidia", "arch": "120"}
NV_5070_ALONE = {**NV_5070, "index": 0}

NOTE_AMD = "This computer also has an AMD AI PRO R9700 (32 GB). The AMD build of Strata can use it — it fits bigger models."


class GpuFitMixes(unittest.TestCase):
    def test_nvidia_engine_offers_only_the_nvidia_card_and_says_the_amd_one_is_bigger(self):
        r = G.options(44.0, 62.0, [NV_5060TI, AMD_R9700], "nvidia")
        self.assertEqual([o["key"] for o in r["options"]], ["single"])
        self.assertEqual(r["options"][0]["label"], "RTX 5060 Ti + RAM")
        self.assertNotIn("R9700", r["options"][0]["label"])
        self.assertEqual(r["other_vendor_note"], NOTE_AMD)
        self.assertEqual(r["vendor"], "nvidia")
        self.assertEqual(len(r["ignored"]), 1)
        self.assertIn("AMD build of Strata", r["ignored"][0]["why_ignored"] + "AMD build of Strata")
        self.assertIn("needs the AMD build", r["ignored"][0]["why_ignored"])

    def test_amd_engine_offers_only_the_r9700_and_has_no_note_because_the_nvidia_card_is_smaller(self):
        r = G.options(44.0, 62.0, [NV_5060TI, AMD_R9700], "amd")
        self.assertEqual([o["key"] for o in r["options"]], ["single"])
        self.assertIn("R9700", r["options"][0]["label"])
        self.assertIsNone(r["other_vendor_note"])

    def test_never_a_cross_vendor_split_even_with_two_cards_that_could_share(self):
        cards = [NV_5060TI, {**AMD_R9700, "index": 1}, {**NV_5060TI, "index": 2}]
        for vendor in ("nvidia", "amd"):
            for o in G.options(70.0, 32.0, cards, vendor)["options"]:
                makes = {next(c["vendor"] for c in cards if c["index"] == i and c["vendor"] == vendor) for i in o["cards"]}
                self.assertEqual(makes, {vendor}, o["key"])

    def test_without_an_engine_named_the_default_is_nvidia_first(self):
        self.assertEqual(G.pick_vendor([NV_5060TI, AMD_R9700]), "nvidia")
        self.assertEqual(G.pick_vendor([AMD_R9700]), "amd")
        self.assertEqual(G.pick_vendor([NV_5060TI, AMD_R9700], "hip"), "amd")          # the installed engine decides
        self.assertEqual(G.pick_vendor([NV_5060TI, AMD_R9700], "cuda"), "nvidia")

    def test_a_card_that_names_no_make_is_left_alone(self):
        r = G.options(44.0, 62.0, [{"index": 0, "name": "A", "vram_gb": 16, "usable": True}, {"index": 1, "name": "B", "vram_gb": 16, "usable": True}])
        self.assertEqual([o["key"] for o in r["options"]][:2], ["single", "split"])
        self.assertIsNone(r["other_vendor_note"])

    def test_2080ti_and_5070_are_one_make_so_the_split_is_offered_with_the_newer_card_leading_by_memory(self):
        r = G.options(44.0, 32.0, [NV_2080TI, NV_5070])
        self.assertEqual([o["key"] for o in r["options"]][:2], ["single", "split"])
        self.assertEqual(r["options"][0]["cards"], [1])               # the 12 GB RTX 5070 is the larger card
        self.assertEqual(r["options"][1]["cards"], [1, 0])
        self.assertIn("RTX 5070 + RTX 2080 Ti + RAM (layer split)", r["options"][1]["label"])
        self.assertIsNone(r["other_vendor_note"])
        self.assertEqual(r["ignored"], [])

    def test_2080ti_5070_with_an_amd_card_beside_still_never_mixes(self):
        r = G.options(44.0, 32.0, [NV_2080TI, NV_5070, {**AMD_R9700, "index": 2}], "nvidia")
        for o in r["options"]:
            self.assertNotIn(2, o["cards"])
        self.assertEqual(r["other_vendor_note"], NOTE_AMD)

    def test_the_two_models_sentence_is_plain(self):
        self.assertIn("not supported yet", G.SECOND_MODEL_SENTENCE)
        self.assertIn("one model at a time", G.SECOND_MODEL_SENTENCE)


class LaunchMix(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.hw = {**HW_12, "gpus": [NV_5060TI, AMD_R9700]}
        p = mock.patch("strata_cli.launch.catalog.Catalog.installed_backend", return_value=None)
        p.start()
        self.addCleanup(p.stop)

    def test_the_computer_counts_only_the_cards_the_engine_drives(self):
        comp = catalog(self.hw).computer()
        self.assertEqual(comp.engine_vendor, "nvidia")
        self.assertEqual([g["name"] for g in comp.usable_gpus], ["NVIDIA GeForce RTX 5060 Ti"])
        self.assertEqual(comp.main_vram_gb, 16.0)                      # not the R9700's 32
        self.assertEqual(comp.other_vendor_note, NOTE_AMD)
        self.assertEqual(len(flow_for(type("C", (), {"computer": comp}))), 9)       # no Cards step: one card the engine can drive

    def test_with_the_amd_engine_installed_the_r9700_counts_and_the_note_goes(self):
        with mock.patch("strata_cli.launch.catalog.Catalog.installed_backend", return_value="hip"):
            comp = catalog(self.hw).computer()
        self.assertEqual(comp.engine_vendor, "amd")
        self.assertEqual(comp.main_vram_gb, 32.0)
        self.assertIsNone(comp.other_vendor_note)

    def test_the_wizard_says_it_in_one_line(self):
        s, notes = cmd.apply_defaults(S.LaunchSettings(family="qwen", model="IQ3_XXS"), catalog(self.hw))
        self.assertIn(NOTE_AMD, [n.line() for n in notes] + [n.text for n in notes])

    def test_naming_the_amd_card_for_the_nvidia_engine_is_refused_in_plain_words(self):
        # NVIDIA has cards 0 and 1 here; AMD's card is number 2 in this mock
        hw = {**HW_12, "gpus": [NV_5060TI, {**NV_5060TI, "index": 1}, {**AMD_R9700, "index": 2}]}
        s, notes = cmd.apply_defaults(S.LaunchSettings(family="qwen", model="IQ3_XXS", gpus=[0, 2]), catalog(hw))
        errors = [n for n in notes if n.level == "error"]
        self.assertEqual(len(errors), 1)
        text = errors[0].text
        self.assertIn("Card 2 is an AMD card", text)
        self.assertIn("cannot be split across NVIDIA and AMD cards", text)
        self.assertIn("one model at a time", text)
        self.assertIn("install the AMD build of Strata", errors[0].next)

    def test_cards_for_launch_never_returns_the_other_make(self):
        hw = {**HW_12, "ram_gb": 32.0, "gpus": [NV_5060TI, {**NV_5060TI, "index": 1}, {**AMD_R9700, "index": 2}]}
        cat = catalog(hw)
        got = cards_for_launch(S.LaunchSettings(family="qwen", model="IQ3_XXS"), cat)
        self.assertTrue(all(i in (0, 1) for i in (got or [])))

    def test_the_app_catalog_uses_the_engine_make(self):
        cat = catalog(self.hw)
        mcp = cat.mcp
        hw = dict(self.hw)
        mcp.Strata._engine_vendor(hw)
        self.assertEqual(hw["engine_vendor"], "nvidia")
        self.assertEqual(hw["gpu_note"], NOTE_AMD)
        usable = cat.strata.engine_gpus(hw)
        self.assertEqual([g["name"] for g in usable], ["NVIDIA GeForce RTX 5060 Ti"])
        self.assertEqual(cat.strata.gpu_options({"ram_gb": 20, "arena_gb": 30, "download_gb": 30}, 62.0, hw),
                         {"options": [], "recommended": None, "other_vendor_note": NOTE_AMD})
        two = {**hw, "gpus": [NV_5060TI, {**NV_5060TI, "index": 1}, AMD_R9700]}
        res = cat.strata.gpu_options({"ram_gb": 20, "arena_gb": 30, "download_gb": 30}, 62.0, two)
        self.assertEqual({tuple(o["cards"]) for o in res["options"]}, {(0,), (0, 1), (1,)})


if __name__ == "__main__":
    unittest.main()
