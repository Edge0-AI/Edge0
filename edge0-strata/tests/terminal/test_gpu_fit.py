"""strata_control/gpu_fit.py: the one rule for which graphics cards a model uses (app, strata launch and setup share it)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from strata_control import gpu_fit as G

ROOT = Path(__file__).resolve().parents[2]


def card(index, name, vram, **kw):
    return {"index": index, "name": name, "vram_gb": vram, "usable": True, **kw}


class GpuFitRule(unittest.TestCase):
    def test_one_card_is_the_only_option(self):
        r = G.options(44.0, 62.0, [card(0, "NVIDIA GeForce RTX 3070 Ti", 8)])
        self.assertEqual([o["key"] for o in r["options"]], ["single"])
        self.assertEqual(r["recommended"], "single")
        self.assertEqual(r["options"][0]["verdict"], "fits")
        self.assertIn("RTX 3070 Ti + RAM", r["options"][0]["label"])

    def test_two_unequal_cards_and_the_model_fits_one_defaults_to_single_and_offers_the_split(self):
        r = G.options(44.0, 62.0, [card(1, "NVIDIA GeForce RTX 3060 Ti", 8), card(0, "NVIDIA GeForce RTX 3070 Ti", 16)])
        keys = [o["key"] for o in r["options"]]
        self.assertEqual(keys[:2], ["single", "split"])             # the largest card first, then the split
        self.assertEqual(r["recommended"], "single")
        self.assertEqual(r["options"][0]["cards"], [0])             # the largest card is the 16 GB one
        split = r["options"][1]
        self.assertEqual(split["cards"], [0, 1])
        self.assertIn("layer split", split["label"])
        self.assertTrue(any("20% faster" in n for n in split["notes"]))
        self.assertEqual(sum(1 for o in r["options"] if o["recommended"]), 1)

    def test_two_small_cards_and_the_model_does_not_fit_one_defaults_to_the_split(self):
        r = G.options(44.0, 32.0, [card(0, "NVIDIA GeForce RTX 3060", 10), card(1, "NVIDIA GeForce RTX 3060", 10)])
        single, split = r["options"][0], r["options"][1]
        self.assertEqual(single["verdict"], "spills")
        self.assertLess(split["spill_gb"], single["spill_gb"])      # the second card takes part of the SSD reading away
        self.assertEqual(r["recommended"], "split")
        self.assertEqual(split["cards"], [0, 1])

    def test_a_tight_single_card_moves_the_default_to_the_split_when_that_fits_with_room(self):
        # 44 GB model, 46 GB of RAM: one 12 GB card + RAM holds it but leaves no RAM spare (tight); two cards leave room
        r = G.options(44.0, 46.0, [card(0, "RTX 4070", 12), card(1, "RTX 4070", 12)])
        self.assertEqual(r["options"][0]["verdict"], "tight")
        self.assertEqual(r["options"][1]["verdict"], "fits")
        self.assertEqual(r["recommended"], "split")

    def test_an_integrated_chip_and_a_tiny_card_are_ignored(self):
        cards = [card(0, "NVIDIA GeForce RTX 3070 Ti", 8), card(1, "AMD Radeon Graphics", 16, integrated=True), card(2, "Old display card", 2)]
        r = G.options(44.0, 62.0, cards)
        self.assertEqual([o["key"] for o in r["options"]], ["single"])
        self.assertEqual({c["index"] for c in r["ignored"]}, {1, 2})
        self.assertTrue(all(c["why_ignored"] for c in r["ignored"]))
        self.assertEqual(G.recommended_cards(44.0, 62.0, cards), [0])

    def test_a_unified_memory_chip_counts_when_it_is_the_only_device(self):
        mac = {"index": 0, "name": "Apple M4", "vendor": "apple", "vram_gb": 0.0, "unified_gb": 32.0, "usable": True}
        keep, ignored = G.usable_cards([mac])
        self.assertEqual((len(keep), ignored), (1, []))
        igpu = {"index": 0, "name": "AMD Radeon 8060S", "vram_gb": 64.0, "uma": True, "usable": True}
        self.assertEqual(len(G.usable_cards([igpu])[0]), 1)                 # a Strix Halo PC: the integrated chip IS the device
        keep, ignored = G.usable_cards([card(0, "RTX 3070 Ti", 8), igpu])   # beside a real card it adds nothing
        self.assertEqual([c["index"] for c in keep], [0] if keep[0]["name"] == "RTX 3070 Ti" else [])
        self.assertEqual(len(ignored), 1)

    def test_a_strix_halo_counts_its_memory_once(self):
        """The 'shared' pool of an APU is the RAM: a 70 GB model on a 128 GB Strix Halo with a 2 GB carve-out is RAM-resident, not 'VRAM + RAM'."""
        igpu = {"index": 0, "name": "AMD Radeon 8060S", "vram_gb": 66.0, "dedicated_gb": 2.0, "uma": True, "usable": True}
        self.assertEqual(G.own_memory_gb(igpu), 2.0)
        self.assertEqual(G.own_memory_gb({"index": 1, "vram_gb": 32.0}), 32.0)
        self.assertEqual(G.own_memory_gb({"index": 0, "vram_gb": 66.0, "uma": True}), 0.0)          # no carve-out figure: none counted
        # 64 GB PC, a 90 GB model: counted twice (66 GB 'VRAM' + 54 GB RAM) it looked like it fit; once, 36 GB come from the SSD
        r = G.options(90.0, 64.0, [dict(igpu, vram_gb=34.0)])
        self.assertEqual(r["options"][0]["verdict"], "spills")
        self.assertGreater(r["options"][0]["spill_gb"], 30.0)
        # 128 GB, 90 GB model: fits in RAM
        self.assertEqual(G.options(90.0, 128.0, [dict(igpu, vram_gb=66.0)])["options"][0]["verdict"], "fits")

    def test_an_unusable_card_is_ignored_with_its_own_reason(self):
        r = G.options(44.0, 62.0, [card(0, "RTX 3070 Ti", 8), card(1, "GTX 1060", 6, usable=False, problem="older than the RTX 20 series")])
        self.assertEqual([o["key"] for o in r["options"]], ["single"])
        self.assertIn("older than the RTX 20 series", r["ignored"][0]["why_ignored"])

    def test_a_card_under_the_split_minimum_is_offered_alone_but_not_in_the_split(self):
        r = G.options(44.0, 32.0, [card(0, "RTX 3060", 12), card(1, "GTX 1660", 6)])
        keys = [o["key"] for o in r["options"]]
        self.assertNotIn("split", keys)
        self.assertEqual(r["recommended"], "single")

    def test_three_cards_offer_the_pair_and_all_of_them(self):
        r = G.options(70.0, 32.0, [card(i, "RTX 3060", 12) for i in range(3)])
        keys = [o["key"] for o in r["options"]]
        self.assertEqual(keys[:3], ["single", "split", "split-all"])

    def test_no_card_gives_no_options(self):
        r = G.options(44.0, 62.0, [])
        self.assertEqual((r["options"], r["recommended"]), ([], None))

    def test_unknown_ram_is_unknown_not_a_guess(self):
        r = G.options(44.0, None, [card(0, "RTX 3070 Ti", 8), card(1, "RTX 3060 Ti", 8)])
        self.assertEqual(r["options"][0]["verdict"], "unknown")
        self.assertEqual(r["recommended"], "single")

    def test_plain_numbers_are_in_the_words(self):
        r = G.options(44.0, 62.0, [card(0, "NVIDIA GeForce RTX 3070 Ti", 8)])
        self.assertIn("needs 44 GB", r["options"][0]["words"])
        self.assertIn("RTX 3070 Ti + RAM", r["options"][0]["summary"])
        r2 = G.options(44.0, 32.0, [card(0, "RTX 3060", 10)])
        self.assertIn("read from the SSD", r2["options"][0]["words"])

    def test_the_constants_equal_the_apps(self):
        text = (ROOT / "apps" / "host" / "src" / "lib" / "native-fit.ts").read_text(encoding="utf-8")
        m = re.search(r"const DENSE_GB = (\d+), SYSTEM_ROOM_GB = (\d+), DENSE_EXTRA_GB = (\d+)", text)
        self.assertIsNotNone(m)
        self.assertEqual((G.DENSE_GB, G.SYSTEM_ROOM_GB, G.DENSE_EXTRA_GB), tuple(float(x) for x in m.groups()))

    def test_fit_matches_the_launch_catalog_port_of_native_fit(self):
        from strata_cli.launch.catalog import fit_on
        for ram, vram in ((62, 16), (32, 10), (16, 0), (128, 24)):
            a = fit_on({"memory_gb": 44.0}, ram, vram)
            b = G.fit(44.0, ram, vram)
            self.assertAlmostEqual(a["spillGb"], b["spill_gb"])
            self.assertAlmostEqual(a["vramUseGb"], b["vram_use_gb"])
            self.assertAlmostEqual(a["ramUseGb"], b["ram_use_gb"])


class GpuFitInTheModelList(unittest.TestCase):
    """The app reads the options from the model list (tools/strata_mcp.py model_table): the same rule, nothing duplicated in the UI."""

    def setUp(self):
        from tests.terminal.launch_support import Home
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)

    def sizes(self, hw):
        from tests.terminal.launch_support import catalog
        table = catalog(hw).strata.model_table(hw)
        return {s["model"]: s for f in table for s in f["sizes"]}

    def test_two_cards_give_options_on_every_model(self):
        from tests.terminal.launch_support import HW_2X16
        s = self.sizes(HW_2X16)["IQ3_XXS"]
        res = s["gpu_options"]
        self.assertEqual([o["key"] for o in res["options"]][:2], ["single", "split"])
        self.assertEqual(res["recommended"], "single")

    def test_one_card_has_no_choice_to_show(self):
        from tests.terminal.launch_support import HW_12
        self.assertIsNone(self.sizes(HW_12)["IQ3_XXS"]["gpu_options"])

    def test_an_integrated_chip_beside_one_card_is_no_choice(self):
        from tests.terminal.launch_support import HW_2X16
        hw = dict(HW_2X16)
        hw["gpus"] = [dict(HW_2X16["gpus"][0]), {"index": 1, "name": "AMD Radeon Graphics", "vram_gb": 16.0, "usable": False, "uma": True, "problem": "integrated graphics"}]
        self.assertIsNone(self.sizes(hw)["IQ3_XXS"]["gpu_options"])


if __name__ == "__main__":
    unittest.main()
