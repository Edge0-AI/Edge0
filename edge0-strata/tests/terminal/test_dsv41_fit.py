"""strata_control/dsv41_fit.py: the memory plan of the official DeepSeek-V4.1-Flash sets (IST-DAS 1.75 / 2.00 / 2.25 bpw), and its
unified-memory branch for a Ryzen AI Max "Strix Halo" (gfx1151) PC, where the graphics memory is the RAM and is counted once.

The numbers of the 64 GB box are the measured ones (Windows, Adrenalin 26.10.1, engine 0.1.51): 2.00 bpw, Faster mode (5 experts),
DSpark auto: answers 14-16 tokens/s on code and 10-11 on chat, a 4K prompt 77-81 tokens/s and 8K 108-111.
"""
from __future__ import annotations

import unittest

from strata_control import dsv41_fit as F

# what setup hands the plan on the owner's Strix Halo: one integrated Radeon, its "VRAM" counting the 64 GB BIOS carve-out plus 32 GB
# shared with the 64 GB Windows sees (128 GB of memory in one pool)
HALO = [{"index": 0, "name": "AMD Radeon 8060S (Ryzen AI Max, Strix Halo, gfx1151)", "vram_gb": 96.0, "arch": "gfx1151",
         "uma": True, "integrated": True, "dedicated_gb": 64.0, "shared_gb": 32.0}]
SMALL = [{"index": 0, "name": "AMD Radeon 8040S (Ryzen AI Max, Strix Halo, gfx1151)", "vram_gb": 32.0, "arch": "gfx1151",
          "uma": True, "integrated": True, "dedicated_gb": 16.0, "shared_gb": 16.0}]
R9700 = [{"index": 0, "name": "Radeon AI PRO R9700", "vram_gb": 32.0, "arch": "gfx1201"}]
TWO_R9700 = [R9700[0], {**R9700[0], "index": 1}]


class UnifiedMemoryPlan(unittest.TestCase):
    def plan(self, model="IST-DAS-2.00bpw", ram=63.6, disk=700.0):
        return F.plan(HALO, ram, disk, 16384, model=model)

    def test_counts_the_whole_pool_once(self):
        p = self.plan()
        self.assertTrue(p["uma"])
        self.assertEqual(p["tp"], 1)                    # no multi-card split words
        self.assertEqual(p["args"], [])                 # and no --tp argument
        resident = 63.6 + 64.0 - F.ISTA_DENSE_GB - F.UMA_RESERVE_GB      # the RAM Windows sees plus the BIOS carve-out
        self.assertAlmostEqual(p["vram_experts_gb"], resident, places=1)
        self.assertAlmostEqual(p["streamed_gb"], 132.07 - resident, places=1)
        self.assertEqual(p["experts_in_vram"], int(resident / (132.07 / F.EXPERTS)))
        self.assertEqual(p["verdict"], "tight")         # it runs, most of it from memory: measured 14-16 tokens/s on code
        text = " ".join(p["lines"])
        self.assertIn("one pool of 128 GB", text)
        self.assertIn("read from the SSD", text)
        self.assertIn("14-16 tokens a second", text)

    def test_it_never_refuses_and_keeps_the_disk_numbers(self):
        p = self.plan("IST-DAS-2.25bpw")
        self.assertTrue(p["disk_ok"])
        self.assertAlmostEqual(p["download_gb"], F.ista_download_bytes("IST-DAS-2.25bpw") / 1e9, places=1)
        self.assertGreater(p["streamed_gb"], 30.0)      # 2.25 needs more from the SSD than 2.00
        self.assertTrue(any("274 GB" in l for l in p["lines"]))   # the set's own download

    def test_a_smaller_unified_pc_is_slower_never_refused(self):
        p = F.plan(SMALL, 16.0, 700.0, 8192, model="IST-DAS-2.00bpw")
        self.assertEqual(p["verdict"], "spills")
        self.assertTrue(any("1.75 bpw" in w for w in p["warnings"]))
        self.assertTrue(p["disk_ok"])

    def test_a_128_gb_unified_pc_nearly_holds_the_smallest_set(self):
        p = F.plan(HALO, 63.6, 700.0, 8192, model="IST-DAS-1.75bpw")
        self.assertGreater(p["vram_experts_gb"], 105.0)
        self.assertLess(p["streamed_gb"], 10.0)
        self.assertEqual(p["verdict"], "tight")         # 1.75 nearly fits the 128 GB Strix Halo: ~5 GB still come from the SSD

    def test_a_large_enough_unified_pc_fits(self):
        p = F.plan(HALO, 160.0, 700.0, 8192, model="IST-DAS-1.75bpw")
        self.assertEqual(p["verdict"], "fits")
        self.assertEqual(p["streamed_gb"], 0.0)

    def test_a_discrete_card_keeps_the_per_card_plan(self):
        p = F.plan(TWO_R9700, 251.0, 700.0, 8192, model="IST-DAS-2.00bpw")
        self.assertFalse(p.get("uma"))
        self.assertEqual(p["tp"], 2)
        self.assertEqual(len(p["cards"]), 2)
        self.assertTrue(any("Two cards share" in l for l in p["lines"]))

    def test_the_smaller_set_is_recommended_for_a_small_unified_pc(self):
        self.assertEqual(F.recommend(32.0), "IST-DAS-1.75bpw")
        self.assertEqual(F.recommend(63.6), "IST-DAS-2.00bpw")


if __name__ == "__main__":
    unittest.main()
