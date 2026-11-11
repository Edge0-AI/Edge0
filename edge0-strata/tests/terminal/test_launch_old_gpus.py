"""`strata launch` on a computer whose only NVIDIA cards are Pascal / Volta (P100, V100): setup.py admits them (the experimental CUDA 12 engine);
the screens must not say "no usable graphics card found", and several of them must offer the Cards step."""

from __future__ import annotations

import unittest
from unittest import mock

from strata_cli.launch.catalog import Catalog
from tests.terminal.launch_support import HW_12, Home, catalog


def v100(index: int, vram: float = 16.0) -> dict:
    return {"index": index, "name": "Tesla V100-SXM2-16GB", "arch": "70", "vram_gb": vram, "vendor": "nvidia", "usable": False,
            "problem": "not supported - older than the RTX 20 series (compute capability 7.0)"}


RTX = {"index": 0, "name": "NVIDIA GeForce RTX 3060", "arch": "86", "vram_gb": 12.0, "vendor": "nvidia", "usable": True, "problem": None}


class OldNvidiaCards(unittest.TestCase):
    def test_only_volta_cards_are_admitted(self):
        got = Catalog._admit_old_nvidia([v100(0), v100(1)])
        self.assertEqual([g["usable"] for g in got], [True, True])
        self.assertTrue(all(g["problem"] is None for g in got))

    def test_a_newer_card_keeps_the_old_one_out(self):
        got = Catalog._admit_old_nvidia([{**RTX}, v100(1)])
        self.assertEqual([g["usable"] for g in got], [True, False])

    def test_an_amd_card_keeps_the_old_one_out(self):
        amd = {"index": 0, "name": "AMD Radeon RX 9070 XT", "vendor": "amd", "usable": True, "vram_gb": 16.0}
        got = Catalog._admit_old_nvidia([amd, v100(0)])
        self.assertEqual([g["usable"] for g in got], [True, False])

    def test_other_cards_are_not_touched(self):
        gtx = {"index": 0, "name": "GeForce GT 1030", "arch": "61", "vram_gb": 2.0, "vendor": "nvidia", "usable": False}
        self.assertEqual(Catalog._admit_old_nvidia([gtx])[0]["usable"], True)    # Pascal is admitted too (setup does the same)
        kepler = {"index": 0, "name": "Tesla K80", "arch": "37", "vram_gb": 12.0, "vendor": "nvidia", "usable": False}
        self.assertEqual(Catalog._admit_old_nvidia([kepler])[0]["usable"], False)
        self.assertEqual(Catalog._admit_old_nvidia([]), [])

    def test_hardware_and_computer_show_the_cards(self):
        with Home():
            cat = catalog()
            cat._hardware_override = None
            hw = {**HW_12, "gpus": [v100(0), v100(1)]}
            with mock.patch.object(type(cat.strata), "hardware", return_value=hw):
                comp = cat.computer(fresh=True)
            self.assertEqual(len(comp.usable_gpus), 2)
            self.assertIn("x2", comp.line())
            self.assertNotIn("no usable graphics card", comp.line())


if __name__ == "__main__":
    unittest.main()
