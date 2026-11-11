"""DeepSeek's plan counts the NVIDIA cards setup runs: a card of 8 GB or less starts Off beside a bigger card (gpu_roles), and the plan says so.
Fake cards only."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strata_control import dsv41_fit                       # noqa: E402


def card(i, name, gb, vendor="nvidia"):
    return {"index": i, "name": name, "vram_gb": gb, "vendor": vendor, "usable": True}


class NvidiaPlan(unittest.TestCase):
    def plan(self, cards, **kw):
        return dsv41_fit.plan(cards, 128.0, None, vendor=cards[0].get("vendor"), model="IST-DAS-2.25bpw", **kw)

    def test_an_8_gb_card_beside_a_bigger_one_is_off_in_the_plan_and_said(self):
        p = self.plan([card(0, "NVIDIA GeForce RTX 4090", 24.0), card(1, "NVIDIA GeForce RTX 3070", 8.0)])
        self.assertEqual([c["index"] for c in p["cards"]], [0])
        text = " ".join(p["lines"])
        self.assertIn("Not used: NVIDIA GeForce RTX 3070 (a card with 8 GB or less starts Off beside a bigger card", text)

    def test_the_card_may_be_listed_first(self):
        p = self.plan([card(0, "NVIDIA GeForce RTX 3070", 8.0), card(1, "NVIDIA GeForce RTX 4090", 24.0)])
        self.assertEqual([c["index"] for c in p["cards"]], [1])

    def test_two_8_gb_cards_and_two_bigger_cards_are_unchanged(self):
        self.assertEqual([c["index"] for c in self.plan([card(0, "A", 8.0), card(1, "A", 8.0)])["cards"]], [0, 1])
        self.assertEqual([c["index"] for c in self.plan([card(0, "A", 24.0), card(1, "B", 12.0)])["cards"]], [0, 1])
        self.assertEqual([c["index"] for c in self.plan([card(i, "A", 24.0) for i in range(3)])["cards"]], [0, 1, 2])

    def test_a_named_card_is_used_as_named(self):
        p = self.plan([card(0, "A", 24.0), card(1, "B", 8.0)], named=True)
        self.assertEqual([c["index"] for c in p["cards"]], [0, 1])

    def test_an_amd_card_of_8_gb_still_counts(self):
        from unittest import mock
        with mock.patch.object(dsv41_fit, "_on_windows", lambda: False):        # (on Windows a model runs on one AMD card)
            p = self.plan([card(0, "AMD Radeon AI PRO R9700", 32.0, "amd"), card(1, "AMD Radeon RX 9060", 8.0, "amd")])
        self.assertEqual([c["index"] for c in p["cards"]], [0, 1])

    def test_cards_with_no_make_are_left_alone(self):
        cs = [{"index": 0, "name": "A", "vram_gb": 24.0}, {"index": 1, "name": "B", "vram_gb": 8.0}]
        self.assertEqual([c["index"] for c in dsv41_fit.plan(cs, 128.0, None, model="IST-DAS-2.25bpw")["cards"]], [0, 1])


if __name__ == "__main__":
    unittest.main()
