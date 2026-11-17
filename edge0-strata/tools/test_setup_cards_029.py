"""A card that works alone but cannot share a model (under 7.5 GB) beside a bigger NVIDIA card: setup says it is not used (a line, no question, no
change to which cards run).  Fake cards through setup.main (tools/test_setup_golden.py's harness)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_setup_golden import install, argv_for       # noqa: E402


def card(i, name, gb, arch="86"):
    return {"index": i, "name": name, "vram_gb": gb, "arch": arch, "driver": "580.97"}


BIG = card(0, "NVIDIA GeForce RTX 4090", 24.0, "89")
SMALL = card(1, "NVIDIA GeForce RTX 2060", 5.9, "75")


class SmallCardSaid(unittest.TestCase):
    def run_it(self, found, extra=()):
        code, out, cfg, asked = install(256.0, found, argv_for("qwen", "IQ3_S") + ["--yes", *extra])
        self.assertEqual(code, 0, out)
        return out, cfg, asked

    def test_the_small_card_is_named_with_the_reason(self):
        out, cfg, asked = self.run_it([BIG, SMALL])
        self.assertIn("GPU 1 (NVIDIA GeForce RTX 2060, 6 GB) is not used", out)
        self.assertIn("needs 8 GB or more", out)
        self.assertIn("runs on GPU 0 alone", out)
        self.assertEqual(asked, [])                                   # printed only
        self.assertEqual((cfg["gpu"], cfg["gpus_asked"]), (0, True))   # the cards used are what they were

    def test_the_small_card_listed_first(self):
        out, cfg, _ = self.run_it([card(0, "NVIDIA GeForce RTX 2060", 5.9, "75"), card(1, "NVIDIA GeForce RTX 4090", 24.0, "89")])
        self.assertIn("GPU 0 (NVIDIA GeForce RTX 2060, 6 GB) is not used", out)
        self.assertEqual(cfg["gpu"], 1)

    def test_nothing_is_said_where_nothing_is_left_out(self):
        for found in ([BIG], [BIG, card(1, "NVIDIA GeForce RTX 3060", 12.0)], [BIG, card(1, "NVIDIA GeForce RTX 3070", 8.0)]):
            out, _cfg, _ = self.run_it(found)
            self.assertNotIn("is not used", out, found)

    def test_naming_a_card_says_nothing_extra(self):
        out, cfg, _ = self.run_it([BIG, SMALL], ["--gpu", "0"])
        self.assertNotIn("is not used", out)
        self.assertEqual(cfg["gpu"], 0)


if __name__ == "__main__":
    unittest.main()
