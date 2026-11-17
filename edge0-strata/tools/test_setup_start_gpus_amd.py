"""#1594: --gpus when starting an installed model on a PC with AMD cards and no NVIDIA ones names the AMD cards (as
setup lists them) instead of failing for want of an NVIDIA GPU.

    python -m unittest tools.test_setup_start_gpus_amd -v
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup  # noqa: E402


def amd(index, arch="gfx1201"):
    return {"index": index, "name": f"Radeon {index}", "arch": arch, "vram_gb": 32.0}


class StartGpusAmd(unittest.TestCase):
    def run_start_gpus(self, text, cards, nvidia=()):
        with mock.patch.object(setup, "WIN", False), mock.patch.object(setup, "gpus", return_value=list(nvidia)), \
                mock.patch.object(setup, "amd_gpus", return_value=cards):
            return setup.start_gpus(text)

    def test_amd_numbers_are_taken_as_setup_lists_them(self):
        self.assertEqual(self.run_start_gpus("1,0", [amd(0), amd(1)]), [1, 0])

    def test_all_still_means_every_supported_amd_card(self):
        self.assertEqual(sorted(self.run_start_gpus("all", [amd(0), amd(1)])), [0, 1])

    def test_a_card_that_cannot_be_used_is_named(self):
        with self.assertRaises(SystemExit):
            self.run_start_gpus("1,0", [amd(0), amd(1, "gfx1010")])

    def test_nothing_named_is_none(self):
        self.assertIsNone(self.run_start_gpus("", [amd(0)]))


if __name__ == "__main__":
    unittest.main()
