"""Step 1's PCIe line: the generation card and board both run, the lanes, and a warning when the card runs on fewer
lanes than it has.  No GPU needed: nvidia-smi's answer is given.

    python -m unittest tools.test_setup_pcie
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class Link(unittest.TestCase):
    def test_parse(self):
        with mock.patch.object(setup, "out", return_value="3, 4, 3, 8, 8\n"):
            self.assertEqual(setup.pcie_link(0), {"gen": 3, "gpu_gen": 4, "host_gen": 3, "width": 8, "max_width": 8})
        for bad in ("", "[N/A], 4, 3, 8, 8\n", "garbage"):
            with mock.patch.object(setup, "out", return_value=bad):
                self.assertIsNone(setup.pcie_link(0), bad)

    def test_board_limit_and_slow_link(self):   # llmserver: RTX 4060 Ti (4.0 x8) on an X99 board (3.0)
        line, warn = setup.pcie_lines({"gen": 3, "gpu_gen": 4, "host_gen": 3, "width": 8, "max_width": 8})
        self.assertIn("PCIe: 3.0 x8 (the card supports 4.0, the board 3.0)", line)
        self.assertIn("~8 GB/s", line)
        self.assertIsNone(warn)

    def test_fast_link_says_nothing_more(self):
        line, warn = setup.pcie_lines({"gen": 4, "gpu_gen": 4, "host_gen": 5, "width": 16, "max_width": 16})
        self.assertEqual(line, "PCIe: 4.0 x16")
        self.assertIsNone(warn)

    def test_fewer_lanes_warns(self):
        _, warn = setup.pcie_lines({"gen": 3, "gpu_gen": 3, "host_gen": 3, "width": 4, "max_width": 16})
        self.assertIn("reads 4 of its 16 lanes right now", warn)
        self.assertIn("narrow it when idle", warn)             # the width is read at idle: a hint, not a verdict


if __name__ == "__main__":
    unittest.main()
