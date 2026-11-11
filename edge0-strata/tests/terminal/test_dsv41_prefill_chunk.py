"""setup.dsv41_prefill_chunk_args: the 8,192-token prompt chunk Strata writes for a unified-memory Ryzen AI Max (Strix Halo) PC.

Measured on the 128 GB box (2.00 bpw, engine 0.1.51, 2026-10-25): 4K prompt 78.7 -> 87.2 tok/s, 8K 108.9 -> 125.1, 32K 155.2 ->
176.9, decode unchanged (16.53 -> 16.59).  The engine's own 1,024-token rule was calibrated on cards that hold most of the experts;
on the shared pool the bigger chunk wins.  A discrete card or a small unified PC keeps the engine's choice.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

import setup


HALO = [{"name": "AMD Radeon 8060S", "uma": True, "vram_gb": 96.0, "dedicated_gb": 64.0, "shared_gb": 32.0}]
SMALL_HALO = [{"name": "AMD Radeon 8040S", "uma": True, "vram_gb": 32.0, "dedicated_gb": 32.0, "shared_gb": 16.0}]
R9700 = [{"name": "Radeon AI PRO R9700", "vram_gb": 32.0}]


class StrixChunkArgs(unittest.TestCase):
    def test_a_128_gb_unified_pc_gets_the_16k_chunk(self):
        # (2026-10-26, the same box: 16K prompt 152.2 -> 158.6 tok/s, 32K 174.0 -> 179.7, 8K flat, decode unchanged)
        self.assertEqual(setup.dsv41_prefill_chunk_args(HALO, 63.6), ["--prefill-chunk", "16384"])

    def test_a_96_gb_pool_is_the_line(self):
        self.assertEqual(setup.dsv41_prefill_chunk_args(HALO, 31.0), ["--prefill-chunk", "8192"])

    def test_a_64_gb_unified_pc_keeps_the_engines_choice(self):
        self.assertEqual(setup.dsv41_prefill_chunk_args(SMALL_HALO, 31.6), [])

    def test_a_discrete_card_is_untouched(self):
        self.assertEqual(setup.dsv41_prefill_chunk_args(R9700, 251.0), [])

    def test_no_card_is_untouched(self):
        self.assertEqual(setup.dsv41_prefill_chunk_args([], 128.0), [])

    def test_the_escape_hatch_turns_it_off(self):
        with mock.patch.dict(os.environ, {"STRATA_DSV41_NO_BIG_CHUNK": "1"}):
            self.assertEqual(setup.dsv41_prefill_chunk_args(HALO, 63.6), [])


if __name__ == "__main__":
    unittest.main()
