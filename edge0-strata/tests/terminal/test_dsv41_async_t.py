"""setup.dsv41_config_env: the engine env Strata writes for a unified-memory Ryzen AI Max (Strix Halo) DeepSeek install.

Measured 2026-10-25 on the 128 GB box (2.00 bpw, engine 0.1.51, greedy, 256-token sets): the engine's async window path is slower
than the synchronous one for the 6-row verify window - code 15.87-16.33 tok/s -> 20.14-21.28 (+31%), agent with the same tokens
a round 15.93 -> 18.48 (+16%).  DS4_ASYNC_T=4 keeps the drafter's T=1 windows async.  A discrete card keeps the engine default.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

import setup


HALO = [{"name": "AMD Radeon 8060S", "uma": True, "vram_gb": 96.0, "dedicated_gb": 64.0, "shared_gb": 32.0}]
R9700 = [{"name": "Radeon AI PRO R9700", "vram_gb": 32.0}]


class StrixAsyncEnv(unittest.TestCase):
    def test_a_unified_pc_gets_the_synchronous_verify_window(self):
        self.assertEqual(setup.dsv41_config_env(HALO), {"DS4_ASYNC_T": "4"})

    def test_a_discrete_card_keeps_the_engine_default(self):
        self.assertEqual(setup.dsv41_config_env(R9700), {})

    def test_no_card_is_untouched(self):
        self.assertEqual(setup.dsv41_config_env([]), {})

    def test_the_value_can_be_overridden(self):
        with mock.patch.dict(os.environ, {"STRATA_DSV41_ASYNC_T": "5"}):
            self.assertEqual(setup.dsv41_config_env(HALO), {"DS4_ASYNC_T": "5"})

    def test_eight_restores_the_engine_default(self):
        with mock.patch.dict(os.environ, {"STRATA_DSV41_ASYNC_T": "8"}):
            self.assertEqual(setup.dsv41_config_env(HALO), {})


if __name__ == "__main__":
    unittest.main()
