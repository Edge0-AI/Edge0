"""strata_control/amd_four_card.py: the serving profile of four Radeon AI PRO R9700 cards, and the places that apply it."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from strata_control import amd_four_card as F

R9700 = {"name": "AMD Radeon AI PRO R9700", "vram_gb": 31.9, "vendor": "amd", "usable": True}
CARDS = [{**R9700, "index": i} for i in range(4)]
CFG = {"model_name": "qwen3.8-flash-next-iq3_s", "backend": "hip", "gpu": [0, 1, 2, 3], "layer_split": "auto",
       "args": ["--pack", "/p/iq3_s", "--spec", "4", "--max-context", "32768"]}


def fresh(**kw):
    c = {**CFG, "args": list(CFG["args"]), **kw}
    return c


class Class(unittest.TestCase):
    def test_four_r9700_are_the_class(self):
        self.assertTrue(F.card_class(CARDS))

    def test_other_hardware_is_not(self):
        self.assertFalse(F.card_class(CARDS[:3]))                                   # three cards
        self.assertFalse(F.card_class(CARDS + [CARDS[0]]))                           # five
        self.assertFalse(F.card_class([{**c, "name": "AMD Radeon RX 7900 XTX", "vram_gb": 24.0} for c in CARDS]))
        self.assertFalse(F.card_class([{**c, "vendor": "nvidia"} for c in CARDS]))
        self.assertFalse(F.card_class([{**c, "vram_gb": 16.0} for c in CARDS]))      # a 16 GB card with the same name
        self.assertFalse(F.card_class([*CARDS[:3], {**R9700, "name": "AMD Radeon RX 9070"}]))

    def test_only_the_measured_model(self):
        for name in ("qwen3.8-flash-next-iq3_xxs", "qwen3.6-35b-a3b-q4", "deepseek-v4.1-flash-2.00bpw"):
            with self.subTest(name):
                self.assertFalse(F.applies(CARDS, fresh(model_name=name, args=["--pack", "/p/x"])))
        self.assertTrue(F.applies(CARDS, fresh()))
        self.assertFalse(F.applies(CARDS, fresh(backend="cuda")))
        self.assertFalse(F.applies(CARDS, fresh(arch="deepseek41")))
        self.assertFalse(F.applies(CARDS, fresh(args=CFG["args"] + ["--peer-device", "1"])))


class Apply(unittest.TestCase):
    def test_fills_the_profile_on_a_default_config(self):
        c = fresh()
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in list(F.BATCH_ENV) + ["STRATA_FOUR_CARD_PROFILE"]:
                os.environ.pop(k, None)
            notes = F.apply(CARDS, c)
        self.assertTrue(notes)
        self.assertEqual(c["layer_split"], "13,26,39")
        self.assertEqual(c["parallel"], 32)
        self.assertEqual(c["args"][c["args"].index("--batch-groups") + 1], "4")
        self.assertEqual(c["args"][c["args"].index("--prefill") + 1], "2048")
        self.assertEqual(c["env"], F.BATCH_ENV)
        self.assertNotIn("--prompt-cache", c["args"])

    def test_prefill_auto_from_setup_is_not_a_choice(self):
        c = fresh(args=CFG["args"] + ["--prefill", "auto"])
        F.apply(CARDS, c)
        self.assertEqual(c["args"][c["args"].index("--prefill") + 1], "2048")
        self.assertEqual(c["args"].count("--prefill"), 1)
        c = fresh(args=CFG["args"] + ["--prefill", "auto:32768"])
        F.apply(CARDS, c)
        self.assertEqual(c["args"][c["args"].index("--prefill") + 1], "auto:32768")

    def test_the_persons_choices_win(self):
        c = fresh(layer_split="12,25,37", parallel=8, args=CFG["args"] + ["--prefill", "8192", "--batch-groups", "2"],
                  env={"STRATA_QSA_SLOTS": "0"})
        F.apply(CARDS, c)
        self.assertEqual(c["layer_split"], "12,25,37")
        self.assertEqual(c["parallel"], 8)
        self.assertEqual(c["args"][c["args"].index("--prefill") + 1], "8192")
        self.assertEqual(c["args"].count("--batch-groups"), 1)
        self.assertEqual(c["env"]["STRATA_QSA_SLOTS"], "0")

    def test_a_number_of_requests_asked_for_keeps_the_groups_to_the_engine(self):
        c = fresh()
        F.apply(CARDS, c, parallel_given=True)
        self.assertNotIn("parallel", c)
        self.assertNotIn("--batch-groups", c["args"])
        self.assertEqual(c["layer_split"], "13,26,39")

    def test_an_explicit_batch_in_the_args_is_kept(self):
        c = fresh(args=CFG["args"] + ["--batch", "6"])
        F.apply(CARDS, c)
        self.assertNotIn("parallel", c)
        self.assertNotIn("--batch-groups", c["args"])

    def test_an_environment_variable_of_the_persons_own_wins(self):
        c = fresh()
        with mock.patch.dict(os.environ, {"STRATA_BATCH_SELF_COMMIT": "0"}):
            F.apply(CARDS, c)
        self.assertNotIn("STRATA_BATCH_SELF_COMMIT", c["env"])
        self.assertEqual(c["env"]["STRATA_QSA_SLOTS"], "1")

    def test_off_switches(self):
        c = fresh()
        with mock.patch.dict(os.environ, {"STRATA_FOUR_CARD_PROFILE": "0"}):
            self.assertEqual(F.apply(CARDS, c), [])
        self.assertEqual(c, fresh())
        c = fresh(four_card_profile=False)
        self.assertEqual(F.apply(CARDS, c), [])
        self.assertNotIn("parallel", c)

    def test_other_hardware_is_untouched(self):
        for cards in (CARDS[:2], [{**x, "name": "NVIDIA RTX A4000", "vendor": "nvidia"} for x in CARDS]):
            c = fresh()
            self.assertEqual(F.apply(cards, c), [])
            self.assertEqual(c, fresh())


class Launcher(unittest.TestCase):
    def test_the_runner_fills_the_profile_and_the_profile_override(self):
        from strata_cli.launch import runner as R, settings as S
        from tests.terminal.launch_support import HW_12, Home, catalog
        hw = {**HW_12, "gpus": [{**x, "usable": True} for x in CARDS]}
        with Home():
            r = R.Runner(S.LaunchSettings(), catalog=catalog(hw))
            cfg = fresh()
            r.apply_four_card(cfg)
            self.assertEqual(cfg["layer_split"], "13,26,39")
            self.assertEqual(r.four_card_parallel, 32)
            r2 = R.Runner(S.LaunchSettings(parallel=3), catalog=catalog(hw))
            cfg2 = fresh()
            r2.apply_four_card(cfg2)
            self.assertNotIn("parallel", cfg2)
            self.assertIsNone(r2.four_card_parallel)

    def test_the_runner_leaves_other_computers_alone(self):
        from strata_cli.launch import runner as R, settings as S
        from tests.terminal.launch_support import HW_2X16, Home, catalog
        with Home():
            r = R.Runner(S.LaunchSettings(), catalog=catalog(HW_2X16))
            cfg = {**fresh(), "gpu": [0, 1]}
            r.apply_four_card(cfg)
            self.assertEqual(cfg, {**fresh(), "gpu": [0, 1]})
            self.assertIsNone(r.four_card_parallel)



class ServerMessage(unittest.TestCase):
    def test_more_than_eight_slots_on_a_layer_split_is_not_cut_in_the_message(self):
        import contextlib
        import io
        from serve import server as S
        split = {"parallel": 32, "gpu": [0, 1, 2, 3], "layer_split": "13,26,39", "args": ["--pack", "/p"]}
        one = {"parallel": 32, "gpu": 0, "args": ["--pack", "/p"]}
        for cfg, said in ((split, False), (one, True)):
            with self.subTest(gpus=len(S.gpu_list(cfg))):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(S.parallel_args(cfg, cfg["args"]), ["--batch", "32"])
                self.assertEqual("runs at most" in out.getvalue(), said)

if __name__ == "__main__":
    unittest.main()
