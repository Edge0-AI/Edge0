"""Tests for setup's UD-* start on a card of about 8 GB (0.1.44): the VRAM estimate, the plan that picks the least harmful
config that starts, and the words (UD-Q4 is slow here: pick IQ3_XXS / IQ3_S).  Measured on an 8 GB Arc A750: setup's
8192 / int8 / reserve 400 start stopped with "no VRAM is left for the expert cache, short 473 MiB".

    python -m unittest tools.test_setup_small_card
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_unsloth import Base  # noqa: E402


class Estimate(unittest.TestCase):
    def test_the_a750_measurement(self):
        # 8192 / int8 / reserve 400 was 473 MiB short; 4608 / q4_0 / reserve 0 started (without the draft layer's slack)
        self.assertAlmostEqual(setup.unsloth_vram_slack_mib(8.0, 8192, "int8", 400), -473, delta=2)
        self.assertGreaterEqual(setup.unsloth_vram_slack_mib(8.0, 4608, "q4_0", 0), 0)

    def test_a_bigger_card_has_room(self):
        self.assertGreater(setup.unsloth_vram_slack_mib(12.0, 32768, "int8", 700), setup.UD_FIT_MIN_SLACK_MIB)

    def test_cuda_and_hip_pay_for_the_runtime_sycl_does_not(self):
        self.assertEqual(setup.unsloth_runtime_mib({"driver": "i915"}), 0)
        self.assertEqual(setup.unsloth_runtime_mib({"driver": "xe"}), 0)
        self.assertEqual(setup.unsloth_runtime_mib({"driver": "580.97"}), setup.UD_FIT_RUNTIME_MIB)
        self.assertEqual(setup.unsloth_runtime_mib({"driver": "amdgpu"}), setup.UD_FIT_RUNTIME_MIB)


class Plan(unittest.TestCase):
    def test_a_card_that_fits_is_not_changed(self):
        self.assertEqual(setup.unsloth_small_card_plan(10.0, 8192, "fp16", None)[:3], (8192, "fp16", True))

    def test_8gb_drops_the_draft_layer_first(self):
        # nothing else frees enough on 8 GB: the context and the KV alone do not fit with the draft layer
        for kv in ("fp16", "int8"):
            plan = setup.unsloth_small_card_plan(8.0, 8192, kv, 300)
            self.assertIsNotNone(plan)
            self.assertFalse(plan[2])
            self.assertEqual(plan[:2], (8192, kv))        # the context and the KV the user would get are kept
            self.assertGreaterEqual(plan[3], setup.UD_FIT_MIN_SLACK_MIB)

    def test_a_card_just_short_changes_the_kv_before_the_context_before_the_draft(self):
        # ~8.9 GB: the 4-bit KV is enough with the draft layer
        vg = (setup.UD_FIT_FIXED_MIB + setup.UD_FIT_MTP_MIB + 8192 * 13 * 576 / 2 ** 20 + 300 + 80) / 1024
        plan = setup.unsloth_small_card_plan(vg, 8192, "fp16", 300)
        self.assertEqual(plan[:3], (8192, "q4_0", True))
        # ... and a smaller context when the user fixed the KV
        vg = (setup.UD_FIT_FIXED_MIB + setup.UD_FIT_MTP_MIB + 4608 * 13 * 2112 / 2 ** 20 + 300 + 80) / 1024
        plan = setup.unsloth_small_card_plan(vg, 8192, "fp16", 300, kv_fixed=True)
        self.assertEqual(plan[:3], (4608, "fp16", True))

    def test_what_the_user_chose_is_not_changed(self):
        plan = setup.unsloth_small_card_plan(8.0, 32768, "int8", 300, ctx_fixed=True, kv_fixed=True)
        self.assertEqual(plan[:2], (32768, "int8"))

    def test_a_6gb_card_has_no_plan(self):
        self.assertIsNone(setup.unsloth_small_card_plan(6.0, 8192, "int8", None))

    def test_a_reserve_the_user_gave_is_counted(self):
        a = setup.unsloth_small_card_plan(8.0, 8192, "int8", 0)
        b = setup.unsloth_small_card_plan(8.0, 8192, "int8", 1500)
        self.assertGreater(a[3], b[3] if b else -1e9)


class Words(unittest.TestCase):
    def test_always_says_to_pick_iq3(self):
        lines = setup.unsloth_small_card_lines("UD-Q4_K_XL", 8.0, (8192, "int8", False, 400.0), False)
        text = " ".join(lines)
        self.assertIn("very slow", text)
        self.assertIn("IQ3_XXS or IQ3_S", text)

    def test_no_plan_says_the_start_will_probably_stop(self):
        lines = setup.unsloth_small_card_lines("UD-Q4_K_XL", 6.0, None, False)
        self.assertTrue(any("will probably" in x for x in lines))


class Main8GB(Base):
    """setup.main() for UD-Q4_K_XL on an 8 GB NVIDIA card: the written config is the one that starts."""

    def arg(self, cfg, name):
        return cfg["args"][cfg["args"].index(name) + 1]

    def test_nvidia_8gb_writes_a_config_that_fits_and_says_so(self):
        code, out, cfg = self.main(["--context", "8192"], vram=8.0)
        self.assertEqual(code, 0, out)
        self.assertNotIn("--mtp", cfg["args"])               # no draft layer on the GPU
        self.assertEqual(self.arg(cfg, "--max-context"), "8192")   # the context the user asked for
        self.assertIn("would not start on this 8 GB card", out)
        self.assertIn("no MTP draft layer", out)
        self.assertIn("IQ3_XXS or IQ3_S", out)
        self.assertIn("very slow", out)

    def test_a_context_and_kv_the_user_chose_are_kept(self):
        code, out, cfg = self.main(["--context", "16384", "--kv", "int8"], vram=8.0)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.arg(cfg, "--max-context"), "16384")
        self.assertEqual(self.arg(cfg, "--kv"), "int8")

    def test_the_default_context_on_8gb_starts_without_asking(self):
        code, out, cfg = self.main([], vram=8.0)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.arg(cfg, "--max-context"), "8192")
        self.assertNotIn("--mtp", cfg["args"])

    def test_a_12gb_card_is_untouched(self):
        code, out, cfg = self.main(["--context", "8192"], vram=11.9)
        self.assertEqual(code, 0, out)
        self.assertIn("--mtp", cfg["args"])
        self.assertNotIn("would not start", out)
        self.assertNotIn("very slow", out)

    def test_amd_8gb(self):
        r = [{"index": 0, "name": "AMD Radeon RX 7600", "vram_gb": 8.0, "arch": "gfx1102", "driver": "amdgpu"}]
        with mock.patch.object(setup, "get_prebuilt_hip", lambda *a, **k: self.t / "engine"), \
                mock.patch.object(setup, "build_engine_hip", lambda *a, **k: self.t / "engine"),                 mock.patch.object(setup, "hip_card", lambda eng, gpu, listed: gpu):
            code, out, cfg = self.main(["--context", "8192", "--backend", "hip", "--vision", "no"], amd=r)
        self.assertEqual(code, 0, out)
        self.assertNotIn("--mtp", cfg["args"])
        self.assertIn("IQ3_XXS or IQ3_S", out)


if __name__ == "__main__":
    unittest.main()
