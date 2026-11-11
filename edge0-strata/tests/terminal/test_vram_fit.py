"""strata_control/vram_fit.py: the fit math with mock card sizes, and its use in setup (sycl/setup_intel.py) and strata launch."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strata_control import vram_fit as V  # noqa: E402


class Measured(unittest.TestCase):
    """The estimate against the 12 starts on the Arc A750 (8 GB): engine slots / MiB short, from /data/022-arcgyro/out/fa_*.txt."""

    def slack(self, ctx, reserve, prefill):
        return V.slack_mib("Gyro-M", 8.0, ctx, "int8", reserve, prefill, "intel")

    def test_the_session_matches_the_engines_three_numbers(self):
        for ctx, mib in ((8192, 234), (16384, 347), (32768, 573)):
            self.assertAlmostEqual(V.session_mib(ctx), mib, delta=8)

    def test_the_default_does_not_start_and_the_estimate_says_how_short(self):
        self.assertAlmostEqual(self.slack(32768, 300, "auto"), -747, delta=40)     # engine: about 748 MiB more
        self.assertAlmostEqual(self.slack(16384, 300, "auto"), -497, delta=40)     # 497
        self.assertAlmostEqual(self.slack(8192, 100, "auto"), -171, delta=40)      # 171

    def test_the_known_good_start_has_room_for_its_slots(self):
        self.assertAlmostEqual(self.slack(8192, 100, 1024), 412 * V.SLOT_MIB, delta=60)    # 412 slots
        self.assertAlmostEqual(self.slack(32768, 100, 1024), 107 * V.SLOT_MIB, delta=60)   # 107 slots
        self.assertAlmostEqual(self.slack(8192, 100, 512), 683 * V.SLOT_MIB, delta=60)     # 683 slots
        self.assertAlmostEqual(self.slack(8192, 100, 256), 820 * V.SLOT_MIB, delta=60)     # 820 slots

    def test_gyro_s_default_fits_as_it_did(self):
        self.assertTrue(V.fits("Gyro-S", 8.0, 32768, "int8", 300, "auto", "intel"))        # 410 slots measured


class Plan(unittest.TestCase):
    def test_gyro_m_on_8gb_is_lowered_until_it_fits_and_says_so(self):
        f = V.plan("Gyro-M", 8.0, 32768, "int8", 300, "auto", "intel", title="Gyro-M")
        self.assertTrue(f.changed and f.fits)
        self.assertEqual((f.ctx, f.reserve_mib, f.prefill), (16384, 100, "1024"))
        self.assertTrue(f.line.startswith("Gyro-M on this 8 GB card: context set to 16K"), f.line)
        self.assertTrue(f.line.endswith("so it fits"))
        self.assertGreaterEqual(f.slack_mib, V.needed_mib())

    def test_a_chosen_context_is_kept_and_the_rest_is_lowered(self):
        f = V.plan("Gyro-M", 8.0, 32768, "int8", 300, "auto", "intel", ctx_fixed=True)
        self.assertEqual(f.ctx, 32768)
        self.assertFalse(f.fits)                           # 32K cannot fit even with 100 / 512: said, not refused
        self.assertIn("still tight", f.line)

    def test_a_chosen_reserve_is_kept(self):
        f = V.plan("Gyro-M", 8.0, 32768, "int8", 500, "auto", "intel", reserve_fixed=True)
        self.assertEqual(f.reserve_mib, 500)
        self.assertLess(f.ctx, 32768)

    def test_a_chosen_chunk_is_kept(self):
        f = V.plan("Gyro-M", 8.0, 32768, "int8", 300, "2048", "intel", prefill_fixed=True)
        self.assertIsNone(f.prefill)

    def test_the_smallest_start_is_returned_when_nothing_fits_never_a_refusal(self):
        f = V.plan("Gyro-M", 6.0, 32768, "int8", 300, "auto", "intel")
        self.assertEqual(f.ctx, 8192)
        self.assertFalse(f.fits)
        self.assertTrue(f.line)

    def test_only_the_ladder_is_used(self):
        for gb in (7.0, 7.5, 8.0, 8.5, 9.0, 10.0):
            f = V.plan("Gyro-M", gb, 32768, "int8", 300, "auto", "intel")
            self.assertIn(f.ctx, (32768, 16384, 8192))

    def test_a_config_that_fits_comes_back_unchanged(self):
        for model, gb, ctx in (("Gyro-M", 12.0, 32768), ("Gyro-M", 16.0, 65536), ("Gyro-S", 8.0, 32768), ("Gyro-S", 12.0, 131072),
                               ("Qwen3.8-Flash-Next-IQ3_XXS", 8.0, 32768), ("Qwen3.8-Flash-Next-IQ3_XXS", 12.0, 131072),
                               ("Qwen3.8-Flash-Next-IQ3_XXS", 16.0, 262144), ("Qwen3.8-Flash-Next-IQ2_XS", 24.0, 262144)):
            f = V.plan(model, gb, ctx, "int8", 300, "auto", "nvidia")
            self.assertEqual((f.ctx, f.reserve_mib, f.prefill, f.changed, f.line), (ctx, 300, None, False, ""), (model, gb, ctx))

    def test_unknown_card_size_and_tiny_cards_are_left_alone(self):
        for gb in (0, None, 4.0):
            f = V.plan("Gyro-M", gb, 32768, "int8", 300, "auto", "intel")
            self.assertFalse(f.changed)

    def test_fit_context_is_the_ladder_step(self):
        self.assertEqual(V.fit_context("Gyro-M", 8.0, 32768), 16384)
        self.assertEqual(V.fit_context("Gyro-M", 12.0, 32768), 32768)
        self.assertEqual(V.fit_context("Gyro-M", 8.0, 8192), 8192)

    def test_generic_model_is_found_short_only_when_far_short(self):
        self.assertFalse(V.plan("Qwen3.8-Flash-Next-IQ3_XXS", 8.0, 32768, "int8", 300, "auto", "intel").changed)
        self.assertTrue(V.plan("Qwen3.8-Flash-Next-IQ3_XXS", 6.0, 131072, "int8", 300, "auto", "intel").changed)


class SetupIntel(unittest.TestCase):
    @staticmethod
    def mod():
        spec = importlib.util.spec_from_file_location("setup_intel_fit", ROOT / "sycl" / "setup_intel.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def cfg(self, m, native="Gyro-M-TQ2_0.gguf", extra=()):
        base = m.MOUNT
        return {"exe": "x", "cwd": "y", "args": ["--pack", str(base / "d" / "pack"), "--native", str(base / "d" / native),
                                                 "--expert-cache", "auto", "--prefill", "auto", "--max-context", "32768",
                                                 "--kv", "int8", "--vram-reserve-mib", "300", *extra]}

    def val(self, c, flag):
        return c["args"][c["args"].index(flag) + 1]

    def run_it(self, m, cfg, vram, drv="i915", **kw):
        notes = []
        c = m.to_sycl(cfg, m.ROOT / "build-sycl" / "strata", 64.0, {}, vram, drv, "56a1", fit_notes=notes, **kw)
        return c, notes

    def test_gyro_m_on_the_a750_gets_a_config_that_starts_and_one_line(self):
        m = self.mod()
        c, notes = self.run_it(m, self.cfg(m), 8.0, auto_reserve=True)
        self.assertEqual(self.val(c, "--max-context"), "16384")
        self.assertEqual(self.val(c, "--vram-reserve-mib"), "100")
        self.assertEqual(self.val(c, "--prefill"), "1024")
        self.assertEqual(c["args"].count("--vram-reserve-mib"), 1)
        self.assertEqual(len(notes), 1)
        self.assertIn("Gyro-M on this 8 GB card: context set to 16K", notes[0])

    def test_a_reserve_the_user_asked_for_is_kept(self):
        m = self.mod()
        c, notes = self.run_it(m, self.cfg(m), 8.0, auto_reserve=False)
        self.assertEqual(self.val(c, "--vram-reserve-mib"), "300")
        self.assertIn(self.val(c, "--max-context"), ("16384", "8192"))

    def test_a_context_the_user_asked_for_is_kept(self):
        m = self.mod()
        c, notes = self.run_it(m, self.cfg(m), 8.0, auto_reserve=True, ctx_asked=True)
        self.assertEqual(self.val(c, "--max-context"), "32768")

    def test_configs_that_fit_are_byte_identical(self):
        m = self.mod()
        for vram, drv in ((8.0, "i915"), (12.0, "xe"), (16.0, "i915"), (24.0, "xe"), (32.0, "xe")):
            for native in ("a.gguf", "Gyro-S-TQ1_0.gguf"):
                off, _ = self.run_it(m, self.cfg(m, native), vram, drv, fit=False)
                on, notes = self.run_it(m, self.cfg(m, native), vram, drv, auto_reserve=True)
                self.assertEqual(off, on, (vram, drv, native))
                self.assertEqual(notes, [])


if __name__ == "__main__":
    unittest.main()
