"""Tests for setup.py's automatic rope configuration (§4 + §6): the factor derives from the FINAL context
actually served - RAM reductions first, then final / trained 262144 clamped at 1 - explicit
--rope-scaling/--rope-scale choices win verbatim, an explicit none is refused past the trained range, and
nothing turns on by itself inside it.  Pure functions - no GPU, no downloads, no prompts.

    python -m unittest tools.test_setup_rope
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class DerivedFactor(unittest.TestCase):
    """The formula itself: max(1.0, final / trained)."""

    def test_the_task_table(self):
        self.assertEqual(setup.derived_factor(131072), 1.0)        # below the trained range: clamped at 1
        self.assertEqual(setup.derived_factor(262144), 1.0)        # exactly the trained range: 1
        self.assertEqual(setup.derived_factor(393216), 1.5)        # 384K
        self.assertEqual(setup.derived_factor(524288), 2.0)        # 512K

    def test_free_form_contexts_get_the_exact_ratio(self):
        # /262144 = /2**18 is exact in binary floating point, so no dust
        self.assertEqual(setup.derived_factor(600000), 600000 / 262144)
        self.assertGreater(setup.derived_factor(600000), 2.0)

    def test_the_clamp_is_what_keeps_128k_at_one(self):
        self.assertEqual(setup.derived_factor(131072), 1.0)        # NOT the raw 0.5


class ResolveRope(unittest.TestCase):
    def test_invalid_explicit_factors_are_rejected(self):
        for ctx in (131072, 262144, 1048576):
            for method in (None, "none", "linear", "yarn"):
                for scale in (float("nan"), float("inf"), float("-inf"), -1.0, 0.0, 0.5):
                    with self.subTest(ctx=ctx, method=method, scale=scale):
                        with self.assertRaisesRegex(ValueError, r"--rope-scale.*finite.*>= 1"):
                            setup.resolve_rope(ctx, method, scale)

    def test_valid_explicit_factors_remain_verbatim(self):
        for ctx in (131072, 262144, 1048576):
            for method in ("linear", "yarn"):
                for scale in (1.0, 1.5, 4.0):
                    with self.subTest(ctx=ctx, method=method, scale=scale):
                        self.assertEqual(setup.resolve_rope(ctx, method, scale), (method, scale))

    # ---- the automatic flow the docs promise (an omitted flag, --yes or the interactive default)
    def test_omitted_384k_gets_yarn_and_1_5(self):
        self.assertEqual(setup.resolve_rope(393216, None, None), ("yarn", 1.5))

    def test_omitted_512k_gets_yarn_and_2(self):
        self.assertEqual(setup.resolve_rope(524288, None, None), ("yarn", 2.0))

    def test_deterministic_for_yes(self):
        # --yes never prompts, so the resolution must be a pure function of its inputs
        for _ in range(3):
            self.assertEqual(setup.resolve_rope(524288, None, None), ("yarn", 2.0))

    def test_omitted_method_with_an_explicit_factor(self):
        # --rope-scale alone still expresses the intent to extend: the method defaults, the factor is kept
        self.assertEqual(setup.resolve_rope(393216, None, 1.8), ("yarn", 1.8))

    # ---- explicit selections are respected, verbatim
    def test_explicit_method_gets_the_derived_factor(self):
        self.assertEqual(setup.resolve_rope(524288, "linear", None), ("linear", 2.0))

    def test_explicit_method_and_factor_are_kept_verbatim(self):
        self.assertEqual(setup.resolve_rope(393216, "linear", 2.0), ("linear", 2.0))
        self.assertEqual(setup.resolve_rope(524288, "yarn", 1.5), ("yarn", 1.5))

    def test_an_explicit_override_survives_a_context_reduction(self):
        # §4: the RAM check reduced the final context to 128K - a user-supplied factor is NOT recalculated
        self.assertEqual(setup.resolve_rope(131072, "yarn", 2.0), ("yarn", 2.0))
        self.assertEqual(setup.resolve_rope(131072, "linear", 1.75), ("linear", 1.75))

    # ---- an explicit none past the trained range is refused with an explanation, not overridden
    def test_explicit_none_past_trained_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            setup.resolve_rope(524288, "none", None)
        self.assertIn("262144", str(cm.exception))
        self.assertIn("yarn", str(cm.exception))                   # the message names the way out

    def test_explicit_none_within_trained_is_the_stock_model(self):
        self.assertEqual(setup.resolve_rope(131072, "none", None), (None, None))

    # ---- inside the trained range nothing turns on by itself; a chosen method gets factor 1
    def test_omitted_within_trained_adds_nothing(self):
        self.assertEqual(setup.resolve_rope(262144, None, None), (None, None))
        self.assertEqual(setup.resolve_rope(131072, None, None), (None, None))

    def test_a_scale_alone_within_trained_is_rejected(self):
        with self.assertRaises(ValueError):
            setup.resolve_rope(131072, None, 2.0)

    def test_a_chosen_method_within_trained_gets_factor_1(self):
        # no more unconditional factor 2: 1 = the trained angles, no automatic expansion
        self.assertEqual(setup.resolve_rope(131072, "yarn", None), ("yarn", 1.0))
        self.assertEqual(setup.resolve_rope(262144, "linear", None), ("linear", 1.0))

    # ---- §4 coordination: the RAM reduction lands BEFORE the rope config, so the factor matches the
    # context actually served - the 512K-on-IQ3_S-with-under-90-GB case from the task, end to end
    def test_512k_on_a_small_ram_machine_resolves_to_128k_factor_1(self):
        # asked 524288 with --rope-scaling yarn and no factor; the RAM check brings the final context to 131072
        self.assertEqual(setup.resolve_rope(131072, "yarn", None), ("yarn", 1.0))

    def test_no_reduction_384k_and_512k_resolve_as_documented(self):
        self.assertEqual(setup.resolve_rope(393216, "yarn", None), ("yarn", 1.5))
        self.assertEqual(setup.resolve_rope(524288, "yarn", None), ("yarn", 2.0))

    # ---- the short-context item's own table: no factor-2 fallback inside the trained window, for either
    # method, with the explicit override column kept verbatim
    def test_the_short_context_table(self):
        cases = [
            (131072, None, "yarn", 1.0),                   # --context 131072 --rope-scaling yarn
            (262144, None, "yarn", 1.0),                   # --context 262144 --rope-scaling yarn
            (393216, None, "yarn", 1.5),
            (524288, None, "yarn", 2.0),
            (131072, 2.0, "yarn", 2.0),                    # --rope-scale 2 is kept inside the window
        ]
        for ctx, scale, method, want in cases:
            self.assertEqual(setup.resolve_rope(ctx, method, scale), (method, want), f"ctx {ctx}")

    def test_linear_gets_the_same_rule_as_yarn(self):
        # the old fallback applied to both methods, so the fix must too
        self.assertEqual(setup.resolve_rope(131072, "linear", None), ("linear", 1.0))
        self.assertEqual(setup.resolve_rope(393216, "linear", None), ("linear", 1.5))


if __name__ == "__main__":
    unittest.main()
