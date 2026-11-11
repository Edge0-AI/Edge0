"""#465: setup's recommendation for "parallel" (several requests decoded together in the engine's batch slots).
No GPU or download needed: python tools/test_setup_parallel.py"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import setup as S  # noqa: E402


class Parallel(unittest.TestCase):
    def test_slot_size(self):
        # measured: a slot's session at 32K, int8 KV: 0.56 GiB = 0.60 GB
        self.assertAlmostEqual(S.parallel_slot_gb(32768, "int8", False), 0.59, delta=0.03)
        # KV streaming keeps only the attention's 32K positions in VRAM, whatever the context
        self.assertEqual(S.parallel_slot_gb(262144, "int8", True), S.parallel_slot_gb(32768, "int8", False))
        self.assertLess(S.parallel_slot_gb(32768, "q4_0", False), S.parallel_slot_gb(32768, "int8", False))

    def test_recommendation_only_where_the_experts_mostly_fit(self):
        q2, iq3s = S.MODELS["Q2_0"]["arena_gb"], S.MODELS["IQ3_S"]["arena_gb"]
        # most experts on the CPU: slots cost speed per request (measured on the 12 GB 5070) - none recommended
        self.assertEqual(S.parallel_recommend(12, q2, 32768, "int8", False), 0)
        self.assertEqual(S.parallel_recommend(16, q2, 32768, "int8", False), 0)
        self.assertEqual(S.parallel_recommend(24, iq3s, 32768, "int8", False), 0)
        # the cache holds most of the experts beside the slots: recommended
        self.assertEqual(S.parallel_recommend(24, q2, 32768, "int8", False), 3)
        self.assertEqual(S.parallel_recommend(48, q2, 32768, "int8", False), 4)       # capped at 4
        self.assertEqual(S.parallel_recommend([24, 24], iq3s, 32768, "int8", False), 4)   # a layer split: every card
        self.assertEqual(S.parallel_recommend([16, 16], q2, 32768, "int8", False), 4)
        self.assertEqual(S.parallel_recommend(32, q2, 262144, "int8", False), 0)      # a 262K KV per slot: no
        self.assertEqual(S.parallel_recommend(32, q2, 262144, "int8", True), 4)       # ... unless it streams

    def test_notes_recommend_never_force(self):
        q2 = S.MODELS["Q2_0"]["arena_gb"]
        # 12 GB: left at one, with the reason
        note = S.parallel_note(None, 12, q2, 32768, "int8", False)
        self.assertEqual(len(note), 1)
        self.assertIn("one at a time", note[0])
        self.assertIn("costs about 10-25% speed per request on this card", note[0])
        self.assertIn("--parallel 3", S.parallel_note(None, 24, q2, 32768, "int8", False)[0])
        # asked on 12 GB: kept, with the cost said
        lines = S.parallel_note(4, 12, q2, 32768, "int8", False)
        self.assertIn("4 at once", lines[0])
        self.assertTrue(any("one at a time" in x and "kept as you chose" in x and "10-25%" in x for x in lines))
        # asked above the recommendation where it pays
        lines = S.parallel_note(4, 24, q2, 32768, "int8", False)
        self.assertTrue(any("recommended for this card: 3" in x for x in lines))
        self.assertTrue(any("at most 8" in x for x in S.parallel_note(12, 48, q2, 32768, "int8", False)))
        self.assertEqual(len(S.parallel_note(3, 24, q2, 32768, "int8", False)), 1)   # as recommended: no warning


if __name__ == "__main__":
    unittest.main()
