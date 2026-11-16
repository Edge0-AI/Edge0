"""Tests for tools/draft_vocab.py's corpus coverage mode (#597): the most frequent tokens of a corpus that together cover
a given share of its occurrences.  No model, no GPU.

    python -m unittest tools.test_draft_vocab
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import draft_vocab as DV  # noqa: E402


class Coverage(unittest.TestCase):
    def test_most_frequent_first_until_the_share_is_covered(self):
        c = Counter({7: 50, 3: 30, 9: 15, 4: 5})              # 100 occurrences
        self.assertEqual(DV.covering(c, 0.5), [7])
        self.assertEqual(DV.covering(c, 0.8), [7, 3])
        self.assertEqual(DV.covering(c, 0.81), [7, 3, 9])
        self.assertEqual(DV.covering(c, 1.0), [7, 3, 9, 4])

    def test_ties_take_the_lower_id_first(self):
        self.assertEqual(DV.covering(Counter({5: 10, 2: 10, 8: 10}), 0.5), [2, 5])

    def test_share_outside_a_subset(self):
        c = Counter({1: 60, 2: 30, 3: 10})
        self.assertAlmostEqual(DV.outside(c, [1]), 0.4)
        self.assertAlmostEqual(DV.outside(c, [1, 2, 3]), 0.0)
        self.assertEqual(DV.outside(Counter(), [1]), 0.0)

    def test_corpus_counts_reads_every_file(self):
        class Tok:                                            # one id per character
            def encode(self, text):
                return [ord(ch) for ch in text]
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.txt", Path(d) / "b.txt"
            a.write_text("été", encoding="utf-8")
            b.write_text("eté", encoding="utf-8")
            self.assertEqual(DV.corpus_counts(Tok(), [a, b]), Counter({ord("é"): 3, ord("t"): 2, ord("e"): 1}))


if __name__ == "__main__":
    unittest.main()
