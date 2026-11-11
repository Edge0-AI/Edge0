"""#4662: a compiled engine gets code for every card Strata can use in the PC, not only the card setup picked.

    python -m unittest tools.test_setup_build_archs
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def card(index, arch, vram=16.0):
    return {"index": index, "arch": arch, "vram_gb": vram, "name": f"card{index}"}


class BuildArchs(unittest.TestCase):
    def test_a_mixed_pair_gets_both_architectures(self):
        found = [card(0, 75, 24), card(1, 86, 12)]
        self.assertEqual(setup.local_build_archs([75], found), [75, 86])

    def test_one_card_stays_one_architecture(self):
        self.assertEqual(setup.local_build_archs([89], [card(0, 89, 24)]), [89])

    def test_a_card_older_than_sm75_is_left_out(self):
        self.assertEqual(setup.local_build_archs([86], [card(0, 86), card(1, 61)]), [86])

    def test_no_card_answer_keeps_the_given_ones(self):
        self.assertEqual(setup.local_build_archs([75, 86], []), [75, 86])
        self.assertEqual(setup.local_build_archs([86], None), [86])
        self.assertEqual(setup.local_build_archs([86], [{"index": 0}]), [86])


if __name__ == "__main__":
    unittest.main()
