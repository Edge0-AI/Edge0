"""A download with no data for two minutes says so (#3567).

    python -m unittest tools.test_setup_stall_note
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class Stall(unittest.TestCase):
    def test_silent_while_bytes_arrive_and_for_the_first_two_minutes(self):
        s = setup.StallNote(0.0)
        self.assertIsNone(s.check(10, 100, 1.0))
        self.assertIsNone(s.check(20, 100, 200.0))                 # moving
        self.assertIsNone(s.check(20, 100, 300.0))                 # 100 s idle

    def test_says_it_once_a_minute_and_stops_when_data_comes(self):
        s = setup.StallNote(0.0)
        s.check(69, 100, 1.0)
        said = s.check(69, 100, 130.0)
        self.assertIn("no data for 2 min at 69%", said)
        self.assertIn("run the same command again", said)
        self.assertIsNone(s.check(69, 100, 150.0))
        self.assertIn("min at 69%", s.check(69, 100, 200.0))
        self.assertIsNone(s.check(70, 100, 201.0))
        self.assertIsNone(s.check(70, 100, 250.0))


if __name__ == "__main__":
    unittest.main()
