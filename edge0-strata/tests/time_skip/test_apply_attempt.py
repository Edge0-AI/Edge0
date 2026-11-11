"""Applying a revision attempt the person chose after review (a run with no deciding checks keeps the baseline)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strata_control import time_skip as ts  # noqa: E402

from test_time_skip import TimeSkipCase  # noqa: E402


class TestApplyChosenAttempt(TimeSkipCase):
    def test_a_reviewed_attempt_can_be_applied_and_undone(self):
        def revise(candidate: Path, prompt: str, index: int) -> str:
            (candidate / "doc.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            return "added three"

        run, result = self.start(revise, checks=[])
        self.assertTrue(result["decision"]["needs_user_review"])
        self.assertEqual((self.ws / "doc.txt").read_text(encoding="utf-8"), "one\ntwo\n")
        outcome = run.apply(attempt=1)
        self.assertTrue(outcome["applied"], outcome)
        self.assertEqual((self.ws / "doc.txt").read_text(encoding="utf-8"), "one\ntwo\nthree\n")
        run.restore_original()
        self.assertEqual((self.ws / "doc.txt").read_text(encoding="utf-8"), "one\ntwo\n")

    def test_a_missing_attempt_is_refused(self):
        run, _ = self.start(lambda c, p, i: None, checks=[])
        with self.assertRaises(ts.TimeSkipError):
            run.apply(attempt=9)

    def test_a_reviewed_attempt_is_still_refused_after_a_user_edit(self):
        def revise(candidate: Path, prompt: str, index: int) -> str:
            (candidate / "doc.txt").write_text("changed\n", encoding="utf-8")
            return "x"

        run, _ = self.start(revise, checks=[])
        (self.ws / "doc.txt").write_text("the user rewrote this\n", encoding="utf-8")
        outcome = run.apply(attempt=1)
        self.assertTrue(outcome["conflict"])
        self.assertEqual((self.ws / "doc.txt").read_text(encoding="utf-8"), "the user rewrote this\n")


if __name__ == "__main__":
    unittest.main()
