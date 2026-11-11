"""atomic.replace: a file another program is reading is waited for, not failed on (Windows "Access is denied")."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strata_control import atomic  # noqa: E402


class TestReplace(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="strata-atomic-"))

    def test_plain_replace(self):
        (self.tmp / "a.tmp").write_text("new")
        (self.tmp / "a").write_text("old")
        atomic.replace(self.tmp / "a.tmp", self.tmp / "a")
        self.assertEqual((self.tmp / "a").read_text(), "new")

    def test_a_held_file_is_waited_for_on_windows_and_the_error_is_kept_elsewhere(self):
        real = os.replace
        calls = {"n": 0}

        def held(src, dst):
            calls["n"] += 1
            if calls["n"] < 4:
                raise PermissionError(5, "Access is denied")
            return real(src, dst)

        (self.tmp / "b.tmp").write_text("new")
        slept = []
        with mock.patch("os.replace", held), mock.patch.object(atomic.os, "name", "nt"):
            atomic.replace(self.tmp / "b.tmp", self.tmp / "b", sleep=slept.append)
        self.assertEqual(calls["n"], 4)
        self.assertEqual(len(slept), 3)
        self.assertEqual((self.tmp / "b").read_text(), "new")
        # on a system without this behaviour a PermissionError is a real error
        (self.tmp / "c.tmp").write_text("x")
        with mock.patch("os.replace", side_effect=PermissionError(13, "denied")), mock.patch.object(atomic.os, "name", "posix"):
            with self.assertRaises(PermissionError):
                atomic.replace(self.tmp / "c.tmp", self.tmp / "c")

    def test_a_file_that_stays_held_is_given_up_on_after_the_patience(self):
        now = [0.0]
        with mock.patch("os.replace", side_effect=PermissionError(5, "Access is denied")), mock.patch.object(atomic.os, "name", "nt"):
            with self.assertRaises(PermissionError):
                atomic.replace("x", "y", patience=1.0, sleep=lambda s: now.__setitem__(0, now[0] + 0.5), clock=lambda: now[0])

    def test_write_text_leaves_no_temporary_file(self):
        atomic.write_text(self.tmp / "run.json", "{}")
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), ["run.json"])


if __name__ == "__main__":
    unittest.main()
