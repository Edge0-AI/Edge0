"""strata_control/folders.problem: a system folder is refused, except inside the running user's own home (root's /root included)."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import folders


class OwnHome(unittest.TestCase):
    def test_a_system_folder_is_still_refused(self):
        root = folders._system_roots()[0]
        with mock.patch.dict(os.environ, {"HOME": tempfile.gettempdir(), "USERPROFILE": tempfile.gettempdir()}):
            p = folders.problem(str(Path(root) / "strata-models"))
        self.assertIsNotNone(p)
        self.assertIn("system folder", p[0])

    def test_the_users_own_home_is_allowed_even_when_it_is_under_a_system_root(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d) / "rootlike"
            home.mkdir()
            with mock.patch.object(folders, "_system_roots", lambda: [d]), \
                    mock.patch.dict(os.environ, {"HOME": str(home), "USERPROFILE": str(home)}):
                self.assertIsNone(folders.problem(str(home / "models")))
                other = Path(d) / "elsewhere"
                other.mkdir()
                self.assertIn("system folder", folders.problem(str(other / "models"))[0])


if __name__ == "__main__":
    unittest.main()
