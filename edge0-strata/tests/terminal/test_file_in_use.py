"""Windows 'file in use': every swap of a state file waits and retries (atomic.replace), and the message does not blame permissions."""

import re
import unittest
from pathlib import Path
from unittest import mock

from strata_control import atomic, private_files

ROOT = Path(__file__).resolve().parents[2]
SITES = ["strata_control/calibration.py", "strata_control/components.py", "strata_control/presets.py",
         "strata_control/preset_marks.py", "strata_control/procs.py", "strata_control/profiles.py",
         "strata_control/tune_apply.py", "strata_control/updater.py", "strata_cli/bench.py", "strata_cli/connections.py",
         "strata_cli/launch/runner.py", "strata_cli/launch/store.py", "strata_cli/tune_commands.py", "serve/runconfig.py"]


class Sites(unittest.TestCase):
    def test_no_bare_os_replace_is_left_in_the_state_file_writers(self):
        for rel in SITES:
            text = (ROOT / rel).read_text(encoding="utf-8")
            bare = [l for l in text.splitlines() if re.search(r"(?<![_\w.])os\.replace\(", l) or re.search(r"[^_]os\.replace\(", l)]
            # updater._move_dir keeps its own longer retry loop
            bare = [l for l in bare if not (rel.endswith("updater.py") and "source, destination" in l)]
            self.assertEqual(bare, [], rel)
            self.assertIn("_atomic_mod.replace(", text, rel)


class Retry(unittest.TestCase):
    def test_a_held_file_is_waited_for_on_windows(self):
        calls = []

        def flaky(src, dst):
            calls.append(1)
            if len(calls) < 3:
                raise PermissionError(13, "in use")
        with mock.patch.object(atomic.os, "replace", flaky), mock.patch.object(atomic.os, "name", "nt"):
            atomic.replace("a", "b", sleep=lambda s: None)
        self.assertEqual(len(calls), 3)


class Message(unittest.TestCase):
    def test_sharing_violation_is_told_from_a_refusal(self):
        in_use = PermissionError(13, "The process cannot access the file because it is being used by another process")
        in_use.winerror = 32
        denied = PermissionError(13, "Access is denied")
        denied.winerror = 5
        self.assertTrue(private_files.is_sharing_violation(in_use))
        self.assertFalse(private_files.is_sharing_violation(denied))
        self.assertNotIn("as yourself", private_files.ACCESS_MESSAGE)
        self.assertIn("another program", private_files.IN_USE_MESSAGE)


if __name__ == "__main__":
    unittest.main()
