"""Tests for the model configs setup.py lists: the server keeps a config's Chat settings next to it as
`strata-<model>.shared-settings.json`, which matches the configs' own `strata-*.json` but is not one (no exe, no
args), so the start menu and the "set up like the last one" lookup must leave it out.

    python -m unittest tools.test_setup_configs
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

CONFIG = {"exe": "engine/strata.exe", "args": ["--max-context", "131072"]}
SHARED = {"reasoning_effort": "high", "temperature": 0.6}


def write(folder: Path, name: str, data: dict) -> Path:
    path = folder / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class ModelConfigs(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.config = write(self.dir, "strata-iq3_s.json", CONFIG)
        self.shared = write(self.dir, "strata-iq3_s.shared-settings.json", SHARED)
        # previous_config also looks into the Strata folders next to this checkout: keep the real ones out
        patch = mock.patch.object(setup, "other_installs", return_value=[])
        patch.start()
        self.addCleanup(patch.stop)

    def test_the_start_menu_lists_the_config_but_not_its_shared_settings(self):
        with mock.patch.object(setup, "ROOT", self.dir):
            self.assertEqual(setup.installed_configs(), [self.config])

    def test_the_last_config_of_another_install_is_never_its_shared_settings(self):
        # the shared settings are written after the config (the first time Chat settings are saved), so they are
        # the newest strata-*.json there
        later = self.config.stat().st_mtime + 60
        os.utime(self.shared, (later, later))
        self.assertEqual(setup.previous_config([self.dir], {}), self.config)

    def test_a_folder_with_only_shared_settings_has_no_config(self):
        self.config.unlink()
        with mock.patch.object(setup, "ROOT", self.dir):
            self.assertEqual(setup.installed_configs(), [])
        self.assertIsNone(setup.previous_config([self.dir], {}))


if __name__ == "__main__":
    unittest.main()
