"""The update's test start must not pick a DeepSeek model: it runs the engine under test (the Strata engine) with the model's options, and a
DeepSeek model runs on strata-dsv41 (a separate package). Seen on a PC with a DeepSeek and a Qwen model: the test start failed with
'unknown argument: --eos' and engine 0.1.50 was refused."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import models, updater


class DiscoverRunConfigCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-discover-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def config(self, name: str, **fields) -> str:
        path = self.root / f"strata-{name}.json"
        path.write_text(json.dumps(fields), encoding="utf-8")
        return str(path)

    def discover(self, *entries):
        listed = {"models": [{"id": i, "component_status": "present", "source_config": c} for i, c in entries]}
        with mock.patch.object(models, "list_installed", return_value=listed), \
                mock.patch("strata_control.profiles.list_profiles", return_value=[]):
            return updater.discover_run_config()

    def test_deepseek_is_skipped_for_a_model_the_engine_under_test_can_run(self):
        deep = self.config("deepseek", exe="C:/e/engine-dsv41/0.1.51/strata-dsv41.exe", arch="deepseek41", args=["--eos", "1"])
        qwen = self.config("qwen", exe="C:/e/engine/strata.exe", args=["--pack", "x"])
        # "deepseek..." sorts first by id, which is how the update picked it before
        self.assertEqual(self.discover(("deepseek-v4.1-flash", deep), ("unsloth-ud-iq4_xs", qwen)), Path(qwen))

    def test_deepseek_is_recognised_by_its_exe_alone(self):
        deep = self.config("deepseek", exe=r"C:\e\engine-dsv41\0.1.51\strata-dsv41.exe", args=[])
        self.assertIsNone(self.discover(("deepseek-v4.1-flash", deep)))

    def test_only_deepseek_installed_means_no_model(self):
        deep = self.config("deepseek", exe="x", arch="deepseek41")
        self.assertIsNone(self.discover(("deepseek-v4.1-flash", deep)))

    def test_other_and_unreadable_configs_are_still_used(self):
        qwen = self.config("qwen", exe="C:/e/engine/strata.exe")
        broken = self.root / "strata-broken.json"
        broken.write_text("{not json", encoding="utf-8")
        self.assertEqual(self.discover(("a-broken", str(broken))), broken)
        self.assertEqual(self.discover(("b-qwen", qwen)), Path(qwen))


if __name__ == "__main__":
    unittest.main()
