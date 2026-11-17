"""serve/test_family_name.py - the API's model name per model family (NEW-MODELS-PLAN.md T0): the config's
"model_name" wins; without one, the family's ("arch": qwen4exp -> qwen3.8-flash-next, qwen35moe -> qwen3.6-35b-a3b),
and Flash-Next's for a config that names neither (every config setup wrote before).

    python -m unittest serve.test_family_name
"""
from __future__ import annotations

import unittest

from serve.server import FAMILY_MODEL_NAMES, default_model_name


class FamilyName(unittest.TestCase):
    def test_config_name_wins(self):
        self.assertEqual(default_model_name({"model_name": "qwen3.8-flash-next-q2_0", "arch": "qwen35moe"}),
                         "qwen3.8-flash-next-q2_0")

    def test_per_family(self):
        self.assertEqual(default_model_name({"arch": "qwen35moe"}), "qwen3.6-35b-a3b")
        self.assertEqual(default_model_name({"arch": "qwen4exp"}), "qwen3.8-flash-next")
        self.assertEqual(default_model_name({"arch": "deepseek41"}), "deepseek-v4.1-flash")

    def test_old_configs(self):
        self.assertEqual(default_model_name({}), "qwen3.8-flash-next")
        self.assertEqual(default_model_name({"arch": "something-else"}), "qwen3.8-flash-next")
        self.assertEqual(set(FAMILY_MODEL_NAMES), {"qwen4exp", "qwen35moe", "deepseek41"})


if __name__ == "__main__":
    unittest.main()
