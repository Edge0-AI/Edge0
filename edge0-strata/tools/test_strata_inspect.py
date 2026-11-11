"""tools/strata_inspect.py's verdict, offline: the fingerprints of the files setup installs (data/gguf_fingerprints.json)
stand in for headers read from the network.

    python -m unittest tools.test_strata_inspect
"""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_inspect as SI  # noqa: E402


class Verdict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = SI.load_fingerprints()

    def fp(self, key):
        return copy.deepcopy({k: v for k, v in self.table[key].items() if k not in ("files", "repo")})

    def test_every_installed_file_is_known(self):
        for key in (k for k in self.table if not k.startswith("_")):
            kind, sentence = SI.verdict(self.fp(key), self.table)
            fam, model = key.split("/")
            self.assertEqual(kind, "known", key)
            self.assertIn(f"--family {fam} --model {model}", sentence)

    def test_same_layout_other_file(self):
        fp = self.fp("qwen/Q2_0")
        fp["bytes"] += 1234                                  # a fine-tune with the same recipe
        kind, sentence = SI.verdict(fp, self.table)
        self.assertEqual(kind, "layout")
        self.assertIn("qwen/Q2_0", sentence)

    def test_other_architecture(self):
        fp = self.fp("qwen/Q2_0")
        fp["arch"] = "qwen3moe"
        self.assertEqual(SI.verdict(fp, self.table)[0], "no")

    def test_qwen36_runs_on_its_own_path(self):
        fp = self.fp("qwen/Q2_0")
        fp["arch"] = "qwen35moe"
        kind, sentence = SI.verdict(fp, self.table)
        self.assertEqual(kind, "family")
        self.assertIn("Qwen3.6-35B-A3B", sentence)
        self.assertIn("--family qwen36", sentence)

    def test_expert_type_without_kernel(self):
        fp = self.fp("qwen/IQ3_S")
        fp["expert_bytes"] = {"IQ1_S": 30_000_000_000}
        kind, sentence = SI.verdict(fp, self.table)
        self.assertEqual(kind, "no")
        self.assertIn("IQ1_S", sentence)

    def test_untested_names_the_new_tensors(self):   # Unsloth UD-Q2_K_XL's case: known expert types, a new head type
        fp = self.fp("qwen/IQ3_XXS")
        fp["expert_bytes"] = {"IQ2_XS": 1, "IQ3_XXS": 2, "IQ4_NL": 3}
        fp["roles"] = {"output.weight": ["Q4_K"], "token_embd.weight": ["Q3_K"]}
        kind, sentence = SI.verdict(fp, self.table)
        self.assertEqual(kind, "untested")
        self.assertIn("output.weight Q4_K", sentence)
        self.assertNotIn("token_embd", sentence)


if __name__ == "__main__":
    unittest.main()
