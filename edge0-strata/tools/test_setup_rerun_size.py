"""0.1.44.1: setup run again over an installed model keeps its size.

0.1.44 made IQ3_S the recommended Qwen size on a PC with 64 GB or more (IQ3_S: about 15% slower decode than IQ3_XXS on an
RTX 5070, docs/MODELS.md, and about 7 GB more RAM).  A first install takes the recommendation; SETUP.bat / --setup on a
folder that already has a model pre-selects the installed size (and `--yes` takes it), so Enter never moves a working
install to a slower size.  The recommended size stays marked, every size stays a choice.

    python -m unittest tools.test_setup_rerun_size
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import test_setup_golden as g  # noqa: E402

PC64 = (63.7, [g.card(0, "NVIDIA GeForce RTX 5070", 11.9, "120")])


def earlier(model: str, family_tag: str = "") -> tuple:
    tag = f"{family_tag}{model.lower()}"
    return (f"strata-{tag}.json", {"exe": "x", "args": ["--pack", f"p/{tag}", "--max-context", "32768"], "cwd": ".",
                                   "tokenizer": "t", "model_name": f"qwen3.8-flash-next-{tag}", "log": "l", "port": 8080})


class RerunKeepsSize(unittest.TestCase):
    def run_setup(self, argv, configs, answers=""):
        ram, found = PC64
        code, out, cfg, asked = g.install(ram, found, argv, answers=answers, configs=configs)
        self.assertEqual(code, 0, out[-2000:])
        return out, cfg, asked

    def test_first_install_takes_the_recommendation(self):
        out, cfg, _ = self.run_setup(["--family", "qwen", "--no-start"], [])
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-iq3_s")
        self.assertNotIn("(installed)", out)

    def test_enter_over_an_installed_iq3_xxs_keeps_it(self):
        out, cfg, asked = self.run_setup(["--family", "qwen", "--setup", "--no-start"], [earlier("IQ3_XXS")])
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-iq3_xxs", out[-1500:])
        line_s = next(x for x in out.splitlines() if " IQ3_S " in x or " IQ3_S\t" in x or "IQ3_S    " in x)
        self.assertIn("(recommended)", line_s)            # still marked, still a choice
        line_xxs = next(x for x in out.splitlines() if "IQ3_XXS" in x and "download" in x)
        self.assertIn("(installed)", line_xxs)
        self.assertTrue(any("Which size?" in a for a in asked))

    def test_yes_over_an_installed_size_keeps_it_too(self):
        _, cfg, asked = self.run_setup(["--family", "qwen", "--setup", "--no-start"], [earlier("Q2_0")], answers=None)
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-q2_0")
        self.assertEqual(asked, [])

    def test_a_named_size_still_wins(self):
        _, cfg, _ = self.run_setup(["--family", "qwen", "--model", "IQ3_S", "--setup", "--no-start"], [earlier("IQ3_XXS")])
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-iq3_s")

    def test_another_family_is_not_touched(self):
        # a Qwen IQ3_XXS install does not pre-select anything for the Unsloth family
        _, cfg, _ = self.run_setup(["--family", "unsloth", "--setup", "--no-start"], [earlier("IQ3_XXS")])
        self.assertIn("unsloth", cfg["model_name"])

    def test_the_helper_prefers_the_most_recent_config_of_the_family(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            paths = []
            for name, cfg in (earlier("IQ3_S"), earlier("Q2_0")):
                p = t / name
                p.write_text(json.dumps(cfg))
                paths.append(p)
            names = [m for m in setup.MODELS if "qwen" in setup.MODELS[m].get("families", ("qwen", "swift"))]
            self.assertEqual(setup.installed_size("qwen", names, paths), "IQ3_S")       # first = most recently used
            self.assertEqual(setup.installed_size("qwen", names, paths[1:]), "Q2_0")
            self.assertIsNone(setup.installed_size("qwen", names, []))


class RerunKeepsOwnEnv(unittest.TestCase):
    """0.1.44 put STRATA_MMQ_KQ / STRATA_HC_Q8 / STRATA_ROUTE_TAIL_SKIP into SETUP_ENV (the entries setup writes, which a
    setup run again does not carry over): a GSQ-RCO config of the user's own STRATA_ROUTE_TAIL_SKIP=0 was dropped with them.
    Those three are setup's only in an Unsloth config."""

    OLD = {"env": {"STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_MMQ_KQ": "1", "STRATA_HC_Q8": "0", "STRATA_RESIDENT_PIN": "0",
                   "MY_VAR": "1"}}

    def test_a_gsq_rco_config_keeps_the_users_values(self):
        new = {"model_name": "qwen3.8-flash-next-iq3_xxs"}
        kept = setup.carry_over(dict(self.OLD), new)
        self.assertEqual(new["env"], {"STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_MMQ_KQ": "1", "STRATA_HC_Q8": "0", "MY_VAR": "1"})
        self.assertIn("env STRATA_ROUTE_TAIL_SKIP", kept)
        self.assertNotIn("env STRATA_RESIDENT_PIN", kept)             # the older entries stay setup's own

    def test_an_unsloth_config_gets_setups_values_not_the_old_ones(self):
        new = {"model_name": "qwen3.8-flash-next-unsloth-ud-q4_k_xl", "env": {"STRATA_HC_Q8": "1"}}
        setup.carry_over(dict(self.OLD), new)
        self.assertEqual(new["env"], {"STRATA_HC_Q8": "1", "MY_VAR": "1"})


if __name__ == "__main__":
    unittest.main()
