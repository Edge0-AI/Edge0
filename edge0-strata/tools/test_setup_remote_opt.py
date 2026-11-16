"""0.1.39b (#578): setup recommends --remote-expert-opt for a config on two or more GPUs, never for one, and keeps it
out when asked (setup's --no-remote-expert-opt, or "remote_expert_opt": false in the config).

    python -m unittest tools.test_setup_remote_opt
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_golden import PROFILES, install  # noqa: E402

FLAG = "--remote-expert-opt"
HELPER = ["--expert-cache-device1", "auto"]


class Rule(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(setup, "ok", lambda *a: None)
        p.start()
        self.addCleanup(p.stop)

    def test_two_gpus_get_it_once(self):
        cfg = {"args": ["--spec", "4"] + HELPER, "gpu": [0, 1]}
        setup.recommend_remote_expert_opt(cfg)
        setup.recommend_remote_expert_opt(cfg)
        self.assertEqual(cfg["args"], ["--spec", "4"] + HELPER + [FLAG])

    def test_no_helper_cache_no_flag(self):
        """#1447: on a plain layer split there is no helper cache for the flag to act on, so setup does not write it."""
        cfg = {"args": ["--spec", "4"], "gpu": [0, 1]}
        setup.recommend_remote_expert_opt(cfg)
        self.assertEqual(cfg["args"], ["--spec", "4"])
        kept = {"args": [FLAG], "gpu": [0, 1]}                # a flag already there is the user's: left alone
        setup.recommend_remote_expert_opt(kept)
        self.assertEqual(kept["args"], [FLAG])
        eq = {"args": ["--expert-cache-device2=auto"], "gpu": [0, 1, 2]}
        setup.recommend_remote_expert_opt(eq)
        self.assertIn(FLAG, eq["args"])

    def test_one_gpu_is_untouched(self):
        for gpu in (None, 0, [0]):
            cfg = {"args": ["--spec", "4"]} if gpu is None else {"args": ["--spec", "4"], "gpu": gpu}
            setup.recommend_remote_expert_opt(cfg)
            self.assertEqual(cfg["args"], ["--spec", "4"], gpu)

    def test_overrides(self):
        cfg = {"args": ["--spec", "4", FLAG], "gpu": [0, 1]}
        setup.recommend_remote_expert_opt(cfg, off=True)
        self.assertEqual(cfg["args"], ["--spec", "4"])
        cfg = {"args": ["--spec", "4"] + HELPER, "gpu": [0, 1, 2], "remote_expert_opt": False}
        setup.recommend_remote_expert_opt(cfg)
        self.assertEqual(cfg["args"], ["--spec", "4"] + HELPER)

    def test_pipeline_windows_wins(self):
        """#1352: the two exclude each other (the engine drops the pipeline beside the helper caches), so setup
        does not add the flag to a config that asks for the pipeline, and says so when both are there."""
        cfg = {"args": ["--spec", "4", "--pipeline-windows", "2"] + HELPER, "gpu": [0, 1]}
        setup.recommend_remote_expert_opt(cfg)
        self.assertEqual(cfg["args"], ["--spec", "4", "--pipeline-windows", "2"] + HELPER)
        both = {"args": [FLAG, "--pipeline-windows", "2"], "gpu": [0, 1]}
        with mock.patch.object(setup, "warn") as w:
            setup.recommend_remote_expert_opt(both)
        self.assertEqual(both["args"], [FLAG, "--pipeline-windows", "2"])    # a recommendation, never forced
        self.assertEqual(w.call_count, 1)
        off = {"args": ["--pipeline-windows", "0"] + HELPER, "gpu": [0, 1]}   # 0: no pipeline asked for
        setup.recommend_remote_expert_opt(off)
        self.assertIn(FLAG, off["args"])

    def test_the_users_key_is_kept_on_a_rerun(self):
        self.assertNotIn("remote_expert_opt", setup.SETUP_KEYS)   # #629: carried over, so the opt-out survives


class Install(unittest.TestCase):
    def test_install_on_two_gpus(self):
        """#1447: a plain two-GPU install has no helper cache, so no --remote-expert-opt is written."""
        ram, found = PROFILES["47GB-2x16GB"]
        args = ["--family", "qwen", "--no-start", "--model", "Q2_0", "--gpus", "0,1"]      # named: since the fit rule the default is the largest card + RAM
        code, text, cfg, _ = install(ram, found, args)
        self.assertEqual(code, 0, text)
        self.assertIsInstance(cfg.get("gpu"), list)
        self.assertNotIn(FLAG, cfg["args"])

    def test_install_on_one_gpu(self):
        ram, found = PROFILES["96GB-1x16GB"]
        code, text, cfg, _ = install(ram, found, ["--family", "qwen", "--no-start", "--model", "Q2_0"])
        self.assertEqual(code, 0, text)
        self.assertNotIn(FLAG, cfg["args"])


if __name__ == "__main__":
    unittest.main()
