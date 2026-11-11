"""tools/dsv41/tune.py: the short DeepSeek tune picks only clear gains, writes them into the config, and never stops a launch."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "dsv41"))
sys.path.insert(0, str(ROOT / "tools"))
import tune  # noqa: E402


class Tune(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = Path(self.tmp.name) / "c.json"
        self.cfg.write_text(json.dumps({"exe": "x", "args": ["--pack", "P", "--max-context", "8192", "--eos", "1", "--dspark", "auto",
                                                              "--dspark-conf", "0.65"]}))
        p = mock.patch.object(tune, "prompt_ids", lambda pack: [1, 2, 3])
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(tune, "physical_cores", lambda: 6)
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(tune.os, "cpu_count", lambda: 12)
        p.start()
        self.addCleanup(p.stop)

    def run_tune(self, speed):
        calls = []

        def fake(exe, args, f, n, timeout, env):
            calls.append(list(args))
            return speed(args)
        return tune.tune(self.cfg, "quick", measure_fn=fake), calls

    def test_parse(self):
        self.assertEqual(tune.parse_speed("x\ndecode: 48 tokens in 9000 ms (5.33 tok/s); CPU lane: 1"), 5.33)
        self.assertIsNone(tune.parse_speed("nothing"))

    def test_a_clear_gain_is_written(self):
        res, calls = self.run_tune(lambda a: 6.0 if "on" in a[a.index("--dspark") + 1:a.index("--dspark") + 2] else 4.5)
        args = json.loads(self.cfg.read_text())["args"]
        self.assertEqual(args[args.index("--dspark") + 1], "on")
        self.assertIn("faster", " ".join(res["lines"]))
        self.assertEqual(len(calls), 4)                                    # baseline, drafter on, block 3, threads
        self.assertIn("tuned", json.loads(self.cfg.read_text()))

    def test_the_result_says_what_was_tried(self):
        res, _ = self.run_tune(lambda a: 4.5)
        text = " ".join(res["lines"])
        self.assertIn("Tried (tokens a second): standard 4.5", text)
        self.assertIn("the drafter on (DSpark) 4.5", text)

    def test_a_single_timing_says_there_was_nothing_to_compare(self):
        def only_baseline(a):
            return 4.5 if "--dspark" in a and a[a.index("--dspark") + 1] == "auto" and "--cpu-threads" not in a else None
        res, _ = self.run_tune(only_baseline)
        self.assertIn("nothing to compare", " ".join(res["lines"]))

    def test_noise_changes_nothing(self):
        before = self.cfg.read_text()
        res, _ = self.run_tune(lambda a: 4.5 + 0.1 * (len(a) % 2))
        self.assertEqual(self.cfg.read_text(), before)
        self.assertEqual(res["changed"], [])
        self.assertIn("nothing changed", " ".join(res["lines"]))

    def test_a_failing_baseline_leaves_the_config_and_says_so(self):
        before = self.cfg.read_text()
        res, calls = self.run_tune(lambda a: None)
        self.assertEqual(self.cfg.read_text(), before)
        self.assertEqual(len(calls), 1)
        self.assertIn("stay as they are", " ".join(res["lines"]))

    def test_a_failing_trial_is_skipped(self):
        res, _ = self.run_tune(lambda a: 4.5 if "--cpu-threads" not in a and "on" != a[a.index("--dspark") + 1] else None)
        self.assertEqual(res["changed"], [])
        self.assertIn("did not finish", " ".join(res["lines"]))

    def test_main_never_raises(self):
        with mock.patch.object(tune, "tune", side_effect=RuntimeError("boom")):
            with mock.patch.object(sys, "argv", ["tune.py", "--config", str(self.cfg)]):
                self.assertEqual(tune.main(), 0)


if __name__ == "__main__":
    unittest.main()
