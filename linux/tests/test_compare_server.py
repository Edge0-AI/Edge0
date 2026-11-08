"""Acceptance checks for the quiet-machine performance gate (no weights)."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import compare_server
from compare_server import decode_rate, summary


def runs(ratio=1.0, load=0.5):
    result = []
    for index in range(3):
        for mode in ("vanilla", "dormant", "active"):
            for threads in (8, 16):
                samples = []
                for streams in ((1, 4, 8) if threads == 8 else (1,)):
                    samples.append({"streams": streams, "tok_s": 20 * (1 if mode == "vanilla" else ratio),
                                    "before": {"load": [load]}, "after": {"load": [load]}})
                result.append(dict(mode=mode, threads=threads, round=index, samples=samples))
    return result


class SpeedGateTests(unittest.TestCase):
    def test_decode_excludes_the_token_sampled_during_prefill(self):
        timing = {"predicted_n": 128, "predicted_ms": 1000}
        self.assertEqual(decode_rate([timing]), 127)
        self.assertEqual(decode_rate([timing] * 8), 1016)

    def test_five_percent_boundary_passes(self):
        self.assertEqual(summary(runs(ratio=0.95))["status"], "PASS")

    def test_one_workload_regression_fails(self):
        values = runs()
        for run in values:
            if run["mode"] == "active" and run["threads"] == 16:
                run["samples"][0]["tok_s"] = 18.99
        self.assertEqual(summary(values)["status"], "FAIL")

    def test_loaded_measurements_cannot_pass_or_fail_the_gate(self):
        for ratio in (0.5, 1.5):
            with self.subTest(ratio=ratio):
                self.assertEqual(summary(runs(ratio=ratio, load=4))["status"], "UNDER_LOAD")

    def test_one_loaded_snapshot_is_enough_to_invalidate_the_gate(self):
        values = runs()
        values[-1]["samples"][0]["after"]["load"][0] = 4.01
        self.assertEqual(summary(values)["status"], "UNDER_LOAD")

    def test_diagnostic_controls_do_not_relax_the_original_gate(self):
        values = runs(ratio=0.8)
        for run in list(values):
            if run["mode"] == "active":
                for mode in ("lora", "accounting"):
                    values.append(dict(run, mode=mode))
        result = summary(values)
        self.assertEqual(result["status"], "FAIL")
        for row in result["rows"]:
            self.assertEqual(row["feature_ratios"]["active_vs_lora"], 1)
            self.assertEqual(row["feature_ratios"]["lora_vs_dormant"], 1)


class QuietPreflightTests(unittest.TestCase):
    def test_busy_host_stops_before_build_checks_or_model_work(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(require_quiet=True, output_dir=Path(directory))
            with patch.object(compare_server, "arguments", return_value=args), \
                    patch.object(compare_server, "snapshot", return_value={"load": [4.0]}), \
                    patch.object(compare_server, "cache_options") as build_check, \
                    patch.object(compare_server, "run_round") as benchmark, redirect_stdout(io.StringIO()):
                self.assertEqual(compare_server.main(), 2)
            build_check.assert_not_called()
            benchmark.assert_not_called()
            result = json.loads((args.output_dir / "compare.json").read_text())
            self.assertEqual(result["summary"]["status"], "UNDER_LOAD")
            self.assertEqual(result["quiet_checks"][0]["load"][0], 4.0)
            self.assertEqual(result["runs"], [])

    def test_a_loaded_sample_stops_even_after_host_load_drops(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(require_quiet=True, output_dir=Path(directory))
            sample = {"before": {"load": [1.0]}, "after": {"load": [4.1]}}
            with patch.object(compare_server, "snapshot", return_value={"load": [1.0]}), \
                    redirect_stdout(io.StringIO()):
                self.assertFalse(compare_server.quiet_check(args, {"runs": []}, [sample]))
            result = json.loads((args.output_dir / "compare.json").read_text())
            self.assertEqual(result["summary"]["status"], "UNDER_LOAD")

    def test_quiet_preflight_continues_without_creating_a_verdict(self):
        args = SimpleNamespace(require_quiet=True)
        result = {"runs": []}
        with patch.object(compare_server, "snapshot", return_value={"load": [3.99]}):
            self.assertTrue(compare_server.quiet_check(args, result))
        self.assertNotIn("summary", result)

    def test_loaded_diagnostic_runs_can_still_be_requested(self):
        with patch.object(compare_server, "snapshot") as observed:
            self.assertTrue(compare_server.quiet_check(SimpleNamespace(require_quiet=False), {}))
        observed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
