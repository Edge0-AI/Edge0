"""Behavior checks for the real-weight runner's memory guard and measurements."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from large_model_server import benchmark_payload, check_trace, command, features, public_command, run_workloads
from resource_watch import ResourceWatch, host_snapshot, memory_fields, watched_process, within_budget


class MemoryTests(unittest.TestCase):
    def test_status_parser_uses_bytes_and_excludes_non_memory_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status"
            path.write_text("Name:\ttest\nVmRSS:\t100 kB\nVmHWM:\t200 kB\nThreads:\t3\n")
            self.assertEqual(memory_fields(path), {"VmRSS": 102400, "VmHWM": 204800})

    def test_lowram_verdict_includes_kernel_peak_between_samples(self):
        summary = dict(samples=5, peak_rss_bytes=3 << 30, peak_hwm_bytes=4 << 30)
        self.assertTrue(within_budget(summary, 4096))
        summary["peak_hwm_bytes"] += 1
        self.assertFalse(within_budget(summary, 4096))
        summary["samples"] = 0
        with self.assertRaises(AssertionError):
            within_budget(summary, 4096)
        with self.assertRaises(AssertionError):
            within_budget(dict(samples=1, peak_rss_bytes=0, peak_hwm_bytes=0), 4096)

    def test_low_available_ram_stops_a_real_process_and_records_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
            watch = ResourceWatch(process, path, minimum=1 << 80)
            watch.worker.start()
            try:
                self.assertIsNotNone(process.wait(timeout=5))
                watch.stop()
                self.assertIn("safety floor", watch.error)
                with self.assertRaises(AssertionError):
                    watch.check()
                self.assertEqual(watch.summary["samples"], 1)
                self.assertGreater(json.loads(path.read_text())["available_bytes"], 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()

    def test_exception_stops_the_process_and_monitor(self):
        with tempfile.TemporaryDirectory() as directory:
            log, samples = (Path(directory) / name for name in ("log", "samples.jsonl"))
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with watched_process([sys.executable, "-c", "import time; time.sleep(30)"],
                                     dict(os.environ), log, samples) as (process, watch):
                    raise RuntimeError("injected")
            self.assertIsNotNone(process.poll())
            self.assertFalse(watch.worker.is_alive())


class MeasurementTests(unittest.TestCase):
    def test_book_at_token_cap_is_recorded_and_benchmarks_continue(self):
        args = SimpleNamespace(output_dir=Path("results"))
        result, watch = {"samples": []}, Mock()
        book = dict(finish_reason="length", generated_tokens=400, answer="An unfinished excerpt")
        with patch("large_model_server.streamed_request", return_value=book), \
                patch("large_model_server.measure", return_value={"streams": 1}) as measure, \
                patch("large_model_server.save_json"):
            run_workloads(args, "active", 1234, result, watch)
        self.assertEqual(result["book"], book)
        self.assertEqual(len(result["samples"]), 3)
        self.assertEqual([call.args[-1] for call in measure.call_args_list], [1, 4, 8])

    def test_lowram_check_uses_a_short_book_reply_and_full_decode_request(self):
        args, result = SimpleNamespace(output_dir=Path("results")), {"samples": []}
        with patch("large_model_server.streamed_request", return_value={"finish_reason": "length"}) as stream, \
                patch("large_model_server.measure", return_value={}) as measure, \
                patch("large_model_server.save_json"):
            run_workloads(args, "lowram", 1234, result, Mock())
        self.assertEqual(stream.call_args.args[1]["max_tokens"], 64)
        measure.assert_called_once_with(args, 1234, "lowram", 1)

    def test_both_builds_share_settings_and_linux_modes_enable_the_features(self):
        args = SimpleNamespace(vanilla_build=Path("vanilla"), linux_build=Path("linux"),
                               model_dir=Path("models/edge0-35b"), converted_dir=Path("converted"))
        for mode in ("vanilla", "active", "lowram"):
            argv, env, budget = command(args, mode, 1234)
            self.assertIn("--no-repack", argv)
            self.assertNotIn("--mlock", argv)
            self.assertEqual(env["OMP_WAIT_POLICY"], "PASSIVE")
            self.assertEqual(argv[argv.index("-np") + 1], "8")
            self.assertEqual(budget, {"vanilla": 0, "active": 12288, "lowram": 4096}[mode])
            self.assertEqual("E0_PREROUTER" in env, mode != "vanilla")
            self.assertEqual(features(args, mode)[2], budget)

    def test_fixed_seed_exact_token_count_and_streaming(self):
        query = benchmark_payload()
        self.assertEqual(query["max_tokens"], 128)
        self.assertEqual(query["seed"], 4242)
        self.assertTrue(query["ignore_eos"])
        self.assertTrue(query["stream"])
        self.assertFalse(query["chat_template_kwargs"]["enable_thinking"])

    def test_35b_trace_requires_adapter_predictions_and_successful_advice(self):
        lines = "loaded 560 tensors\nINIT ok: H=2048 E=256 K=4\n"
        trace = "steps=5 heads=80 experts=320 advice=960 failures=0\n"
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text(lines + trace)
            self.assertEqual(check_trace(log)["steps"], 5)
            for content in (lines, trace, lines + trace.replace("failures=0", "failures=1"),
                            lines.replace("H=2048", "H=1536") + trace,
                            lines.replace("K=4", "K=8") + trace):
                log.write_text(content)
                with self.assertRaises(AssertionError):
                    check_trace(log)

    def test_memory_budget_must_be_confirmed_in_the_real_server_log(self):
        text = "loaded 560 tensors\nINIT ok: H=2048 E=256 K=4\n"
        text += "steps=5 heads=80 experts=320 advice=960 failures=0\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text(text)
            with self.assertRaises(AssertionError):
                check_trace(path, 4096)
            path.write_text("[mem-budget] enabled: 4096 MiB\n" + text)
            self.assertEqual(check_trace(path, 4096)["failures"], 0)


class SavedEvidenceTests(unittest.TestCase):
    def test_saved_commands_shorten_the_home_folder(self):
        home = str(Path.home())
        self.assertEqual(public_command([f"{home}/bin/llama-server", "-m", f"{home}/m.gguf", 8080]),
                         ["~/bin/llama-server", "-m", "~/m.gguf", "8080"])

    def test_host_snapshots_hold_no_process_list_or_user_count(self):
        self.assertEqual(set(host_snapshot()), {"time", "load", "memory_bytes"})


if __name__ == "__main__":
    unittest.main()
