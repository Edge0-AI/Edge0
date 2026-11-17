"""Engine measurement tests.

The parsing is tested against the engine's real line formats, taken from
`src/program/generate.cpp`, and the failure handling is tested with a stub engine that speaks the
same protocol. That combination is deliberate: real hardware needs a model that does not fit this
machine, but the *contract* — what counts as a measurement and what must be rejected — is fully
testable here, and it is the part that decides whether a tuning result can be trusted.

The risk these tests exist for is specific: a tuning harness that scores a failed or incomplete run
as a slow one will happily recommend a configuration that does not work.
"""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strata_control import tune_engine as te  # noqa: E402


class TestDoneParsing(unittest.TestCase):
    """`DONE` is what the engine reports; the field order is the engine's, not ours."""

    def test_a_normal_done_line(self):
        # DONE produced prompt_tokens prompt_ms decode_ms finish drafts_acc drafts_off ...
        d = te.parse_done("DONE 128 512 210.5 1830.0 stop 40 55 0 0 0 12 3 3.4 12 0")
        self.assertEqual(d["produced"], 128)
        self.assertEqual(d["prompt_tokens"], 512)
        self.assertAlmostEqual(d["prompt_ms"], 210.5)
        self.assertAlmostEqual(d["decode_ms"], 1830.0)
        self.assertEqual(d["finish"], "stop")
        self.assertEqual(d["drafts_accepted"], 40)

    def test_the_variable_tail_is_kept_not_required(self):
        """The engine appends lookup-chain counters only when that option is on."""
        short = te.parse_done("DONE 8 16 10.0 20.0 stop 1 2")
        self.assertEqual(short["tail"], [])
        long = te.parse_done("DONE 8 16 10.0 20.0 stop 1 2 3 4 5 6 7 8.0 9 10 11 12 13")
        self.assertEqual(long["tail"], ["3", "4", "5", "6", "7", "8.0", "9", "10", "11", "12", "13"])

    def test_a_short_or_malformed_line_is_an_error_not_a_zero(self):
        for bad in ("DONE 1 2", "DONE a b c d stop", "not a done line", "DONE 1 2 three 4.0 stop"):
            with self.assertRaises(te.MeasurementError, msg=bad):
                te.parse_done(bad)

    def test_unreadable_optional_fields_do_not_fail_the_parse(self):
        d = te.parse_done("DONE 8 16 10.0 20.0 stop x y")
        self.assertIsNone(d["drafts_accepted"])
        self.assertIsNone(d["drafts_offered"])


class TestReadyAndPP(unittest.TestCase):
    def test_ready_carries_the_context(self):
        self.assertEqual(te.parse_ready("READY 32768 stop"), 32768)

    def test_a_ready_without_a_context_is_an_error(self):
        with self.assertRaises(te.MeasurementError):
            te.parse_ready("READY")

    def test_pp_with_and_without_the_rate(self):
        a = te.parse_pp("PP 1024 4096 800.0 1280.0")
        self.assertEqual(a["processed"], 1024)
        self.assertEqual(a["tok_s"], 1280.0)
        b = te.parse_pp("PP 1024 4096 800.0")
        self.assertNotIn("tok_s", b)


class TestMeasurements(unittest.TestCase):
    """The numbers, and the cases that must never produce one."""

    def done(self, **kw):
        base = {"produced": 100, "prompt_tokens": 512, "prompt_ms": 200.0, "decode_ms": 1000.0,
                "finish": "stop"}
        base.update(kw)
        return base

    def test_rates_come_from_the_engines_own_timings(self):
        m = te.measurement_from_protocol(self.done(), tokens_seen=100)
        self.assertAlmostEqual(m.decode_tokens_per_s, 100.0)          # 100 tokens / 1.0 s
        self.assertAlmostEqual(m.prompt_tokens_per_s, 2560.0)         # 512 / 0.2 s
        self.assertAlmostEqual(m.tokens_per_s, 100 * 1000 / 1200)     # whole request
        self.assertEqual(m.finish, "stop")

    def test_ttft_is_the_prompt_plus_one_decode_step(self):
        m = te.measurement_from_protocol(self.done(), tokens_seen=100)
        # 200 ms prompt + (1000 ms / 100 tokens) = 210 ms
        self.assertAlmostEqual(m.ttft_ms, 210.0)

    def test_client_ttft_is_recorded_separately_and_labelled(self):
        m = te.measurement_from_protocol(self.done(), tokens_seen=100, client_ttft_ms=375.0)
        self.assertAlmostEqual(m.ttft_ms, 210.0)
        self.assertAlmostEqual(m.client_ttft_ms, 375.0)

    def test_a_cancelled_run_is_not_a_measurement(self):
        with self.assertRaises(te.MeasurementError):
            te.measurement_from_protocol(self.done(finish="cancel"), tokens_seen=100)

    def test_a_nonfinite_run_is_not_a_measurement(self):
        with self.assertRaises(te.MeasurementError):
            te.measurement_from_protocol(self.done(finish="nonfinite"), tokens_seen=100)

    def test_zero_tokens_is_not_a_measurement_even_with_a_stop_finish(self):
        with self.assertRaises(te.MeasurementError) as ctx:
            te.measurement_from_protocol(self.done(produced=0), tokens_seen=0)
        self.assertIn("no tokens", str(ctx.exception))

    def test_a_token_count_that_disagrees_with_the_stream_is_refused(self):
        """If the stream and the summary disagree, neither can be trusted for a rate."""
        with self.assertRaises(te.MeasurementError) as ctx:
            te.measurement_from_protocol(self.done(produced=100), tokens_seen=42)
        self.assertIn("refusing to score", str(ctx.exception))

    def test_a_length_finish_is_a_legitimate_measurement(self):
        """Hitting the output cap is a real measurement of the cap, not a failure."""
        m = te.measurement_from_protocol(self.done(finish="length"), tokens_seen=100)
        self.assertEqual(m.finish, "length")

    def test_missing_metrics_stay_missing_rather_than_becoming_zero(self):
        m = te.measurement_from_protocol(
            {"produced": 10, "prompt_tokens": 0, "prompt_ms": 0.0, "decode_ms": 500.0,
             "finish": "stop"}, tokens_seen=10)
        metrics = m.to_metrics()
        self.assertIn("tokens_per_s", metrics)
        self.assertNotIn("prompt_tokens_per_s", metrics,
                         "an unmeasurable prompt rate must be absent, not zero")


STUB_PRELUDE = (
    "import sys\n"
    "def say(text):\n    print(text, flush=True)\n"
    "def wait():\n    sys.stdin.readline()\n")


class TestEngineHarnessWithAStub(unittest.TestCase):
    """The harness against a stub that speaks the real protocol.

    This cannot prove the engine measures what we think. It proves the harness's own contract: a
    boot failure, an early exit and an error line all become errors, and a good response becomes a
    measurement with the engine's numbers.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="strata-tune-engine-"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def stub(self, body: str) -> Path:
        """Write a stub engine that speaks the serve protocol.

        The stub is a Python script (the harness starts it as ``[sys.executable, stub.py]``, so it
        runs the same on every OS; a ``.sh`` stub is WinError 193 on Windows). ``body`` is Python
        using ``say(text)`` (print a line and flush) and ``wait()`` (block until the harness sends
        a line). Each body is explicit about how it blocks: ``wait()`` keeps the process, and so the
        stdout pipe, alive without leaving a child behind.
        """
        p = self.tmp / "stub-engine.py"
        p.write_text(STUB_PRELUDE + body)
        return p

    def test_a_good_response_becomes_a_measurement(self):
        stub = self.stub(
            'say("READY 4096 stop")\n'
            'wait()\n'
            'say("PP 512 512 200.0 2560.0")\n'
            'say("T 1"); say("T 2"); say("T 3")\n'
            'say("DONE 3 512 200.0 30.0 stop 0 0")\n'
            'wait()\n')
        h = te.EngineHarness(stub, {}, workload={"prompt_tokens": 8, "max_new": 3})
        try:
            h.start()
            self.assertEqual(h._context, 4096)
            m = h.measure()
            self.assertEqual(m.produced, 3)
            self.assertAlmostEqual(m.decode_tokens_per_s, 100.0)   # 3 / 0.030 s
            self.assertEqual(m.finish, "stop")
        finally:
            h.close()

    def test_an_engine_that_never_reports_ready_times_out(self):
        # The stub must block *in its own process* with stdout open. A shell `sleep` child would
        # hold the pipe open after the shell exits, so the harness would keep waiting on a process
        # that is already gone -- which made this test hang rather than fail.
        stub = self.stub('wait()\n')
        h = te.EngineHarness(stub, {}, boot_timeout_s=1.0)
        try:
            with self.assertRaises(te.MeasurementError) as ctx:
                h.start()
            self.assertIn("READY", str(ctx.exception))
        finally:
            h.close()

    def test_an_engine_that_exits_without_ready_is_reported_as_an_early_exit(self):
        # It prints something and *then* dies without READY, which is the realistic shape: the
        # engine got far enough to log, then failed.
        stub = self.stub('say("loading weights")\nsys.exit(0)\n')
        h = te.EngineHarness(stub, {}, boot_timeout_s=10.0)
        try:
            with self.assertRaises(te.MeasurementError) as ctx:
                h.start()
            self.assertIn("before it was ready", str(ctx.exception))
        finally:
            h.close()

    def test_an_engine_that_exits_immediately_reports_the_code(self):
        stub = self.stub('say("starting up")\nsys.exit(7)\n')
        h = te.EngineHarness(stub, {})
        try:
            with self.assertRaises(te.MeasurementError) as ctx:
                h.start()
            self.assertIn("7", str(ctx.exception))
        finally:
            h.close()

    def test_an_error_line_from_the_engine_is_raised_verbatim(self):
        stub = self.stub('say("READY 4096 stop")\nwait()\nsay("ERR out of memory")\nwait()\n')
        h = te.EngineHarness(stub, {}, workload={"prompt_tokens": 4, "max_new": 1},
                             request_timeout_s=3.0)
        try:
            h.start()
            with self.assertRaises(te.MeasurementError) as ctx:
                h.measure()
            self.assertIn("out of memory", str(ctx.exception))
        finally:
            h.close()

    def test_the_shared_stub_engine_speaks_the_protocol_and_reacts_to_kv(self):
        """tests/tuning/stub_engine.py is the engine every sweep test uses, on every OS."""
        stub_path = Path(__file__).with_name("stub_engine.py")
        speeds = {}
        for kv in ("int8", "fp16"):
            h = te.EngineHarness(stub_path, {"--kv": kv}, workload={"prompt_tokens": 8, "max_new": 4})
            try:
                h.start()
                m = h.measure()
                self.assertEqual(m.produced, 4)
                speeds[kv] = m.decode_tokens_per_s
            finally:
                h.close()
        self.assertAlmostEqual(speeds["int8"], 80.0)     # 4 tokens in 50 ms
        self.assertAlmostEqual(speeds["fp16"], 40.0)     # 4 tokens in 100 ms

    def test_a_py_engine_runs_through_the_current_interpreter(self):
        from strata_control import tuning
        self.assertEqual(tuning.engine_command(Path("x") / "stub.py"), [sys.executable, str(Path("x") / "stub.py")])
        self.assertEqual(tuning.engine_command(Path("x") / "strata"), [str(Path("x") / "strata")])

    def test_a_missing_binary_is_an_error_before_any_process_starts(self):
        h = te.EngineHarness(self.tmp / "nope", {})
        with self.assertRaises(te.MeasurementError) as ctx:
            h.start()
        self.assertIn("not at", str(ctx.exception))

    def test_the_declared_flags_are_passed_to_the_engine(self):
        stub = self.stub('say("READY 4096 stop")\nwait()\nsay("DONE 1 1 1.0 1.0 stop 0 0")\nwait()\n')
        h = te.EngineHarness(stub, {"--kv": "int8", "--kv-grow": "1"})
        argv = h._argv()
        self.assertIn("--kv", argv)
        self.assertIn("int8", argv)
        self.assertIn("--kv-grow", argv)
        try:
            h.start()
            h.measure()
        finally:
            h.close()

    def test_make_measure_returns_the_metric_dict_the_job_expects(self):
        stub = self.stub('say("READY 4096 stop")\nwait()\nsay("T 1")\nsay("DONE 1 16 8.0 4.0 stop 0 0")\nwait()\n')
        measure = te.make_measure(stub, workload={"prompt_tokens": 4, "max_new": 1})

        class C:
            args: dict = {}

        metrics = measure(C())
        self.assertIn("tokens_per_s", metrics)
        self.assertIsInstance(metrics["tokens_per_s"], float)
        self.assertGreater(metrics["tokens_per_s"], 0)


class TestWorkloadIsFixed(unittest.TestCase):
    """A sweep is only comparable if every candidate did the same work."""

    def test_the_default_workload_is_deterministic(self):
        w = te.DEFAULT_WORKLOAD
        self.assertEqual(w["seed"], 1)
        self.assertEqual(w["temperature"], 0.0)
        self.assertGreater(w["prompt_tokens"], 0)
        self.assertGreater(w["max_new"], 0)

    def test_the_same_prompt_ids_are_produced_every_time(self):
        h = te.EngineHarness(Path("/nonexistent"), {}, workload={"prompt_tokens": 16})
        self.assertEqual(list(h._prompt_tokens()), list(h._prompt_tokens()))
        self.assertEqual(len(list(h._prompt_tokens())), 16)


if __name__ == "__main__":
    unittest.main()
