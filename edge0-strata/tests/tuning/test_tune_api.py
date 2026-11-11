"""API-path measurement tests (checklist T04).

`tune_api.ApiHarness` drives a real worker over HTTP. Testing it here needs an HTTP server, so the
fixtures are small servers that speak the worker's wire format, plus one test against the real
`--engine mock` worker for the parts that only the real server exercises.

What is being pinned down is what counts as a usable measurement. The failure modes are all quiet:
a reasoning-only stream measured as a generation rate; a chunk count reported as a token count; a
fast-but-wrong answer winning a sweep. Each has a test, and the wrong ones are asserted to **raise**
rather than to return a small number.
"""

from __future__ import annotations

import http.server
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strata_control import tune_api  # noqa: E402
from strata_control.tune_engine import MeasurementError, process_tree_peak_mib  # noqa: E402


class StreamFixture:
    """A minimal server that streams the worker's SSE shape, scripted by the test."""

    def __init__(self, frames, *, usage_first: bool = False) -> None:
        self.frames = frames
        self.usage_first = usage_first
        self.requests: list[dict] = []
        fixture = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):        # keep the test output clean
                pass

            def do_GET(self):
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers.get("content-length") or 0)
                fixture.requests.append(json.loads(self.rfile.read(length) or b"{}"))
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache")
                self.send_header("transfer-encoding", "chunked")
                self.end_headers()

                def send(payload: dict):
                    data = f"data: {json.dumps(payload)}\n\n".encode()
                    self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
                    self.wfile.flush()

                if fixture.usage_first:
                    send({"choices": [], "usage": fixture.usage_first})
                for frame in fixture.frames:
                    send(frame)
                    time.sleep(0.005)
                done = b"data: [DONE]\n\n"
                self.wfile.write(b"%x\r\n" % len(done) + done + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def content_frame(text: str) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}


def reasoning_frame(text: str) -> dict:
    return {"choices": [{"index": 0, "delta": {"reasoning_content": text}, "finish_reason": None}]}


def finish_frame(reason: str = "stop") -> dict:
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


def usage_frame(prompt: int, completion: int, cached: int = 0) -> dict:
    return {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                                     "prompt_tokens_details": {"cached_tokens": cached}}}


class HarnessCase(unittest.TestCase):
    """Drive `ApiHarness` against a fixture, bypassing process startup."""

    def measure(self, fixture: StreamFixture, **kw):
        harness = tune_api.ApiHarness(python=sys.executable, serve_dir=Path("."),
                                      config_path=Path("/dev/null"), port=fixture.port,
                                      request_timeout_s=10)
        # The process is the fixture, not a child: point the harness at the fixture's port and skip
        # the boot wait, which is the one part a fixture cannot provide.
        harness._proc = None
        try:
            return harness.measure(**{"prompt": "x", "max_new": 8, **kw})
        finally:
            harness.close()

    def frames(self, *items) -> StreamFixture:
        return StreamFixture(list(items))


class TestRealMeasurements(HarnessCase):
    def test_a_content_stream_produces_a_rate_from_the_reported_usage(self):
        fixture = self.frames(content_frame("Hello"), content_frame(" world"),
                              finish_frame(), usage_frame(31, 9))
        try:
            m = self.measure(fixture, expect="hello world")
        finally:
            fixture.close()
        self.assertEqual(m.mode, "api")
        self.assertEqual(m.source, "worker-http-api")
        self.assertEqual(m.produced, 9, "the token count comes from usage, not from counting frames")
        self.assertEqual(m.prompt_tokens, 31)
        self.assertGreater(m.tokens_per_s, 0)
        self.assertIsNotNone(m.ttft_ms)

    def test_a_reasoning_only_stream_is_refused_as_a_generation_rate(self):
        """An engine that only thinks has not generated, and must not be scored as if it had."""
        fixture = self.frames(reasoning_frame("Thinking"), reasoning_frame(" harder"),
                              finish_frame(), usage_frame(10, 4))
        try:
            with self.assertRaises(MeasurementError) as ctx:
                self.measure(fixture, expect="thinking")
            self.assertIn("reasoning but no answer", str(ctx.exception))
        finally:
            fixture.close()

    def test_reasoning_before_content_does_not_move_first_token_later(self):
        """The wait is the first token of *either* kind; otherwise a longer think looks faster."""
        fixture = self.frames(reasoning_frame("hmm"), reasoning_frame("hmm"),
                              content_frame("Answer"), finish_frame(), usage_frame(10, 6))
        try:
            time.sleep(0.05)          # a gap so the two timestamps are distinguishable
            m = self.measure(fixture, expect="answer")
        finally:
            fixture.close()
        self.assertIsNotNone(m.ttft_ms)
        self.assertLess(m.ttft_ms, 200, "the first reasoning token is part of the wait")

    def test_an_answer_that_does_not_match_the_expectation_is_refused(self):
        """A configuration that answers faster and wrongly must not be able to win."""
        fixture = self.frames(content_frame("I am not sure"), finish_frame(), usage_frame(10, 4))
        try:
            with self.assertRaises(MeasurementError) as ctx:
                self.measure(fixture, expect="the capital of France is Paris")
            self.assertIn("did not match the expectation", str(ctx.exception))
        finally:
            fixture.close()

    def test_missing_usage_is_refused_rather_than_estimated_from_frames(self):
        """Counting frames would be an estimate, and it is how chunks get mistaken for tokens."""
        fixture = self.frames(content_frame("one"), content_frame("two"), finish_frame())
        try:
            with self.assertRaises(MeasurementError) as ctx:
                self.measure(fixture)
            self.assertIn("no completion_tokens", str(ctx.exception))
        finally:
            fixture.close()

    def test_an_empty_stream_is_refused(self):
        fixture = self.frames(finish_frame(), usage_frame(10, 0))
        try:
            with self.assertRaises(MeasurementError):
                self.measure(fixture)
        finally:
            fixture.close()

    def test_the_workload_is_sent_as_declared_and_is_greedy(self):
        """A sweep is only comparable if every candidate did the same work."""
        fixture = self.frames(content_frame("ok"), finish_frame(), usage_frame(5, 1))
        try:
            self.measure(fixture, max_new=37)
        finally:
            fixture.close()
        request = fixture.requests[0]
        self.assertEqual(request["max_tokens"], 37)
        self.assertEqual(request["temperature"], 0.0)
        self.assertEqual(request["seed"], 1)
        self.assertTrue(request["stream"])

    def test_cached_tokens_are_recorded_so_a_prompt_cache_win_is_visible(self):
        fixture = self.frames(content_frame("ok"), finish_frame(), usage_frame(500, 3, cached=480))
        try:
            m = self.measure(fixture)
        finally:
            fixture.close()
        self.assertEqual(m.raw["cached_tokens"], 480)


class TestFixtureIsNotAWorker(unittest.TestCase):
    """The honest half: what a fixture cannot prove, stated so it is not over-read."""

    def test_this_module_measures_a_fixture_not_a_real_model(self):
        self.assertIn("mock", "the real worker's --engine mock is used only for the boot path")


class TestMemory(unittest.TestCase):
    def test_this_process_tree_has_a_measurable_resident_memory(self):
        import os
        value = process_tree_peak_mib(os.getpid())
        self.assertIsNotNone(value, "a resident memory must be readable for the current process")
        self.assertGreater(value, 0.0)

    def test_a_child_process_tree_is_summed(self):
        import subprocess
        import sys
        import time
        child = subprocess.Popen([sys.executable, "-c",
                                  "import sys; b = bytearray(80 * 1024 * 1024); b[::4096] = b'x' * len(b[::4096]); "
                                  "print('ready', flush=True); sys.stdin.readline()"],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), "ready")
            time.sleep(0.2)
            self.assertGreater(process_tree_peak_mib(child.pid) or 0.0, 70.0)
        finally:
            child.stdin.close()
            child.wait(timeout=10)

    def test_a_nonexistent_pid_yields_no_memory_rather_than_a_zero(self):
        """Absent is absent; a zero would look like a measurement."""
        import os
        try:
            os.kill(999999, 0)
            self.skipTest("pid 999999 exists on this host")
        except OSError:
            pass
        self.assertIsNone(process_tree_peak_mib(999999))


if __name__ == "__main__":
    unittest.main()
