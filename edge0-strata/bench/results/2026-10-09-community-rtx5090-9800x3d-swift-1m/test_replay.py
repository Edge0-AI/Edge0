"""Tiny stdlib loopback fixtures; these never contact an inference server."""
import gzip
import hashlib
import http.server
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest

HELPER = Path(__file__).with_name("replay.py")


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-replay-fixture-")
        self.root = Path(self.temp.name)
        shutil.copyfile(HELPER, self.root / "replay.py")
        (self.root / "requests").mkdir()
        self.received = []
        self.headers = []
        self.behavior = "success"
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                body = json.dumps({"loaded": True, "model": "fixture-only"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                behavior = owner.behavior
                owner.received.append((self.path, body))
                owner.headers.append(dict(self.headers))
                if self.path.startswith("/slots/0?"):
                    payload = b'{"n_saved":1000,"n_restored":1000}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                request = json.loads(body)
                if behavior == "timeout":
                    time.sleep(1.2)
                if behavior == "http-error":
                    payload = b'{"error":{"message":"HTTP fixture error"}}'
                    self.send_response(503)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                if behavior == "malformed":
                    chunks = [b"data: not-json\n\n"]
                elif behavior == "error":
                    chunks = [b'data: {"error":{"message":"fixture API error"}}\n\n']
                else:
                    prompt = 1001 if behavior == "wrong-prompt" else 1000
                    reuse = 900 if request["messages"][0]["content"] == "disk-continuation" else 0
                    if request["messages"][0]["content"].startswith("decode"):
                        reuse = 999
                    if behavior == "reused":
                        reuse = 1
                    generated = request["max_tokens"]
                    if behavior == "short":
                        generated -= 1
                    finish = "stop" if request["messages"][0]["content"].startswith("disk") else "length"
                    values = [
                        {"choices": [{"delta": {"role": "assistant"}}]},
                        {"choices": [{"delta": {"reasoning_content": "actual reasoning"}}]},
                        {"choices": [{"delta": {"content": "2 + 2 = 5"}}]},
                        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call1", "type": "function", "function": {"name": "calculate", "arguments": "{\"x\":"}}]}}]},
                        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "5}"}}]}, "finish_reason": finish}],
                         "usage": {"prompt_tokens": prompt, "completion_tokens": generated},
                         "timings": {"cache_n": reuse, "prompt_n": prompt - reuse, "prompt_ms": 200, "predicted_n": generated, "predicted_ms": 400}},
                    ]
                    chunks = [("data: " + json.dumps(value) + "\r\n\r\n").encode() for value in values]
                    if behavior != "missing-done":
                        chunks.append(b"data: [DONE]\n\n")
                try:
                    for chunk in chunks:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:" + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def entry(self, ident, role="measured", suite="confirmation", round=0):
        request = {"model": "unchanged-fixture-name", "stream": True,
                   "max_tokens": 64 if role == "warmup" else 512,
                   "messages": [{"role": "user", "content": ident}]}
        if suite == "disk":
            request["max_tokens"] = 256
        body = json.dumps(request, separators=(",", ":")).encode()
        path = "requests/" + ident + ".json.gz"
        (self.root / path).write_bytes(gzip.compress(body, mtime=0))
        return {"id": ident, "suite": suite, "role": role, "depth": 1000, "round": round,
                "path": path, "request_sha256": hashlib.sha256(body).hexdigest(),
                "expected_prompt_tokens": 1000}, body

    def run_helper(self, entries, *args):
        (self.root / "requests/index.json").write_text(json.dumps({"requests": entries}))
        out = self.root / ("out-" + str(len(list(self.root.glob("out-*")))))
        env = dict(os.environ, STRATA_BENCHMARK_API_KEY="fixture-secret-do-not-record")
        proc = subprocess.run([sys.executable, str(self.root / "replay.py"), "--url", self.url,
                               "--suite", "confirmation", "--out", str(out), *args],
                              capture_output=True, text=True, env=env, timeout=15)
        self.assertTrue((out / "run.json").exists(), "Replay must record run.json: " + proc.stderr)
        return proc, out, json.loads((out / "run.json").read_text())

    def test_frozen_order_and_actual_output(self):
        measured, measured_body = self.entry("measured")
        warm, warm_body = self.entry("warm", role="warmup")
        ignored, _ = self.entry("other-round", round=1)
        proc, out, run = self.run_helper([measured, ignored, warm], "--round", "0")
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        self.assertEqual([body for _, body in self.received], [warm_body, measured_body])
        row = run["results"][1]
        self.assertEqual(row["content"], "2 + 2 = 5")
        self.assertEqual(row["reasoning_content"], "actual reasoning")
        self.assertEqual(row["tool_calls"][0]["function"]["arguments"], '{"x":5}')
        self.assertEqual(row["engine_prefill_tok_s"], 5000)
        self.assertEqual(row["engine_decode_tok_s"], 1280)
        self.assertIsNotNone(row["client_ttft_s"])
        self.assertEqual(len(row["harness_sha256"]), 64)
        self.assertTrue(row["eligible"])
        self.assertTrue(any(b"[DONE]" in path.read_bytes() for path in out.rglob("*.sse")))
        self.assertEqual(self.headers[0]["Authorization"], "Bearer fixture-secret-do-not-record")
        self.assertNotIn("fixture-secret-do-not-record", (out / "run.json").read_text())

    def test_stream_failures_recorded(self):
        for behavior in ("missing-done", "error", "malformed", "timeout", "http-error"):
            with self.subTest(behavior=behavior):
                self.behavior = behavior
                entry, _ = self.entry(behavior)
                proc, _, run = self.run_helper([entry], "--timeout", "1")
                self.assertEqual(proc.returncode, 1, proc.stderr + proc.stdout)
                self.assertFalse(run["results"][0]["eligible"])
                self.assertEqual(run["results"][0]["status"], "failed")
                self.assertTrue(run["results"][0]["error"])
                if behavior == "timeout":
                    self.assertIn("timeout", run["results"][0]["error"].lower())

    def test_ineligible_counts_are_failures(self):
        for behavior in ("wrong-prompt", "reused", "short"):
            with self.subTest(behavior=behavior):
                self.behavior = behavior
                entry, _ = self.entry(behavior)
                proc, _, run = self.run_helper([entry])
                self.assertEqual(proc.returncode, 1)
                self.assertFalse(run["results"][0]["eligible"])
                self.assertEqual(run["results"][0]["content"], "2 + 2 = 5")

    def test_sha_mismatch_never_sends(self):
        entry, _ = self.entry("bad-sha")
        entry["request_sha256"] = "0" * 64
        proc, _, run = self.run_helper([entry])
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(self.received, [])
        self.assertIn("SHA", run["error"])

    def test_disk_phase_and_native_slot_actions(self):
        initial, _ = self.entry("disk-initial", role="initial", suite="disk")
        continuation, body = self.entry("disk-continuation", role="continuation", suite="disk")
        continuation["previous_prefix_tokens"] = 900
        proc, _, run = self.run_helper([initial, continuation], "--suite", "disk", "--disk-phase", "continuation", "--depth", "1000")
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        self.assertEqual(self.received[0][1], body)
        self.assertEqual(run["results"][0]["previous_prefix_reuse_fraction"], 1)
        for action in ("save", "restore"):
            proc, _, run = self.run_helper([], "--slot-action", action, "--slot-filename", "orchestrator.bin")
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            self.assertEqual(self.received[-1][0], "/slots/0?action=" + action)
            self.assertEqual(json.loads(self.received[-1][1]), {"filename": "orchestrator.bin"})
            self.assertEqual(run["slot_action"]["response"]["n_saved"], 1000)
            self.assertGreaterEqual(run["slot_action"]["client_total_s"], 0)

    def test_unsafe_slot_basename_never_sends(self):
        proc, _, run = self.run_helper([], "--slot-action", "save", "--slot-filename", "../oops.bin")
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(self.received, [])
        self.assertIn("basename", run["error"])

    def test_decode_preserves_sequence_and_cache_eligibility(self):
        warm0, body0 = self.entry("decode-warm0", role="warmup", suite="decode")
        measured0, body1 = self.entry("decode-measured0", suite="decode")
        warm1, _ = self.entry("decode-warm1", role="warmup", suite="decode", round=1)
        measured1, _ = self.entry("decode-measured1", suite="decode", round=1)
        proc, _, run = self.run_helper([warm0, measured0, warm1, measured1], "--suite", "decode", "--round", "0")
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        self.assertEqual([body for _, body in self.received], [body0, body1])
        self.assertTrue(run["results"][1]["eligible"])
        self.behavior = "reused"
        proc, _, run = self.run_helper([measured0], "--suite", "decode")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("99.9%", run["results"][0]["ineligible_reasons"][0])

    def test_disk_wrong_needles_remain_actual(self):
        entry, _ = self.entry("disk-initial", role="initial", suite="disk")
        entry["expected_needles"] = {"AUDIT_ROUTE_1": "fixture expected value"}
        proc, _, run = self.run_helper([entry], "--suite", "disk")
        self.assertEqual(proc.returncode, 1)
        self.assertFalse(run["results"][0]["needle_grade"]["passed"])
        self.assertEqual(run["results"][0]["content"], "2 + 2 = 5")


if __name__ == "__main__":
    unittest.main(verbosity=2)
