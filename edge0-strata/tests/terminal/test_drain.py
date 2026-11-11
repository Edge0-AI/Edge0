"""Opt-in serving drain: reject new work, finish owned calls, cancel on deadline."""

import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve
from strata_cli.client import Endpoint
from strata_cli.render import render_view


class DrainTests(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "Hello.", max_context=4096), tok,
                           ChatTemplate(Path(__file__).resolve().parents[2] / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def test_drain_rejects_new_inference_and_reports_state(self):
        self.assertEqual(self.svc.begin_drain(), 0)
        request = urllib.request.Request(self.url + "/v1/chat/completions", method="POST",
                                         headers={"Content-Type": "application/json"},
                                         data=json.dumps({"messages": [{"role": "user", "content": "Hi"}]}).encode())
        with self.assertRaises(urllib.error.HTTPError) as captured:
            urllib.request.urlopen(request, timeout=3)
        with captured.exception as response:
            self.assertEqual(response.status, 503)
            self.assertEqual(json.loads(response.read())["error"]["code"], "server_draining")
        snapshot = Endpoint(self.url).snapshot()
        self.assertEqual(snapshot["instance"]["state"], "draining")
        self.assertIn("DRAINING", render_view(snapshot, "overview", 80, 24))
        self.assertEqual(self.svc.admission_rejected, 0)

    def test_drain_waits_for_reserved_call_then_releases(self):
        self.assertTrue(self.svc.reserve_inference())
        self.assertEqual(self.svc.begin_drain(), 1)
        self.assertIsNone(self.svc.reserve_inference())
        timer = threading.Timer(0.08, self.svc.release_inference)
        timer.start()
        self.addCleanup(timer.join)
        start = time.monotonic()
        self.assertEqual(self.svc.wait_for_drain(1), 0)
        self.assertGreaterEqual(time.monotonic() - start, 0.05)

    def test_deadline_and_owned_cancellation(self):
        self.assertTrue(self.svc.reserve_inference())
        cancel = threading.Event()
        with self.svc.status_lock:
            self.svc.active_cancels.add(cancel)
        self.svc.begin_drain()
        self.assertEqual(self.svc.wait_for_drain(0), 1)
        self.assertEqual(self.svc.cancel_owned_requests(), 1)
        self.assertTrue(cancel.is_set())
        self.svc.release_inference()
        self.assertEqual(self.svc.wait_for_drain(0), 0)


if __name__ == "__main__":
    unittest.main()
