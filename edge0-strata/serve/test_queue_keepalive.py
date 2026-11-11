"""#1619: a request waiting for the single request turn sends keep-alive comments (a client with a stream idle
timeout must not abort a healthy queued request), and one cancelled while queued leaves without taking the turn.

    python -m unittest serve.test_queue_keepalive -v
"""
import http.client
import json
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import serve.server as server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]


class QueueKeepAlive(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        p = mock.patch.object(server, "QUEUE_BEAT_S", 0.1)
        p.start()
        self.addCleanup(p.stop)

    def test_a_queued_stream_gets_keep_alives_then_its_answer(self):
        self.svc.fifo.acquire()                           # another request holds the turn
        c = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=10)
        self.addCleanup(c.close)
        body = {"stream": True, "messages": [{"role": "user", "content": "hi"}]}
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        seen = b""
        t0 = time.monotonic()
        while seen.count(b": keep-alive") < 2 and time.monotonic() - t0 < 5:
            seen += r.fp.readline()
        self.assertGreaterEqual(seen.count(b": keep-alive"), 2, seen)
        self.assertEqual(self.svc.status["queued"], 1)
        self.svc.fifo.release()
        rest = r.read()
        self.assertIn(b"[DONE]", rest)
        self.assertIn(b'"finish_reason": "stop"', rest)     # the answer ran once the turn was free
        self.assertEqual(self.svc.status["queued"], 0)

    def test_a_request_cancelled_while_queued_never_takes_the_turn(self):
        self.svc.fifo.acquire()
        self.addCleanup(self.svc.fifo.release)
        cancel = threading.Event()
        ids = self.svc.tok.encode("hi")
        run = self.svc.run(ids, False, None, 8, {}, cancel)
        first = next(run)
        self.assertEqual(first, ("ping", None))
        self.assertEqual(self.svc.status["queued"], 1)
        cancel.set()
        rest = list(run)
        self.assertEqual([k for k, _ in rest], ["done"])
        self.assertEqual(rest[0][1]["finish"], "cancel")
        self.assertEqual(self.svc.status["queued"], 0)
        self.assertTrue(self.svc.fifo.locked())            # still the other request's: we never held it

    def test_closing_a_queued_request_is_no_longer_queued(self):
        self.svc.fifo.acquire()
        self.addCleanup(self.svc.fifo.release)
        run = self.svc.run(self.svc.tok.encode("hi"), False, None, 8, {}, threading.Event())
        next(run)
        run.close()
        self.assertEqual(self.svc.status["queued"], 0)


if __name__ == "__main__":
    unittest.main()
