"""Calls to this computer's own servers must ignore the system proxy (a company proxy cannot reach 127.0.0.1)."""

import http.server
import os
import threading
import unittest
from unittest import mock

from strata_control import loopback


class Hello(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


class LoopbackTests(unittest.TestCase):
    def test_is_loopback(self):
        for u in ("http://127.0.0.1:1/x", "http://localhost:2/", "http://[::1]:3/"):
            self.assertTrue(loopback.is_loopback(u), u)
        for u in ("https://example.com/", "http://192.168.1.5:8080/", "not a url"):
            self.assertFalse(loopback.is_loopback(u), u)

    def test_local_server_is_reached_with_a_dead_proxy_set(self):
        srv = http.server.HTTPServer(("127.0.0.1", 0), Hello)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            env = {"HTTP_PROXY": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9", "NO_PROXY": "", "no_proxy": ""}
            with mock.patch.dict(os.environ, env):
                with loopback.urlopen(f"http://127.0.0.1:{srv.server_port}/", timeout=5) as r:
                    self.assertEqual(r.read(), b"ok")
        finally:
            srv.shutdown()

    def test_other_addresses_still_use_urllib(self):
        with mock.patch("urllib.request.urlopen", return_value="remote") as m:
            self.assertEqual(loopback.urlopen("https://example.com/", timeout=3), "remote")
            m.assert_called_once()


if __name__ == "__main__":
    unittest.main()
