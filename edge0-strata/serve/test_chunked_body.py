"""#893 / #894: a request body sent as Transfer-Encoding: chunked (a relay or proxy) is decoded, on every route that
reads a body; a malformed or oversized one is a 400 / 413, not an empty body. Incomplete chunked framing must not
apply an otherwise valid JSON body.

    python -m unittest serve.test_chunked_body -v
"""
import json
import socket
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve
import serve.server as server

ROOT = Path(__file__).resolve().parent.parent
CRLF = b"\r\n"


def chunked(data: bytes, size: int = 7, trailers: bytes = b"") -> bytes:
    out = b""
    for i in range(0, len(data), size):
        piece = data[i:i + size]
        out += b"%x;ext=1" % len(piece) + CRLF + piece + CRLF
    return out + b"0" + CRLF + trailers + CRLF


class ChunkedBody(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = MockEngine(tok, "hello there", max_context=4096)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def send(self, head: bytes, body: bytes):
        """-> (status, JSON body) of one raw request."""
        with socket.create_connection(("127.0.0.1", self.port), timeout=20) as s:
            s.sendall(b"POST " + head + b"Host: 127.0.0.1" + CRLF + b"Connection: close" + CRLF + CRLF + body)
            s.shutdown(socket.SHUT_WR)
            data = b""
            while True:
                piece = s.recv(65536)
                if not piece:
                    break
                data += piece
        status = int(data.split(b" ", 2)[1])
        payload = data.split(CRLF + CRLF, 1)[1]
        return status, (json.loads(payload) if payload.strip().startswith(b"{") else payload)

    def chat(self, extra: bytes, body: bytes):
        return self.send(b"/v1/chat/completions HTTP/1.1" + CRLF + b"Content-Type: application/json" + CRLF + extra, body)

    def test_a_chunked_chat_request_is_read(self):
        req = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 20}).encode()
        status, body = self.chat(b"Transfer-Encoding: chunked" + CRLF, chunked(req))
        self.assertEqual(status, 200, body)
        self.assertIn("choices", body)

    def test_chunked_wins_over_content_length(self):
        req = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 20}).encode()
        status, body = self.chat(b"Transfer-Encoding: chunked" + CRLF + b"Content-Length: 3" + CRLF,
                                 chunked(req, trailers=b"X-Trailer: 1" + CRLF))
        self.assertEqual(status, 200, body)

    def test_a_content_length_request_is_unchanged(self):
        req = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 20}).encode()
        status, body = self.chat(b"Content-Length: %d" % len(req) + CRLF, req)
        self.assertEqual(status, 200, body)

    def test_a_bad_chunk_size_is_a_400(self):
        status, body = self.chat(b"Transfer-Encoding: chunked" + CRLF, b"zz" + CRLF + b"{}" + CRLF + b"0" + CRLF + CRLF)
        self.assertEqual(status, 400, body)
        self.assertIn("chunk", body["error"]["message"])

    def test_a_truncated_chunked_body_is_a_400(self):
        status, body = self.chat(b"Transfer-Encoding: chunked" + CRLF, b"10" + CRLF + b"{}")
        self.assertEqual(status, 400, body)

    def test_an_oversized_chunked_body_is_a_413_and_is_never_allocated(self):
        old = server.CHUNKED_BODY_MAX
        server.CHUNKED_BODY_MAX = 1000
        try:
            status, body = self.chat(b"Transfer-Encoding: chunked" + CRLF, b"FFFFFFFF" + CRLF)   # announces 4 GiB
            self.assertEqual(status, 413, body)
            status, body = self.chat(b"Transfer-Encoding: chunked" + CRLF, chunked(b"x" * 3000, size=500))
            self.assertEqual(status, 413, body)
        finally:
            server.CHUNKED_BODY_MAX = old

    def test_a_chunked_settings_body_is_read_too(self):
        body = json.dumps({"defaults": {"temperature": 0.5}}).encode()
        head = (b"/settings HTTP/1.1" + CRLF + b"Content-Type: application/json" + CRLF +
                b"Transfer-Encoding: chunked" + CRLF)
        status, got = self.send(head, chunked(body))
        self.assertEqual(status, 200, got)

    def assert_settings_body_rejected(self, extra: bytes, body: bytes):
        previous = dict(self.svc.shared)
        self.addCleanup(self.svc.set_shared, previous)
        before = self.svc.set_shared({"temperature": 0.7})
        head = b"/settings HTTP/1.1" + CRLF + b"Content-Type: application/json" + CRLF + extra
        status, got = self.send(head, body)
        self.assertEqual((status, self.svc.shared), (400, before), got)

    def test_incomplete_chunked_trailers_do_not_change_settings(self):
        body = json.dumps({"defaults": {"temperature": 0.5}}).encode()
        prefix = chunked(body)[:-len(CRLF)]                 # the zero chunk, without the terminating blank line
        for tail in (b"", b"X-Trailer: 1", b"X-Trailer: 1" + CRLF, b"\r", b"\n", b" " + CRLF):
            with self.subTest(tail=tail):
                self.assert_settings_body_rejected(b"Transfer-Encoding: chunked" + CRLF, prefix + tail)

    def test_chunked_trailer_limit_does_not_change_settings(self):
        body = json.dumps({"defaults": {"temperature": 0.5}}).encode()
        prefix = chunked(body)[:-len(CRLF)]
        for end in (b"", CRLF):
            with self.subTest(end=end):
                trailers = (b"X-Trailer: 1" + CRLF) * 64 + end
                self.assert_settings_body_rejected(b"Transfer-Encoding: chunked" + CRLF, prefix + trailers)

    def test_an_overlong_chunked_trailer_does_not_change_settings(self):
        body = json.dumps({"defaults": {"temperature": 0.5}}).encode()
        trailers = b"X-Trailer: " + b"x" * 8193 + CRLF
        self.assert_settings_body_rejected(b"Transfer-Encoding: chunked" + CRLF, chunked(body, trailers=trailers))

    def test_complete_settings_bodies_still_apply(self):
        previous = dict(self.svc.shared)
        self.addCleanup(self.svc.set_shared, previous)
        body = json.dumps({"defaults": {"temperature": 0.5}}).encode()
        for extra, framed in ((b"Content-Length: %d" % len(body) + CRLF, body),
                              (b"Transfer-Encoding: chunked" + CRLF, chunked(body)),
                              (b"Transfer-Encoding: chunked" + CRLF,
                               chunked(body, trailers=(b"X-Trailer: 1" + CRLF) * 63))):
            with self.subTest(extra=extra, size=len(framed)):
                self.svc.set_shared({"temperature": 0.7})
                head = b"/settings HTTP/1.1" + CRLF + b"Content-Type: application/json" + CRLF + extra
                status, got = self.send(head, framed)
                self.assertEqual((status, self.svc.shared), (200, {"temperature": 0.5}), got)

    def test_a_chunked_control_body_is_consumed(self):
        head = (b"/unload HTTP/1.1" + CRLF + b"Content-Type: application/json" + CRLF +
                b"Transfer-Encoding: chunked" + CRLF)
        status, got = self.send(head, chunked(b"{}"))
        self.assertIn(status, (200, 409), got)


if __name__ == "__main__":
    unittest.main()
