"""serve/test_anthropic_base_url.py - an Anthropic client reaches the server by the root address, and by the "/v1" one too.

The Anthropic SDKs (and Claude Code) add /v1/messages to the base URL themselves. The app used to show the OpenAI-style base
(.../v1) in its Anthropic tab, so those clients posted /v1/v1/messages and got a 404. The address is now shown without /v1, and
the server accepts /v1/v1/messages (and /v1/v1/messages/count_tokens) as the same request, for people who already pasted the old one.

Uses the real `anthropic` Python client against the mock-engine server (no network, no key). Skipped when the package is missing.

    python -m unittest serve.test_anthropic_base_url -v
"""
from __future__ import annotations

import http.client
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

try:
    import anthropic
except ImportError:                                           # pragma: no cover
    anthropic = None


REPLY = "<think>\nhm\n</think>\n\nhello from the mock"


class AnthropicBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = MockEngine(tok, REPLY, max_context=4096)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]
        cls.root = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def raw(self, path, body):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        try:
            c.request("POST", path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            r = c.getresponse()
            return r.status, json.loads(r.read() or b"{}")
        finally:
            c.close()

    @unittest.skipIf(anthropic is None, "the anthropic package is not installed")
    def test_sdk_with_the_root_address_works(self):
        client = anthropic.Anthropic(base_url=self.root, api_key="not-needed", max_retries=0)
        msg = client.messages.create(model="strata", max_tokens=32, messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(msg.type, "message")
        self.assertIn("hello", "".join(b.text for b in msg.content if b.type == "text"))

    @unittest.skipIf(anthropic is None, "the anthropic package is not installed")
    def test_sdk_with_the_old_v1_address_works_through_the_alias(self):
        client = anthropic.Anthropic(base_url=self.root + "/v1", api_key="not-needed", max_retries=0)
        msg = client.messages.create(model="strata", max_tokens=32, messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(msg.type, "message")
        self.assertIn("hello", "".join(b.text for b in msg.content if b.type == "text"))

    @unittest.skipIf(anthropic is None, "the anthropic package is not installed")
    def test_sdk_streaming_through_the_alias(self):
        client = anthropic.Anthropic(base_url=self.root + "/v1", api_key="not-needed", max_retries=0)
        with client.messages.stream(model="strata", max_tokens=32, messages=[{"role": "user", "content": "hi"}]) as stream:
            text = "".join(stream.text_stream)
        self.assertIn("hello", text)

    def test_count_tokens_alias(self):
        body = {"model": "strata", "messages": [{"role": "user", "content": "hi"}]}
        code, real = self.raw("/v1/messages/count_tokens", body)
        code2, aliased = self.raw("/v1/v1/messages/count_tokens", body)
        self.assertEqual((code, code2), (200, 200))
        self.assertEqual(real, aliased)

    def test_only_the_two_messages_paths_are_aliased(self):
        body = {"model": "strata", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
        for path in ("/v1/v1/chat/completions", "/v1/v1/responses", "/v1/v1/v1/messages", "/v1/v2/messages"):
            code, _ = self.raw(path, body)
            self.assertEqual(code, 404, path)


if __name__ == "__main__":
    unittest.main()
