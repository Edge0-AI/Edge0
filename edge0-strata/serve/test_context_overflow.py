"""#1615: recognizable context-overflow errors over HTTP, without a GPU or model pack.

    python -m unittest serve.test_context_overflow -v
"""
import http.client
import json
import unittest
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import CTX_SLACK, ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]
CTX = 131072
APIS = (("/v1/chat/completions", "max_tokens"),
        ("/v1/messages", "max_tokens"),
        ("/v1/responses", "max_output_tokens"))


class ContextOverflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = MockEngine(tok, "ok", max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.svc.fit_max_tokens = False

    def post(self, path, budget_key, prompt_tokens, max_tokens=None, stream=False):
        body = {"input": "hi"} if path == "/v1/responses" else {
            "messages": [{"role": "user", "content": "hi"}]}
        body["stream"] = stream
        if max_tokens is not None:
            body[budget_key] = max_tokens
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            with mock.patch.object(self.svc, "encode_prompt", return_value=[ord("x")] * prompt_tokens):
                c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
                r = c.getresponse()
                return r.status, r.getheader("Content-Type"), json.loads(r.read())
        finally:
            c.close()

    def test_both_overflow_errors_are_recognizable_before_streaming(self):
        # The full prompt case reproduces #1615. fit_max_tokens cannot rescue it.
        cases = ((150722, None, False, "no room to answer"),
                 (150722, 16, True, "no room to answer"),
                 (150722, 16, False, "+ max tokens (16)"),
                 (1024, CTX, False, f"+ max tokens ({CTX})"))
        for path, key in APIS:
            for stream in (False, True):
                for prompt, budget, fit, detail in cases:
                    with self.subTest(path=path, stream=stream, prompt=prompt, budget=budget, fit=fit):
                        self.svc.fit_max_tokens = fit
                        with mock.patch.object(self.engine, "generate") as generate:
                            status, content_type, body = self.post(path, key, prompt, budget, stream)
                        generate.assert_not_called()
                        self.assertEqual(status, 400, body)
                        self.assertIn("application/json", content_type)
                        error = body["error"]
                        self.assertEqual(error["type"], "invalid_request_error")
                        message = error["message"]
                        # OpenAI's established wording, recognized by client overflow classifiers.
                        self.assertIn("exceeds the context window", message)
                        self.assertIn(f"prompt ({prompt} tokens)", message)
                        self.assertIn(detail, message)
                        self.assertIn(f"({CTX})", message)
                        self.assertIn("requests are never truncated", message)
                        if "+ max tokens" in detail:
                            room = max(0, CTX - CTX_SLACK - prompt)
                            self.assertIn(f"max_tokens (at most {room} here)", message)
                            self.assertIn('"fit_max_tokens": true', message)
                        if path == "/v1/responses":
                            self.assertEqual(error["code"], "context_length_exceeded")
                            self.assertEqual(error["param"], "input")
                        else:
                            self.assertEqual(set(error), {"type", "message"})

    def test_default_service_shortens_an_overshooting_budget_in_every_dialect(self):
        # the Service default (fit_max_tokens on): a prompt that fits is answered, streaming or not; the prompt that
        # alone leaves no room is the 400 of the strict test above
        self.assertTrue(Service(self.engine, ByteTokenizer(), self.svc.template).fit_max_tokens)
        self.svc.fit_max_tokens = True
        for path, key in APIS:
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    if not stream:      # (a 200 stream is SSE, not the JSON post() reads)
                        with mock.patch.object(self.engine, "generate", wraps=self.engine.generate) as generate:
                            status, _, body = self.post(path, key, 1024, CTX, stream)
                        self.assertEqual(status, 200, body)
                        self.assertEqual(generate.call_args.args[1], CTX - CTX_SLACK - 1024)
                    status, _, body = self.post(path, key, CTX - CTX_SLACK, CTX, stream)
                    self.assertEqual(status, 400, body)
                    self.assertIn("no room to answer", body["error"]["message"])

    def test_one_token_of_room_still_answers_but_zero_is_rejected(self):
        for path, key in APIS:
            for fit in (False, True):
                with self.subTest(path=path, fit=fit):
                    self.svc.fit_max_tokens = fit
                    status, _, body = self.post(path, key, CTX - CTX_SLACK - 1, 1)
                    self.assertEqual(status, 200, body)
                    status, _, body = self.post(path, key, CTX - CTX_SLACK, 1)
                    self.assertEqual(status, 400, body)
                    self.assertIn("exceeds the context window", body["error"]["message"])

    def test_prepare_preserves_prompt_and_budget_limits(self):
        ids = [ord("x")] * 1024
        room = CTX - CTX_SLACK - len(ids)
        for fit in (False, True):
            self.svc.fit_max_tokens = fit
            for budget in (None, 0, -1, 1, room, room + 1):
                with self.subTest(fit=fit, budget=budget), \
                        mock.patch.object(self.svc, "encode_prompt", return_value=ids):
                    if budget == room + 1 and not fit:
                        with self.assertRaises(ValueError):
                            self.svc.prepare([], None, {}, budget)
                    else:
                        prepared, _, maximum = self.svc.prepare([], None, {}, budget)
                        self.assertEqual(prepared, ids)
                        self.assertEqual(maximum, 1 if budget == 1 else room)


if __name__ == "__main__":
    unittest.main()
