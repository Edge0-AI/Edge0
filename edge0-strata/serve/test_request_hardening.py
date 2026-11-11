"""0.1.41 server hardening: a JSON body of the wrong shape is a 400 (not a dropped connection), an oversized body is a
413 before it is read, a bad Content-Length is a 400, and a sampling field of the wrong type is a 400 that names it.

    python -m unittest serve.test_request_hardening -v
"""
import http.client
import json
import os
import unittest
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, body_limit, serve

ROOT = Path(__file__).resolve().parent.parent


class Hardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "hello there", max_context=4096), tok,
                          ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        try:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            c.request("POST", path, body=data, headers={"Content-Type": "application/json", **(headers or {})})
            r = c.getresponse()
            return r.status, json.loads(r.read() or b"{}")
        finally:
            c.close()

    def test_wrong_shapes_are_400(self):
        for path, body in (("/v1/chat/completions", {"messages": 5}),
                           ("/v1/chat/completions", {"messages": [7]}),
                           ("/v1/chat/completions", {"messages": [{"role": "user", "content": 5}]}),
                           ("/v1/messages", {"messages": [None], "max_tokens": 4}),
                           ("/v1/messages", {"messages": "hi", "max_tokens": 4}),
                           ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "tools": 3})):
            with mock.patch("builtins.print"):
                code, out = self.post(path, body)
            self.assertEqual(code, 400, (path, body, out))
            self.assertIn("error", out)

    def test_not_json_is_400(self):
        with mock.patch("builtins.print"):
            code, out = self.post("/v1/chat/completions", b"{nope")
        self.assertEqual(code, 400, out)

    def test_a_good_request_still_answers(self):
        code, out = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
        self.assertEqual(code, 200, out)

    def test_oversized_content_length_is_413_unread(self):
        with mock.patch.dict(os.environ, {"STRATA_MAX_BODY_MIB": "1"}):
            self.assertEqual(body_limit(), 1 << 20)
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
            c.putrequest("POST", "/v1/chat/completions")
            c.putheader("Content-Type", "application/json")
            c.putheader("Content-Length", str(10 << 40))          # 10 TiB announced, nothing sent
            c.endheaders()
            r = c.getresponse()
            self.assertEqual(r.status, 413)
            self.assertIn("STRATA_MAX_BODY_MIB", json.loads(r.read())["error"]["message"])
            c.close()

    def test_negative_or_bad_content_length_is_400(self):
        for value in ("-5", "abc"):
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
            c.putrequest("POST", "/v1/chat/completions")
            c.putheader("Content-Type", "application/json")
            c.putheader("Content-Length", value)
            c.endheaders()
            r = c.getresponse()
            self.assertEqual(r.status, 400, value)
            c.close()

    def test_default_limit_is_generous(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATA_MAX_BODY_MIB", None)
            self.assertEqual(body_limit(), 256 << 20)
        with mock.patch.dict(os.environ, {"STRATA_MAX_BODY_MIB": "junk"}):
            self.assertEqual(body_limit(), 256 << 20)

    # A request whose sampling field has the wrong JSON type is a 400 that names the field, not a greedy reply.
    # Only types are checked: in-type values of any range, and null fields, answer as before.
    def test_a_wrong_type_is_a_400_that_names_the_field(self):
        chat = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
        responses = {"model": "m", "input": "hi", "store": False, "max_output_tokens": 8}
        for path, body, field in (
                ("/v1/chat/completions", {**chat, "temperature": "abc"}, "temperature"),
                ("/v1/chat/completions", {**chat, "top_k": 40.0}, "top_k"),
                ("/v1/chat/completions", {**chat, "seed": "x7"}, "seed"),
                ("/v1/messages", {**chat, "top_k": [40]}, "top_k"),
                ("/v1/messages", {**chat, "repetition_penalty": {"a": 1}}, "repetition_penalty"),
                ("/v1/messages", {**chat, "presence_penalty": True}, "presence_penalty"),
                ("/v1/responses", {**responses, "top_p": True}, "top_p"),
                ("/v1/responses", {**responses, "temperature": "hot"}, "temperature"),
                ("/v1/responses", {**responses, "seed": 7.5}, "seed"),
                ("/v1/messages/count_tokens", {"messages": chat["messages"], "temperature": "abc"},
                 "temperature")):
            with self.subTest(path=path, field=field):
                with mock.patch("builtins.print"):
                    code, out = self.post(path, body)
                self.assertEqual(code, 400, (path, body, out))
                self.assertIn(field, json.dumps(out))

    def test_numeric_strings_are_read_and_used(self):
        from serve.server import check_request_sampling
        req = {"temperature": "0.7", "top_p": " 1 ", "top_k": "40", "seed": "7", "top_k_x": "keep", "min_p": "0"}
        check_request_sampling(req)
        self.assertEqual((req["temperature"], req["top_p"], req["top_k"], req["seed"], req["min_p"]),
                         (0.7, 1, 40, 7, 0))
        self.assertEqual(req["top_k_x"], "keep")
        for bad in ({"top_k": "40.5"}, {"top_k": "abc"}, {"temperature": "nan"}, {"temperature": "true"},
                    {"seed": ""}, {"temperature": [1]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check_request_sampling(dict(bad))
        code, out = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                                       "max_tokens": 8, "temperature": "0.7", "top_k": "40"})
        self.assertEqual(code, 200, out)

    def test_llama_cpp_names_for_the_repetition_penalty(self):
        """#1819: repeat_penalty / repeat_last_n are repetition_penalty / penalty_last_n; the OpenAI-style name wins."""
        from serve.server import SAMPLING_ALIASES, check_request_sampling, sampling_defaults_from_config
        req = {"repeat_penalty": 1.1, "repeat_last_n": 128}
        check_request_sampling(req)
        self.assertEqual((req["repetition_penalty"], req["penalty_last_n"]), (1.1, 128))
        req = {"repeat_penalty": 1.3, "repetition_penalty": 1.05, "repeat_last_n": 9, "penalty_last_n": 32}
        check_request_sampling(req)
        self.assertEqual((req["repetition_penalty"], req["penalty_last_n"]), (1.05, 32))
        req = {"repeat_last_n": 0}                      # off / -1: the engine's default window, nothing is set
        check_request_sampling(req)
        self.assertNotIn("penalty_last_n", req)
        with self.assertRaises(ValueError):
            check_request_sampling({"repeat_penalty": "abc"})
        self.assertEqual(sampling_defaults_from_config({"sampling": {"repeat_penalty": 1.1, "repeat_last_n": 256}}),
                         {"repetition_penalty": 1.1, "penalty_last_n": 256})
        self.assertEqual(sampling_defaults_from_config({"sampling": {"temperature": 0.6}}), {"temperature": 0.6})
        with self.assertRaises(SystemExit):
            sampling_defaults_from_config({"sampling": {"repeat_penalty": 1.1, "repetition_penalty": 1.2}})
        with self.assertRaises(SystemExit):
            sampling_defaults_from_config({"sampling": {"repeat_penalty": -1}})
        self.assertEqual(set(SAMPLING_ALIASES.values()), {"repetition_penalty", "penalty_last_n"})
        code, out = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                                       "max_tokens": 8, "repeat_penalty": 1.1, "repeat_last_n": 64})
        self.assertEqual(code, 200, out)

    def test_in_type_values_and_nulls_still_answer(self):
        chat = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "temperature": 0, "top_k": 0,
                "seed": 7}
        code, out = self.post("/v1/chat/completions", chat)
        self.assertEqual(code, 200, out)
        code, out = self.post("/v1/messages", {**chat, "temperature": -1, "top_k": 100, "presence_penalty": 0.5,
                                               "penalty_last_n": 64})
        self.assertEqual(code, 200, out)
        code, out = self.post("/v1/responses", {"model": "m", "input": "hi", "store": False, "max_output_tokens": 8,
                                                "temperature": None, "top_p": None, "seed": None})
        self.assertEqual(code, 200, out)
        code, out = self.post("/v1/messages/count_tokens", {"messages": [{"role": "user", "content": "hi"}],
                                                            "temperature": 0.7})
        self.assertEqual(code, 200, out)


if __name__ == "__main__":
    unittest.main()
