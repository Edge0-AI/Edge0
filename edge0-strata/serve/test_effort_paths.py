"""serve/test_effort_paths.py - every way a client sets the thinking level reaches the model's prompt (mock engine, no GPU).

The level is nothing but a sentence in the rendered prompt (the template: xhigh / low sentences, none = the empty think
block, medium = no sentence), so "the effort does not work" is always a request form whose level never reached the template.
This pins the prompt each form produces, over the three APIs, the Chat settings shared with apps and the config's
"reasoning_effort" (#1641).  0.1.44.1: the Responses API (/v1/responses, Codex) ignored the config's level.

    python -m unittest serve.test_effort_paths -v
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MSG = [{"role": "user", "content": "hello"}]


def level_of(prompt: str) -> str:
    """What the template made of the level: xhigh and low have their sentence, none ends in the empty think block, and
    medium is the prompt with no sentence at all (the model's own wording)."""
    if prompt.rstrip().endswith("</think>"):
        return "none"
    if "Reasoning effort is set to low." in prompt:
        return "low"
    if "Reasoning effort is set to xhigh." in prompt:
        return "xhigh"
    return "medium"


class EffortPaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = MockEngine(cls.tok, "<think>\nhm\n</think>\n\nok", max_context=8192)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.svc.config_effort = None
        self.svc.shared = {}

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        self.engine.last_prompt = []
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                status = r.status
                r.read()
        except urllib.error.HTTPError as e:
            with e:
                status = e.code
        return status

    def level(self, path, body):
        status = self.post(path, body)
        self.assertEqual(status, 200, (path, body))
        return level_of(self.tok.decode(self.engine.last_prompt))

    # ---------------------------------------------------------------- the request's own level, every spelling
    def test_openai_reasoning_effort(self):
        for given, want in ((None, "xhigh"), ("none", "none"), ("off", "none"), ("minimal", "none"), ("low", "low"),
                            ("medium", "medium"), ("high", "xhigh"), ("xhigh", "xhigh"), ("max", "xhigh"),
                            ("LOW", "low"), (False, "none")):
            body = {"messages": MSG, **({} if given is None else {"reasoning_effort": given})}
            with self.subTest(given=given):
                self.assertEqual(self.level("/v1/chat/completions", body), want)

    def test_openai_reasoning_object_and_template_kwargs(self):
        for given, want in (("low", "low"), ("medium", "medium"), ("high", "xhigh"), ("none", "none")):
            with self.subTest(given=given):
                self.assertEqual(self.level("/v1/chat/completions", {"messages": MSG, "reasoning": {"effort": given}}), want)
                self.assertEqual(self.level("/v1/chat/completions",
                                            {"messages": MSG, "chat_template_kwargs": {"reasoning_effort": given}}), want)
        self.assertEqual(self.level("/v1/chat/completions",
                                    {"messages": MSG, "chat_template_kwargs": {"enable_thinking": False}}), "none")

    def test_an_unknown_level_is_a_400_not_the_default(self):
        self.assertEqual(self.post("/v1/chat/completions", {"messages": MSG, "reasoning_effort": "huge"}), 400)

    def test_anthropic_output_config_and_thinking(self):
        for given, want in (("low", "low"), ("medium", "medium"), ("high", "xhigh")):
            with self.subTest(effort=given):
                self.assertEqual(self.level("/v1/messages", {"messages": MSG, "max_tokens": 50,
                                                             "output_config": {"effort": given}}), want)
        for budget, want in ((500, "low"), (2047, "low"), (2048, "medium"), (8191, "medium"), (8192, "xhigh"), (31999, "xhigh")):
            with self.subTest(budget=budget):
                self.assertEqual(self.level("/v1/messages", {"messages": MSG, "max_tokens": 50,
                                                             "thinking": {"type": "enabled", "budget_tokens": budget}}), want)
        self.assertEqual(self.level("/v1/messages", {"messages": MSG, "max_tokens": 50, "thinking": {"type": "disabled"}}),
                         "none")
        self.assertEqual(self.level("/v1/messages", {"messages": MSG, "max_tokens": 50}), "xhigh")   # nothing asked: it thinks

    def test_responses_reasoning_effort(self):
        for given, want in (("low", "low"), ("medium", "medium"), ("high", "xhigh"), ("none", "none"), ("minimal", "none")):
            with self.subTest(given=given):
                self.assertEqual(self.level("/v1/responses", {"input": "hello", "reasoning": {"effort": given}}), want)
        self.assertEqual(self.level("/v1/responses", {"input": "hello"}), "xhigh")

    # ---------------------------------------------------------------- the level of a request that names none
    def test_config_level_reaches_every_api(self):
        for cfg, want in (("low", "low"), ("medium", "medium"), ("high", "xhigh"), ("none", "none")):
            self.svc.config_effort = cfg
            with self.subTest(config=cfg):
                self.assertEqual(self.level("/v1/chat/completions", {"messages": MSG}), want)
                self.assertEqual(self.level("/v1/messages", {"messages": MSG, "max_tokens": 50}), want)
                self.assertEqual(self.level("/v1/responses", {"input": "hello"}), want)      # 0.1.44.1: it ignored the config

    def test_the_requests_own_level_beats_the_config(self):
        self.svc.config_effort = "low"
        self.assertEqual(self.level("/v1/chat/completions", {"messages": MSG, "reasoning_effort": "high"}), "xhigh")
        self.assertEqual(self.level("/v1/messages", {"messages": MSG, "max_tokens": 50, "output_config": {"effort": "high"}}),
                         "xhigh")
        self.assertEqual(self.level("/v1/messages", {"messages": MSG, "max_tokens": 50,
                                                     "thinking": {"type": "enabled", "budget_tokens": 20000}}), "xhigh")
        self.assertEqual(self.level("/v1/responses", {"input": "hello", "reasoning": {"effort": "high"}}), "xhigh")
        self.assertEqual(self.level("/v1/chat/completions", {"messages": MSG, "reasoning_effort": "none"}), "none")
        self.assertEqual(self.level("/v1/chat/completions", {"messages": MSG, "reasoning_effort": False}), "none")

    def test_the_chat_settings_shared_with_apps_beat_the_config(self):
        self.svc.config_effort = "low"
        self.svc.shared = {"reasoning_effort": "high"}
        for path, body in (("/v1/chat/completions", {"messages": MSG}), ("/v1/messages", {"messages": MSG, "max_tokens": 50}),
                           ("/v1/responses", {"input": "hello"})):
            with self.subTest(path=path):
                self.assertEqual(self.level(path, body), "xhigh")
        self.svc.shared = {"reasoning_effort": "none"}
        for path, body in (("/v1/chat/completions", {"messages": MSG}), ("/v1/messages", {"messages": MSG, "max_tokens": 50}),
                           ("/v1/responses", {"input": "hello"})):
            with self.subTest(path=path, shared="none"):
                self.assertEqual(self.level(path, body), "none")

    def test_the_web_apps_request_shape(self):
        # serve/web/app.js: {model, messages, stream, reasoning_effort: settings.thinking} with none / low / medium / high
        for given, want in (("none", "none"), ("low", "low"), ("medium", "medium"), ("high", "xhigh")):
            with self.subTest(given=given):
                self.assertEqual(self.level("/v1/chat/completions", {"model": "m", "messages": MSG, "stream": True,
                                                                     "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                                                                     "reasoning_effort": given}), want)

    def test_tools_and_a_system_message_keep_the_level(self):
        tools = [{"type": "function", "function": {"name": "f", "description": "d",
                                                    "parameters": {"type": "object", "properties": {}}}}]
        body = {"messages": [{"role": "system", "content": "be brief"}] + MSG, "tools": tools}
        self.assertEqual(self.level("/v1/chat/completions", {**body, "reasoning_effort": "low"}), "low")
        self.assertEqual(self.level("/v1/chat/completions", {**body, "reasoning_effort": "high"}), "xhigh")
        self.assertEqual(self.level("/v1/chat/completions", {**body, "reasoning_effort": "none"}), "none")

    def test_the_level_follows_a_conversation(self):
        history = MSG + [{"role": "assistant", "content": "hi", "reasoning_content": "greet"},
                         {"role": "user", "content": "and now?"}]
        for given, want in (("low", "low"), ("medium", "medium"), ("none", "none")):
            with self.subTest(given=given):
                self.assertEqual(self.level("/v1/chat/completions", {"messages": history, "reasoning_effort": given}), want)


    def test_the_agents_request_shape_with_thinking_off(self):
        """The app's agent (Harness) sends no reasoning_effort when Thinking is Off, only chat_template_kwargs {enable_thinking: false}, with tools, a
        system message, earlier reasoning and a token cap. That one flag must reach the template as the empty think block, whatever else is in the
        request or configured (a thinking model that went on reasoning with the menu on Off was this request form).  The same body with
        enable_thinking true / an effort keeps thinking."""
        tools = [{"type": "function", "function": {"name": "pwsh", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
        history = [{"role": "developer", "content": "You are a helpful assistant."}, {"role": "user", "content": "first"},
                   {"role": "assistant", "content": "ok", "reasoning_content": "earlier thoughts"}, {"role": "user", "content": "now?"}]
        agent = {"model": "m", "messages": history, "tools": tools, "stream": True, "stream_options": {"include_usage": True},
                 "max_completion_tokens": 4096}
        for config in (None, "high", "low"):
            self.svc.config_effort = config
            with self.subTest(config=config):
                self.assertEqual(self.level("/v1/chat/completions", {**agent, "chat_template_kwargs": {"enable_thinking": False}}), "none")
        self.svc.config_effort = None
        self.assertEqual(self.level("/v1/chat/completions", {**agent, "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}}), "low")
        self.assertEqual(self.level("/v1/chat/completions", {**agent, "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}}), "xhigh")
        # the same flag in the plain (no tools) body the direct chat sends alongside a budget
        self.assertEqual(self.level("/v1/chat/completions", {"messages": MSG, "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}), "none")


if __name__ == "__main__":
    unittest.main()
