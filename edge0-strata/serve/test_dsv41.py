"""DeepSeek-V4.1 through the server: its tokenizer, its chat template, the DSML tool calls, the effort words, the end
token and the model-card defaults, over both APIs against a fake engine (no GPU, no pack, no model).

    python -m unittest serve.test_dsv41 -v

STRATA_DSV41_PACK=<dir> (the pack's tokenizer folder: vocab.json, merges.txt, token_type.json, tokenizer.json) adds
the check of the real tokenizer against the ids llama.cpp (PR 30817 b72050d0) made of the same rendered prompts.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST  # noqa: E402
from serve import frontend, server  # noqa: E402
from serve.dsml import DsmlOutputParser, DsmlTranslator, to_dsml_force  # noqa: E402
from serve.frontend import ChatTemplate, OutputParser, openai_to_messages  # noqa: E402
from serve.server import (FAMILY_MODEL_NAMES, MockEngine, REASONING_WRAP_UP, Service, default_model_name, model_card_sampling,  # noqa: E402
                          serve)

FIX = ROOT / "serve" / "fixtures" / "deepseek41"
TEMPLATE = FIX / "chat_template.jinja"
GOLDEN = json.loads((FIX / "golden.json").read_text(encoding="utf-8"))

BOS, EOS_TEXT = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
SPECIALS = [(BOS, 3), (EOS_TEXT, 3), ("<｜User｜>", 3), ("<｜Assistant｜>", 3), ("<｜System｜>", 3),
            ("<｜latest_reminder｜>", 3), ("<think>", 4), ("</think>", 4), ("｜DSML｜", 4), ("<｜deepseek_image｜>", 3)]
EOS = 1


def mini_tokenizer() -> ST.Tokenizer:
    """The real Tokenizer class over a tiny vocabulary: DeepSeek's special tokens, then the 256 byte tokens, no merges."""
    b2u = ST.bytes_to_unicode()
    tokens = [s for s, _ in SPECIALS] + [b2u[b] for b in range(256)]
    types = [t for _, t in SPECIALS] + [1] * 256
    return ST.Tokenizer(tokens, [], types, "deepseek3", {"bos": 0, "eos": EOS})


class DsEngine(MockEngine):
    """Replays scripted completions as ids, then DeepSeek's end token; keeps what it was asked."""

    def __init__(self, tok, script, max_context=8192):
        super().__init__(tok, script, max_context=max_context)
        self.scripts = [tok.encode(x, parse_special=True) + [EOS] for x in ([script] if isinstance(script, str) else script)]
        self.script = self.scripts[0]
        self.last_sampling = None
        self.last = {}

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_sampling = dict(sampling or {})
        got = 0
        try:
            for t in super().generate(ids, max_new, sampling, cancel, embeddings):
                got += 1
                yield t
        finally:      # (the server stops reading at the end token: the engine's DONE comes while it is closed)
            self.last = {"generated": got, "prompt_tokens": len(ids), "prompt_ms": 50.0, "decode_ms": got * 10.0,
                         "finish": "stop"}


def call(*invokes, string="true") -> str:
    """The text DeepSeek writes for calls: [(name, [(param, value, is_string)...])...]."""
    out = ["\n\n<｜DSML｜ calls>"]
    for name, params in invokes:
        out.append(f'<｜DSML｜ invoke name="{name}">')
        for p, v, is_str in params:
            out.append(f'<｜DSML｜ parameter name="{p}" string="{"true" if is_str else "false"}">{v}</｜DSML｜ parameter>')
        out.append("</｜DSML｜ invoke>")
    out.append("</｜DSML｜ calls>")
    return "\n".join(out)


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "weather",
                                              "parameters": {"type": "object", "properties": {
                                                  "city": {"type": "string"}, "days": {"type": "integer"}},
                                                  "required": ["city"]}}}


class Pre(unittest.TestCase):
    """The pre-tokenizer is llama.cpp's DEEPSEEK3_LLM list: a sequence of splits, not one pattern."""

    def split(self, text):
        return ST._sequence_findall([ST.regex.compile(p) for p in ST.DEEPSEEK3_PATTERNS], text)

    def test_digits_in_runs_of_three(self):
        self.assertEqual(self.split("a12345678b"), ["a", "123", "456", "78", "b"])

    def test_cjk_runs(self):
        self.assertEqual(self.split("你好, world"), ["你好", ",", " world"])

    def test_punctuation_then_letters_and_whitespace(self):
        self.assertEqual(self.split("x_y.z  \n\nq"), ["x", "_y", ".z", "  \n\n", "q"])

    def test_every_character_is_kept(self):
        text = "Hello, 世界! 123\t\r\n  é \U0001f600 ｜DSML｜"
        self.assertEqual("".join(self.split(text)), text)


class Tokenizer(unittest.TestCase):
    def test_a_special_token_decodes_to_its_text(self):
        tok = mini_tokenizer()
        ids = tok.encode("<｜User｜>hi｜DSML｜", parse_special=True)
        self.assertEqual(ids[0], 2)
        self.assertEqual(tok.decode(ids), "<｜User｜>hi｜DSML｜")
        self.assertEqual(tok.token_bytes(8), "｜DSML｜".encode())

    def test_the_dsml_token_is_matched_without_parse_special_and_a_control_token_is_not(self):
        tok = mini_tokenizer()
        self.assertEqual(tok.encode("｜DSML｜").__len__(), 1)
        self.assertGreater(len(tok.encode("<｜User｜>")), 1)

    def test_qwen_tokens_are_unchanged(self):
        b2u = ST.bytes_to_unicode()
        tok = ST.Tokenizer([b2u[b] for b in range(256)] + ["<|im_end|>"], [], [1] * 256 + [3])
        self.assertEqual(tok.token_bytes(256), b"<|im_end|>")
        self.assertEqual(tok.encode("a<|im_end|>", parse_special=True), [97, 256])

    @unittest.skipUnless(os.environ.get("STRATA_DSV41_PACK"), "STRATA_DSV41_PACK (the pack's tokenizer folder) not set")
    def test_the_real_tokenizer_makes_the_ids_llama_cpp_made(self):
        d = Path(os.environ["STRATA_DSV41_PACK"])
        vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        cfg = json.loads((d / "tokenizer.json").read_text())
        tok = ST.Tokenizer(tokens, (d / "merges.txt").read_text(encoding="utf-8").split("\n"),
                           json.loads((d / "token_type.json").read_text()), cfg["pre"], cfg["special_ids"])
        for case in GOLDEN:
            self.assertEqual(tok.encode(case["text"], parse_special=True), case["ids"], case["id"] + " " + case["mode"])


class Template(unittest.TestCase):
    """The GGUF's own chat template (the same bytes as DeepSeek's encoding.py on the cases checked), through the
    server's request reader and renderer: the text llama.cpp tokenized for the reference run."""

    @classmethod
    def setUpClass(cls):
        cls.tpl = ChatTemplate(TEMPLATE)

    def test_the_golden_prompts(self):
        for case in GOLDEN:
            messages, tools, kw = openai_to_messages(case["request"])
            self.assertEqual(self.tpl.render(messages, tools=tools, **kw), case["text"], case["id"] + " " + case["mode"])

    def test_thinking_is_the_default(self):
        text = self.tpl.render([{"role": "user", "content": "hi"}])
        self.assertTrue(text.endswith("<｜Assistant｜><think>"))
        text = self.tpl.render([{"role": "user", "content": "hi"}], enable_thinking=False)
        self.assertTrue(text.endswith("<｜Assistant｜></think>"))

    def test_a_tool_call_and_its_result_render_in_dsml(self):
        msgs = [{"role": "user", "content": "weather?"},
                {"role": "assistant", "content": "", "reasoning_content": "call it",
                 "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Zagreb", "days": 3}}}]},
                {"role": "tool", "content": "sunny"}]
        text = self.tpl.render(msgs, tools=[WEATHER["function"]], enable_thinking=False)
        self.assertIn('<｜DSML｜ invoke name="get_weather">', text)
        self.assertIn('<｜DSML｜ parameter name="city" string="true">Zagreb</｜DSML｜ parameter>', text)
        self.assertIn('<｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>', text)
        self.assertIn("<tool_result>sunny</tool_result>", text)


class Translator(unittest.TestCase):
    RAW = call(("get_weather", [("city", "Zagreb", True), ("days", "3", False)]), ("ping", []))

    def all_at_once(self, text):
        t = DsmlTranslator()
        return t.feed(text) + t.flush()

    def test_a_call_becomes_the_qwen_form(self):
        got = self.all_at_once("Sure." + self.RAW)
        self.assertEqual(got, "Sure.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nZagreb\n</parameter>\n"
                              "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>\n"
                              "<tool_call>\n<function=ping>\n</function>\n</tool_call>\n")

    def test_the_split_of_the_stream_changes_nothing(self):
        want = self.all_at_once("a < b and 1<2" + self.RAW + "tail")
        for step in (1, 2, 3, 5, 8, 13):
            t = DsmlTranslator()
            text = "a < b and 1<2" + self.RAW + "tail"
            got = "".join(t.feed(text[i:i + step]) for i in range(0, len(text), step)) + t.flush()
            self.assertEqual(got, want, step)

    def test_text_with_angle_brackets_passes(self):
        text = "x <div> y </｜DSML｜ nope> <｜User｜> <｜DSML｜ unknown>"
        self.assertEqual(self.all_at_once(text), text)

    def test_the_end_of_the_output_inside_a_tag_is_text(self):
        t = DsmlTranslator()
        got = t.feed("ok <｜DSML｜ inv") + t.flush()
        self.assertEqual(got, "ok <｜DSML｜ inv")

    def test_a_value_that_looks_like_a_tag(self):
        raw = call(("write", [("text", "<b>bold</b> and < 3", True)]))
        self.assertIn("<parameter=text>\n<b>bold</b> and < 3\n</parameter>", self.all_at_once(raw))

    def test_a_missing_closing_tag_is_closed(self):
        raw = '<｜DSML｜ calls>\n<｜DSML｜ invoke name="f">\n<｜DSML｜ parameter name="a" string="true">x</｜DSML｜ invoke>'
        self.assertEqual(self.all_at_once(raw),
                         "<tool_call>\n<function=f>\n<parameter=a>\nx\n</parameter>\n</function>\n</tool_call>\n")


class Parser(unittest.TestCase):
    TOOLS = [WEATHER["function"], {"name": "lookup", "description": "", "parameters": {
        "type": "object", "properties": {"id": {}}}}]

    def run_parser(self, text, thinking=True, step=7, cls=DsmlOutputParser):
        p = cls(thinking=thinking, tools=self.TOOLS, stream_tools=True)
        evs = []
        for i in range(0, len(text), step):
            evs += p.feed(text[i:i + step])
        evs += p.finish("stop")
        return evs

    def test_reasoning_then_two_calls(self):
        raw = call(("get_weather", [("city", "Zagreb", True), ("days", "3", False)]), ("lookup", [("id", "123", True)]))
        for step in (1, 4, 1000):
            evs = self.run_parser("Think < ok.</think>" + raw, step=step)
            self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), "Think < ok.")
            calls = [e.call for e in evs if e.kind == "tool_call"]
            self.assertEqual([(c.name, c.arguments) for c in calls],
                             [("get_weather", {"city": "Zagreb", "days": 3}), ("lookup", {"id": "123"})])
            self.assertEqual("".join(e.text for e in evs if e.kind == "tool_args"),
                             '{"city":"Zagreb","days":3}{"id":"123"}')

    def test_a_string_value_stays_a_string_even_when_the_schema_says_nothing(self):
        evs = self.run_parser(call(("lookup", [("id", "007", True)])), thinking=False)
        self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"id": "007"}])
        evs = self.run_parser(call(("lookup", [("id", "7", False)])), thinking=False)
        self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"id": 7}])

    def test_a_non_string_value_is_json(self):
        evs = self.run_parser(call(("lookup", [("id", '{"a": [1, 2]}', False)])), thinking=False)
        self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"id": {"a": [1, 2]}}])

    def test_content_before_the_call_and_none_after(self):
        evs = self.run_parser("Checking.\n\n" + call(("lookup", [("id", "x", True)]))[2:], thinking=False)
        self.assertEqual("".join(e.text for e in evs if e.kind == "content").strip(), "Checking.")
        self.assertEqual(sum(e.kind == "tool_call" for e in evs), 1)

    def test_a_call_cut_off_by_the_length_limit_is_not_a_call(self):
        raw = call(("get_weather", [("city", "Zagr", True)]))
        raw = raw[:raw.index("Zagr") + 4]
        p = DsmlOutputParser(thinking=False, tools=self.TOOLS, stream_tools=True)
        evs = p.feed(raw) + p.finish("length")
        self.assertEqual(sum(e.kind == "tool_call" for e in evs), 0)

    def test_the_plain_parser_is_unchanged(self):
        p = OutputParser(thinking=False, tools=self.TOOLS)
        evs = p.feed("<tool_call>\n<function=lookup>\n<parameter=id>\n5\n</parameter>\n</function>\n</tool_call>")
        self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"id": 5}])
        self.assertTrue(p.clean)

    def test_clean_counts_a_held_tag(self):
        p = DsmlOutputParser(thinking=False, tools=self.TOOLS)
        p.feed("hello <｜DS")
        self.assertFalse(p.clean)
        p.feed("ML｜ calls>")
        self.assertTrue(p.clean)

    def test_forced_opening(self):
        self.assertEqual(to_dsml_force("<tool_call>\n<function=get_weather>\n"),
                         '<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n')
        self.assertEqual(to_dsml_force("<tool_call>\n<function="), '<｜DSML｜ calls>\n<｜DSML｜ invoke name="')
        self.assertIsNone(to_dsml_force(None))
        # the opening and the model's continuation are one call
        p = DsmlOutputParser(thinking=False, tools=self.TOOLS, stream_tools=True)
        evs = p.feed(to_dsml_force("<tool_call>\n<function=get_weather>\n"))
        evs += p.feed('<｜DSML｜ parameter name="city" string="true">Rome</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')
        evs += p.finish("stop")
        self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
                         [("get_weather", {"city": "Rome"})])


class ReadyTimeout(unittest.TestCase):
    """A 300+ GB model read from a slow disk needs more than the default 900 s before READY: the config's "engine_ready_s"."""

    def test_the_config_sets_it_and_a_bad_value_is_said(self):
        from serve import server
        before = server.ENGINE_READY_S
        self.addCleanup(setattr, server, "ENGINE_READY_S", before)
        server.apply_ready_timeout({})
        self.assertEqual(server.ENGINE_READY_S, before)
        server.apply_ready_timeout({"engine_ready_s": 3600})
        self.assertEqual(server.ENGINE_READY_S, 3600.0)
        for bad in (0, -5, "soon", True):
            with self.assertRaises(ValueError):
                server.apply_ready_timeout({"engine_ready_s": bad})


class ApiBase(unittest.TestCase):
    """Both APIs against a fake engine that replays scripted DeepSeek text (the helpers; Api has the tests)."""

    @classmethod
    def setUpClass(cls):
        cls.tok = mini_tokenizer()

    def start(self, script):
        tok = self.tok
        engine = DsEngine(tok, script)
        svc = Service(engine, tok, ChatTemplate(TEMPLATE), model_name=default_model_name({"arch": "deepseek41"}))
        svc.model_arch = "deepseek41"
        svc.set_family("deepseek41")
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.addCleanup(frontend.use_family, None)
        self.engine, self.svc = engine, svc
        self.base = f"http://127.0.0.1:{httpd.server_address[1]}"
        return engine

    def post(self, path, body, raw=False):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read().decode()
                return r.status, (data if raw else json.loads(data))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())

    def prompt(self):
        return self.tok.decode(self.engine.last_prompt)

    def chat(self, **extra):
        body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 400, **extra}
        return self.post("/v1/chat/completions", body)


class Api(ApiBase):
    # -- thinking --------------------------------------------------------------------------------
    def test_thinking_by_default(self):
        self.start("The user greets me.</think>\n\nHello!")
        status, r = self.chat()
        self.assertEqual(status, 200, r)
        self.assertTrue(self.prompt().endswith("<｜User｜>hi<｜Assistant｜><think>"))
        msg = r["choices"][0]["message"]
        self.assertEqual(msg["reasoning_content"], "The user greets me.")
        self.assertEqual(msg["content"].strip(), "Hello!")
        self.assertEqual(r["choices"][0]["finish_reason"], "stop")

    ANSWER = "17 x 23 = 391. The answer is 391."

    def test_no_closing_think_tag_makes_the_text_the_answer(self):
        """The 2-bit sets write the answer after <think> and stop without </think>: it must not arrive as reasoning only (P2003)."""
        for effort in (None, "low", "medium", "high"):
            with self.subTest(effort=effort, api="openai"):
                self.start(self.ANSWER)
                status, r = self.chat(**({"reasoning_effort": effort} if effort else {}))
                self.assertEqual(status, 200, r)
                msg = r["choices"][0]["message"]
                self.assertEqual(msg["content"].strip(), self.ANSWER)
                self.assertFalse(msg.get("reasoning_content"))
                self.assertEqual(r["choices"][0]["finish_reason"], "stop")
            with self.subTest(effort=effort, api="anthropic"):
                self.start(self.ANSWER)
                status, r = self.post("/v1/messages", {"model": "x", "max_tokens": 400,
                                                       "messages": [{"role": "user", "content": "hi"}],
                                                       **({"output_config": {"effort": effort}} if effort else {})})
                self.assertEqual(status, 200, r)
                self.assertEqual([b["type"] for b in r["content"]], ["text"])
                self.assertEqual(r["content"][0]["text"].strip(), self.ANSWER)

    def test_no_closing_think_tag_streaming_ends_with_the_answer_as_content(self):
        self.start(self.ANSWER)
        status, raw = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 400,
                                                         "stream": True}, raw=True)
        self.assertEqual(status, 200)
        chunks = [json.loads(l[6:]) for l in raw.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
        content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
        self.assertEqual(content.strip(), self.ANSWER)
        status, raw = self.post("/v1/messages", {"model": "x", "max_tokens": 400, "stream": True,
                                                 "messages": [{"role": "user", "content": "hi"}]}, raw=True)
        self.assertEqual(status, 200)
        self.assertIn('"text_delta"', raw)
        self.assertIn("391", raw)

    def test_a_turn_cut_by_max_tokens_is_not_turned_into_an_answer(self):
        self.start(self.ANSWER * 30)
        status, r = self.chat(max_tokens=8)
        msg = r["choices"][0]["message"]
        self.assertEqual(r["choices"][0]["finish_reason"], "length")
        self.assertFalse(msg.get("content"))
        self.assertTrue(msg.get("reasoning_content"))

    def test_a_normal_closed_think_is_unchanged(self):
        self.start("Reasoning here.</think>The answer.")
        status, r = self.chat()
        msg = r["choices"][0]["message"]
        self.assertEqual((msg["reasoning_content"], msg["content"].strip()), ("Reasoning here.", "The answer."))

    def test_thinking_off(self):
        self.start("Hello!")
        for extra in ({"chat_template_kwargs": {"enable_thinking": False}}, {"reasoning_effort": "none"}):
            status, r = self.chat(**extra)
            self.assertEqual(status, 200, r)
            self.assertTrue(self.prompt().endswith("<｜User｜>hi<｜Assistant｜></think>"), extra)
            msg = r["choices"][0]["message"]
            self.assertEqual(msg["content"].strip(), "Hello!")
            self.assertFalse(msg.get("reasoning_content"))

    def test_the_end_token_is_deepseeks_not_qwens(self):
        self.start("one<|im_end|>two")                # (not a token here: just text, like any other)
        status, r = self.chat(chat_template_kwargs={"enable_thinking": False})
        self.assertEqual(r["choices"][0]["message"]["content"], "one<|im_end|>two")

    def test_model_name_and_card_sampling(self):
        self.start("ok")
        self.assertEqual(FAMILY_MODEL_NAMES["deepseek41"], "deepseek-v4.1-flash")
        self.assertEqual(default_model_name({"arch": "deepseek41"}), "deepseek-v4.1-flash")
        self.assertEqual(model_card_sampling("deepseek41", True), {"temperature": 1.0, "top_p": 1.0})
        self.chat()
        self.assertEqual(self.engine.last_sampling.get("temperature"), 1.0)

    # -- effort ------------------------------------------------------------------------------------
    def test_effort_words(self):
        self.start("ok")
        for given, line in (("low", "Reasoning Effort: 50 "), ("medium", "Reasoning Effort: 62 "),
                            ("high", "Reasoning Effort: 75 "), ("max", "Reasoning Effort: 100 "),
                            ("xhigh", "Reasoning Effort: 100 "), (40, "Reasoning Effort: 40 ")):
            status, r = self.chat(reasoning_effort=given)
            self.assertEqual(status, 200, (given, r))
            self.assertIn(line, self.prompt(), given)
        status, r = self.chat()
        self.assertIn("Reasoning Effort: 75 ", self.prompt())
        status, r = self.chat(reasoning_effort="extreme")
        self.assertEqual(status, 400)

    def test_anthropic_budget_picks_the_effort(self):
        self.start("ok")
        status, r = self.post("/v1/messages", {"model": "x", "max_tokens": 300, "thinking": {"type": "enabled", "budget_tokens": 1000},
                                               "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200, r)
        self.assertIn("Reasoning Effort: 50 ", self.prompt())

    # -- tool calls -------------------------------------------------------------------------------
    def test_openai_tool_call(self):
        self.start("Need the weather.</think>" + call(("get_weather", [("city", "Zagreb", True), ("days", "2", False)])))
        status, r = self.chat(tools=[WEATHER])
        self.assertEqual(status, 200, r)
        self.assertIn('"name": "get_weather"', self.prompt())
        ch = r["choices"][0]
        self.assertEqual(ch["finish_reason"], "tool_calls")
        tc = ch["message"]["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "get_weather")
        self.assertEqual(json.loads(tc["function"]["arguments"]), {"city": "Zagreb", "days": 2})
        self.assertEqual(ch["message"]["reasoning_content"], "Need the weather.")

    def test_two_calls_in_one_turn(self):
        self.start("</think>" + call(("get_weather", [("city", "A", True)]), ("get_weather", [("city", "B", True)])))
        status, r = self.chat(tools=[WEATHER])
        names = [json.loads(c["function"]["arguments"])["city"] for c in r["choices"][0]["message"]["tool_calls"]]
        self.assertEqual(names, ["A", "B"])

    def test_streamed_tool_call(self):
        self.start("</think>" + call(("get_weather", [("city", "Zagreb", True)])))
        status, text = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "w?"}], "tools": [WEATHER],
                                                          "stream": True, "max_tokens": 300}, raw=True)
        self.assertEqual(status, 200)
        chunks = [json.loads(x[6:]) for x in text.split("\n\n") if x.startswith("data: ") and x != "data: [DONE]"]
        calls = [tc for c in chunks for tc in c["choices"][0]["delta"].get("tool_calls", [])]
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads("".join(tc["function"].get("arguments", "") for tc in calls)), {"city": "Zagreb"})
        self.assertEqual([c["choices"][0]["finish_reason"] for c in chunks if c["choices"][0].get("finish_reason")],
                         ["tool_calls"])

    def test_a_tool_result_turn_renders_for_the_model(self):
        self.start("The weather is sunny.")
        msgs = [{"role": "user", "content": "weather in Zagreb?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "get_weather", "arguments": "{\"city\": \"Zagreb\"}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "sunny, 21C"}]
        status, r = self.post("/v1/chat/completions", {"messages": msgs, "tools": [WEATHER], "max_tokens": 200,
                                                       "chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(status, 200, r)
        p = self.prompt()
        self.assertIn('<｜DSML｜ parameter name="city" string="true">Zagreb</｜DSML｜ parameter>', p)
        self.assertIn("<tool_result>sunny, 21C</tool_result>", p)
        self.assertEqual(r["choices"][0]["message"]["content"].strip(), "The weather is sunny.")

    def test_forced_tool_choice(self):
        self.start('<｜DSML｜ parameter name="city" string="true">Rome</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')
        status, r = self.chat(tools=[WEATHER], tool_choice={"type": "function", "function": {"name": "get_weather"}},
                              chat_template_kwargs={"enable_thinking": False})
        self.assertEqual(status, 200, r)
        self.assertTrue(self.prompt().endswith('<｜Assistant｜></think><｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n'), self.prompt()[-120:])
        tc = r["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual((tc["function"]["name"], json.loads(tc["function"]["arguments"])), ("get_weather", {"city": "Rome"}))

    # -- anthropic ---------------------------------------------------------------------------------
    def test_anthropic_tool_use(self):
        self.start("Weather needed.</think>" + call(("get_weather", [("city", "Oslo", True), ("days", "1", False)])))
        status, r = self.post("/v1/messages", {"model": "x", "max_tokens": 400,
                                               "tools": [{"name": "get_weather", "description": "w", "input_schema":
                                                          WEATHER["function"]["parameters"]}],
                                               "messages": [{"role": "user", "content": "Oslo?"}]})
        self.assertEqual(status, 200, r)
        kinds = [b["type"] for b in r["content"]]
        self.assertEqual(kinds, ["thinking", "tool_use"])
        self.assertEqual(r["content"][1]["input"], {"city": "Oslo", "days": 1})
        self.assertEqual(r["stop_reason"], "tool_use")

    def test_anthropic_text_without_thinking(self):
        self.start("Hello there.")
        status, r = self.post("/v1/messages", {"model": "x", "max_tokens": 100, "thinking": {"type": "disabled"},
                                               "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200, r)
        self.assertTrue(self.prompt().endswith("<｜Assistant｜></think>"))
        self.assertEqual([b["type"] for b in r["content"]], ["text"])
        self.assertEqual(r["content"][0]["text"], "Hello there.")
        self.assertEqual(r["stop_reason"], "end_turn")

    # -- metrics -----------------------------------------------------------------------------------
    def test_metrics_use_the_one_answer_speed(self):
        self.start("</think>Hello, how are you today?")
        self.chat()
        status = urllib.request.urlopen(self.base + "/metrics", timeout=10)
        m = json.loads(status.read().decode())
        last = m["requests"][-1]
        # produced tokens over the engine's decode time (10 ms a token in the fake), never over the prompt's time
        self.assertEqual(last["decode_tok_s"], 100.0)


    # -- the shared start of an agent's and its sub-agents' prompts (the engine keeps its state there) ------------
    def test_the_system_prompt_and_tools_are_the_shared_start(self):
        self.start("ok")
        self.auto_pin("1")
        sysmsg = {"role": "system", "content": "You are an agent. " * 20}
        body = {"messages": [sysmsg, {"role": "user", "content": "task one"}], "tools": [WEATHER], "max_tokens": 40}
        self.assertEqual(self.post("/v1/chat/completions", body)[0], 200)
        n = self.engine.last_sampling["strata_prefix"]["tokens"]
        ids = self.engine.last_prompt
        self.assertTrue(0 < n < len(ids) - 1)
        self.assertIn("get_weather", self.tok.decode(ids[:n]))            # the tools are in the shared start
        self.assertNotIn("task one", self.tok.decode(ids[:n]))            # the first user message is not
        # another agent behind the same system prompt and tools, another task: the same boundary, the same start
        body["messages"] = [sysmsg, {"role": "user", "content": "a different task for the sub-agent"}]
        self.assertEqual(self.post("/v1/chat/completions", body)[0], 200)
        self.assertEqual(self.engine.last_sampling["strata_prefix"]["tokens"], n)
        self.assertEqual(self.engine.last_prompt[:n], ids[:n])

    def auto_pin(self, value):
        old = os.environ.get("STRATA_DS4_AUTO_PIN")
        if value is None:
            os.environ.pop("STRATA_DS4_AUTO_PIN", None)
        else:
            os.environ["STRATA_DS4_AUTO_PIN"] = value
        self.addCleanup(lambda: os.environ.pop("STRATA_DS4_AUTO_PIN", None) if old is None
                        else os.environ.__setitem__("STRATA_DS4_AUTO_PIN", old))

    def test_no_shared_start_without_a_system_prompt_or_when_switched_off(self):
        self.start("ok")
        self.auto_pin("1")
        self.chat()
        self.assertNotIn("strata_prefix", self.engine.last_sampling)     # no system message
        body = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], "max_tokens": 40}
        self.auto_pin("0")                                                # switched off
        self.post("/v1/chat/completions", body)
        self.assertNotIn("strata_prefix", self.engine.last_sampling)
        self.auto_pin(None)                                               # on by default
        self.post("/v1/chat/completions", body)
        self.assertIn("strata_prefix", self.engine.last_sampling)

    def test_a_clients_own_prefix_wins(self):
        self.start("ok")
        self.auto_pin("1")
        body = {"messages": [{"role": "system", "content": "s " * 30}, {"role": "user", "content": "u"}],
                "max_tokens": 40, "strata_prefix": {"tokens": 7}}
        self.post("/v1/chat/completions", body)
        self.assertEqual(self.engine.last_sampling["strata_prefix"], {"tokens": 7})

    def test_the_qwen_path_asks_for_no_prefix(self):
        # the same request on a server that is not DeepSeek's: nothing is added (Qwen keeps its own cache logic)
        svc = Service(MockEngine(self.tok, "ok"), self.tok, ChatTemplate(ROOT / "serve" / "chat_template.jinja"))
        req = {}
        svc.prepare([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], None, {}, 16, req=req)
        self.assertEqual(req, {})


class SlotEnv(unittest.TestCase):
    """The engine's kept conversations: the folder and the switch reach the DeepSeek engine only."""

    KEYS = ("STRATA_DS4_SLOT_KEEP", "STRATA_DS4_SLOT_DIR")

    def env(self, cfg):
        saved = {k: os.environ.pop(k, None) for k in self.KEYS}
        try:
            return server.child_env(cfg)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_deepseek_gets_a_folder_in_the_data_folder_and_the_default_switch(self):
        env = self.env({"arch": "deepseek41", "log": str(Path("data") / "strata-deepseek.log"), "args": []})
        self.assertEqual(Path(env["STRATA_DS4_SLOT_DIR"]), Path("data") / "ds4-slots")
        self.assertEqual(env["STRATA_DS4_SLOT_KEEP"], "1")
        self.assertEqual(env["STRATA_DS4_SLOT_GB"], "2")

    def test_the_configs_own_settings_win(self):
        env = self.env({"arch": "deepseek41", "log": "x/y.log",
                        "env": {"STRATA_DS4_SLOT_KEEP": "1", "STRATA_DS4_SLOT_DIR": "elsewhere"}})
        self.assertEqual((env["STRATA_DS4_SLOT_KEEP"], env["STRATA_DS4_SLOT_DIR"]), ("1", "elsewhere"))

    def test_other_families_get_nothing(self):
        for arch in ("qwen4exp", None, "gyro"):
            env = self.env({"arch": arch, "log": "x/y.log"} if arch else {"log": "x/y.log"})
            self.assertFalse([k for k in env if k.startswith("STRATA_DS4_")], arch)

class ThinkForever(DsEngine):
    """Thinks without end (random letters) until the thinking is closed in the prompt, then answers."""

    ANSWER = "Four."

    def __init__(self, tok):
        super().__init__(tok, "")
        self.prompts = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        self.last_prompt = list(ids)
        got = 0
        try:
            if self.tok.decode(list(ids[-14:])).endswith("</think>\n\n"):
                out = self.tok.encode(self.ANSWER) + [EOS]
            else:
                import random
                rng = random.Random(len(ids))          # (no repeated token: the server's loop guard would end it)
                out = self.tok.encode("".join(chr(97 + rng.randrange(26)) for _ in range(max_new)))
            for t in out[:max_new]:
                if cancel.is_set():
                    return
                got += 1
                yield t
        finally:
            self.last = {"generated": got, "prompt_tokens": len(ids), "prompt_ms": 50.0, "decode_ms": got * 10.0,
                         "finish": "stop"}


class SmallCap(ApiBase):
    """A short max_tokens (200-600) with the thinking on: it was all spent on the thinking and no text block came.
    With no budget of the client's own the thinking is capped at half of max_tokens (this family only)."""

    def start_forever(self):
        self.start("")
        engine = ThinkForever(self.tok)
        self.svc.engine = engine
        self.engine = engine
        return engine

    def text_of(self, r):
        return "".join(c.get("text", "") for c in r["content"] if c["type"] == "text")

    def test_every_short_cap_gets_an_answer_on_both_apis(self):
        for cap in (200, 400, 600):
            for api in ("anthropic", "openai"):
                with self.subTest(cap=cap, api=api):
                    self.start_forever()
                    if api == "anthropic":
                        status, r = self.post("/v1/messages", {"model": "x", "max_tokens": cap, "thinking": {"type": "enabled"},
                                                               "messages": [{"role": "user", "content": "2+2?"}]})
                        self.assertEqual(status, 200, r)
                        self.assertEqual(self.text_of(r), ThinkForever.ANSWER)
                        self.assertLessEqual(r["usage"]["output_tokens"], cap)
                    else:
                        status, r = self.chat(max_tokens=cap)
                        self.assertEqual(status, 200, r)
                        self.assertEqual(r["choices"][0]["message"]["content"], ThinkForever.ANSWER)
                        self.assertLessEqual(r["usage"]["completion_tokens"], cap)
                    first = self.engine.prompts[0]
                    self.assertEqual(len(self.engine.prompts), 2)          # the thinking was wrapped up, then answered
                    self.assertEqual(len(self.engine.prompts[1]) - len(first), cap // 2 - 0 + len(self.tok.encode(
                        REASONING_WRAP_UP, parse_special=True)))

    def test_thinking_off_is_untouched(self):
        for cap in (200, 400, 600):
            with self.subTest(cap=cap):
                self.start("Plain answer.")
                status, r = self.chat(max_tokens=cap, reasoning_effort="none")
                self.assertEqual(status, 200, r)
                self.assertEqual(r["choices"][0]["message"]["content"], "Plain answer.")
                status, r = self.post("/v1/messages", {"model": "x", "max_tokens": cap, "thinking": {"type": "disabled"},
                                                       "messages": [{"role": "user", "content": "hi"}]})
                self.assertEqual(self.text_of(r), "Plain answer.")

    def test_a_client_budget_wins(self):
        self.start_forever()
        status, r = self.chat(max_tokens=600, reasoning_budget_tokens=20)
        self.assertEqual(status, 200, r)
        self.assertEqual(len(self.engine.prompts[1]) - len(self.engine.prompts[0]), 20 + len(self.tok.encode(
            REASONING_WRAP_UP, parse_special=True)))

    def test_a_tool_call_under_a_short_cap_still_parses(self):
        for cap in (200, 400, 600):
            with self.subTest(cap=cap):
                self.start("Need it.</think>" + call(("f", [("a", "1", True)])))
                status, r = self.chat(max_tokens=cap, tools=[{"type": "function", "function": {
                    "name": "f", "description": "f", "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}}])
                self.assertEqual(status, 200, r)
                calls = r["choices"][0]["message"]["tool_calls"]
                self.assertEqual(calls[0]["function"]["name"], "f")
                self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"a": "1"})
                self.assertEqual(len(self.engine.last_prompt) > 0, True)

    def test_the_family_default_thinking_budget(self):
        """DeepSeek-V4.1 only, as Maya ships it: 32768 thinking tokens when neither the request nor the config names a budget."""
        self.start("ok")
        self.assertEqual(self.svc.reasoning_budget({}), 32768)
        self.assertIsNone(self.svc.reasoning_budget({"reasoning_budget_tokens": 0}))        # a request turns it off
        self.assertEqual(self.svc.reasoning_budget({"reasoning_budget_tokens": 50}), 50)    # and its own wins
        self.assertEqual(self.svc.reasoning_budget({"thinking_budget_tokens": 70}), 70)
        self.svc.reasoning_budget_tokens, self.svc.reasoning_budget_configured = 90, True
        self.assertEqual(self.svc.reasoning_budget({}), 90)                                 # the config's wins
        self.svc.reasoning_budget_tokens = 0                                                # the config says none
        self.assertIsNone(self.svc.reasoning_budget({}))
        self.svc.reasoning_budget_configured = False
        self.svc.set_family(None)
        self.assertIsNone(self.svc.reasoning_budget({}))                                    # other families: none

    def test_the_default_budget_closes_a_long_thinking(self):
        self.start_forever()
        status, r = self.chat(max_tokens=5000)
        self.assertEqual(status, 200, r)
        self.assertEqual(r["choices"][0]["message"]["content"], ThinkForever.ANSWER)
        status, r = self.chat(max_tokens=300, reasoning_effort="none")                      # thinking off: no budget, no wrap-up
        self.assertEqual(len(self.engine.prompts), 3)

    def test_other_families_keep_the_old_behaviour(self):
        self.start_forever()
        self.svc.set_family(None)
        status, r = self.chat(max_tokens=300)
        self.assertEqual(len(self.engine.prompts), 1)                       # no wrap-up: Qwen's serving is as it was


class PickTemplateTests(unittest.TestCase):
    """A pack made from another GGUF can ship the V4-Flash template (`tool_calls`, no space): the V4.1 model then
    writes malformed calls (P2003 install test).  The server swaps in the V4.1 template for such a pack."""

    def test_v4flash_pack_template_is_replaced(self):
        import tempfile
        from serve import dsml
        with tempfile.TemporaryDirectory() as d:
            old = Path(d) / "old.jinja"
            old.write_text("{%- set dsml_token = '｜DSML｜' -%}{{ '<' + dsml_token + 'tool_calls>' }}", encoding="utf-8")
            self.assertEqual(dsml.pick_template(old), (dsml.FIXTURE_TEMPLATE, True))
            same = Path(d) / "same.jinja"
            same.write_text(dsml.FIXTURE_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
            self.assertEqual(dsml.pick_template(same), (same, False))
            self.assertEqual(dsml.pick_template(Path(d) / "none.jinja"), (dsml.FIXTURE_TEMPLATE, False))

    def test_fixture_is_accepted(self):
        from serve import dsml
        self.assertTrue(dsml.v41_template_ok(dsml.FIXTURE_TEMPLATE.read_text(encoding="utf-8")))


class ArchFromTokenizer(unittest.TestCase):
    """#4137: a config without "arch" (serve/server.py run on its own) must not serve DeepSeek with the Qwen stop ids."""

    def test_a_deepseek_tokenizer_without_an_arch_is_deepseek41(self):
        self.assertEqual(server.arch_from_tokenizer(None, "deepseek3"), "deepseek41")
        self.assertEqual(server.arch_from_tokenizer("", "deepseek3"), "deepseek41")

    def test_an_arch_the_config_names_is_kept(self):
        self.assertEqual(server.arch_from_tokenizer("qwen4exp", "deepseek3"), "qwen4exp")
        self.assertEqual(server.arch_from_tokenizer("deepseek41", "qwen35"), "deepseek41")

    def test_a_qwen_tokenizer_names_nothing(self):
        self.assertIsNone(server.arch_from_tokenizer(None, "qwen35"))
        self.assertIsNone(server.arch_from_tokenizer(None, None))

    def test_without_the_family_the_pieces_of_im_end_are_not_stop_ids_and_the_end_token_is(self):
        # what the reporter's config got (#4137, #4576): the DeepSeek tokenizer splits <|im_end|> into pieces, so "|", ">" and "im" (Nerazim)
        # stopped a reply, and the end token (id 1) did not stop it at all
        tok = mini_tokenizer()
        wrong = Service(DsEngine(tok, "x"), tok, ChatTemplate(TEMPLATE))
        self.assertIn(EOS, wrong.stop_ids)
        self.assertFalse(wrong.stop_ids & set(tok.encode("|", parse_special=True)))
        right = Service(DsEngine(tok, "x"), tok, ChatTemplate(TEMPLATE))
        right.set_family("deepseek41")
        self.addCleanup(frontend.use_family, None)
        self.assertEqual(right.stop_ids, {EOS})


if __name__ == "__main__":
    unittest.main()
