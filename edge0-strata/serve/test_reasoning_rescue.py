"""serve/test_reasoning_rescue.py - #804 / #1058: when a `<tool_call>` written inside the reasoning is a real call.

A call inside the reasoning counts only when ALL hold: tools are declared and the name is one of them; the opener is
at the start of a line; it is outside a Markdown code fence and outside inline code; the call closes with
</tool_call> and nothing but whitespace (or more calls) follows it before the end of the turn or `</think>`; and the
turn did not end by max tokens (or a cancel or an error).  Anything else stays reasoning text.

The corpus in serve/fixtures/rcall_specimens.json holds 25 entries from PR #525's corpus and the 12 live strandings
recovered by 47Hunter47 (gist 4abcd3a8...): 16 act, 21 must not (the reporter's 20 negative controls
plus the live one cut by max tokens).

    python -m unittest serve.test_reasoning_rescue -v
"""
from __future__ import annotations

import json
import random
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, OutputParser  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402
from serve.test_server import CTX  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPECIMENS = json.loads((ROOT / "serve/fixtures/rcall_specimens.json").read_text(encoding="utf-8"))["specimens"]
CALL = "<tool_call>\n<function=write>\n<parameter=path>\nfile.txt\n</parameter>\n</function>\n</tool_call>"
RM = CALL.replace("file.txt", "rm -rf build")
PROPS = {"path": {"type": "string"}}
SCHEMA = [{"name": "write", "parameters": {"properties": PROPS}}]


def parse(text, width, tools=SCHEMA, finish=None, stream_tools=True):
    """Events for `text` fed `width` characters at a time (0 = random 1-9, seeded)."""
    p = OutputParser(thinking=True, tools=tools, stream_tools=stream_tools)
    evs, rnd, i = [], random.Random(7), 0
    while i < len(text):
        k = width or rnd.randint(1, 9)
        evs += p.feed(text[i:i + k])
        i += k
    return evs + p.finish(finish)


def calls(evs):
    return [e.call.name for e in evs if e.kind == "tool_call"]


def reasoning(evs):
    return "".join(e.text for e in evs if e.kind == "reasoning")


WIDTHS = (1, 2, 3, 7, 0, 100_000)


class Corpus(unittest.TestCase):
    def test_every_specimen_whole_and_in_chunks(self):
        for sp in SPECIMENS:
            tools = [{"name": n, "parameters": {"properties": {}}} for n in sp["tools"]]
            seen = set()
            for width in WIDTHS:
                for st in (True, False):
                    with self.subTest(name=sp["name"], width=width, stream_tools=st):
                        evs = parse(sp["input"], width, tools, sp["finish"], st)
                        self.assertEqual(calls(evs), sp["calls"])
                        seen.add((reasoning(evs), tuple(e.call.name for e in evs if e.kind == "tool_call")))
            self.assertEqual(len(seen), 1, sp["name"])        # streamed and whole agree

    def test_the_corpus_has_the_reported_split(self):
        self.assertEqual((len(SPECIMENS), sum(bool(s["calls"]) for s in SPECIMENS)), (37, 16))


class Gates(unittest.TestCase):
    def check(self, text, expect, finish=None, tools=SCHEMA):
        for width in WIDTHS:
            with self.subTest(text=text[:40], width=width):
                evs = parse(text, width, tools, finish)
                self.assertEqual(len(calls(evs)), expect)
                if not expect:
                    self.assertEqual(reasoning(evs), text)       # nothing lost, nothing turned into content

    def test_a_stranded_call_is_recovered(self):
        self.check("I will write it now.\n\n" + CALL, 1)
        self.check(CALL, 1)
        self.check("Hmm.\n  " + CALL + "\n", 1)                    # indented opener and trailing blank lines
        self.check("Two steps.\n" + CALL + "\n\n" + CALL, 2)
        self.check("Two steps.\n" + CALL + "\n</think>\n\ndone", 1)

    def test_recovered_call_keeps_the_text_before_it(self):
        evs = parse("Let me think.\n\n" + CALL, 3)
        self.assertEqual(reasoning(evs), "Let me think.\n\n")

    def test_the_reporters_negative_controls(self):
        self.check("I could run\n" + RM + "\nbut first let me check what build holds...", 0)
        self.check("I could run\n" + RM, 0, finish="length")             # cut by max tokens: the destructive one
        self.check("Example:\n```xml\n" + RM + "\n```\nok", 0)
        self.check("Example:\n~~~\n" + RM + "\n~~~\nok", 0)
        self.check("The format is " + RM + " ok.", 0)
        self.check("I must emit `<tool_call>` then more", 0)
        self.check("Use `\n" + RM, 0)                                       # inside inline code spanning a line
        self.check("> " + RM, 0)
        self.check("Plan: " + RM, 0)

    def test_a_length_cut_keeps_even_a_clean_stranded_call_as_text(self):
        for finish in ("length", "cancel", "error", "disconnect"):
            self.check("Now.\n" + CALL, 0, finish=finish)
        self.check("Now.\n" + CALL, 1, finish="stop")

    def test_after_a_closed_fence_a_call_counts_again(self):
        self.check("```\nx\n```\n" + CALL, 1)
        self.check("~~~\nx\n~~~\n" + CALL, 1)
        self.check("a `b` c\n" + CALL, 1)

    def test_a_fence_needs_the_same_marker_to_close(self):
        self.check("~~~\n```\n" + CALL, 0)

    def test_fence_split_across_chunks(self):
        text = "Format:\n```xml\n" + CALL + "\n```\n"
        for cut in range(1, len(text)):
            with self.subTest(cut=cut):
                p = OutputParser(thinking=True, tools=SCHEMA, stream_tools=True)
                evs = p.feed(text[:cut]) + p.feed(text[cut:]) + p.finish()
                self.assertEqual(calls(evs), [])
                self.assertEqual(reasoning(evs), text)

    def test_opener_split_across_chunks(self):
        text = "Go.\n" + CALL
        for cut in range(1, len(text)):
            with self.subTest(cut=cut):
                p = OutputParser(thinking=True, tools=SCHEMA, stream_tools=True)
                evs = p.feed(text[:cut]) + p.feed(text[cut:]) + p.finish("stop")
                self.assertEqual(calls(evs), ["write"])
        quoted = "Go " + CALL
        for cut in range(1, len(quoted)):
            with self.subTest(cut=cut, quoted=True):
                p = OutputParser(thinking=True, tools=SCHEMA, stream_tools=True)
                evs = p.feed(quoted[:cut]) + p.feed(quoted[cut:]) + p.finish("stop")
                self.assertEqual(calls(evs), [])

    def test_the_end_of_the_turn_splits_a_held_tail(self):
        p = OutputParser(thinking=True, tools=SCHEMA)
        evs = p.feed("ok\n" + CALL + "\n<tool_c")           # a half opener after the call: it was prose
        evs += p.finish("stop")
        self.assertEqual(calls(evs), [])
        self.assertEqual(reasoning(evs), "ok\n" + CALL + "\n<tool_c")

    def test_a_call_waits_and_is_not_streamed_early(self):
        p = OutputParser(thinking=True, tools=SCHEMA)
        evs = p.feed("go\n" + CALL + "\n")
        self.assertEqual(calls(evs), [])                    # not yet: the turn could still say it was a quote
        self.assertEqual(calls(p.finish("stop")), ["write"])

    def test_no_tools_declared_is_unchanged(self):
        text = "before\n" + CALL + "\n</think>\n\nanswer"
        for tools in (None, []):
            evs = parse(text, 5, tools)
            self.assertEqual(calls(evs), [])
            self.assertEqual(reasoning(evs), "before\n" + CALL + "\n")

    def test_an_undeclared_name_stays_reasoning(self):
        self.check("go\n" + CALL.replace("write", "other"), 0)

    def test_the_counters(self):
        p = OutputParser(thinking=True, tools=SCHEMA)
        p.feed("go\n" + CALL)
        p.finish("length")
        self.assertEqual((p.rescued, p.refused), (0, 1))
        p = OutputParser(thinking=True, tools=SCHEMA)
        p.feed("go\n" + CALL)
        p.finish("stop")
        self.assertEqual((p.rescued, p.refused), (1, 0))


class VisibleAnswer(unittest.TestCase):
    """#1058: the same fence / inline-code rule for a call in the visible answer (after </think>)."""
    HEAD = "ok</think>\n\n"

    def run_all(self, text, finish=None):
        res = set()
        for width in WIDTHS:
            evs = parse(text, width, SCHEMA, finish)
            res.add((tuple(calls(evs)), "".join(e.text for e in evs if e.kind == "content")))
        self.assertEqual(len(res), 1, res)               # every chunking agrees
        return res.pop()

    def test_quoted_calls_stay_text(self):
        for body in ("Format:\n```xml\n" + RM + "\n```\nDone.", "Format:\n~~~\n" + RM + "\n~~~\nDone.",
                     "Use `" + RM + "` like this.", "Use `x`, then `\n" + RM + "`", "```\n" + RM,
                     "Text\n\n```python\nx = 1\n```\n```\n" + CALL):
            with self.subTest(body=body[:30]):
                got, text = self.run_all(self.HEAD + body)
                self.assertEqual(got, ())
                self.assertEqual(text, body)             # nothing lost

    def test_real_calls_still_fire(self):
        for body in (CALL, "Let me do it.\n\n" + CALL, "Let me do it. " + CALL, "```\ncode\n```\n" + CALL,
                     "`a` and `b` " + CALL, "~~~\nx\n~~~\n\n" + CALL):
            with self.subTest(body=body[:30]):
                self.assertEqual(self.run_all(self.HEAD + body)[0], ("write",))
        self.assertEqual(self.run_all(self.HEAD + CALL + "\n" + CALL)[0], ("write", "write"))

    def test_a_closed_call_in_a_length_cut_answer_stays_a_call(self):
        self.assertEqual(self.run_all(self.HEAD + "Now.\n" + CALL, finish="length")[0], ("write",))

    def test_without_thinking(self):
        p = OutputParser(thinking=False, tools=SCHEMA)
        evs = p.feed("```\n" + RM + "\n```") + p.finish()
        self.assertEqual(calls(evs), [])


class OverHttp(unittest.TestCase):
    """The same rules through the real server: OpenAI chat (whole and streamed), Anthropic and Responses."""

    def ask(self, script, max_tokens=800):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, script, max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        msgs = [{"role": "user", "content": "save my notes"}]
        reqs = {
            "openai": ("/v1/chat/completions", {"tools": [{"type": "function", "function": {
                "name": "write", "parameters": {"type": "object", "properties": PROPS}}}], "messages": msgs,
                "max_tokens": max_tokens}),
            "anthropic": ("/v1/messages", {"tools": [{"name": "write", "input_schema": {
                "type": "object", "properties": PROPS}}], "messages": msgs, "max_tokens": max_tokens}),
            "responses": ("/v1/responses", {"tools": [{"type": "function", "name": "write", "parameters": {
                "type": "object", "properties": PROPS}}], "input": "save my notes", "max_output_tokens": max_tokens}),
        }
        out = {}
        try:
            for api, (path, body) in reqs.items():
                for stream in (False, True):
                    req = urllib.request.Request(base + path, data=json.dumps(dict(body, model="x", stream=stream)).encode(),
                                                 headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
                    with urllib.request.urlopen(req, timeout=30) as r:
                        raw = r.read().decode()
                    out[api, stream] = self.tool_uses(api, stream, raw)
        finally:
            httpd.shutdown()
            httpd.server_close()
        return out

    @staticmethod
    def tool_uses(api, stream, raw):
        """How many tool calls the answer carries."""
        if not stream:
            m = json.loads(raw)
            if api == "openai":
                return len(m["choices"][0]["message"].get("tool_calls") or [])
            if api == "anthropic":
                return sum(b["type"] == "tool_use" for b in m["content"])
            return sum(o["type"] == "function_call" for o in m["output"])
        evs = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
        if api == "openai":
            return len({tc["index"] for e in evs for c in e["choices"] for tc in c["delta"].get("tool_calls") or []})
        if api == "anthropic":
            return sum(e["type"] == "content_block_start" and e["content_block"]["type"] == "tool_use" for e in evs)
        return sum(e["type"] == "response.output_item.added" and e["item"]["type"] == "function_call" for e in evs)

    def test_a_stranded_call_is_a_call_in_every_api(self):
        a = self.ask("Writing the file now.\n\n" + CALL)
        self.assertEqual(set(a.values()), {1}, a)

    def test_quotes_and_cuts_are_not(self):
        for name, script, tokens in (
                ("fence", "Format:\n```xml\n" + RM + "\n```\nand that is all.", 800),
                ("midsentence", "I could run " + RM + " but no.", 800),
                ("prose after", "I could run\n" + RM + "\nbut first let me check what build holds.", 800),
                ("cut", "I could run\n" + RM + "\nthen more thinking " * 40, len("I could run\n" + RM)),
                ("cut exactly at the call", "Now.\n" + CALL + "\nmore thought " * 40, len("Now.\n" + CALL))):
            with self.subTest(name):
                a = self.ask(script, max_tokens=tokens)
                self.assertEqual(set(a.values()), {0}, a)

    def test_visible_answer_fences_on_all_three_apis(self):
        for name, body, want in (("fence", "Format:\n```xml\n" + RM + "\n```\nDone.", 0),
                                 ("inline", "Use `" + RM + "` here.", 0),
                                 ("real", "Doing it.\n\n" + CALL, 1)):
            with self.subTest(name):
                a = self.ask( "ok</think>\n\n" + body)
                self.assertEqual(set(a.values()), {want}, a)


if __name__ == "__main__":
    unittest.main()
