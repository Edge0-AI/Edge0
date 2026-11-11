"""OpenAI logprobs on /v1/chat/completions, from the engine's LP lines: python -m unittest serve.test_logprobs.

The engine (src/program/generate.cpp) writes an `LP <logprob> [id:logprob ...]` line right after
each `T <id>` when the request carries `logprobs=K`; this is the server half that turns those
lines into OpenAI's logprobs, and the parts that are easy to get wrong:

  - `allowed_token_ids`: only the named ids, renormalised over them (vLLM's semantics).
  - `response_format: json_schema` rewrites the content, so the tokens no longer spell what is
    returned; the entries are re-aligned or dropped, never returned wrong.
"""
from __future__ import annotations

import json
import math
import queue
import threading
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from serve.frontend import ChatTemplate
from serve.server import (ByteTokenizer, LogprobEntries, LPLines, MockEngine, Service, StrataEngine, batch_slots_on,
                          logprob_span, serve, top_logprobs, allowed_ids)

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ChatTemplate(ROOT / "serve" / "chat_template.jinja")


class Tokenizer(ByteTokenizer):
    def token_bytes(self, token):
        return self.SPECIALS[token - 256].encode() if token >= 256 else bytes([token])


class ThinkTokenizer(Tokenizer):
    # two extra special tokens that decode as the thinking marks; ALWAYS so the byte-level
    # encode() maps them wherever they appear ("" would loop forever in startswith, so they
    # are real strings, like the engine's own think tokens are)
    SPECIALS = ByteTokenizer.SPECIALS + ["<think>", "</think>"]
    ALWAYS = ("<think>", "</think>")


class LpEngine(MockEngine):
    """MockEngine with the LP line of each token on lp_lines before the token is yielded, as the
    engine leaves it; token i has the log-probability -(i + 1) / 100 and four named alternatives."""

    def __init__(self, tok, script, lines=True):
        super().__init__(tok, script, max_context=16384)
        self.with_lines, self.sampling = lines, None

    def line(self, i, t):
        """The LP line of token t, the i-th of the reply: the same four ids whatever was asked."""
        return (f"LP {-(i + 1) / 100:.6f} 65:-0.500000 72:-1.250000 "
                f"88:-3.000000 91:-4.200000\n")

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.sampling, self.lp_lines = dict(sampling or {}), LPLines(cancel)
        for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
            if self.with_lines:
                self.lp_lines.append((t, self.line(i, t)))
            yield t


class NamedEngine(LpEngine):
    """LpEngine that prints the request's own `allowed_token_ids`, as the engine's emit_lp_line does:
    `LP <logprob> <named id>:<logprob> ...`, the chosen id left out (it is on the head)."""

    def line(self, i, t):
        named = [int(x) for x in (self.sampling or {}).get("allowed_token_ids") or [] if int(x) != t]
        alts = "".join(f" {x}:{-0.700000 - 0.010000 * j:.6f}" for j, x in enumerate(named))
        return f"LP {-(i + 1) / 100:.6f}{alts}\n"


class NextEngine(LpEngine):
    """LpEngine for two requests: the second one stops after its first token (its LP line is in lp_lines)
    until the first request's reply has been read."""

    def __init__(self, tok, scripts, first_out):
        super().__init__(tok, scripts)
        self.first_out, self.calls = first_out, 0
        self.first_in, self.first_generated, self.second_started = (threading.Event(), threading.Event(),
                                                                    threading.Event())

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls += 1
        mine = self.calls
        if mine == 1:
            self.first_in.set()
        try:
            for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
                if mine == 2 and i == 0:
                    self.second_started.set()
                    self.first_out.wait(10)
                yield t
        finally:
            if mine == 1:
                self.first_generated.set()


class HandOver:
    """Service.fifo that, once the first request has generated, lets its turn go and then waits until the
    next request is generating before the first one goes on."""

    def __init__(self, engine):
        self.lock, self.engine, self.waited = threading.Lock(), engine, False

    def acquire(self, *a, **k):
        return self.lock.acquire(*a, **k)

    def release(self):
        self.lock.release()
        if self.engine.first_generated.is_set() and not self.waited:
            self.waited = True
            self.engine.second_started.wait(10)

    def locked(self):
        return self.lock.locked()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


SCHEMA = {"type": "json_schema", "json_schema": {"name": "a", "strict": True, "schema": {
    "type": "object", "properties": {"rent": {"type": "string"}, "quote": {"type": "string"}},
    "required": ["rent", "quote"], "additionalProperties": False}}}


class Keys(unittest.TestCase):
    """What the request asks for, and what reaches the engine line."""

    def test_what_the_request_asks(self):
        self.assertEqual(top_logprobs({"logprobs": True}), 0)
        self.assertEqual(top_logprobs({"logprobs": True, "top_logprobs": 3}), 3)
        self.assertEqual(top_logprobs({"logprobs": True, "top_logprobs": 50}), 20)
        self.assertEqual(top_logprobs({"logprobs": 2}), 2)          # the legacy completions form
        for req in ({}, {"logprobs": False}, {"logprobs": None}, {"top_logprobs": 3}):
            self.assertIsNone(top_logprobs(req))

    def test_allowed_ids(self):
        self.assertEqual(allowed_ids({"allowed_token_ids": [1, 2, 3]}), [1, 2, 3])
        for req in ({}, {"allowed_token_ids": "1,2"}, {"allowed_token_ids": [True]}, {"allowed_token_ids": [-1]}):
            self.assertEqual(allowed_ids(req), [])

    def test_the_engine_key_only_when_asked(self):
        self.assertIn(" logprobs=0", StrataEngine.sampling_keys({"logprobs": True}))
        self.assertIn(" logprobs=5", StrataEngine.sampling_keys({"logprobs": True, "top_logprobs": 5}))
        self.assertNotIn("logprobs", StrataEngine.sampling_keys({}))
        self.assertNotIn("logprobs", StrataEngine.sampling_keys({"logprobs": False, "temperature": 0}))

    def test_allowed_reaches_the_engine_line(self):
        keys = StrataEngine.sampling_keys({"logprobs": True, "top_logprobs": 2,
                                           "allowed_token_ids": [7, 8, 9]})
        self.assertIn(" logprobs=2 ", keys)
        self.assertIn(" allowed=7,8,9", keys)

    def test_unreadable_lines_give_no_entry(self):
        a = threading.Event()
        eng = SimpleNamespace(lp_lines=LPLines(a))
        for line in ("LP nan", "LP -nan(ind)", "LP inf", "LP -0.1 65:x", "LP -0.1 65:-inf", "LP -0.1 65",
                     "LP -0.200000 65:-1.0"):
            eng.lp_lines.append((72, line))
        got = LogprobEntries(SimpleNamespace(tok=Tokenizer(), stop_ids=set()), owner=a).take(eng)
        self.assertEqual([e["logprob"] for e in got], [-0.2])

    def test_a_filtered_line_with_nothing_left_keeps_its_token(self):
        # none of the named ids on the line: the entry keeps the chosen token and its own logprob, and no
        # alternatives.  It used to be None, appended all the same, and the reply failed on e["token"].
        a = threading.Event()
        eng = SimpleNamespace(lp_lines=LPLines(a))
        eng.lp_lines.append((72, "LP -0.300000 65:-0.500000"))
        got = LogprobEntries(SimpleNamespace(tok=Tokenizer(), stop_ids=set()), [70], a).take(eng)
        self.assertEqual([(e["token"], e["logprob"], e["top_logprobs"]) for e in got], [("H", -0.3, [])])

    def test_a_named_token_the_line_does_not_repeat_is_its_own_distribution(self):
        # the engine prints the chosen token's logprob on the head and the named ids after it, never repeating
        # the chosen one: naming only that id leaves it alone in the distribution, at 1 (logprob 0)
        a = threading.Event()
        eng = SimpleNamespace(lp_lines=LPLines(a))
        eng.lp_lines.append((72, "LP -0.300000 65:-0.500000"))
        got = LogprobEntries(SimpleNamespace(tok=Tokenizer(), stop_ids=set()), [72], a).take(eng)
        self.assertEqual(got[0]["top_logprobs"], [{"token": "H", "logprob": 0.0, "bytes": [72]}])
        self.assertEqual(got[0]["logprob"], 0.0)


class Generate(unittest.TestCase):
    """The token's LP line is read next to it, and only when it was asked for."""

    def engine(self, *lines):
        e = StrataEngine.__new__(StrataEngine)
        e.proc = __import__("types").SimpleNamespace(
            stdin=__import__("io").StringIO(), poll=lambda: None)
        e.lines, e.can_stop, e.max_context, e.silence_s = queue.Queue(), False, 262144, 0
        for line in lines:
            e.lines.put(line)
        return e

    def test_a_token_is_yielded_with_its_lp_line_read(self):
        e = self.engine("PP 3 3", "T 72", "LP -0.100000 72:-0.1 65:-2.5", "T 105",
                        "LP -0.200000 105:-0.2 65:-3", "T 257", "LP -0.010000",
                        "DONE 3 3 0 3 stop 0 0 0")
        seen = []
        for t in e.generate([1, 2, 3], 8, {"logprobs": True, "top_logprobs": 2}, threading.Event()):
            if t is not None:
                seen.append((t, [x[0] for x in e.lp_lines]))
        self.assertEqual(seen, [(72, [72]), (105, [72, 105]), (257, [72, 105, 257])])
        self.assertIn(" logprobs=2 ", e.proc.stdin.getvalue())
        self.assertEqual(e.lp_lines[1][1], "LP -0.200000 105:-0.2 65:-3")

    def test_an_engine_without_lp_lines(self):
        e = self.engine("T 72", "T 105", "DONE 3 3 0 2 stop 0 0 0")
        toks = [t for t in e.generate([1], 8, {"logprobs": True}, threading.Event()) if t is not None]
        self.assertEqual(toks, [72, 105])
        self.assertEqual(list(e.lp_lines), [])
        self.assertEqual(e.last["finish"], "stop")

    def test_not_asked_reads_as_before(self):
        e = self.engine("T 72", "LP -0.1", "T 105", "DONE 3 3 0 2 stop 0 0 0")
        toks = [t for t in e.generate([1], 8, {}, threading.Event()) if t is not None]
        self.assertEqual(toks, [72, 105])
        self.assertEqual(list(e.lp_lines), [])

    def test_a_consumer_that_stops_early_does_not_lose_the_done_read_after_the_token(self):
        # the DONE line was read while looking for the token's LP line: the drain must take it,
        # not wait for another
        e = self.engine("T 72", "DONE 3 3 0 1 length 0 0 0", None)
        gen = e.generate([1], 1, {"logprobs": True}, threading.Event())
        self.assertEqual(next(gen), 72)
        gen.close()
        self.assertEqual(e.last["finish"], "length")


class Api(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer()
        self.engine = LpEngine(self.tok, "Café costs €5.")
        self.svc = Service(self.engine, self.tok, TEMPLATE)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, body, stream=False):
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 200,
                "chat_template_kwargs": {"enable_thinking": False}, **body}
        if stream:
            body["stream"] = True
        req = urllib.request.Request(self.base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
        if not stream:
            return json.loads(raw)
        return [json.loads(x[6:]) for x in raw.splitlines() if x.startswith("data: {")]

    def test_the_tokens_spell_the_content(self):
        out = self.post({"logprobs": True, "top_logprobs": 1})
        choice = out["choices"][0]
        content = choice["message"]["content"]
        lps = choice["logprobs"]["content"]
        self.assertEqual(content, "Café costs €5.")
        self.assertEqual("".join(e["token"] for e in lps), content)
        self.assertEqual(len(lps), len(content.encode()))       # one per byte token; no stop token
        self.assertEqual(b"".join(bytes(e["bytes"]) for e in lps), content.encode())
        e_acute = [e for e in lps if len(e["bytes"]) == 1 and e["bytes"][0] >= 0x80]
        self.assertEqual([e["token"] for e in e_acute[:2]], ["", "é"])   # split across two tokens
        self.assertEqual(lps[0]["logprob"], -0.01)
        self.assertEqual(lps[0]["top_logprobs"][0]["token"], "A")
        self.assertEqual(self.engine.sampling.get("logprobs"), True)

    def test_a_request_reads_only_its_own_lines(self):
        # A's last "." is held by its stop string and sent after A has left the queue, when B is
        # already generating on the same engine: each reply's tokens must spell that reply
        first_out = threading.Event()
        engine = NextEngine(self.tok, ["Café costs €5.", "Bonjour!"], first_out)
        svc = Service(engine, self.tok, TEMPLATE)
        svc.fifo = HandOver(engine)
        httpd = serve(svc, port=0)
        got = {}
        try:
            self.base = f"http://127.0.0.1:{httpd.server_address[1]}"
            a = threading.Thread(target=lambda: got.update(a=self.post({"logprobs": True, "stop": [".!"]})))
            b = threading.Thread(target=lambda: got.update(b=self.post({"logprobs": True})))
            a.start()
            self.assertTrue(engine.first_in.wait(10))
            b.start()
            a.join(30)
            first_out.set()
            b.join(30)
        finally:
            first_out.set()
            httpd.shutdown()
            httpd.server_close()
        for key, text in (("a", "Café costs €5."), ("b", "Bonjour!")):
            choice = got[key]["choices"][0]
            self.assertEqual(choice["message"]["content"], text)
            self.assertEqual("".join(e["token"] for e in choice["logprobs"]["content"]), text, key)

    def test_logprobs_take_any_max_tokens(self):
        # No cap of our own: the reply is already bounded by max_tokens (the rest of the context when it is not
        # sent), which is the only bound OpenAI and vLLM put on it too -- openai-node's max_tokens is optional
        # and vLLM's defaults to 16 or None, neither tying it to logprobs.
        for max_tokens in (200, 5000, 0, -1, None):        # 0 / -1: "the rest of the context"; None: not sent
            choice = self.post({"logprobs": True, "top_logprobs": 1, "max_tokens": max_tokens})["choices"][0]
            self.assertEqual("".join(e["token"] for e in choice["logprobs"]["content"]),
                             "Café costs €5.", max_tokens)
        self.assertNotIn("logprobs", self.post({"max_tokens": 0})["choices"][0])   # not asked: as before

    def test_a_streamed_reply_needs_no_bound(self):
        # A streamed reply hands each entry out with its chunk and keeps none, so it is bounded by the reply's own
        # length.  OpenAI itself bounds a reply only by the model's context, and its SDKs send no max_tokens by
        # default.  5000 and "the rest of the context" must both be answered, as they are without logprobs.
        for max_tokens in (5000, 0, None):                 # None: the field is not sent at all
            chunks = self.post({"logprobs": True, "max_tokens": max_tokens}, stream=True)
            text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))
            toks = [e["token"] for c in chunks if c.get("choices")
                    for e in (c["choices"][0].get("logprobs") or {}).get("content", [])]
            self.assertEqual(text, "Café costs €5.", max_tokens)
            self.assertEqual("".join(toks), text, max_tokens)   # an entry for every token, streamed or not

    def test_a_json_schema_reply_takes_any_max_tokens(self):
        # structured_chunks buffers the reply to validate it, logprobs included: it holds the entries until the
        # end, as a reply that is not streamed does.  That memory is bounded by max_tokens, not by a cap of ours.
        written = '```json\n{"rent": "$1,250", "quote": "x"}\n```'
        self.engine = self.svc.engine = LpEngine(self.tok, written)
        content = '{"rent": "$1,250", "quote": "x"}'
        for max_tokens in (200, 0, 5000):
            choice = self.post({"logprobs": True, "max_tokens": max_tokens, "response_format": SCHEMA})["choices"][0]
            self.assertEqual(choice["message"]["content"], content, max_tokens)
            self.assertEqual("".join(e["token"] for e in choice["logprobs"]["content"]), content, max_tokens)
            chunks = self.post({"logprobs": True, "max_tokens": max_tokens, "response_format": SCHEMA}, stream=True)
            toks = [e["token"] for c in chunks if c.get("choices")
                    for e in (c["choices"][0].get("logprobs") or {}).get("content", [])]
            self.assertEqual("".join(toks), content, (max_tokens, "streamed"))

    def test_logprobs_are_refused_on_the_batch_path(self):
        # --parallel/--batch: the engine's batched loop prints "BT <slot> <id>" and never calls emit_lp_line
        # (src/program/generate.cpp), so no LP line reaches the server and the field would simply be absent.
        # Refused up front instead, before anything is generated; a request that does not ask is untouched.
        self.engine.batch = 2
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post({"logprobs": True})
        self.assertEqual(cm.exception.code, 400)
        self.assertIn("--parallel", cm.exception.read().decode())
        self.engine.batch = 0
        self.assertIn("logprobs", self.post({"logprobs": True})["choices"][0])

    def test_allowed_token_ids_outside_the_vocabulary_is_a_400(self):
        # emit_lp_line indexes its logits row with each named id, so an id past the vocabulary would read out of
        # bounds; the request is refused before anything is generated
        self.tok.tokens = [str(i) for i in range(300)]      # a tokenizer that knows its vocabulary
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post({"logprobs": True, "top_logprobs": 1, "allowed_token_ids": [300]})
        self.assertEqual(cm.exception.code, 400)
        self.assertIn("allowed_token_ids", cm.exception.read().decode())
        self.assertIn("logprobs", self.post({"logprobs": True, "top_logprobs": 1,
                                            "allowed_token_ids": [299]})["choices"][0])

    def test_streamed(self):
        chunks = self.post({"logprobs": True}, stream=True)
        content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))
        toks = [e["token"] for c in chunks if c.get("choices")
                for e in (c["choices"][0].get("logprobs") or {}).get("content", [])]
        self.assertEqual(content, "Café costs €5.")
        self.assertEqual("".join(toks), content)

    def test_with_thinking_each_part_is_spelled_by_its_tokens(self):
        tok = ThinkTokenizer()
        svc = Service(LpEngine(tok, "Two times three.</think>\n\nSix."), tok,
                      TEMPLATE)
        httpd = serve(svc, port=0)
        try:
            self.base = f"http://127.0.0.1:{httpd.server_address[1]}"
            for stream in (False, True):
                out = self.post({"logprobs": True, "chat_template_kwargs": {"enable_thinking": True}},
                                 stream)
                chunks = out if stream else [{"choices": [{"delta": out["choices"][0]["message"],
                                                           "logprobs": out["choices"][0]["logprobs"]}]}]
                for key in ("content", "reasoning_content"):
                    text = "".join(c["choices"][0]["delta"].get(key) or "" for c in chunks if c.get("choices"))
                    toks = "".join(e["token"] for c in chunks if c.get("choices")
                                   for e in (c["choices"][0].get("logprobs") or {}).get(key, []))
                    self.assertTrue(text)
                    self.assertEqual(toks, text, (key, stream))
                    self.assertNotIn("think>", toks)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_not_asked_no_logprobs(self):
        out = self.post({})
        self.assertNotIn("logprobs", out["choices"][0])

    def test_an_engine_without_lp_lines_gives_none(self):
        self.engine.with_lines = False
        out = self.post({"logprobs": True})
        self.assertEqual(out["choices"][0]["message"]["content"], "Café costs €5.")
        self.assertNotIn("logprobs", out["choices"][0])

    def test_allowed_token_ids_are_renormalised_over_the_named_ids(self):
        out = self.post({"logprobs": True, "allowed_token_ids": [65, 72]})
        lps = out["choices"][0]["logprobs"]["content"]
        self.assertTrue(lps)
        for e in lps:
            ids = [t["token"] for t in e["top_logprobs"]]
            self.assertEqual(ids, ["A", "H"])                    # only the named ones
        import math
        total = sum(math.exp(t["logprob"]) for t in lps[0]["top_logprobs"])
        self.assertAlmostEqual(total, 1.0, places=6)             # renormalised: they sum to 1

    def test_allowed_token_ids_with_nothing_to_renormalise(self):
        # the mock's alternatives are 65/72/88/91: naming another id leaves every line with no survivor.  Each
        # token keeps its own logprob and an empty candidate list, and the reply still comes back whole (this
        # used to be a 400 on the unstreamed path and a reply cut after the role chunk when streamed).
        for body in ({"logprobs": True, "allowed_token_ids": [66]},
                     {"logprobs": True, "allowed_token_ids": [66, 70, 71]}):
            choice = self.post(body)["choices"][0]
            lps = choice["logprobs"]["content"]
            self.assertEqual(choice["message"]["content"], "Café costs €5.")
            self.assertEqual("".join(e["token"] for e in lps), "Café costs €5.")
            self.assertEqual([e["top_logprobs"] for e in lps], [[]] * len(lps))
            self.assertEqual(lps[0]["logprob"], -0.01)           # the token's own, as the engine reported it
            streamed = self.post(body, stream=True)
            toks = [e["token"] for c in streamed if c.get("choices")
                    for e in (c["choices"][0].get("logprobs") or {}).get("content", [])]
            self.assertEqual("".join(toks), "Café costs €5.")

    def test_a_named_token_the_model_chose_is_in_its_distribution(self):
        # 46 is the reply's final ".": the engine does not repeat the chosen token among the named ids, so
        # naming only it used to leave nothing to normalise over.  It is the whole distribution now, at 0.
        lps = self.post({"logprobs": True, "allowed_token_ids": [46]})["choices"][0]["logprobs"]["content"]
        dot = [e for e in lps if e["token"] == "."]
        self.assertEqual(len(dot), 1)
        self.assertEqual(dot[0]["top_logprobs"], [{"token": ".", "logprob": 0.0, "bytes": [46]}])
        self.assertEqual(dot[0]["logprob"], 0.0)
        self.assertEqual([e["top_logprobs"] for e in lps if e["token"] != "."], [[]] * (len(lps) - 1))

    def test_a_choice_of_ten_that_only_the_ended_reply_settles(self):
        # ten named ids, and a reply that is two of them: "1" alone is already one of the choices, so which
        # one was picked is only known when the reply ends.  Here "11": each token's entries cover the ten
        # named ids (the chosen one included, because the engine never repeats it), and they sum to 1.
        self.engine = self.svc.engine = NamedEngine(self.tok, "11")
        digits = [ord(c) for c in "0123456789"]
        for stream in (False, True):
            out = self.post({"logprobs": True, "top_logprobs": 10, "allowed_token_ids": digits}, stream)
            chunks = out if stream else [{"choices": [{"delta": out["choices"][0]["message"],
                                                       "logprobs": out["choices"][0]["logprobs"]}]}]
            content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))
            lps = [e for c in chunks if c.get("choices")
                   for e in (c["choices"][0].get("logprobs") or {}).get("content", [])]
            self.assertEqual(content, "11")
            self.assertEqual([e["token"] for e in lps], ["1", "1"])     # the first token alone is a choice too
            self.assertEqual("".join(e["token"] for e in lps), content)
            for e in lps:
                ids = [t["token"] for t in e["top_logprobs"]]
                self.assertEqual(sorted(ids), sorted("0123456789"))    # the ten named ids, and only them
                self.assertIn(e["token"], ids)                         # the chosen one is one of them
                self.assertEqual(ids[0], e["token"])                   # ... and the most likely, here
                self.assertEqual([t["logprob"] for t in e["top_logprobs"]],
                                 sorted((t["logprob"] for t in e["top_logprobs"]), reverse=True))
                self.assertAlmostEqual(sum(math.exp(t["logprob"]) for t in e["top_logprobs"]), 1.0, places=6)

    def test_a_held_tail_of_more_than_one_token(self):
        # "5." is the start of the stop string, so the parser holds two tokens and sends them as one piece
        # after the request has left the queue: both entries come from this request's own lines all the same
        chunks = self.post({"logprobs": True, "stop": ["5.!"]}, stream=True)
        content = [c for c in chunks if c.get("choices") and c["choices"][0]["delta"].get("content")]
        self.assertEqual(content[-1]["choices"][0]["delta"]["content"], "5.")      # held, then flushed whole
        self.assertEqual([e["token"] for e in content[-1]["choices"][0]["logprobs"]["content"]], ["5", "."])
        text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))
        self.assertEqual(text, "Café costs €5.")
        self.assertEqual("".join(e["token"] for c in chunks if c.get("choices")
                                 for e in (c["choices"][0].get("logprobs") or {}).get("content", [])), text)
        choice = self.post({"logprobs": True, "stop": ["5.!"]})["choices"][0]         # and not streamed
        self.assertEqual(choice["message"]["content"], "Café costs €5.")
        self.assertEqual("".join(e["token"] for e in choice["logprobs"]["content"]), "Café costs €5.")
        self.assertEqual("".join(e["token"] for e in choice["logprobs"]["content"][-2:]), "5.")

    def test_json_schema_answer_as_written_with_its_logprobs(self):
        # MockEngine appends the stop token to the script it is built with, and reports
        # finish_reason=length when it runs out -- so the engine is rebuilt with this answer
        # rather than re-pointed at a raw token list (which would drop the stop token).
        written = '```json\n{"rent": "$1,250", "quote": "x"}\n```'
        self.engine = self.svc.engine = LpEngine(self.tok, written)
        out = self.post({"logprobs": True, "temperature": 0, "response_format": SCHEMA})
        choice = out["choices"][0]
        content = choice["message"]["content"]
        self.assertEqual(content, '{"rent": "$1,250", "quote": "x"}')   # the JSON as written
        self.assertEqual("".join(e["token"] for e in choice["logprobs"]["content"]), content)
        streamed = self.post({"logprobs": True, "response_format": SCHEMA}, stream=True)
        toks = [e["token"] for c in streamed if c.get("choices")
                for e in (c["choices"][0].get("logprobs") or {}).get("content", [])]
        self.assertEqual("".join(toks), content)
        plain = self.post({"temperature": 0, "response_format": SCHEMA})    # not asked: as before
        self.assertEqual(plain["choices"][0]["message"]["content"],
                         '{"rent":"$1,250","quote":"x"}')
        self.assertNotIn("logprobs", plain["choices"][0])


class Replicas(unittest.TestCase):
    """`--replicas`: a ReplicaEngine's own `batch` is how many requests its engines run at once and counts a replica
    without slots as one, so asking it "does this engine use batch slots" refused logprobs on a server that has
    none -- one that reads every reply by the GEN path, which does write LP lines."""

    @staticmethod
    def replica(*batches):
        from serve.replicas import ReplicaEngine
        return ReplicaEngine([SimpleNamespace(gen=1, batch=b) for b in batches])

    def test_no_slots_behind_replicas(self):
        self.assertFalse(batch_slots_on(self.replica(0, 0)))
        self.assertFalse(batch_slots_on(SimpleNamespace(batch=0)))      # no replicas: the engine's own figure

    def test_slots_behind_replicas(self):
        self.assertTrue(batch_slots_on(self.replica(2, 3)))             # every one of them
        self.assertTrue(batch_slots_on(self.replica(0, 2)))             # one is enough
        self.assertTrue(batch_slots_on(SimpleNamespace(batch=4)))


class Span(unittest.TestCase):
    def entries(self, *toks):
        return [{"token": t} for t in toks]

    def test_span(self):
        es = self.entries("```", "json", "\n", "{", "}", "\n", "```")
        self.assertEqual([e["token"] for e in logprob_span("```json\n{}\n```", "{}", es)], ["{", "}"])
        self.assertEqual(len(logprob_span("{}", "{}", self.entries("{", "}"))), 2)

    def test_no_span(self):
        self.assertIsNone(logprob_span("x{}", "{}", self.entries("x{", "}")))   # no boundary at the start
        self.assertIsNone(logprob_span("{}x", "{}", self.entries("{", "}x")))   # none at the end
        self.assertIsNone(logprob_span("{}", "{}", self.entries("{")))           # do not spell it


if __name__ == "__main__":
    unittest.main()
