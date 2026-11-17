"""The end of a reply and the text held back in front of a stop string (#4576: "Nerazim" came out as "Neraz").

DeepSeek-V4.1's tokenizer has no single token for Qwen's <|im_end|> / <|endoftext|>: it splits them into '<', '|', 'im', '_end',
'|', '>' and 'end', 'of', 'text'.  A server that took the ids of those pieces as its stop ids (a config without "arch") ended the
reply in front of the first "im" it wrote: Nerazim = N, er, az, im.  These tests keep three things true, for the OpenAI and the
Anthropic route, streamed and not, with and without the family named:

  * the reply that has no stop string comes out whole (every word that ends in a piece of a stop string or of an end marker);
  * a stop string cuts the reply exactly there, whatever the chunking;
  * the text held back for a stop string that does not come is given back at the end.

    python -m unittest serve.test_stop_holdback -v
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST  # noqa: E402
from serve import frontend, server  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Detokenizer, Service, StopMatcher, end_token_ids, serve  # noqa: E402
from serve.test_dsv41 import EOS, SPECIALS, TEMPLATE, DsEngine  # noqa: E402

QWEN_STOPS = ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "\nUser:"]
DEEPSEEK_STOPS = ["<｜end▁of▁sentence｜>", "<｜User｜>", "</s>", "\n\nHuman:"]
# words and endings that look like the start of a stop string or a piece of an end marker
WORDS = ["Nerazim", "Zim", "swim", "Neraz", "im", "im_end", "Nerazimville", "a|b|c|", "x > y and a < b", "the end of the text",
         "<", "<|", "<|im", "<|im_end", "|>", "friend", "legend", "\nUser", "\n\nHuman", "</", "</s", "<｜User", "<｜", "Nerazim<",
         "Nerazim|", "Nerazim <|im_start|"]


def nerazim_tokenizer() -> ST.Tokenizer:
    """DeepSeek's specials, the 256 byte tokens and the pieces of the real vocabulary that matter here: N er az im ('im' is id
    328 in the real one), and the pieces its tokenizer cuts Qwen's end markers into."""
    b2u = ST.bytes_to_unicode()
    extra = ["er", "az", "im", "_e", "_en", "_end", "en", "end", "of", "te", "tex", "text"]
    merges = ["e r", "a z", "i m", "_ e", "_e n", "_en d", "e n", "en d", "o f", "t e", "te x", "tex t"]
    tokens = [s for s, _ in SPECIALS] + [b2u[b] for b in range(256)] + extra
    types = [t for _, t in SPECIALS] + [1] * (256 + len(extra))
    return ST.Tokenizer(tokens, merges, types, "deepseek3", {"bos": 0, "eos": EOS})


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = nerazim_tokenizer()

    def test_the_fixture_has_the_real_tokenizer_shape(self):
        self.assertEqual([self.tok.tokens[i] for i in self.tok.encode("Nerazim")], ["N", "er", "az", "im"])
        pieces = [self.tok.tokens[i] for i in self.tok.encode("<|im_end|>", parse_special=True)]
        self.assertEqual(pieces, ["<", "|", "im", "_end", "|", ">"])
        pieces = [self.tok.tokens[i] for i in self.tok.encode("<|endoftext|>", parse_special=True)]
        self.assertEqual(pieces, ["<", "|", "end", "of", "text", "|", ">"])


class EndTokenIds(Fixture):
    def test_pieces_of_the_qwen_markers_are_not_stop_ids(self):
        ids = end_token_ids(self.tok)
        self.assertEqual(ids, {EOS})
        pieces = set(self.tok.encode("<|im_end|>", parse_special=True)) | set(self.tok.encode("<|endoftext|>", parse_special=True))
        self.assertFalse(ids & pieces)

    def test_qwen_markers_that_are_one_token_each_are_the_stop_ids(self):
        tok = ByteTokenizer()
        self.assertEqual(end_token_ids(tok), set(tok.encode("<|im_end|>", parse_special=True))
                         | set(tok.encode("<|endoftext|>", parse_special=True)))
        self.assertEqual(len(end_token_ids(tok)), 2)

    def test_no_marker_and_no_eos_means_no_stop_id(self):
        class Bare:
            def encode(self, text, parse_special=False):
                return list(text.encode())
        self.assertEqual(end_token_ids(Bare()), set())

    def test_a_service_without_a_family_does_not_stop_on_im(self):
        svc = Service(DsEngine(self.tok, "x"), self.tok, ChatTemplate(TEMPLATE))
        im = self.tok.encode("im")[0]
        for piece in ("im", "<", "|", ">", "end", "of", "text", "_end"):
            self.assertNotIn(self.tok.encode(piece)[0], svc.stop_ids, piece)
        self.assertIn(im, set(self.tok.encode("<|im_end|>", parse_special=True)))
        self.assertEqual(svc.stop_ids, {EOS})


class StopMatcherSplits(unittest.TestCase):
    """StopMatcher alone: every way to cut the text in two, and one character at a time."""

    @staticmethod
    def run_matcher(stops, chunks):
        m = StopMatcher(stops)
        out = "".join(m.push(c) for c in chunks)
        if m.hit is None:
            out += m.flush()
        return out, m.hit

    def chunkings(self, text):
        yield [text]
        for i in range(1, len(text)):
            yield [text[:i], text[i:]]
        yield list(text)

    def test_text_without_a_stop_string_comes_out_whole_at_every_split(self):
        for stops in (QWEN_STOPS, DEEPSEEK_STOPS, ["im_end"], ["im"], ["zim"], ["|", ">"], ["end"]):
            for word in WORDS:
                text = "My friend " + word
                if any(s in text for s in stops):
                    continue
                for chunks in self.chunkings(text):
                    with self.subTest(stops=stops, text=text, chunks=chunks):
                        self.assertEqual(self.run_matcher(stops, chunks), (text, None))

    def test_a_stop_string_cuts_exactly_there_at_every_split(self):
        for stops in (QWEN_STOPS, DEEPSEEK_STOPS):
            for stop in stops:
                for word in ("Nerazim", "swim"):
                    text = f"My friend {word}{stop}never seen"
                    for chunks in self.chunkings(text):
                        with self.subTest(stops=stops, text=text, chunks=chunks):
                            first = min((text.find(s), s) for s in stops if s in text)
                            self.assertEqual(self.run_matcher(stops, chunks), (text[:first[0]], first[1]))

    def test_the_held_back_tail_is_given_back_at_the_end(self):
        for stops in (QWEN_STOPS, DEEPSEEK_STOPS):
            for text in ("Nerazim <|im", "Nerazim <｜User", "Nerazim\nUser", "Nerazim <", "ends with a pipe |"):
                with self.subTest(stops=stops, text=text):
                    self.assertEqual(self.run_matcher(stops, list(text)), (text, None))

    def test_a_stop_string_that_starts_like_the_word_costs_nothing_but_the_wait(self):
        # "im_end" as a stop: "Nerazim" holds "im" until the next character says it is not the stop string
        m = StopMatcher(["im_end"])
        self.assertEqual(m.push("Neraz"), "Neraz")
        self.assertEqual(m.push("im"), "")
        self.assertEqual(m.held, "im")
        self.assertEqual(m.flush(), "im")


class ByteWise(Fixture):
    """The text path as the server runs it: one token per byte through the Detokenizer (a UTF-8 character split across tokens is held
    until it is whole), then the StopMatcher."""

    def run_bytes(self, stops, text):
        tok = ByteTokenizer()
        detok, m, out = Detokenizer(tok), StopMatcher(stops), ""
        for b in text.encode("utf-8"):
            out += m.push(detok.push(b))
            if m.hit is not None:
                break
        if m.hit is None:
            out += m.flush()
        return out, m.hit

    def test_every_word_comes_out_whole_byte_by_byte(self):
        for stops in (QWEN_STOPS, DEEPSEEK_STOPS):
            for word in WORDS + ["Zażółć", "日本語の名前"]:
                text = "Hello " + word
                if any(s in text for s in stops):
                    continue
                with self.subTest(stops=stops, text=text):
                    self.assertEqual(self.run_bytes(stops, text), (text, None))

    def test_a_multibyte_stop_string_is_cut_byte_by_byte(self):
        text = "Nerazim<｜end▁of▁sentence｜>not shown"
        self.assertEqual(self.run_bytes(DEEPSEEK_STOPS, text), ("Nerazim", "<｜end▁of▁sentence｜>"))


class Routes(Fixture):
    """The four ways out of the server (OpenAI and Anthropic, streamed and whole), with the family named and without it."""

    def start(self, script, named):
        """A server for `named` (one per test and setting, reused); the engine replays `script` for the next request."""
        if named in self.servers:
            engine = self.servers[named]
            engine.script = self.tok.encode(script) + [EOS]     # (a marker in the script is text the model wrote)
            self.base = self.bases[named]
            return engine
        engine = self.servers[named] = DsEngine(self.tok, script)
        engine.script = self.tok.encode(script) + [EOS]
        svc = Service(engine, self.tok, ChatTemplate(TEMPLATE), model_name="deepseek-v4.1-flash")
        if named:
            svc.model_arch = "deepseek41"
            svc.set_family("deepseek41")
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.addCleanup(frontend.use_family, None)
        self.base = self.bases[named] = f"http://127.0.0.1:{httpd.server_address[1]}"
        return engine

    def send(self, route, stream, stops):
        if route == "openai":
            path = "/v1/chat/completions"
            body = {"model": "m", "max_tokens": 200, "stream": stream, "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": "Use the name Nerazim in a sentence."}]}
            if stops:
                body["stop"] = stops[:4]
        else:
            path = "/v1/messages"
            body = {"model": "m", "max_tokens": 200, "stream": stream, "thinking": {"type": "disabled"},
                    "messages": [{"role": "user", "content": "Use the name Nerazim in a sentence."}]}
            if stops:
                body["stop_sequences"] = stops
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8")
        text, finish, stop_seq = "", None, None
        if not stream:
            d = json.loads(raw)
            if route == "openai":
                text, finish = d["choices"][0]["message"].get("content") or "", d["choices"][0]["finish_reason"]
            else:
                text = "".join(b.get("text", "") for b in d["content"] if b["type"] == "text")
                finish, stop_seq = d["stop_reason"], d.get("stop_sequence")
            return text, finish, stop_seq
        for line in raw.splitlines():
            if not line.startswith("data:") or line.strip() == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if route == "openai":
                for ch in d.get("choices", []):
                    text += (ch.get("delta") or {}).get("content") or ""
                    finish = ch.get("finish_reason") or finish
            elif d.get("type") == "content_block_delta":
                text += d["delta"].get("text", "")
            elif d.get("type") == "message_delta":
                finish, stop_seq = d["delta"].get("stop_reason"), d["delta"].get("stop_sequence")
        return text, finish, stop_seq

    def setUp(self):
        self.servers = {}
        self.bases = {}

    def every_route(self):
        for named in (False, True):
            for route in ("openai", "anthropic"):
                for stream in (False, True):
                    yield named, route, stream

    def test_nerazim_and_its_kin_come_out_whole_with_no_stop_string(self):
        for word in ("Nerazim", "Nerazimville and Nerazim", "swim", "Zim", "a|b|c|", "x > y", "the end of the text", "im_end", "<", "Nerazim<|im"):
            for named, route, stream in self.every_route():
                with self.subTest(word=word, named=named, route=route, stream=stream):
                    self.start(word, named)
                    text, finish, _ = self.send(route, stream, None)
                    self.assertEqual(text.strip(), word)
                    self.assertIn(finish, ("stop", "end_turn"))

    def test_the_held_tail_is_given_back_when_the_stop_string_never_comes(self):
        for stops in (QWEN_STOPS, DEEPSEEK_STOPS, ["im_end", "|>"]):
            for named, route, stream in self.every_route():
                for word in ("My friend Nerazim", "Nerazim <|im", "Nerazim <｜User", "ends with a pipe |"):
                    with self.subTest(stops=stops, word=word, named=named, route=route, stream=stream):
                        self.start(word, named)
                        text, finish, stop_seq = self.send(route, stream, stops)
                        self.assertEqual(text.strip(), word)
                        self.assertIn(finish, ("stop", "end_turn"))
                        self.assertIsNone(stop_seq)

    def test_a_stop_string_cuts_the_reply_there(self):
        for stops in (QWEN_STOPS, DEEPSEEK_STOPS):
            for named, route, stream in self.every_route():
                for stop in stops:
                    with self.subTest(stops=stops, stop=stop, named=named, route=route, stream=stream):
                        self.start("My friend Nerazim" + stop + "and more", named)
                        text, finish, stop_seq = self.send(route, stream, stops)
                        self.assertEqual(text.strip(), "My friend Nerazim")
                        self.assertIn(finish, ("stop", "end_turn", "stop_sequence"))
                        if route == "anthropic":
                            self.assertEqual(stop_seq, stop)


if __name__ == "__main__":
    unittest.main()
