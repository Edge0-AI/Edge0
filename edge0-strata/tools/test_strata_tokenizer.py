"""Tests for tools/strata_tokenizer.py: llama.cpp's reference vectors, the piece cache (0.1.41: the ids are the ones the
merge loop gives, cache cold, warm or switched off), and the incremental prompt encoder, which must give exactly the
ids of a plain encode.

    python -m unittest tools.test_strata_tokenizer -v

The real vocabulary comes from $STRATA_TOKENIZER (a pack's tokenizer/ directory) when set, else from llama.cpp's
`models/ggml-vocab-qwen35.gguf` next to the gguf-py the tools use ($STRATA_GGUF_PY, third_party/llama.cpp); the
tests that need it are skipped without one.  The chat template's USER_DEFINED tokens (`<think>`, `<tool_call>`,
...) are added to the llama.cpp vocabulary, which does not have them, as the pack's has.
"""
from __future__ import annotations

import json
import os
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import strata_tokenizer as ST  # noqa: E402

ADDED = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
         "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]


def vocab_gguf() -> Path | None:
    cands = [os.environ.get("STRATA_GGUF_PY"), ROOT / "third_party" / "llama.cpp" / "gguf-py", "/opt/llama.cpp/gguf-py"]
    for c in cands:
        if c and (Path(c).parent / "models" / "ggml-vocab-qwen35.gguf").is_file():
            return Path(c).parent / "models" / "ggml-vocab-qwen35.gguf"
    return None


def load_tokenizer():
    pack = os.environ.get("STRATA_TOKENIZER")
    if pack and (Path(pack) / "vocab.json").is_file():
        t = Path(pack)
        vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for s, i in vocab.items():
            tokens[i] = s
        return ST.Tokenizer(tokens, (t / "merges.txt").read_text(encoding="utf-8").split("\n"),
                            json.loads((t / "token_type.json").read_text()))
    path = vocab_gguf()
    return None if path is None else gguf_tokenizer(path)


def gguf_tokenizer(path):
    """llama.cpp's qwen35 vocabulary with the template's added tokens: the one its test vectors were made with."""
    from gguf_reader import GGUFFile
    md = GGUFFile(path).metadata
    tokens, types = list(md["tokenizer.ggml.tokens"]), list(md["tokenizer.ggml.token_type"])
    for s in ADDED:
        if s not in tokens:
            tokens.append(s)
            types.append(4)
    return ST.Tokenizer(tokens, list(md["tokenizer.ggml.merges"]), types)


def byte_tokenizer(specials: list[str]) -> ST.Tokenizer:
    """A vocabulary of the 256 byte tokens and `specials` (CONTROL), no merges: small enough to build per test."""
    tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)] + specials
    return ST.Tokenizer(tokens, [], [1] * 256 + [3] * len(specials))


WORDS = ["the", "tokenizer", " ", "  ", "\n", "\n\n", "\t", "def", "f(x):", "return", "x", "==", "->", "café",
         "你好", "\U0001f600", "é", "1234", "don't", "I'm", "<|im_", "<think", "</", "tool", "_call>",
         " ", "\r\n", "Hello,", "world!", "<", ">", "|", "      ", "x" * 70]


def random_text(rnd: random.Random, n: int, specials=()) -> str:
    parts = []
    for _ in range(n):
        parts.append(rnd.choice(WORDS) if not specials or rnd.random() > 0.05 else rnd.choice(specials))
        if rnd.random() < 0.5:
            parts.append(" ")
    return "".join(parts)


def conversation_prompts(template, rnd: random.Random) -> list[tuple[str, str]]:
    """Rendered prompts, in the order a server would see them, from clients that grow a conversation turn after turn,
    use tools, send images, change the start of the prompt, edit an earlier turn, send a shorter conversation, and
    interleave with each other: (what happens, prompt)."""
    out: list[tuple[str, str]] = []
    specials = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_response>"]
    weather = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather",
                                                  "parameters": {"type": "object", "properties": {
                                                      "city": {"type": "string"}}, "required": ["city"]}}}]

    def add(what, msgs, tools=None, **kw):
        out.append((what, template.render(msgs, tools=tools, **kw)))

    # several turns, with unicode, a literal special token inside the text, and reasoning kept or not
    msgs = [{"role": "system", "content": "Be brief. Réponds en français si on te parle en français."}]
    for turn in range(6):
        msgs.append({"role": "user", "content": random_text(rnd, 60, specials) + " é你\U0001f600 " * turn})
        add(f"turn {turn}", msgs)
        answer = {"role": "assistant", "content": random_text(rnd, 30)}
        if turn % 2:
            answer["reasoning_content"] = random_text(rnd, 30)
        msgs.append(answer)
    grown = list(msgs)

    # tools: definitions, a call, its response, then a following turn
    tmsgs = [{"role": "user", "content": "Weather in Oslo?"}]
    add("tools 1", tmsgs, weather)
    tmsgs += [{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
        "name": "get_weather", "arguments": {"city": "Oslo"}}}]},
        {"role": "tool", "content": "{\"temp\": 3}", "name": "get_weather"}]
    add("tools 2", tmsgs, weather)
    tmsgs += [{"role": "assistant", "content": "3 degrees."}, {"role": "user", "content": "And in Bergen? " * 20}]
    add("tools 3", tmsgs, weather)

    # images: one per user turn, the placeholder tokens are specials
    imsgs = []
    for turn in range(3):
        imsgs.append({"role": "user", "content": [{"type": "image", "source": f"img{turn}"},
                                                   {"type": "text", "text": f"What is on picture {turn}?"}]})
        add(f"images {turn}", imsgs)
        imsgs.append({"role": "assistant", "content": "A cat."})

    # the start of the prompt changes (a thinking level, a system message), then the old conversation grows again
    add("effort changed", grown, reasoning_effort="low")
    add("system changed", [dict(grown[0], content=grown[0]["content"] + " Be kind.")] + grown[1:])
    grown.append({"role": "user", "content": "one more question"})
    add("grown again", grown)

    # an earlier turn edited, then a conversation that got shorter, then the full one again
    edited = list(grown)
    edited[3] = dict(edited[3], content=edited[3]["content"] + " (edited)")
    add("edited", edited)
    add("shorter", grown[:5])
    add("empty user", [{"role": "user", "content": ""}])
    add("full again", grown)
    return out


class Oracle(unittest.TestCase):
    """llama.cpp's own tokenizer test vectors for qwen35."""

    def test_llama_cpp_vectors(self):
        path = vocab_gguf()
        if path is None or not path.with_suffix(".gguf.inp").is_file():
            self.skipTest("llama.cpp's ggml-vocab-qwen35.gguf test vectors are not here")
        tk = gguf_tokenizer(path)                  # not $STRATA_TOKENIZER: the vectors belong to this vocabulary
        inp = path.with_suffix(".gguf.inp").read_text(encoding="utf-8").split("\n__ggml_vocab_test__\n")
        out = path.with_suffix(".gguf.out").read_text(encoding="utf-8").split("\n")
        n = 0
        for text, want in zip(inp, out):
            if text.endswith("\n__ggml_vocab_test__"):
                text = text[:-len("\n__ggml_vocab_test__")]
            self.assertEqual(tk.encode(text), [int(x) for x in want.split()], repr(text[:40]))
            n += 1
        self.assertGreater(n, 40)


class Marks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tk = load_tokenizer()
        if cls.tk is None:
            raise unittest.SkipTest("no tokenizer (set STRATA_TOKENIZER or STRATA_GGUF_PY)")

    def test_marks(self):
        text = "a<|im_start|>user\nhi<|im_end|>\n<think>\n"
        ids, marks = self.tk.encode_marked(text, parse_special=True)
        self.assertEqual(ids, self.tk.encode(text, parse_special=True))
        self.assertEqual([m[0] for m in marks], [text.index("user"), text.index("<|im_end|>") + len("<|im_end|>"),
                                                  len(text) - 1])
        for end, n in marks:                            # each mark: the ids of the text up to it
            self.assertEqual(ids[:n], self.tk.encode(text[:end], parse_special=True))


class Prompts(unittest.TestCase):
    """PromptEncoder: the ids of a plain encode, whatever the earlier prompts were."""

    @classmethod
    def setUpClass(cls):
        cls.tk = load_tokenizer()
        if cls.tk is None:
            raise unittest.SkipTest("no tokenizer (set STRATA_TOKENIZER or STRATA_GGUF_PY)")
        from serve.frontend import ChatTemplate
        cls.template = ChatTemplate(ROOT / "serve" / "chat_template.jinja")

    def check(self, enc, text):
        self.assertEqual(enc.encode(text), self.tk.encode(text, parse_special=True))

    def test_chat_golden(self):
        cases = json.loads((ROOT / "serve" / "chat_golden.json").read_text(encoding="utf-8"))
        enc = ST.PromptEncoder(self.tk)
        for c in cases + cases:                         # the second round finds its own earlier prompts
            if c.get("rendered"):
                with self.subTest(case=c["name"]):
                    self.check(enc, c["rendered"])

    def test_conversations(self):
        for seed in range(3):
            prompts = conversation_prompts(self.template, random.Random(seed))
            wants = [self.tk.encode(prompt, parse_special=True) for _, prompt in prompts]
            for keep in (1, 2, 4):
                enc = ST.PromptEncoder(self.tk, keep=keep)
                for (what, prompt), want in zip(prompts, wants):
                    with self.subTest(seed=seed, keep=keep, what=what):
                        self.assertEqual(enc.encode(prompt), want)

    def test_growing_turn_reuses_the_previous_prompt(self):
        enc = ST.PromptEncoder(self.tk)
        msgs = [{"role": "user", "content": "first question " * 50}]
        prev = ""
        for turn in range(4):
            prompt = self.template.render(msgs)
            self.check(enc, prompt)
            if turn:
                self.assertGreater(enc.last_reused, len(prev) - 100)
            prev = prompt
            msgs += [{"role": "assistant", "content": f"answer {turn}"}, {"role": "user", "content": f"question {turn}"}]

    def test_prefix_of_an_earlier_prompt_and_unrelated_prompts(self):
        enc = ST.PromptEncoder(self.tk, keep=2)
        a = self.template.render([{"role": "user", "content": "first question " * 50}])
        b = self.template.render([{"role": "user", "content": "another conversation"}])
        c = self.template.render([{"role": "user", "content": "a third one"}])
        for text in (a, a[:len(a) // 2], b, c, a, a + "tail", "", "<|im_end|>", "<|im_end|><|im_end|>x"):
            with self.subTest(text=text[:30]):
                self.check(enc, text)

    def test_a_read_only_encode_gives_the_ids_and_stores_nothing(self):
        enc = ST.PromptEncoder(self.tk)
        a = self.template.render([{"role": "user", "content": "first question " * 50}])
        enc.encode(a)
        kept = list(enc.entries)
        for text in (a + "<|im_start|>user\nand a follow-up<|im_end|>\n", a[:len(a) // 2], "<|im_end|>x",
                     self.template.render([{"role": "user", "content": "another conversation"}])):
            with self.subTest(text=text[:30]):
                self.assertEqual(enc.encode(text, store=False), self.tk.encode(text, parse_special=True))
                self.assertTrue(enc.entries == kept, "a read-only encode wrote an entry")
        self.assertEqual(enc.encode(a + "tail", store=False), self.tk.encode(a + "tail", parse_special=True))
        self.assertGreater(enc.last_reused, len(a) // 2)

    def test_the_returned_ids_are_the_callers(self):
        enc = ST.PromptEncoder(self.tk)
        text = "<|im_start|>user\nhi<|im_end|>\n" + "and a tail longer than the look-ahead margin. " * 3
        want = self.tk.encode(text, parse_special=True)
        got = enc.encode(text)
        got[0] = -1                                     # inside the part the next call takes from this one
        got.append(-1)
        self.assertEqual(enc.encode(text), want)
        self.assertGreater(enc.last_reused, 0)


class Margin(unittest.TestCase):
    """A special literal that is a prefix of a longer one: cutting at the shorter one's end without the look-ahead
    margin would give different ids."""

    def test_longer_literal_past_the_common_prefix(self):
        tk = byte_tokenizer(["<|a|>", "<|a|>zz"])
        enc = ST.PromptEncoder(tk)
        first = "x<|a|>y" + "<|a|>" * 3 + "q"
        self.assertEqual(enc.encode(first), tk.encode(first, parse_special=True))
        second = "x<|a|>y" + "<|a|>" * 3 + "zz"             # shares everything up to the last <|a|>
        got = enc.encode(second)
        self.assertEqual(got, tk.encode(second, parse_special=True))
        self.assertEqual(got[-1], tk.special_tokens["<|a|>zz"])
        self.assertGreater(enc.last_reused, 0)              # the earlier boundaries were still reused

    def test_literal_ending_inside_another(self):
        tk = byte_tokenizer(["<|a|>", "b<|a|>c"])
        enc = ST.PromptEncoder(tk)
        for text in ("zzb<|a|>d", "zzb<|a|>c", "zzb<|a|>cz<|a|>", "zzb<|a|>cz<|a|>b<|a|>c"):
            with self.subTest(text=text):
                self.assertEqual(enc.encode(text), tk.encode(text, parse_special=True))

    def test_random_specials(self):
        """Literals that extend or end inside one another, texts that change right after one of them."""
        rnd = random.Random(3)

        def word(lo, hi):
            return "".join(rnd.choice("<|ab>") for _ in range(rnd.randint(lo, hi)))
        for trial in range(60):
            a, b = word(2, 4), word(2, 4)
            lits = sorted({a, a + word(1, 3), b, word(1, 2) + b + word(0, 2)})
            tk = byte_tokenizer(lits)
            enc = ST.PromptEncoder(tk, keep=3)
            pieces = lits + [x[:rnd.randint(1, len(x))] for x in lits] + ["x", "y", "a", "|"]
            base = "".join(rnd.choice(pieces) for _ in range(60))
            for _ in range(30):
                cut = rnd.randint(0, len(base))
                text = base[:cut] + "".join(rnd.choice(pieces) for _ in range(rnd.randint(0, 6)))
                self.assertEqual(enc.encode(text), tk.encode(text, parse_special=True), (lits, text))
                if rnd.random() < 0.3:
                    base = text + base[cut:]


class CommonPrefix(unittest.TestCase):
    def test_against_a_loop(self):
        rnd = random.Random(5)
        for _ in range(500):
            a = "".join(rnd.choice("ab") for _ in range(rnd.randint(0, 9000)))
            k = rnd.randint(0, len(a))
            b = a[:k] + rnd.choice(["", "c", "cd"]) + a[k:k + rnd.randint(0, 50)]
            want = 0
            while want < min(len(a), len(b)) and a[want] == b[want]:
                want += 1
            self.assertEqual(ST.common_prefix_len(a, b), want)
            self.assertEqual(ST.common_prefix_len(b, a), want)



def toy() -> ST.Tokenizer:
    base = [ST.BYTE_TO_UNICODE[b] for b in range(256)]
    merges = ["t h", "th e", "i n", "a n", "an d", "Ġ t", "Ġt h", "Ġth e", "e r", "o n"]
    merges = [m.replace("Ġ", ST.BYTE_TO_UNICODE[32]) for m in merges]
    tokens = list(base)
    for m in merges:
        a, b = m.split(" ")
        tokens.append(a + b)
    return ST.Tokenizer(tokens, merges)


class PieceCache(unittest.TestCase):
    def test_same_ids_cold_warm_and_off(self):
        rng = random.Random(7)
        words = ["the", "and", "in", "on", "there", "other", "x" * 70, "你好", "def", "return", "\n\n", "  "]
        text = "".join(rng.choice(words) + rng.choice([" ", "", "\n", "\t"]) for _ in range(3000))
        a = toy()
        a.PIECE_CACHE_MAX = 0
        want = a.encode(text)
        b = toy()
        cold = b.encode(text)
        warm = b.encode(text)
        self.assertEqual(cold, want)
        self.assertEqual(warm, want)
        self.assertGreater(len(b._piece_ids), 0)
        self.assertEqual(b.decode(warm), text)

    def test_the_cache_is_bounded(self):
        t = toy()
        t.PIECE_CACHE_MAX = 5
        t.encode(" ".join("w%d" % i for i in range(100)))
        self.assertLessEqual(len(t._piece_ids), 5)


class ByteCache(unittest.TestCase):
    def test_decode_before_encode_cold_and_warm(self):
        t = toy()
        text = "café 日本語 🙂\r\n\x00"
        ids = list(text.encode("utf-8"))  # toy's first 256 token IDs are the raw bytes
        self.assertEqual(t.decode(ids, errors="strict"), text)
        self.assertEqual(t.decode(ids, errors="strict"), text)
        self.assertEqual(t.decode([]), "")
        for i in range(256):
            self.assertEqual(t.token_bytes(i), bytes([i]))

    def test_invalid_ids_before_and_after_cache_use(self):
        t = toy()
        for _ in range(2):
            for i in (-1, len(t.tokens), len(t.tokens) + 100):
                with self.assertRaises(IndexError):
                    t.token_bytes(i)
            self.assertEqual(t.token_bytes(65), b"A")

    def test_instances_with_different_vocabularies_stay_independent(self):
        tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)]
        a, b = ST.Tokenizer(tokens, []), ST.Tokenizer(tokens[::-1], [])
        text = "the café 日本語 🙂\n"
        for t in (a, b, a, b):
            expected = [t.ids[ST.BYTE_TO_UNICODE[byte]] for byte in text.encode("utf-8")]
            self.assertEqual(t.encode(text), expected)
            self.assertEqual(t.decode(expected, errors="strict"), text)
        self.assertEqual(a.token_bytes(0), b"\x00")
        self.assertEqual(b.token_bytes(0), b"\xff")

    def test_shared_tokenizer_concurrent_first_decode(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        t = toy()
        texts = ["café", "日本語", "🙂\r\n", "\x00plain"]
        barrier = threading.Barrier(len(texts))
        def decode(text):
            ids = list(text.encode("utf-8"))
            barrier.wait(timeout=5)
            for _ in range(100):
                if t.decode(ids, errors="strict") != text:
                    raise AssertionError("concurrent decode changed the text")
            return text
        with ThreadPoolExecutor(len(texts)) as pool:
            self.assertEqual(list(pool.map(decode, texts)), texts)


if __name__ == "__main__":
    unittest.main()
