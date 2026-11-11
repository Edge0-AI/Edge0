"""The streaming detokenizer (perf-review F-1): the incremental one emits exactly what the old re-decode did, token
by token, and its cost per token does not grow with the reply.

    python -m unittest serve.test_detok      (needs a pack's tokenizer; skipped without one)
"""
import os
import random
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from serve.server import Detokenizer  # noqa: E402


def find_tokenizer():
    cands = [os.environ.get("STRATA_TOKENIZER", "")]
    cands += [str(p) for p in ROOT.glob("packs/*/tokenizer")] + [str(p) for p in ROOT.glob("pack/*/tokenizer")]
    cands += [r"C:\Users\AI-Server\Desktop\Strata\Public\Engine\pack\full\tokenizer"]
    for c in cands:
        if c and (Path(c) / "vocab.json").exists():
            return Path(c)
    return None


class OldDetokenizer:
    """The pre-0.1.22 algorithm: re-decode every generated id, hold a trailing U+FFFD."""

    def __init__(self, tok):
        self.tok, self.ids, self.sent = tok, [], 0

    def push(self, t):
        self.ids.append(t)
        text = self.tok.decode(self.ids)
        if text.endswith("\ufffd"):
            return ""
        delta, self.sent = text[self.sent:], len(text)
        return delta


@unittest.skipIf(find_tokenizer() is None, "no pack tokenizer here")
class Detok(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        import strata_tokenizer as ST
        t = find_tokenizer()
        vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
        toks = [None] * len(vocab)
        for s, i in vocab.items():
            toks[i] = s
        cls.tok = ST.Tokenizer(toks, (t / "merges.txt").read_text(encoding="utf-8").split("\n"),
                               json.loads((t / "token_type.json").read_text()))

    def same_stream(self, ids):
        """The same text as the old re-decode, never behind it (the old one held complete characters too while
        the tail was an incomplete one; the new one holds only the incomplete bytes), and never ahead of what the
        ids so far decode to."""
        a, b = OldDetokenizer(self.tok), Detokenizer(self.tok)
        self.assertIsNotNone(b.inc)
        old = new = ""
        for i, t in enumerate(ids):
            old += a.push(t)
            new += b.push(t)
            self.assertTrue(new.startswith(old), f"token {i} ({t}): behind the old stream")
            self.assertTrue(self.tok.decode(ids[:i + 1]).startswith(new), f"token {i} ({t}): ahead of the ids")
        if not self.tok.decode(ids).endswith("�"):
            self.assertEqual(old, new)

    def test_text_with_split_characters(self):
        text = ("Grüße aus Zürich! 你好，世界。日本語のテキスト 🌊🦊👩‍💻 and ASCII code: def f(x): return x**2\n" * 40)
        self.same_stream(self.tok.encode(text))

    def test_random_ids_and_broken_bytes(self):
        rng = random.Random(5)
        n = len(self.tok.tokens)
        for _ in range(20):   # random ids: split and invalid UTF-8 sequences everywhere (no U+FFFD token itself)
            ids = [rng.randrange(n) for _ in range(300)]
            ids = [t for t in ids if "\ufffd" not in self.tok.decode([t], errors="strict" if False else "replace")
                   or len(self.tok.token_bytes(t)) < 3]
            self.same_stream(ids)

    def test_constant_cost(self):
        ids = self.tok.encode("The quick brown fox jumps over the lazy dog. " * 2000)[:16384]
        d = Detokenizer(self.tok)
        for t in ids[:1024]:
            d.push(t)
        t0 = time.perf_counter()
        for t in ids[1024:1124]:
            d.push(t)
        early = time.perf_counter() - t0
        for t in ids[1124:16000]:
            d.push(t)
        t0 = time.perf_counter()
        for t in ids[16000:16100]:
            d.push(t)
        late = time.perf_counter() - t0
        self.assertLess(late / 100, 0.0002, "a token after 16K tokens costs more than 0.2 ms")
        self.assertLess(late, early * 5 + 0.002)


# ------------------------------------------------------------------ #268: the heap BPE for long words
_REAL = []


def real_tokenizer():
    """The pack's tokenizer, loaded once for every class that needs it (None when there is no pack here)."""
    if not _REAL:
        t = find_tokenizer()
        if t is None:
            _REAL.append(None)
        else:
            import json
            import strata_tokenizer as ST
            vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
            toks = [None] * len(vocab)
            for s, i in vocab.items():
                toks[i] = s
            _REAL.append(ST.Tokenizer(toks, (t / "merges.txt").read_text(encoding="utf-8").split("\n"),
                                      json.loads((t / "token_type.json").read_text())))
    return _REAL[0]


def random_text(rng, n_pieces=40):
    """Strings shaped to make one pre-token piece long or awkward: CJK and Hangul runs (syllables and conjoining
    jamo), letters under stacks of combining marks, URLs, minified code, digits, emoji, whitespace, specials."""
    def run(lo, hi, k):
        return "".join(chr(rng.randint(lo, hi)) for _ in range(k))

    kinds = [
        lambda: run(0x4E00, 0x9FFF, rng.randint(1, 400)),                       # CJK
        lambda: run(0xAC00, 0xD7A3, rng.randint(1, 200)),                       # Hangul syllables
        lambda: "".join(chr(0x1100 + rng.randrange(19)) + chr(0x1161 + rng.randrange(21))
                        for _ in range(rng.randint(1, 60))),                    # conjoining jamo
        lambda: "".join(rng.choice("aeiouxyz") + run(0x0300, 0x036F, rng.randint(0, 4))
                        for _ in range(rng.randint(1, 80))),                    # combining marks
        lambda: "https://example.com/" + "/".join(run(0x61, 0x7A, rng.randint(1, 12))
                                                  for _ in range(rng.randint(1, 20))) + "?q=" + run(0x30, 0x39, 8),
        lambda: "".join(rng.choice(["var ", "a=", "b", "(", ")", "{", "}", ";", "function", "return", "x", "+",
                                    "!0", "void 0", ",", ".", "[", "]", "=>", "&&"]) for _ in range(rng.randint(1, 300))),
        lambda: run(0x30, 0x39, rng.randint(1, 50)),
        lambda: rng.choice([" ", "  ", "\n", "\n\n", "\t", "\r\n", " \n "]) * rng.randint(1, 30),
        lambda: rng.choice(["\U0001f600", "\U0001f30a", "\U0001f469‍\U0001f4bb", "é", "ß"]) * rng.randint(1, 30),
        lambda: rng.choice(["<|im_end|>", "<|im_start|>", "<think>", "</think>", "<tool_call>", "<|im_sta"]),
        lambda: "a" * rng.randint(60, 300),
        lambda: run(0x20, 0x7E, rng.randint(1, 200)),
    ]
    return "".join(rng.choice(kinds)() for _ in range(rng.randint(1, n_pieces)))


def encode_with(tok, text, parse_special, heap_min):
    tok.HEAP_MIN = heap_min
    try:
        return tok.encode(text, parse_special=parse_special)
    finally:
        del tok.HEAP_MIN


def synthetic_tokenizer(seed=268, n_merges=1500):
    """A small byte-level BPE vocabulary with merges learned (randomly) from random_text, and a few specials of
    both kinds - so the parity test runs without a pack."""
    import strata_tokenizer as ST
    rng = random.Random(seed)
    tokens = sorted(set(ST.BYTE_TO_UNICODE.values()))
    have, merges = set(tokens), []
    words = []
    for _ in range(60):
        for piece in ST.regex.compile(ST.QWEN35_PATTERN).findall(random_text(rng, 10)):
            words.append([ST.BYTE_TO_UNICODE[b] for b in piece.encode("utf-8")])
    words = [w for w in words if len(w) > 1]
    while len(merges) < n_merges:
        w = rng.choice(words)
        if len(w) < 2:
            continue
        i = rng.randrange(len(w) - 1)
        a, b = w[i], w[i + 1]
        w[i:i + 2] = [a + b]                        # learn in place, so later merges build on earlier ones
        if a + b in have or f"{a} {b}" in merges:
            continue
        merges.append(f"{a} {b}")
        tokens.append(a + b)
        have.add(a + b)
    types = [1] * len(tokens)
    for lit, ty in [("<|im_end|>", 3), ("<|im_start|>", 3), ("<think>", 4), ("</think>", 4), ("<tool_call>", 4)]:
        tokens.append(lit)
        types.append(ty)
    return ST.Tokenizer(tokens, merges, types)


class ThreadedEncode(unittest.TestCase):
    """#1385: 'int' object has no attribute 'ranks' in _bpe, seen once under load.  Could not be reproduced here (it
    looks like an interpreter fault); this pins that many threads sharing one Tokenizer, as the server does, get
    exactly the single-threaded ids, and that _bpe works with a tokenizer whose own state is only read."""

    def test_threads_match_serial(self):
        import threading
        tok = synthetic_tokenizer()
        rng = random.Random(1385)
        texts = [random_text(rng, 6) for _ in range(40)]
        want = [tok.encode(t) for t in texts]
        errors = []

        def work():
            try:
                for _ in range(15):
                    for t, w in zip(texts, want):
                        if tok.encode(t) != w:
                            errors.append("mismatch")
            except Exception as e:                  # noqa: BLE001
                errors.append(repr(e))

        ts = [threading.Thread(target=work) for _ in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(errors, [])


class HeapBpe(unittest.TestCase):
    """#268: words longer than HEAP_MIN symbols merge with a heap; the ids must be exactly the scan's."""

    def parity(self, tok, texts):
        for k, text in enumerate(texts):
            for ps in (False, True):
                old = encode_with(tok, text, ps, 10 ** 9)           # the scan for every word
                self.assertEqual(encode_with(tok, text, ps, 0), old, f"text {k} parse_special={ps}")   # the heap
                self.assertEqual(tok.encode(text, parse_special=ps), old, f"text {k} parse_special={ps}")

    def test_words_directly(self):
        tok = synthetic_tokenizer()
        rng = random.Random(1)
        import strata_tokenizer as ST
        for _ in range(300):
            for piece in ST.regex.compile(ST.QWEN35_PATTERN).findall(random_text(rng, 8)):
                word = "".join(ST.BYTE_TO_UNICODE[b] for b in piece.encode("utf-8"))
                tok.HEAP_MIN = 10 ** 9
                try:
                    old = tok._bpe(word)
                finally:
                    del tok.HEAP_MIN
                self.assertEqual(tok._bpe_heap(word), old, repr(piece[:40]))

    def test_synthetic_vocab(self):
        tok = synthetic_tokenizer()
        rng = random.Random(2)
        self.parity(tok, [random_text(rng) for _ in range(60)] + ["a" * 5000, "ab" * 3000, ""])

    @unittest.skipIf(find_tokenizer() is None, "no pack tokenizer here")
    def test_pack_vocab(self):
        tok = real_tokenizer()
        rng = random.Random(3)
        fixed = ["你好，世界" * 150, "한국어 텍스트" * 100,
                 "é́̂" * 200, "x" * 5000,
                 "https://example.com/" + "a/b-c_d.e?f=g&h=i#j" * 60,
                 "!function(e){var t={};function n(r){if(t[r])return t[r].exports}}" * 30,
                 "<|im_start|>user\n日本語" * 40 + "<|im_end|><think>" + "漢" * 300]
        self.parity(tok, fixed + [random_text(rng) for _ in range(40)])

    def check_speed(self, tok):
        rng = random.Random(4)
        text = "".join(chr(rng.randint(0x4E00, 0x9FFF)) for _ in range(8192))       # one 24K-byte piece
        t0 = time.perf_counter()
        ids = tok.encode(text)
        took = time.perf_counter() - t0
        self.assertEqual(tok.decode(ids), text)
        self.assertLess(took, 1.0, f"an 8K-character CJK run took {took:.2f} s")

    def test_speed_synthetic(self):
        self.check_speed(synthetic_tokenizer())

    @unittest.skipIf(find_tokenizer() is None, "no pack tokenizer here")
    def test_speed_pack(self):
        self.check_speed(real_tokenizer())


if __name__ == "__main__":
    unittest.main()
