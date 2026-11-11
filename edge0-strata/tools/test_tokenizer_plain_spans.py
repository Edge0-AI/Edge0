"""Exact token-ID parity for protected literals, including general caller span inputs."""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_tokenizer as ST


class ReferenceTokenizer(ST.Tokenizer):
    def _encode_matching(self, text, pat, plain=()):
        if pat is None:
            return self._encode_plain(text)
        out, pos = [], 0
        for m in pat.finditer(text):
            if plain and any(a <= m.start() < b for a, b in plain):
                continue
            if m.start() > pos:
                out.extend(self._encode_plain(text[pos:m.start()]))
            out.append(self.special_tokens[m.group(0)])
            pos = m.end()
        if pos < len(text):
            out.extend(self._encode_plain(text[pos:]))
        return out


def tokenizer(cls=ST.Tokenizer, specials=True, merges=False):
    tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)]
    types = [1] * 256
    rules = ["t h", "th i", "thi n", "thin k", "p l", "pl a", "pla i", "plai n", "< /"] if merges else []
    for rule in rules:
        left, right = rule.split(" ")
        tokens.append(left + right)
        types.append(1)
    if specials:
        tokens += ["<think>", "</think>", "<|im_start|>", "<|im_end|>"]
        types += [4, 4, 3, 3]
    return cls(tokens, rules, types)


class PlainSpans(unittest.TestCase):
    def test_boundaries_overlaps_order_and_no_input_mutation(self):
        text = "🙂 a<think>literal</think> b<|im_start|>x<|im_end|><think>z"
        first = text.index("<think>")
        spans_cases = [[], [(first, first + 1)], [(first + 1, first + 7)],
                       [(0, first)], [(0, first + 1)], [(0, len(text))],
                       [(first + 2, first + 9), (first, first + 2)],
                       [(first, first + 20), (first + 1, first + 2)],
                       [(first, first), (first + 20, first)],
                       [(-10, 0), (len(text) + 1, len(text) + 20)]]
        actual, reference = tokenizer(), tokenizer(ReferenceTokenizer)
        for spans in spans_cases:
            saved = list(spans)
            for seq in (spans, tuple(spans)):
                for parse_special in (False, True):
                    self.assertEqual(actual.encode(text, parse_special, plain=seq),
                                     reference.encode(text, parse_special, plain=seq))
            self.assertEqual(spans, saved)

    def test_whole_protected_text_is_literal_bytes(self):
        t = tokenizer()
        text = "<think>日本語🙂</think><|im_start|>"
        expected = [t.ids[ST.BYTE_TO_UNICODE[b]] for b in text.encode("utf-8")]
        for parse_special in (False, True):
            self.assertEqual(t.encode(text, parse_special, plain=[(0, len(text))]), expected)
        self.assertIn(t.special_tokens["<think>"], t.encode(text))

    def test_randomized_spans_match_reference(self):
        rng = random.Random(1031)
        actual, reference = tokenizer(merges=True), tokenizer(ReferenceTokenizer, merges=True)
        pieces = ["<think>", "</think>", "<|im_start|>", "<|im_end|>", "plain", " café🙂 ", "\n"]
        for _ in range(200):
            text = "".join(rng.choice(pieces) for _ in range(30))
            spans = [(rng.randrange(-5, len(text) + 5), rng.randrange(-5, len(text) + 5))
                     for _ in range(rng.randrange(50))]
            for parse_special in (False, True):
                self.assertEqual(actual.encode(text, parse_special, plain=spans),
                                 reference.encode(text, parse_special, plain=spans))

    def test_one_shot_iterables_preserve_consumption_behavior(self):
        text = "<think>a</think><|im_start|>b<|im_end|>"
        spans = [(10, 30), (0, 7), (8, 12)]
        actual, reference = tokenizer(), tokenizer(ReferenceTokenizer)
        for parse_special in (False, True):
            self.assertEqual(actual.encode(text, parse_special, plain=iter(spans)),
                             reference.encode(text, parse_special, plain=iter(spans)))

    def test_no_special_pattern_and_empty_text(self):
        t = tokenizer(specials=False)
        text = "café <think>"
        self.assertEqual(t.encode(text, True, plain=[(0, len(text))]), t.encode(text))
        self.assertEqual(tokenizer().encode("", True, plain=[(-1, 100)]), [])


if __name__ == "__main__":
    unittest.main()
