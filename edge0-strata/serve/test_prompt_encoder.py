"""serve/test_prompt_encoder.py - #567's incremental prompt encoding with #537's plain spans: every prompt's ids
are those of a full encode, and a turn reuses the previous one's.

    python -m unittest serve.test_prompt_encoder -v
"""
from __future__ import annotations

import random
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402
from strata_tokenizer import PromptEncoder, plain_agree_until  # noqa: E402


class ThinkTokenizer(ByteTokenizer):
    """ByteTokenizer whose <think> / </think> are specials matched even without parse_special (type 4), as in the
    Qwen vocabulary: the case #537's plain spans exist for."""
    SPECIALS = ByteTokenizer.SPECIALS + ["<think>", "</think>"]
    ALWAYS = ("<think>", "</think>")


PIECES = ["<|im_start|>user\n", "<|im_end|>\n", "<think>", "</think>", "hello ", "é你 ", "<|im_sta", "x" * 7,
          "<|im_start|>assistant\n", "</thi", "\n\n", "<|endoftext|>", "<|vision_start|>", "<|image_pad|>"]


def spans_of(text, rng):
    """Random plain spans over the specials' texts (<think>, </think>, <|im_start|>, ...), as unmark_think_literals
    marks the ones quoted inside a message."""
    matches = re.finditer("|".join(map(re.escape, ThinkTokenizer.SPECIALS)), text)
    return tuple((m.start(), m.end()) for m in matches if rng.random() < 0.5)   # finditer: left to right, no overlap


class Units(unittest.TestCase):
    def test_plain_agree_until(self):
        self.assertEqual(plain_agree_until((), (), 50), 50)
        self.assertEqual(plain_agree_until(((5, 13),), ((5, 13),), 50), 50)
        self.assertEqual(plain_agree_until(((5, 13),), (), 50), 5)
        self.assertEqual(plain_agree_until(((5, 13),), ((5, 9),), 50), 9)
        self.assertEqual(plain_agree_until(((60, 70),), (), 50), 50)          # beyond the shared text: no matter
        self.assertEqual(plain_agree_until(((5, 13), (20, 28)), ((5, 13),), 50), 20)

    def test_same_ids_as_a_full_encode(self):
        tok = ThinkTokenizer()
        rng = random.Random(7)
        for conv in range(300):
            enc = PromptEncoder(tok)
            text, plain = "", ()
            for turn in range(8):
                text += "".join(rng.choice(PIECES) for _ in range(rng.randint(1, 12)))
                if rng.random() < 0.6:
                    plain = spans_of(text, rng)                     # the spans may change in the shared part too
                with self.subTest(conv=conv, turn=turn):
                    self.assertEqual(enc.encode(text, plain), tok.encode(text, True, plain))
                if rng.random() < 0.2 and len(text) > 10:              # an edit: a client re-rendering its history
                    cut = rng.randrange(len(text))
                    text = text[:cut]
                    plain = tuple((a, b) for a, b in plain if b <= cut)

    def test_a_turn_reuses_the_previous_one(self):
        tok = ThinkTokenizer()
        enc = PromptEncoder(tok)
        # as a chat turn: the earlier prompt goes on past its last special (the answer being written), and the
        # next one shares it up to there - the resume point needs max_special_len characters after it
        base = "<|im_start|>user\n" + "quoted </think> here " * 200 + "<|im_end|>\n<|im_start|>assistant\nan answer"
        plain = tuple((i, i + 8) for i in range(len(base)) if base.startswith("</think>", i))
        enc.encode(base, plain)
        more = base + " that goes on<|im_end|>\n<|im_start|>user\nnext<|im_end|>\n"
        self.assertEqual(enc.encode(more, plain), tok.encode(more, True, plain))
        self.assertGreater(enc.last_reused, len(base) // 2)
        # the same text with the quoted </think> no longer plain: nothing before it may be reused past its start
        self.assertEqual(enc.encode(more, ()), tok.encode(more, True, ()))
        self.assertLessEqual(enc.last_reused, plain[0][0])


class Served(unittest.TestCase):
    def test_quoted_think_across_turns(self):
        tok = ThinkTokenizer()
        template = ChatTemplate(ROOT / "serve/chat_template.jinja")
        svc = Service(MockEngine(tok, "</think>\n\nok", max_context=1 << 20), tok, template)
        full = Service(MockEngine(tok, "</think>\n\nok", max_context=1 << 20), tok, template)
        full.prompts = None                                        # the reference: a full encode every time
        self.assertIsNotNone(svc.prompts)
        msgs = [{"role": "system", "content": "Explain tags."}]
        for turn in range(5):
            msgs.append({"role": "user", "content": f"turn {turn}: what does </think> mean? <think> too " * 3})
            got, want = svc.encode_prompt(msgs, None, {}), full.encode_prompt(msgs, None, {})
            self.assertEqual(got, want)
            if turn:
                self.assertGreater(svc.prompts.last_reused, 0)
            msgs.append({"role": "assistant", "content": "It closes the reasoning: </think> is a marker."})

    def service(self, max_context=1 << 22):
        tok = ByteTokenizer()
        return Service(MockEngine(tok, "ok", max_context=max_context), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))

    def test_another_conversation_does_not_evict_the_first(self):
        """A subagent beside the main conversation: every rendered prompt starts with the template's header, which
        must not make the second client's prompt a continuation of the first's entry."""
        svc = self.service()
        turn2 = [{"role": "user", "content": "a long question " * 500}, {"role": "assistant", "content": "answer"},
                 {"role": "user", "content": "next"}]
        svc.encode_prompt(turn2, None, {})
        svc.encode_prompt([{"role": "user", "content": "one unrelated prompt " * 300}], None, {})
        self.assertEqual(len(svc.prompts.entries), 2)
        turn3 = turn2 + [{"role": "assistant", "content": "x"}, {"role": "user", "content": "more"}]
        svc.encode_prompt(turn3, None, {})
        self.assertGreater(svc.prompts.last_reused, len(turn2[0]["content"]))
        self.assertEqual(len(svc.prompts.entries), 2)           # turn 3 replaced turn 2, the other one stayed

    def test_a_refused_prompt_is_not_kept(self):
        svc = self.service(max_context=4096)
        svc.fit_max_tokens = False   # the strict mode refuses an over-long max_tokens (the default now shortens it)
        msgs = [{"role": "user", "content": "kept " * 20}]
        svc.prepare(msgs, None, {}, 16)
        refused = msgs + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "more " * 20}]
        full = self.service(max_context=4096)
        full.prompts = None
        refused_ids = full.encode_prompt(refused, None, {})
        with self.assertRaisesRegex(ValueError, "exceeds the context"):
            svc.prepare(refused, None, {}, 1 << 20)
        self.assertNotIn(refused_ids, [e[2] for e in svc.prompts.entries])
        self.assertNotIn(full.render_prompt(refused, None, {}), [e[0] for e in svc.prompts.entries])

    def test_random_messages_equal_a_full_encode(self):
        """Messages quoting control-token texts and <think> tags at random, growing, edited and cut back between
        requests: the incremental encoder's ids are the full encode's, 2400 prompts."""
        tok = ThinkTokenizer()
        template = ChatTemplate(ROOT / "serve/chat_template.jinja")
        svc = Service(MockEngine(tok, "ok", max_context=1 << 20), tok, template)
        full = Service(MockEngine(tok, "ok", max_context=1 << 20), tok, template)
        full.prompts = None
        rng = random.Random(567)
        quoted = ThinkTokenizer.SPECIALS + ["<|im_sta", "</thi"]
        words = ["hello ", "é你 ", "x" * 9, "\n", "tool "]

        def text():
            return "".join(rng.choice(quoted if rng.random() < 0.4 else words) for _ in range(rng.randint(1, 10)))

        reused = 0
        for conv in range(300):
            msgs = [{"role": "system", "content": text()}]
            for turn in range(8):
                msgs.append({"role": "user", "content": text()})
                if rng.random() < 0.15:                             # an earlier turn edited by the client
                    msgs[rng.randrange(len(msgs))]["content"] = text()
                if rng.random() < 0.1 and len(msgs) > 2:            # a shorter conversation
                    msgs = msgs[:rng.randrange(1, len(msgs))] + msgs[-1:]
                with self.subTest(conv=conv, turn=turn):
                    self.assertEqual(svc.encode_prompt(msgs, None, {}), full.encode_prompt(msgs, None, {}))
                reused += svc.prompts.last_reused > 0
                msgs.append({"role": "assistant", "content": text()})
        self.assertGreater(reused, 1000)


if __name__ == "__main__":
    unittest.main()
