"""Partial-tag parity, including overlapping prefixes and arbitrary chunk boundaries."""
import itertools
import random
import unittest

from serve.frontend import OutputParser


def reference_hold(text, tags):
    return max((n for tag in tags for n in range(1, len(tag)) if text.endswith(tag[:n])), default=0)


class ReferenceParser(OutputParser):
    def _hold(self, text, tags):
        return reference_hold(text, tags)


class HoldTests(unittest.TestCase):
    def test_overlapping_and_empty_tags(self):
        parser = OutputParser()
        tags = ("", "a", "aaa", "abab", "baab", "<tool_call>", "</think>")
        for size in range(8):
            for chars in itertools.product("ab<", repeat=size):
                text = "".join(chars)
                self.assertEqual(parser._hold(text, tags), reference_hold(text, tags), text)

    def test_every_real_tag_split(self):
        tags = ("<tool_call>", "</think>", "</parameter>")
        parser = OutputParser()
        for tag in tags:
            for n in range(len(tag) + 1):
                for prefix in ("", "prose\n", "<", "<<<", "🙂日本語"):
                    text = prefix + tag[:n]
                    self.assertEqual(parser._hold(text, tags), reference_hold(text, tags), text)

    def test_streamed_events_match_reference(self):
        texts = ["Reasoning.\n</think>\nAnswer café 🙂\n",
                 "Quoted `</think>` text and <tool_call> prose.",
                 "\n```xml\n<tool_call>\n```\nAnswer\n<tool_",
                 "<tool_call><function=lookup><parameter=q>abc</parameter></function></tool_call>\nDone",
                 "<tool_call><function=lookup><parameter=q>abc</parameter></function></tool_call></think>Done"]
        rng = random.Random(1028)
        for text, thinking, recover, stream_tools in itertools.product(texts, (False, True), (False, True), (False, True)):
            for _ in range(5):
                args = dict(thinking=thinking, tools=[{"name": "lookup"}], recover=recover, stream_tools=stream_tools)
                actual, expected = OutputParser(**args), ReferenceParser(**args)
                def normalized(events):
                    # Tool call IDs are deliberately random; compare their semantic contents.
                    return [(e.kind, e.text, (e.call.name, e.call.arguments) if e.call else None) for e in events]
                pos = 0
                while pos < len(text):
                    end = pos + rng.randint(1, 9)
                    delta = text[pos:end]
                    self.assertEqual(normalized(actual.feed(delta)), normalized(expected.feed(delta)))
                    pos = end
                self.assertEqual(normalized(actual.finish()), normalized(expected.finish()))


if __name__ == "__main__":
    unittest.main()
