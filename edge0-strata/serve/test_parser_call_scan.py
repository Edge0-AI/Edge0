"""Wrapped-call precheck parity against the original structural walk."""
import itertools
import random
import unittest
from unittest.mock import patch

from serve import frontend


class ReferenceParser(frontend.OutputParser):
    def _wrapped_call_end(self):
        return frontend.call_end(self.buf)


def events_without_ids(events):
    return [(e.kind, e.text, (e.call.name, e.call.arguments) if e.call else None) for e in events]


class CallScanTests(unittest.TestCase):
    def test_chunk_boundaries_literal_tags_multiple_calls_and_recovery(self):
        def call(value):
            return "<tool_call><function=write><parameter=text>" + value + "</parameter></function></tool_call>"
        bodies = [call("ordinary"), call("literal </tool_call> text </function> here"),
                  call("literal </parameter> prose"), call("a") + call("b"),
                  call("unfinished")[:-5], call("incomplete </tool_call> prose")[:-30],
                  "<tool_call><function=write></function></tool_call>",
                  "<function=write><parameter=text>bare</parameter></function>",
                  "<tool_call><function=write><parameter=text>a</parameter></function>"
                  "<parameter=write><parameter=text>b</parameter></function></tool_call>"]
        tools = [{"name": "write", "parameters": {"properties": {"text": {"type": "string"}}}}]
        rng = random.Random(1030)
        for text, stream, recover in itertools.product(bodies, (False, True), (False, True)):
            # Every fixed chunk width, plus random partitions; compare after EACH feed.
            for width in range(1, 16):
                a = frontend.OutputParser(thinking=False, tools=tools, stream_tools=stream, recover=recover)
                b = ReferenceParser(thinking=False, tools=tools, stream_tools=stream, recover=recover)
                pos = 0
                while pos < len(text):
                    end = pos + (rng.randint(1, 20) if width == 15 else width)
                    delta = text[pos:end]
                    self.assertEqual(events_without_ids(a.feed(delta)), events_without_ids(b.feed(delta)))
                    pos = end
                self.assertEqual(events_without_ids(a.finish()), events_without_ids(b.finish()))

    def test_no_structural_rescans_until_closer_and_reset_for_next_call(self):
        parser = frontend.OutputParser(thinking=False)
        with patch.object(frontend, "call_end", wraps=frontend.call_end) as walk:
            for _ in range(2):
                parser.feed("<tool_call><function=write><parameter=text>")
                before = walk.call_count
                for _ in range(100):
                    parser.feed("abcdefgh")
                self.assertEqual(walk.call_count, before)
                for char in "</parameter></function></tool_call>":
                    parser.feed(char)
                self.assertEqual(walk.call_count, before + 1)
                self.assertEqual(parser._call_end_checked, 0)


if __name__ == "__main__":
    unittest.main()
