"""serve/test_tool_name_space.py - a tool call written `<function= NAME>` (a space after the `=`) is the call NAME.

Seen live (Qwen3.8-Flash-Next IQ2_XS, Zed's agent panel): the model wrote `<function= capture>` inside its thinking;
the name was read as " capture", matched no declared tool, stayed reasoning text (#804), and the turn ended with no
call, so the agent stopped.  After the thinking the same call came out named " capture", and its arguments were
typed without the tool's schema.

    python -m unittest serve.test_tool_name_space -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser  # noqa: E402

# "42" for a string parameter stays a string only when the call finds its tool's schema
SCHEMA = [{"name": "capture", "parameters": {"properties": {"content": {"type": "string"}}}}]
CALL = "<tool_call>\n<function= capture>\n<parameter=content>\n42\n</parameter>\n</function>\n</tool_call>"


def calls(text, stream_tools):
    p = OutputParser(thinking=True, tools=SCHEMA, stream_tools=stream_tools)
    evs = []
    for i in range(0, len(text), 3):                   # fed in small pieces, as a stream arrives
        evs += p.feed(text[i:i + 3])
    evs += p.finish("stop")
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


class SpacedName(unittest.TestCase):
    def check(self, text):
        for stream_tools in (False, True):
            with self.subTest(stream_tools=stream_tools):
                self.assertEqual(calls(text, stream_tools), [("capture", {"content": "42"})])

    def test_inside_the_thinking(self):
        self.check("Let me save this.\n" + CALL + "\n")

    def test_after_the_thinking(self):
        self.check("Saving it.</think>\n\n" + CALL)


if __name__ == "__main__":
    unittest.main()
