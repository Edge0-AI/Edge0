"""A missing wrapper closer ends before a following call, but prose alone does not confirm a call.

A call's own </function> separates it from the next wrapper without merging their arguments.
Parameter values may still contain call-format tags; recovery batches stay opt-in.

    python -m unittest serve.test_tool_call_unclosed -v
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser  # noqa: E402

TOOLS = [{"name": "read", "parameters": {"properties": {"path": {"type": "string"}}}},
         {"name": "write", "parameters": {"properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}]
READ = "<function=read>\n<parameter=path>\na.py\n</parameter>\n</function>"
WRITE = "<function=write>\n<parameter=path>\nb.py\n</parameter>\n<parameter=content>\nx = 1\n</parameter>\n</function>"


def run(text, stream_tools, recover, step):
    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=stream_tools, recover=recover)
    evs = []
    for i in range(0, len(text), step):                # fed in small pieces, as a stream arrives
        evs += p.feed(text[i:i + step])
    evs += p.finish("stop")
    calls = [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]
    if stream_tools and not calls:
        arguments = "".join(e.text for e in evs if e.kind == "tool_args")
        if arguments:
            try:
                json.loads(arguments)
            except ValueError:
                pass
            else:
                raise AssertionError("an unconfirmed call has complete streamed arguments")
    return calls, "".join(e.text for e in evs if e.kind == "content")


class UnclosedCall(unittest.TestCase):
    def check(self, text, want_calls, want_content=""):
        for stream_tools in (False, True):
            for recover in (False, True):
                for step in (1, 3, 7, 4096):
                    with self.subTest(stream_tools=stream_tools, recover=recover, step=step):
                        calls, content = run(text, stream_tools, recover, step)
                        self.assertEqual(calls, want_calls)
                        self.assertEqual(content.strip(), want_content)

    def test_closed_calls_are_unchanged(self):
        self.check("<tool_call>\n" + READ + "\n</tool_call>\n<tool_call>\n" + WRITE + "\n</tool_call>",
                   [("read", {"path": "a.py"}), ("write", {"path": "b.py", "content": "x = 1"})])

    def test_a_second_call_after_an_unclosed_one_is_its_own_call(self):
        self.check("<tool_call>\n" + READ + "\n<tool_call>\n" + WRITE + "\n</tool_call>",
                   [("read", {"path": "a.py"}), ("write", {"path": "b.py", "content": "x = 1"})])

    def test_quoted_incomplete_wrapper_stays_literal(self):
        text = "Format: <tool_call>\n" + READ + "\nand then I explain it."
        self.check(text, [], text)

    def test_following_literal_marker_does_not_complete_an_unclosed_call(self):
        text = "<tool_call>\n" + READ + "\n<tool_call> is a format marker."
        self.check(text, [], text)

    def test_omitted_wrapper_recovers_only_on_a_natural_stop(self):
        for closed in (True, False):
            text = "<tool_call>\n" + READ + ("\n</tool_call>" if closed else "")
            for reason in (None, "stop", "length", "cancel", "disconnect"):
                for step in (1, 7, 4096):
                    with self.subTest(closed=closed, reason=reason, step=step):
                        p = OutputParser(thinking=False, tools=TOOLS, stream_tools=True)
                        events = []
                        for offset in range(0, len(text), step):
                            events += p.feed(text[offset:offset + step])
                        events += p.finish(reason)
                        calls = [e for e in events if e.kind == "tool_call"]
                        complete = closed or reason in (None, "stop")
                        self.assertEqual(len(calls), int(complete))
                        if not complete:
                            with self.assertRaises(ValueError):
                                json.loads("".join(e.text for e in events if e.kind == "tool_args"))

    def test_a_value_that_holds_the_tags_still_ends_at_the_calls_own_closer(self):
        value = "a </function> and a <tool_call> in a string"
        call = ("<tool_call>\n<function=write>\n<parameter=path>\nc.py\n</parameter>\n<parameter=content>\n" + value +
                "\n</parameter>\n</function>\n</tool_call>")
        self.check(call, [("write", {"path": "c.py", "content": value})])


if __name__ == "__main__":
    unittest.main()
