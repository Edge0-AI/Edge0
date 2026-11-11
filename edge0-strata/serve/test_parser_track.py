"""Tracking state must be identical at every streamed fragment, not just at the end."""
import itertools
import random
import unittest

from serve.frontend import OutputParser


class ReferenceParser(OutputParser):
    # Keep the original string storage as well as the original tracking algorithm.
    line = ""

    def _track(self, text):
        parts = text.split("\n")
        for k, part in enumerate(parts):
            if k < len(parts) - 1:
                line, self.line = self.line + part, ""
                s = line.lstrip()
                was_fenced = bool(self.fence)
                if self.fence:
                    if len(s) - len(s.lstrip(self.fence[0])) >= len(self.fence):   # #4702: as many fence characters as opened it
                        self.fence = ""
                elif s[:3] in ("```", "~~~") and (s[0] * 3) not in s[3:]:
                    self.fence = s[0] * (len(s) - len(s.lstrip(s[0])))
                elif not s:
                    self.ticks = 0
                else:
                    self.ticks += line.count("`")
                if was_fenced or self.fence:                                       # #4703: indented code blocks
                    self._prev_blank = self._indented = False
                elif not s:
                    self._prev_blank = True
                elif line.startswith(("    ", "\t")):
                    self._indented = self._prev_blank or self._indented
                    self._prev_blank = False
                else:
                    self._prev_blank = self._indented = False
            else:
                self.line += part
        return text


class TrackTests(unittest.TestCase):
    def assert_state(self, actual, expected):
        self.assertEqual((actual.line, actual.fence, actual.ticks, actual._prev_blank, actual._indented),
                         (expected.line, expected.fence, expected.ticks, expected._prev_blank, expected._indented))
        self.assertEqual(actual._opener_ok(), expected._opener_ok())
        self.assertEqual(actual._in_code(), expected._in_code())

    def test_fragment_state_from_varied_initial_states(self):
        fragments = ("", "a", "é🙂", "`", "``", "```", "~~~", "\r", "\n", "\r\n",
                     "\n\n", "\n    code\n", "    x\n", "  ```python\n", "````md\n", "```quoted```\n", "~\ntext", "\t\nrest")
        for fence, line, ticks, text in itertools.product(("", "`", "~"),
                                                         ("", "  ", "``", "code `"),
                                                         (0, 1, 2, 3), fragments):
            actual, expected = OutputParser(), ReferenceParser()
            for parser in (actual, expected):
                parser.fence, parser.line, parser.ticks = fence, line, ticks
            self.assertEqual(actual._track(text), expected._track(text))
            self.assert_state(actual, expected)

    def test_random_streaming_boundaries(self):
        text = ("Reasoning `inline` café 🙂\n\n  ```python\n<tool_call>quoted\n```\n"
                "~~~xml\r\n<function=test>\n~~~\n\nAnswer\n" * 8)
        rng = random.Random(1029)
        for _ in range(50):
            actual, expected = OutputParser(), ReferenceParser()
            pos = 0
            while pos < len(text):
                end = pos + rng.randint(1, 17)
                fragment = text[pos:end]
                self.assertEqual(actual._track(fragment), expected._track(fragment))
                self.assert_state(actual, expected)
                pos = end

    def test_long_line_snapshots_restore_and_append(self):
        actual, expected = OutputParser(), ReferenceParser()
        for parser in (actual, expected):
            # Include Unicode and backticks: string materialization must preserve
            # characters and the code decision, even after a restored snapshot.
            for _ in range(4096):
                parser._track("abc café 🙂 `")
        self.assert_state(actual, expected)
        for parser in (actual, expected):
            saved = (parser.fence, parser.line, parser.ticks)
            parser.buf = "\n```python\nignored"
            parser._ok_at(len(parser.buf))
            self.assertEqual((parser.fence, parser.line, parser.ticks), saved)
            parser._track("more text")
            self.assertEqual(saved[1] + "more text", parser.line)
            parser._track("\n~~~\ninside\n~~~\n")
        self.assert_state(actual, expected)

    def test_feed_events_with_long_lines_and_quoted_tool_tags(self):
        text = ("prefix " * 1024 + "`<tool_call>`\n\n```xml\n" + "x" * 16384
                + "<tool_call><function=quoted></function></tool_call>\n```\n"
                + "<tool_call><function=write><parameter=text>ok</parameter></function></tool_call>\nDone")
        for recover, stream in itertools.product((False, True), repeat=2):
            kwargs = dict(thinking=False, tools=[{"name": "write"}], recover=recover, stream_tools=stream)
            actual, expected = OutputParser(**kwargs), ReferenceParser(**kwargs)
            def normalized(events):
                return [(e.kind, e.text, (e.call.name, e.call.arguments) if e.call else None) for e in events]
            for pos in range(0, len(text), 17):
                delta = text[pos:pos + 17]
                self.assertEqual(normalized(actual.feed(delta)), normalized(expected.feed(delta)))
            self.assertEqual(normalized(actual.finish()), normalized(expected.finish()))

    def test_compaction_boundaries_reads_and_line_replacement(self):
        for count in (63, 64, 65, 127, 128, 129, 1025):
            actual, expected = OutputParser(), ReferenceParser()
            for i in range(count):
                text = f"{i}: café 🙂 `"
                actual._track(text)
                expected._track(text)
            saved = actual.line
            self.assert_state(actual, expected)
            # Reading a joined line must not lose it when another block fills.
            for i in range(129):
                text = f"{i}: more"
                actual._track(text)
                expected._track(text)
            self.assert_state(actual, expected)
            # Replacing the line discards every old block, including a partial tail.
            for parser in (actual, expected):
                parser.line = saved
                parser._track("\n```python\n")
                for _ in range(count):
                    parser._track("code")
                parser._track("\n```\n\n")
                parser._track("new line")
            self.assert_state(actual, expected)
            self.assertEqual(actual.line, "new line")


if __name__ == "__main__":
    unittest.main()
