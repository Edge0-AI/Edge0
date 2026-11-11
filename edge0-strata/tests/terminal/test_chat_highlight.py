"""highlight.py: fenced code blocks in chat.py's streamed output get syntax highlighting (tags, strings, comments,
keywords, numbers), prose passes through unchanged, and a fence split across streamed chunks still opens.

    python -m pytest tests/terminal/test_chat_highlight.py      (no GPU, no downloads)
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from highlight import StreamHighlighter  # noqa: E402

RESET = "\033[0m"


def colored(text, code):
    return f"\033[{code}m{text}{RESET}"


class TestHighlight(unittest.TestCase):
    def test_prose_unchanged_with_color(self):
        hl = StreamHighlighter(color=True)
        self.assertEqual(hl.feed("hello world\n"), "hello world\n")
        self.assertEqual(hl.flush(), "")

    def test_prose_unchanged_without_color(self):
        hl = StreamHighlighter(color=False)
        self.assertEqual(hl.feed("def f(x):\n    return x\n"), "def f(x):\n    return x\n")

    def test_python_block_is_highlighted(self):
        hl = StreamHighlighter(color=True)
        out = hl.feed("```python\nfor i in range(3):  # loop\n    print(i)\n```\n")
        self.assertIn(colored("for", "35;1"), out)
        self.assertIn(colored("# loop", "90"), out)
        self.assertIn(colored("in", "35;1"), out)
        lines = out.split("\n")
        self.assertTrue(lines[-1] == "" and lines[-2] == colored("```", "90"))
        # prose after the fence is unchanged
        self.assertEqual(hl.feed("plain again\n"), "plain again\n")

    def test_html_tags_highlighted(self):
        hl = StreamHighlighter(color=True)
        out = hl.feed("```html\n<div class=\"box\">hi</div>\n```\n")
        self.assertIn(colored("<div", "96"), out)
        self.assertIn(colored("</div>", "96"), out)
        self.assertIn(colored("class", "94"), out)

    def test_fence_split_across_chunks(self):
        hl = StreamHighlighter(color=True)
        out = hl.feed("code:\n``") + hl.feed("`py\nimport os\n```\n")
        self.assertIn(colored("import", "35;1"), out)
        self.assertIn(colored("```", "90"), out)
        self.assertIn(colored("py", "94"), out)

    def test_partial_line_held_back_then_flushed(self):
        hl = StreamHighlighter(color=False)
        self.assertEqual(hl.feed("abc"), "")
        self.assertEqual(hl.feed("def\nxyz"), "abcdef\n")
        self.assertEqual(hl.flush(), "xyz")

    def test_unknown_language_passes_through(self):
        hl = StreamHighlighter(color=True)
        out = hl.feed("```brainfuck\n+++++\n```\n")
        self.assertIn("+++++", out)
        self.assertIn(colored("```brainfuck", "90"), out)

    def test_streaming_bytes_preserved(self):
        text = "```python\ndef f(x):\n    return x**2\n```\ndone\n"
        hl = StreamHighlighter(color=True)
        out = hl.feed(text)
        self.assertEqual(out.replace(RESET, "").replace("\033[35;1m", "").replace("\033[90m", "")
                         .replace("\033[33m", "").replace("\033[34m", "").replace("\033[94m", ""), text)
        self.assertEqual(hl.flush(), "")


if __name__ == "__main__":
    unittest.main()
