"""#4702 and #4703: a call quoted in a Markdown code form is text, not a call.  A four-backtick fence is closed only by a fence of
at least four backticks (an inner ``` block stays inside it), and an indented (4-space) code block after a blank line is a quote.
A call the model writes plainly, or indents inside a paragraph, is still a call.

    python -m unittest serve.test_parser_markdown_code
"""
import unittest

from serve.frontend import OutputParser


def tool(name, **props):
    return {"name": name, "parameters": {"type": "object", "properties": props}}


S = {"type": "string"}
TOOLS = [tool("read", path=S), tool("write_file", path=S, content=S)]


def qwen(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def parse(text, piece=0, recover=False):
    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=False, recover=recover, declared_only=True)
    if piece:
        evs = []
        for i in range(0, len(text), piece):
            evs += p.feed(text[i:i + piece])
        evs += p.finish("stop")
    else:
        evs = p.feed(text) + p.finish("stop")
    return ([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
            "".join(e.text for e in evs if e.kind == "content"))


class MarkdownCode(unittest.TestCase):
    def each(self, text):
        for recover in (False, True):
            for piece in (0, 1, 7):
                yield parse(text, piece, recover)

    def test_an_inner_fence_does_not_close_a_longer_one(self):
        text = ("Save this as docs/TOOLS.md:\n\n````markdown\n# Tool calls\n\nA call:\n\n```xml\n"
                + qwen("read", path="EXAMPLE.txt") + "\n```\n````\n\nThat is the file.")
        for calls, answer in self.each(text):
            self.assertEqual(calls, [])
            self.assertEqual(answer, text)

    def test_a_call_after_a_closed_longer_fence_is_a_call(self):
        text = "Here:\n\n````md\nexample\n````\n\n" + qwen("read", path="a.txt")
        for calls, _ in self.each(text):
            self.assertEqual(calls, [("read", {"path": "a.txt"})])

    def test_a_tilde_fence_is_not_closed_by_backticks(self):
        text = "Quote:\n\n~~~\n```\n" + qwen("read", path="EXAMPLE.txt") + "\n```\n~~~\n"
        for calls, _ in self.each(text):
            self.assertEqual(calls, [])

    def test_an_indented_code_block_is_a_quote(self):
        block = "\n".join("    " + line for line in qwen("read", path="EXAMPLE.txt").split("\n"))
        text = "The format, indented as code:\n\n" + block + "\n\nIndentation makes it a code block in Markdown."
        for calls, answer in self.each(text):
            self.assertEqual(calls, [])
            self.assertEqual(answer, text)

    def test_a_plain_call_after_a_paragraph_is_still_a_call(self):
        text = "Let me read it.\n" + qwen("read", path="a.txt")
        for calls, _ in self.each(text):
            self.assertEqual(calls, [("read", {"path": "a.txt"})])

    def test_a_call_after_an_indented_block_ends_it(self):
        text = "Example:\n\n    x = 1\n\nNow the real one:\n\n" + qwen("read", path="a.txt")
        for calls, _ in self.each(text):
            self.assertEqual(calls, [("read", {"path": "a.txt"})])


if __name__ == "__main__":
    unittest.main()
