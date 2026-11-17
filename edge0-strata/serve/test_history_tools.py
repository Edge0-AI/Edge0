"""#4663: a call to a tool that this turn's `tools` no longer lists, but that the conversation already called, is a tool call (not raw text).

    python -m unittest serve.test_history_tools
"""
import unittest

import serve.server as server
from serve.frontend import OutputParser

TODO = {"name": "TodoWrite", "parameters": {"type": "object", "properties": {"todos": {"type": "array"}}}}
CALL = "<tool_call>\n<function=Bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>"


def history(*names):
    return {"messages": [{"role": "user", "content": "go"},
                         *[{"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function",
                           "function": {"name": n, "arguments": "{}"}}]} for i, n in enumerate(names)]]}


def run(tools, text=CALL):
    p = OutputParser(thinking=False, tools=tools, stream_tools=True, recover=True, declared_only=True)
    evs = p.feed(text) + p.finish("stop")
    return [e.call.name for e in evs if e.kind == "tool_call"], "".join(e.text for e in evs if e.kind == "content")


class HistoryTools(unittest.TestCase):
    def test_a_tool_the_conversation_called_is_a_tool_call_again(self):
        names, answer = run(server.with_history_tools([TODO], history("Bash")))
        self.assertEqual(names, ["Bash"])
        self.assertEqual(answer, "")

    def test_without_the_history_it_stays_text(self):
        names, answer = run(server.with_history_tools([TODO], history("Read")))
        self.assertEqual(names, [])
        self.assertIn("<tool_call>", answer)

    def test_declared_names_are_not_doubled_and_no_tools_stays_no_tools(self):
        tools = [TODO, {"name": "Bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}]
        self.assertIs(server.with_history_tools(tools, history("Bash", "TodoWrite")), tools)
        self.assertEqual(server.with_history_tools([], history("Bash")), [])
        self.assertIsNone(server.with_history_tools(None, history("Bash")))

    def test_odd_requests_are_left_alone(self):
        self.assertEqual(server.with_history_tools([TODO], None), [TODO])
        self.assertEqual(server.with_history_tools([TODO], {"messages": [None, {"role": "assistant", "tool_calls": [None, {}]}]}), [TODO])


if __name__ == "__main__":
    unittest.main()
