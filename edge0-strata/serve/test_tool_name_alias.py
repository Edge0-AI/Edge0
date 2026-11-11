"""Codex MCP names that are not the declared namespace.member form.

Observed with Qwen3.8-Flash-Next through Codex CLI against Strata v0.2.2. The model
wrote <function=mcp__filesystem=write_file>, <function=mcp__filesystem_edit_file>,
and later a bare <function=write_file>. The declared tool was
mcp__filesystem.write_file. Stock main already accepts namespace__member. The
equals sign, the dropped underscore, and a unique bare member have to as well.
A name that matches nothing stays text (#3075). Two tools that share a member
are not guessed.

    python -m unittest serve.test_tool_name_alias -v
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser  # noqa: E402
from serve.responses import Assembler, request_tools  # noqa: E402

REQ = {
    "tools": [{
        "type": "namespace",
        "name": "mcp__filesystem",
        "tools": [
            {"type": "function", "name": "write_file", "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"]}},
            {"type": "function", "name": "edit_file", "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "edits": {"type": "string"}},
                "required": ["path", "edits"]}},
            {"type": "function", "name": "list_allowed_directories", "parameters": {
                "type": "object", "properties": {}}},
        ],
    }],
}


def events(text, tools, thinking=False):
    parser = OutputParser(thinking=thinking, tools=tools, stream_tools=True)
    evs = parser.feed(text)
    evs += parser.finish("stop")
    return evs


def text_of(evs):
    return "".join(e.text or "" for e in evs if e.kind in ("content", "reasoning"))


def response_items(text, tools, names, req=REQ):
    asm = Assembler(req, "qwen", 10, names, encrypted=False)
    out = []
    parser_events = events(text, tools)
    for ev in parser_events:
        out += asm.feed(ev)
    out += asm.finish({"finish": "stop", "completion_tokens": 1, "prompt_tokens": 10})
    items = [ev["item"] for ev in out if ev.get("type") == "response.output_item.done"]
    return items, parser_events


class DriftedMcpNames(unittest.TestCase):
    def setUp(self):
        self.tools, self.names, _skipped = request_tools(REQ)

    def _call(self, text, tools=None, names=None):
        items, evs = response_items(text, tools or self.tools, names or self.names)
        calls = [i for i in items if i["type"] == "function_call"]
        return calls, evs

    def test_equals_write_is_the_namespace_tool(self):
        raw = (
            "<tool_call>\n<function=mcp__filesystem=write_file>\n<parameter=path>\n/tmp/x.py\n</parameter>\n"
            "<parameter=content>\nprint(\"not used\")\n</parameter>\n</function>\n</tool_call>"
        )
        calls, evs = self._call(raw)
        self.assertEqual(text_of(evs).strip(), "")
        self.assertEqual(len(calls), 1)
        self.assertEqual((calls[0]["namespace"], calls[0]["name"]), ("mcp__filesystem", "write_file"))
        args = json.loads(calls[0]["arguments"])
        self.assertEqual(args["path"], "/tmp/x.py")
        self.assertIn('print("not used")', args["content"])

    def test_single_underscore_edit_keeps_a_string_argument(self):
        raw = (
            "<tool_call>\n<function=mcp__filesystem_edit_file>\n<parameter=path>\n/tmp/a.txt\n</parameter>\n"
            "<parameter=edits>\nsay \"hi\"\n</parameter>\n</function>\n</tool_call>"
        )
        calls, evs = self._call(raw)
        self.assertEqual(text_of(evs).strip(), "")
        self.assertEqual((calls[0]["namespace"], calls[0]["name"]), ("mcp__filesystem", "edit_file"))
        args = json.loads(calls[0]["arguments"])
        self.assertEqual(args["path"], "/tmp/a.txt")
        self.assertIsInstance(args["edits"], str)
        self.assertIn('"hi"', args["edits"])

    def test_bare_write_file_is_the_one_namespace_tool(self):
        raw = (
            "<tool_call>\n<function=write_file>\n<parameter=path>\n/tmp/patch-tencut.py\n</parameter>\n"
            "<parameter=content>\nprint(\"ten\")\n</parameter>\n</function>\n</tool_call>"
        )
        calls, evs = self._call(raw)
        self.assertEqual(text_of(evs).strip(), "")
        self.assertEqual((calls[0]["namespace"], calls[0]["name"]), ("mcp__filesystem", "write_file"))
        args = json.loads(calls[0]["arguments"])
        self.assertEqual(args["path"], "/tmp/patch-tencut.py")
        self.assertEqual(args["content"], 'print("ten")')

    def test_bare_list_allowed_directories_is_a_call(self):
        raw = "<tool_call>\n<function=list_allowed_directories>\n</function>\n</tool_call>"
        calls, evs = self._call(raw)
        self.assertEqual(text_of(evs).strip(), "")
        self.assertEqual((calls[0]["namespace"], calls[0]["name"]),
                         ("mcp__filesystem", "list_allowed_directories"))
        self.assertEqual(json.loads(calls[0]["arguments"]), {})

    def test_double_underscore_still_maps(self):
        raw = (
            "<tool_call>\n<function=mcp__filesystem__write_file>\n<parameter=path>\n/tmp/z\n</parameter>\n"
            "<parameter=content>\nok\n</parameter>\n</function>\n</tool_call>"
        )
        calls, _evs = self._call(raw)
        self.assertEqual((calls[0]["namespace"], calls[0]["name"]), ("mcp__filesystem", "write_file"))
        self.assertEqual(json.loads(calls[0]["arguments"])["content"], "ok")

    def test_equals_inside_thinking_is_still_a_call(self):
        raw = (
            "<tool_call>\n<function=mcp__filesystem=write_file>\n<parameter=path>\n/tmp/x.py\n</parameter>\n"
            "<parameter=content>\nok\n</parameter>\n</function>\n</tool_call>"
        )
        evs = events(raw + "</think>\n", self.tools, thinking=True)
        found = [e for e in evs if e.kind == "tool_call"]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].call.arguments["path"], "/tmp/x.py")

    def test_an_unknown_name_stays_text(self):
        raw = "<tool_call>\n<function=mcp__filesystem=not_a_tool>\n</function>\n</tool_call>"
        calls, evs = self._call(raw)
        self.assertEqual(calls, [])
        self.assertIn("not_a_tool", text_of(evs))

    def test_two_namespaces_sharing_a_member_stay_text(self):
        req = {"tools": [
            {"type": "namespace", "name": "mcp__filesystem", "tools": [
                {"type": "function", "name": "write_file", "parameters": {
                    "type": "object", "properties": {"path": {"type": "string"}}}}]},
            {"type": "namespace", "name": "mcp__other", "tools": [
                {"type": "function", "name": "write_file", "parameters": {
                    "type": "object", "properties": {"path": {"type": "string"}}}}]},
        ]}
        tools, names, _ = request_tools(req)
        raw = "<tool_call>\n<function=write_file>\n<parameter=path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>"
        calls, evs = self._call(raw, tools, names)
        self.assertEqual(calls, [])
        self.assertIn("write_file", text_of(evs))

    def test_a_flat_tool_with_the_member_name_wins(self):
        req = {"tools": [
            {"type": "function", "name": "write_file", "parameters": {
                "type": "object", "properties": {"path": {"type": "string"}}}},
            {"type": "namespace", "name": "mcp__filesystem", "tools": [
                {"type": "function", "name": "write_file", "parameters": {
                    "type": "object", "properties": {"path": {"type": "string"},
                                                     "content": {"type": "string"}}}}]},
        ]}
        tools, names, _ = request_tools(req)
        self.assertEqual(names["write_file"], (None, "write_file", "function"))
        raw = "<tool_call>\n<function=write_file>\n<parameter=path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>"
        calls, evs = self._call(raw, tools, names)
        self.assertEqual(text_of(evs).strip(), "")
        self.assertEqual(calls[0]["name"], "write_file")
        self.assertNotIn("namespace", calls[0])


    def test_a_flat_tool_named_like_the_underscore_alias_wins(self):
        req = {"tools": [
            {"type": "function", "name": "mcp__filesystem_write_file", "parameters": {
                "type": "object", "properties": {"path": {"type": "string"}}}},
            {"type": "namespace", "name": "mcp__filesystem", "tools": [
                {"type": "function", "name": "write_file", "parameters": {
                    "type": "object", "properties": {"path": {"type": "string"},
                                                     "content": {"type": "string"}}}}]},
        ]}
        tools, names, _ = request_tools(req)
        raw = ("<tool_call>\n<function=mcp__filesystem_write_file>\n<parameter=path>\n/tmp/x\n</parameter>\n</function>"
               "\n</tool_call>")
        calls, evs = self._call(raw, tools, names)
        self.assertEqual(text_of(evs).strip(), "")
        self.assertEqual(calls[0]["name"], "mcp__filesystem_write_file")
        self.assertNotIn("namespace", calls[0])


def plain_tool(name):
    return {"name": name, "parameters": {"type": "object", "properties": {"url": {"type": "string"}}}}


def call_text(name, param="url", value="https://example.com"):
    return f"<tool_call>\n<function={name}>\n<parameter={param}>\n{value}\n</parameter>\n</function>\n</tool_call>"


def run_parser(text, tools, width=0):
    parser = OutputParser(thinking=False, tools=tools, stream_tools=True, declared_only=True)
    evs = []
    step = width or len(text) or 1
    for k in range(0, len(text), step):
        evs += parser.feed(text[k:k + step])
    evs += parser.finish("stop")
    return evs, parser


class CollapsedUnderscoreNames(unittest.TestCase):
    """#3796: `mcp__server__tool` written as `mcp_server_tool` is a spelling of the declared tool (the same class of
    difference as namespace__member). A name that is only similar stays text, and is named for the client."""
    MCP = [plain_tool("mcp__playwright__browser_navigate"), plain_tool("mcp__playwright__browser_snapshot"),
           plain_tool("mcp__playwright__browser_tabs"), plain_tool("todo_write")]

    def test_the_collapsed_spelling_is_the_declared_tool(self):
        for name in ("browser_navigate", "browser_snapshot", "browser_tabs"):
            for width in (0, 1, 5):
                with self.subTest(name=name, width=width):
                    evs, parser = run_parser("Opening it.\n" + call_text("mcp_playwright_" + name), self.MCP, width)
                    found = [e.call for e in evs if e.kind == "tool_call"]
                    self.assertEqual([c.name for c in found], ["mcp__playwright__" + name])   # the client gets the name it declared
                    self.assertEqual(found[0].arguments, {"url": "https://example.com"})
                    self.assertEqual(text_of(evs).strip(), "Opening it.")
                    self.assertEqual((parser.unread, parser.unread_names), (0, []))

    def test_the_declared_spelling_and_a_plain_tool_are_unchanged(self):
        evs, _ = run_parser(call_text("mcp__playwright__browser_tabs") + "\n" + call_text("todo_write"), self.MCP)
        self.assertEqual([e.call.name for e in evs if e.kind == "tool_call"], ["mcp__playwright__browser_tabs", "todo_write"])

    def test_a_tool_declared_in_the_collapsed_spelling_wins(self):
        tools = self.MCP + [plain_tool("mcp_playwright_browser_tabs")]
        evs, _ = run_parser(call_text("mcp_playwright_browser_tabs"), tools)
        self.assertEqual([e.call.name for e in evs if e.kind == "tool_call"], ["mcp_playwright_browser_tabs"])

    def test_two_tools_with_the_same_collapsed_spelling_are_not_guessed(self):
        tools = [plain_tool("a__b_c"), plain_tool("a_b__c")]
        evs, parser = run_parser(call_text("a_b_c"), tools)
        self.assertEqual([e for e in evs if e.kind == "tool_call"], [])
        self.assertIn("a_b_c", text_of(evs))
        self.assertEqual(parser.unread_names, ["a_b_c"])

    def test_a_reordered_or_renamed_tool_stays_text_and_is_named(self):
        """The LibreChat case (`server1_label_values` for `label_values_mcp_server1`) and the bash/shell case: no guessing."""
        for declared, written in (("label_values_mcp_server1", "server1_label_values"), ("shell", "bash"),
                                  ("godot_execute_gdscript", "execute_gdscript")):
            for width in (0, 3):
                with self.subTest(written=written, width=width):
                    raw = "Done.\n" + call_text(written)
                    evs, parser = run_parser(raw, [plain_tool(declared)], width)
                    self.assertEqual([e for e in evs if e.kind == "tool_call"], [])
                    self.assertEqual(text_of(evs), raw)                                    # the text is exactly what the model wrote
                    self.assertEqual((parser.unread, parser.unread_names), (1, [written]))


if __name__ == "__main__":
    unittest.main()
