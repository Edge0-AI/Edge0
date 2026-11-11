"""#3441 (a call whose envelope drifted by a token) and #3075 (a call for a tool the request never declared)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontend  # noqa: E402

TOOLS = [{"name": "functions.exec", "parameters": {"properties": {"input": {"type": "string"}}}},
         {"name": "shell", "parameters": {"properties": {"cmd": {"type": "string"}}}},
         {"name": "ns.web_search", "parameters": {"properties": {"q": {"type": "string"}}}}]


def body(opener, param="input", value="1+1", name=None):
    return f"<tool_call>\n{opener}\n<parameter={param}>\n{value}\n</parameter>\n</function>\n</tool_call>"


def run(text, width=0, **kw):
    """-> (events, parser); `width` 0 feeds the text whole, else that many characters at a time."""
    p = frontend.OutputParser(thinking=False, stream_tools=True, **kw)
    evs = []
    if width:
        for k in range(0, len(text), width):
            evs += p.feed(text[k:k + width])
    else:
        evs += p.feed(text)
    evs += p.finish()
    return evs, p


def calls(evs):
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


def text(evs):
    return "".join(e.text or "" for e in evs if e.kind == "content")


class DriftedEnvelope(unittest.TestCase):
    def test_a_dropped_function_keyword_is_still_the_call(self):
        for opener in ("<=functions.exec>", "<function:functions.exec>", "<function =functions.exec>",
                       "<functions.exec>", "<= functions.exec >"):
            for width in (0, 1, 2, 3, 7):
                with self.subTest(opener=opener, width=width):
                    evs, p = run(body(opener), width, tools=TOOLS)
                    self.assertEqual(calls(evs), [("functions.exec", {"input": "1+1"})])
                    self.assertEqual(text(evs), "")
                    self.assertEqual(p.unread, 0)

    def test_the_template_form_is_unchanged(self):
        evs, p = run(body("<function=functions.exec>"), 0, tools=TOOLS)
        self.assertEqual(calls(evs), [("functions.exec", {"input": "1+1"})])

    def test_an_undeclared_name_in_a_drifted_opener_stays_text(self):
        for width in (0, 1, 4):
            with self.subTest(width=width):
                raw = body("<=missing>")
                evs, p = run(raw, width, tools=TOOLS)
                self.assertEqual(calls(evs), [])
                self.assertEqual(text(evs), raw)
                self.assertEqual(p.unread, 1)
                self.assertFalse([e for e in evs if e.kind == "tool_start"])

    def test_prose_about_the_tag_is_not_a_drift(self):
        for raw in ("Use a <tool_call> block for that.", "<tool_call> then the name"):
            evs, p = run(raw, 3, tools=TOOLS)
            self.assertEqual(calls(evs), [])
            self.assertEqual(text(evs), raw)
            self.assertEqual(p.unread, 0)

    def test_streamed_and_whole_agree(self):
        raw = "Let me run it.\n" + body("<=shell>", "cmd", "ls") + "\nDone."
        whole, _ = run(raw, 0, tools=TOOLS)
        for width in (1, 2, 5, 11):
            streamed, _ = run(raw, width, tools=TOOLS)
            self.assertEqual(calls(streamed), calls(whole))
            self.assertEqual(text(streamed), text(whole))
        self.assertEqual(calls(whole), [("shell", {"cmd": "ls"})])
        self.assertEqual(text(whole), "Let me run it.Done.")


class OnlyDeclaredTools(unittest.TestCase):
    def test_a_call_for_another_tool_stays_text(self):
        raw = body("<function=get_indicators>", "symbol", "600076")
        for width in (0, 1, 3, 9):
            with self.subTest(width=width):
                evs, p = run("I will check.\n" + raw, width, tools=TOOLS)
                self.assertEqual(calls(evs), [])
                self.assertFalse([e for e in evs if e.kind in ("tool_start", "tool_args")])
                self.assertEqual(text(evs), "I will check.\n" + raw)

    def test_an_unparsed_call_is_counted_and_named_for_the_log(self):
        """T1 rail: a call that went back as text is no longer silent; the server prints its count and the names the model wrote."""
        raw = body("<function=get_indicators>", "symbol", "600076")
        evs, p = run("I will check."+chr(10) + raw, 4, tools=TOOLS)
        self.assertEqual((p.unread, p.unread_names), (1, ["get_indicators"]))
        evs, p = run(body("<function=shell>", "cmd", "ls"), 4, tools=TOOLS)         # a declared call is no unread call
        self.assertEqual((p.unread, p.unread_names), (0, []))

    def test_a_declared_call_after_an_undeclared_one_is_still_a_call(self):
        raw = body("<function=get_indicators>") + "\n" + body("<function=shell>", "cmd", "ls")
        evs, p = run(raw, 5, tools=TOOLS)
        self.assertEqual(calls(evs), [("shell", {"cmd": "ls"})])
        self.assertIn("get_indicators", text(evs))

    def test_the_responses_spelling_of_a_namespace_tool_counts(self):
        evs, p = run(body("<function=ns__web_search>", "q", "x"), 4, tools=TOOLS)
        # PR #3556: the call carries the declared spelling (ns.web_search), the one the client offered, not the model's flat one
        self.assertEqual(calls(evs), [("ns.web_search", {"q": "x"})])

    def test_no_tools_declared(self):
        raw = body("<function=search>", "q", "2+2")
        evs, p = run(raw, 3, tools=None, declared_only=True)
        self.assertEqual((calls(evs), text(evs)), ([], raw))
        evs, p = run(raw, 3, tools=[], declared_only=True)
        self.assertEqual((calls(evs), text(evs)), ([], raw))
        # without the flag (the Responses route: a Codex compaction request keeps its tools in the prompt only)
        evs, p = run(raw, 3, tools=None)
        self.assertEqual(calls(evs), [("search", {"q": "2+2"})])

    def test_the_reasoning_state_is_as_before(self):
        p = frontend.OutputParser(thinking=True, tools=TOOLS, stream_tools=True)
        evs = p.feed("think " + body("<function=shell>", "cmd", "ls") + "</think>\n\nok") + p.finish()
        self.assertEqual(calls(evs), [])


if __name__ == "__main__":
    unittest.main()
