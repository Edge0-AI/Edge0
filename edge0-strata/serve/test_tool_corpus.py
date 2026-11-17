"""A corpus of model outputs for the tool-call and reasoning parser (#4427, #4078, #4077).

Each specimen is the text a model wrote, the tools the request declared, and what a client must get: the calls (name and
arguments), the answer text, the reasoning text.  The sources are named in each entry: the reporters' own examples in
the issues, the Qwen3.8-Flash-Next form (`<tool_call><function=NAME><parameter=P>`), the drift shapes of the 1,462-turn
corpus the recovery switch was written for, and DeepSeek-V4.1's DSML form.

Every specimen is read whole, one byte at a time, in pieces of 3 and 7 bytes, and cut in two at every byte; with
streaming of the arguments on and off.  All of them must give the same events, and the streamed arguments of a call
must be its final arguments.

    python -m unittest serve.test_tool_corpus -v
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.dsml import DsmlOutputParser  # noqa: E402
from serve.frontend import OutputParser  # noqa: E402


def tool(name, **props):
    return {"name": name, "parameters": {"type": "object", "properties": props}}


S, I, A, O = {"type": "string"}, {"type": "integer"}, {"type": "array"}, {"type": "object"}
TOOLS = [tool("todo_write", todos=A), tool("run_command", command=S, timeout=I), tool("write_file", path=S, content=S),
         tool("read", path=S), tool("bash", command=S), tool("edit", path=S, edits=A, options=O)]


def qwen(name, **params):
    """The call as Qwen3.8-Flash-Next writes it (the chat template's own form)."""
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def dsml(*calls):
    """DeepSeek-V4.1's DSML block: calls are (name, [(parameter, value, is_string)])."""
    out = ["<｜DSML｜ calls>\n"]
    for name, params in calls:
        out.append(f'<｜DSML｜ invoke name="{name}">\n')
        for k, v, is_str in params:
            out.append(f'<｜DSML｜ parameter name="{k}" string="{str(is_str).lower()}">{v}</｜DSML｜ parameter>\n')
        out.append("</｜DSML｜ invoke>\n")
    return "".join(out) + "</｜DSML｜ calls>"


NESTED = qwen("read", path="nested.txt")
HEREDOC = f"python3 - <<'PY'\ntext = '''{NESTED}'''\nprint('tail that must not be lost')\nPY"
REOPEN = "first line\n</parameter>\n<parameter=content>\nlast line"
TODO_OK = ('[{"content": "Generate lorem ipsum page texture atlas (PIL) + inspect", "status": "in_progress"}, '
           '{"content": "Blender script: book geometry + materials", "status": "pending"}]')
TODO_STRAY = ('[{"content": "Init git repo with LFS", "status": "in_progress", "pending": false, "completed": false}, '
              '{"content": "Write Blender build script", "status": "pending", "in_progress", "pending": false, '
              '"completed": false}]')
TODO_STRAY_WANT = [{"content": "Init git repo with LFS", "status": "in_progress", "pending": False, "completed": False},
                   {"content": "Write Blender build script", "status": "pending", "pending": False, "completed": False}]
STRAY_VALUE = "reasoning sample:\n</think>\nanswer sample"

# name: dict(text, calls, content, reasoning, thinking, recover, finish, tools, unread, notes, source)
CORPUS = {}


def add(name, text, calls=(), content="", reasoning="", thinking=False, recover=False, finish="stop", tools=None,
        unread=0, notes=0, source="", parser="qwen", stream_only=False, announced=False):
    assert name not in CORPUS, name
    CORPUS[name] = dict(text=text, calls=[tuple(c) for c in calls], content=content, reasoning=reasoning,
                        thinking=thinking, recover=recover, finish=finish, tools=tools or TOOLS, unread=unread,
                        notes=notes, source=source, parser=parser, stream_only=stream_only, announced=announced)


# ---- the reporters' examples ------------------------------------------------------------------------------------
add("4427-todos-valid", qwen("todo_write", todos=TODO_OK), [("todo_write", {"todos": json.loads(TODO_OK)})],
    source="#4427 FraYoshi, first example (the Harness todo_write call)")
add("4427-todos-stray-items", "Old tree project files are present but the task is fresh.\n\n" + qwen("todo_write", todos=TODO_STRAY),
    [("todo_write", {"todos": TODO_STRAY_WANT})], content="Old tree project files are present but the task is fresh.",
    notes=1, source="#4427 FraYoshi, second example: `\"status\": \"pending\", \"in_progress\", \"pending\": false`")
add("4427-todos-single-quotes", qwen("todo_write", todos="[{'content': 'a', 'status': 'pending'}]"),
    [("todo_write", {"todos": [{"content": "a", "status": "pending"}]})], notes=1, source="#4427 shape: Python quotes")
add("4427-todos-trailing-comma-and-open-bracket", qwen("todo_write", todos='[{"content": "a", "status": "pending"},'),
    [("todo_write", {"todos": [{"content": "a", "status": "pending"}]})], notes=1, source="#4427 shape: a cut array")
add("4427-todos-json-string-of-an-array", qwen("todo_write", todos=json.dumps(TODO_OK)),
    [("todo_write", {"todos": json.loads(TODO_OK)})], source="#4427 shape: the array written as a JSON string")
add("4427-todos-not-json-stays-text", qwen("todo_write", todos="buy milk"), [("todo_write", {"todos": "buy milk"})],
    source="#4427: text that is no array stays the text it is")
add("4427-one-line", "Let me run ls and read the images.\n\n<tool_call> <function=run_command> <parameter=command> "
    "ls -la /home/user/imap-attachments/ 2>&1 | head -30 </parameter> </function> </tool_call>",
    [("run_command", {"command": " ls -la /home/user/imap-attachments/ 2>&1 | head -30 "})],
    content="Let me run ls and read the images.", source="#4427 miscdebris: the tags on one line")
add("4427-one-line-in-thinking", "Let me run ls and read the images.\n\n<tool_call> <function=run_command> <parameter=command> "
    "ls -la 2>&1 | head -30 </parameter> </function> </tool_call>", [("run_command", {"command": " ls -la 2>&1 | head -30 "})],
    reasoning="Let me run ls and read the images.\n\n", thinking=True,
    source="#4427 miscdebris: written inside the thinking, the turn ends on it")
add("4078-nested-call", qwen("write_file", path="fixture.txt", content="first line\n" + NESTED + "\nlast line"),
    [("write_file", {"path": "fixture.txt", "content": "first line\n" + NESTED + "\nlast line"})],
    source="#4078 LispEngineer / ackalker: a value holding a whole call")
add("4078-reopened-parameter", qwen("write_file", path="fixture.txt", content=REOPEN),
    [("write_file", {"path": "fixture.txt", "content": REOPEN})], source="#4078: a value that reopens its own parameter")
add("4078-heredoc", qwen("bash", command=HEREDOC), [("bash", {"command": HEREDOC})],
    source="#4078 Tsury: a heredoc that prints a quoted call")
add("4078-call-format-in-a-file", qwen("write_file", path="DOC.md", content="Close a call with </function> and then </tool_call>.\nlike this."),
    [("write_file", {"path": "DOC.md", "content": "Close a call with </function> and then </tool_call>.\nlike this."})],
    source="#210: a file that documents the format")
add("4077-stray-think-line", "The user wants a file with a literal.\n" + STRAY_VALUE + "\nSo I write it.\n</think>\n\n" +
    qwen("write_file", path="out.txt", content=STRAY_VALUE),
    [("write_file", {"path": "out.txt", "content": STRAY_VALUE})], thinking=True,
    reasoning="The user wants a file with a literal.\nreasoning sample:\n",
    content="answer sample\nSo I write it.", source="#4077 Tsury: the model quoted the closing tag at a line start")
add("4077-stray-think-in-a-fence", "Content:\n```\n" + STRAY_VALUE + "\n```\nNow call it.\n</think>\n\n" +
    qwen("write_file", path="out.txt", content=STRAY_VALUE),
    [("write_file", {"path": "out.txt", "content": STRAY_VALUE})], thinking=True,
    reasoning="Content:\n```\nreasoning sample:\n", content="answer sample\n```\nNow call it.",
    source="#4077: the quote sat in a code fence, so the fence closes in the 'answer'")
add("4077-stray-think-after-a-backtick", "The text is `" + STRAY_VALUE + "`. Call it.\n</think>\n\n" +
    qwen("write_file", path="out.txt", content=STRAY_VALUE),
    [("write_file", {"path": "out.txt", "content": STRAY_VALUE})], thinking=True,
    reasoning="The text is `reasoning sample:\n", content="answer sample`. Call it.", source="#4077: an open backtick")
add("4077-quoted-think-stays-reasoning", "I will not write `</think>` yet, I keep thinking.\nDone thinking.\n</think>\n\nThe answer.",
    thinking=True, reasoning="I will not write `</think>` yet, I keep thinking.\nDone thinking.\n", content="The answer.",
    source="#537: a closing tag after a backtick is still reasoning (control)")
add("4077-closer-in-the-answer-text-is-kept-mid-line", "Reasoning.\n</think>\n\nWrite </think> to end the thinking.",
    thinking=True, reasoning="Reasoning.\n", content="Write </think> to end the thinking.",
    source="a closing tag inside a sentence of the answer is text (control)")
add("4077-no-second-closer-is-the-normal-case", "Reasoning.\n</think>\n\n" + qwen("read", path="a.txt"),
    [("read", {"path": "a.txt"})], thinking=True, reasoning="Reasoning.\n", source="control")

# ---- the Qwen3.8-Flash-Next form and its slips ------------------------------------------------------------------
add("qwen-one-call", "I will read it.\n\n" + qwen("read", path="a.txt"), [("read", {"path": "a.txt"})],
    content="I will read it.", source="the template's own form")
add("qwen-int-argument", qwen("run_command", command="make -j8", timeout="30"),
    [("run_command", {"command": "make -j8", "timeout": 30})], source="a declared integer")
add("qwen-after-thinking", "Plan: read the file.\n</think>\n\n" + qwen("read", path="a.txt"), [("read", {"path": "a.txt"})],
    thinking=True, reasoning="Plan: read the file.\n", source="the normal agent turn")
add("qwen-call-in-the-thinking", "Plan: read the file.\n\n" + qwen("read", path="a.txt"), [("read", {"path": "a.txt"})],
    thinking=True, reasoning="Plan: read the file.\n\n", source="#804: the turn ends inside the thinking, on the call")
add("qwen-three-calls", "Reading all three.\n\n" + "\n".join(qwen("read", path=f"{n}.txt") for n in "abc"),
    [("read", {"path": "a.txt"}), ("read", {"path": "b.txt"}), ("read", {"path": "c.txt"})], content="Reading all three.",
    source="parallel calls, one wrapper each")
add("qwen-calls-with-prose-between", qwen("read", path="a.txt") + "\nand now the other one:\n" + qwen("read", path="b.txt"),
    [("read", {"path": "a.txt"}), ("read", {"path": "b.txt"})], content="and now the other one:", source="prose between calls")
add("qwen-three-calls-in-the-thinking", "Plan.\n\n" + "\n".join(qwen("read", path=f"{n}.txt") for n in "ab"),
    [("read", {"path": "a.txt"}), ("read", {"path": "b.txt"})], thinking=True, reasoning="Plan.\n\n\n",
    source="#804: two calls at the end of the thinking")
add("qwen-function-close-left-out", "<tool_call>\n<function=write_file>\n<parameter=path>\nout.txt\n</parameter>\n"
    "<parameter=content>\nhello\n</parameter>\n</tool_call>", [("write_file", {"path": "out.txt", "content": "hello"})],
    source="#4427 shape: no </function> (PR #3960's Coder-L slip)")
add("qwen-function-close-left-out-in-thinking", "Writing.\n\n<tool_call>\n<function=write_file>\n<parameter=path>\nout.txt\n"
    "</parameter>\n<parameter=content>\nhello\n</parameter>\n</tool_call>", [("write_file", {"path": "out.txt", "content": "hello"})],
    thinking=True, reasoning="Writing.\n\n", source="the same inside the thinking")
add("qwen-wrapper-close-left-out", "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>",
    [("read", {"path": "a.txt"})], stream_only=True, source="the output ends on </function> (the server always streams)")
add("qwen-wrapper-close-left-out-in-thinking", "Plan.\n\n<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>",
    [("read", {"path": "a.txt"})], thinking=True, reasoning="Plan.\n\n", source="the same inside the thinking")
add("qwen-wrapper-close-left-out-two-calls", qwen("read", path="a.txt").replace("\n</tool_call>", "") + "\n" + qwen("read", path="b.txt"),
    [("read", {"path": "a.txt"}), ("read", {"path": "b.txt"})], source="a following opener confirms the missing closer")
add("qwen-wrapper-close-left-out-two-calls-in-thinking", "Plan.\n\n" + qwen("read", path="a.txt").replace("\n</tool_call>", "") +
    "\n" + qwen("read", path="b.txt"), [("read", {"path": "a.txt"}), ("read", {"path": "b.txt"})], thinking=True,
    reasoning="Plan.\n\n\n", source="the same inside the thinking")
add("qwen-undeclared-name-is-text", qwen("format_disk", target="/"), content=qwen("format_disk", target="/"), unread=1,
    source="by design: a call for a tool the request did not declare comes back as text (and is counted)")
add("qwen-cut-inside-a-value", "<tool_call>\n<function=write_file>\n<parameter=path>\nout.txt\n</parameter>\n<parameter=content>\nhal",
    content="<tool_call>\n<function=write_file>\n<parameter=path>\nout.txt\n</parameter>\n<parameter=content>\nhal",
    finish="length", announced=True, source="an unterminated tag: the output stopped at the token limit; no call")
add("qwen-cut-inside-the-opener", "Now.\n\n<tool_call>\n<function=writ", content="Now.\n\n<tool_call>\n<function=writ",
    finish="length", source="cut inside the name")
add("qwen-cut-after-function-close-on-length", "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>",
    finish="length", content="<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>",
    source="a turn cut by the token limit is not a natural stop: no call")
add("qwen-quoted-in-a-fence", "Example:\n```\n" + qwen("read", path="a.txt") + "\n```", content="Example:\n```\n" + qwen("read", path="a.txt") + "\n```",
    source="#1058: a call in a code fence is a quote")
add("qwen-prose-names-the-format", "Calls are written as <tool_call> blocks.", content="Calls are written as <tool_call> blocks.",
    source="prose that names the tag")

# ---- the drift shapes (tool_call_recovery on; the corpus of 1,462 agent turns) ----------------------------------
PW = "<parameter=path>\na.md\n</parameter>\n<parameter=content>\nhi\n</parameter>\n</function>"
WRITE_TOOLS = [tool("write", path=S, content=S)]
add("drift-parameter-as-opener", f"<tool_call>\n<parameter=write>\n{PW}\n</tool_call>", [("write", {"path": "a.md", "content": "hi"})],
    recover=True, tools=WRITE_TOOLS, source="corpus 300dfba2 (x11)")
add("drift-json-in-the-wrapper", '<tool_call>\n{"name": "write", "arguments": {"path": "a.md", "content": "hi"}}\n</tool_call>',
    [("write", {"path": "a.md", "content": "hi"})], recover=True, tools=WRITE_TOOLS, source="corpus 5e847539 (x6)")
add("drift-bare-function", f"Writing it now.\n\n<function=write>\n{PW}", [("write", {"path": "a.md", "content": "hi"})],
    content="Writing it now.", recover=True, tools=WRITE_TOOLS, source="corpus 395efd4c / 6e71eed4 (x5 each)")
add("drift-batch-in-one-wrapper", f"<tool_call>\n<function=write>\n{PW}\n<function=write>\n{PW.replace('a.md', 'b.md')}\n</tool_call>",
    [("write", {"path": "a.md", "content": "hi"}), ("write", {"path": "b.md", "content": "hi"})], recover=True,
    tools=WRITE_TOOLS, source="corpus 0912870b / 51b155c9")

# ---- DeepSeek-V4.1 (DSML) ---------------------------------------------------------------------------------------
add("dsml-one-call", dsml(("write_file", [("path", "out.txt", True), ("content", "hello\nworld", True)])),
    [("write_file", {"path": "out.txt", "content": "hello\nworld"})], parser="dsml", source="the model's own template")
add("dsml-after-thinking", "Plan.</think>\n\n" + dsml(("read", [("path", "a.txt", True)])), [("read", {"path": "a.txt"})],
    thinking=True, reasoning="Plan.", parser="dsml", source="a thinking turn")
add("dsml-typed-arguments", dsml(("run_command", [("command", "make", True), ("timeout", "30", False)]),
                                 ("todo_write", [("todos", TODO_OK, False)])),
    [("run_command", {"command": "make", "timeout": 30}), ("todo_write", {"todos": json.loads(TODO_OK)})], parser="dsml",
    source="string=\"false\" values are JSON")
add("dsml-damaged-array", dsml(("todo_write", [("todos", TODO_STRAY, False)])), [("todo_write", {"todos": TODO_STRAY_WANT})],
    notes=1, parser="dsml", source="#4427 shape inside DSML")
add("dsml-several-calls-and-text", "Reading both.\n\n" + dsml(("read", [("path", "a.txt", True)]), ("read", [("path", "b.txt", True)])),
    [("read", {"path": "a.txt"}), ("read", {"path": "b.txt"})], content="Reading both.", parser="dsml", source="a batch")
add("dsml-value-holds-a-qwen-call", dsml(("write_file", [("path", "x.txt", True), ("content", NESTED, True)])),
    [("write_file", {"path": "x.txt", "content": NESTED})], parser="dsml", source="#4078 shape in DSML")
add("dsml-stray-think", "Quote:\n</think>\nmore thinking.\n</think>\n\n" + dsml(("read", [("path", "a.txt", True)])),
    [("read", {"path": "a.txt"})], thinking=True, reasoning="Quote:\n", content="more thinking.", parser="dsml",
    source="#4077 shape in DSML")
add("dsml-cut-call", dsml(("read", [("path", "a.txt", True)]))[:60], finish="length", content=None,
    parser="dsml", source="a DSML call cut by the token limit is no call (its text is the translator's, not checked)")


def run(spec, pieces, stream_tools):
    cls = DsmlOutputParser if spec["parser"] == "dsml" else OutputParser
    p = cls(thinking=spec["thinking"], tools=spec["tools"], stream_tools=stream_tools, recover=spec["recover"],
            declared_only=True)
    evs = []
    for piece in pieces:
        evs += p.feed(piece)
    evs += p.finish(spec["finish"])
    return evs, p


def cut(text, size):
    return [text[i:i + size] for i in range(0, len(text), size)]


class ToolCorpus(unittest.TestCase):
    def check(self, name, spec, pieces, stream_tools, label):
        evs, p = run(spec, pieces, stream_tools)
        calls = [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]
        text = lambda kind: "".join(e.text for e in evs if e.kind == kind)   # noqa: E731
        where = f"{name} [{label}, stream_tools={stream_tools}]"
        self.assertEqual(calls, spec["calls"], where)
        # a call announced to the client and then cut stays unfinished there (#211): its text is not repeated as content
        if spec["content"] is not None:
            self.assertEqual(text("content").strip(), "" if spec["announced"] and stream_tools else spec["content"], where)
        self.assertEqual(text("reasoning").strip(), spec["reasoning"].strip(), where)
        self.assertEqual(p.unread, spec["unread"], where)
        self.assertEqual(len(p.notes), spec["notes"], where)
        if stream_tools:
            for e in (x for x in evs if x.kind == "tool_call"):
                streamed = "".join(x.text for x in evs if x.kind == "tool_args" and x.call is not None and x.call.id == e.call.id)
                if streamed:
                    self.assertEqual(json.loads(streamed), e.call.arguments, where)
        return evs

    def test_every_specimen_whole_and_in_pieces(self):
        for name, spec in CORPUS.items():
            for stream_tools in (True,) if spec["stream_only"] else (False, True):
                self.check(name, spec, [spec["text"]], stream_tools, "whole")
                for size in (1, 3, 7):
                    self.check(name, spec, cut(spec["text"], size), stream_tools, f"pieces of {size}")

    def test_every_specimen_cut_in_two_at_every_byte(self):
        for name, spec in CORPUS.items():
            text = spec["text"]
            for at in range(1, len(text)):
                self.check(name, spec, [text[:at], text[at:]], True, f"cut at {at}")

    def test_the_same_events_whatever_the_pieces(self):
        """Text and calls of a specimen do not depend on how the stream was cut (the ids aside)."""
        for name, spec in CORPUS.items():
            whole, _ = run(spec, [spec["text"]], True)
            want = [(e.kind, e.text) for e in whole if e.kind in ("content", "reasoning")]
            joined = lambda evs, k: "".join(t for kind, t in evs if kind == k)   # noqa: E731
            for size in (1, 5):
                got, _ = run(spec, cut(spec["text"], size), True)
                got = [(e.kind, e.text) for e in got if e.kind in ("content", "reasoning")]
                for kind in ("content", "reasoning"):
                    self.assertEqual(joined(got, kind), joined(want, kind), f"{name} pieces of {size}")

    def test_the_sources_are_named(self):
        for name, spec in CORPUS.items():
            self.assertTrue(spec["source"], name)


if __name__ == "__main__":
    unittest.main()
