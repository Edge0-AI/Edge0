"""#4427, #4077, #4078 through the HTTP routes: what a client gets for the damaged calls and the stray closing think tag.

The model's text is replayed by the mock engine; the reply is read from Chat Completions, the Anthropic Messages API and
the Responses API, each whole and streamed.  Every route must hand over the same calls with the same arguments.

    python -m unittest serve.test_tool_routes -v
"""
from __future__ import annotations

import contextlib
import http.client
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ChatTemplate(ROOT / "serve/chat_template.jinja")
PROPS = {"todos": {"type": "array"}, "path": {"type": "string"}, "content": {"type": "string"}}
NAMES = ("todo_write", "write_file")


def call(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


class Routes(unittest.TestCase):
    def read(self, script, **service):
        """{route: (finish, [(name, arguments)], answer text)} for the model's `script`, on every route."""
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, script, max_context=16384), tok, TEMPLATE)
        for k, v in service.items():
            setattr(svc, k, v)
        httpd = serve(svc, port=0)
        port = httpd.server_address[1]
        out = {}

        def post(path, body):
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    c.request("POST", path, body=json.dumps(body).encode(),
                              headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
                    return c.getresponse().read().decode()
            finally:
                c.close()

        def sse(raw):
            return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]

        chat_tools = [{"type": "function", "function": {"name": n, "parameters": {"type": "object", "properties": PROPS}}}
                      for n in NAMES]
        try:
            for stream in (False, True):
                raw = post("/v1/chat/completions", {"model": "x", "max_tokens": 900, "stream": stream, "tools": chat_tools,
                                                    "messages": [{"role": "user", "content": "go"}]})
                if stream:
                    evs, calls, text, finish = sse(raw), {}, "", None
                    for e in evs:
                        ch = e["choices"][0]
                        text += ch["delta"].get("content") or ""
                        finish = ch.get("finish_reason") or finish
                        for tc in ch["delta"].get("tool_calls") or []:
                            slot = calls.setdefault(tc["index"], {"name": None, "args": ""})
                            slot["name"] = (tc.get("function") or {}).get("name") or slot["name"]
                            slot["args"] += (tc.get("function") or {}).get("arguments") or ""
                    got = [(c["name"], json.loads(c["args"])) for _, c in sorted(calls.items())]
                else:
                    d = json.loads(raw)["choices"][0]
                    text, finish = d["message"].get("content") or "", d["finish_reason"]
                    got = [(t["function"]["name"], json.loads(t["function"]["arguments"])) for t in d["message"].get("tool_calls") or []]
                out["chat", stream] = (finish, got, text.strip())

                raw = post("/v1/messages", {"model": "x", "max_tokens": 900, "stream": stream,
                                            "tools": [{"name": n, "input_schema": {"type": "object", "properties": PROPS}} for n in NAMES],
                                            "messages": [{"role": "user", "content": "go"}]})
                if stream:
                    evs, blocks, text, finish = sse(raw), {}, "", None
                    for e in evs:
                        if e["type"] == "content_block_start" and e["content_block"]["type"] == "tool_use":
                            blocks[e["index"]] = {"name": e["content_block"]["name"], "json": ""}
                        elif e["type"] == "content_block_delta":
                            if e["delta"]["type"] == "input_json_delta":
                                blocks[e["index"]]["json"] += e["delta"]["partial_json"]
                            elif e["delta"]["type"] == "text_delta":
                                text += e["delta"]["text"]
                        elif e["type"] == "message_delta":
                            finish = e["delta"]["stop_reason"]
                    got = [(b["name"], json.loads(b["json"] or "{}")) for _, b in sorted(blocks.items())]
                else:
                    d = json.loads(raw)
                    finish = d["stop_reason"]
                    text = "".join(b.get("text", "") for b in d["content"] if b["type"] == "text")
                    got = [(b["name"], b["input"]) for b in d["content"] if b["type"] == "tool_use"]
                out["anthropic", stream] = (finish, got, text.strip())

                raw = post("/v1/responses", {"model": "x", "input": "go", "stream": stream, "store": False,
                                             "tools": [{"type": "function", "name": n, "parameters": {"type": "object", "properties": PROPS}}
                                                       for n in NAMES]})
                if stream:
                    items = [e["item"] for e in sse(raw) if e["type"] == "response.output_item.done"]
                else:
                    items = json.loads(raw)["output"]
                got = [(i["name"], json.loads(i["arguments"])) for i in items if i["type"] == "function_call"]
                text = "".join(c["text"] for i in items if i["type"] == "message" for c in i["content"] if c["type"] == "output_text")
                out["responses", stream] = ("-", got, text.strip())
        finally:
            httpd.shutdown()
            httpd.server_close()
        return out

    def every_route(self, script, calls, text="", finish=("tool_calls", "tool_use"), **service):
        got = self.read(script, **service)
        for (route, stream), (fin, got_calls, got_text) in got.items():
            with self.subTest(route=route, stream=stream):
                self.assertEqual(got_calls, calls)
                self.assertEqual(got_text, text)
                if route == "chat":
                    self.assertEqual(fin, finish[0])
                elif route == "anthropic":
                    self.assertEqual(fin, finish[1])

    def test_a_damaged_array_argument_arrives_as_the_array(self):
        stray = ('[{"content": "Init", "status": "in_progress", "pending": false}, '
                 '{"content": "Write", "status": "pending", "in_progress", "pending": false}]')
        want = [{"content": "Init", "status": "in_progress", "pending": False},
                {"content": "Write", "status": "pending", "pending": False}]
        self.every_route("Plan.\n</think>\n\n" + call("todo_write", todos=stray), [("todo_write", {"todos": want})])

    def test_a_call_without_its_function_closer(self):
        text = ("Plan.\n</think>\n\n<tool_call>\n<function=write_file>\n<parameter=path>\nout.txt\n</parameter>\n"
                "<parameter=content>\nhello\n</parameter>\n</tool_call>")
        self.every_route(text, [("write_file", {"path": "out.txt", "content": "hello"})])

    def test_a_call_without_its_wrapper_closer_at_the_end(self):
        text = "Plan.\n</think>\n\n" + call("write_file", path="out.txt", content="hello").removesuffix("\n</tool_call>")
        self.every_route(text, [("write_file", {"path": "out.txt", "content": "hello"})])

    def test_a_value_that_quotes_a_whole_call(self):
        value = "first line\n" + call("write_file", path="nested.txt") + "\nlast line"
        self.every_route("Plan.\n</think>\n\n" + call("write_file", path="fixture.txt", content=value),
                         [("write_file", {"path": "fixture.txt", "content": value})])

    def test_a_stray_closing_think_tag_keeps_the_call(self):
        value = "reasoning sample:\n</think>\nanswer sample"
        script = ("The content is\n```\n" + value + "\n```\nCall it.\n</think>\n\n" + call("write_file", path="out.txt", content=value))
        got = self.read(script)
        for (route, stream), (fin, calls, text) in got.items():
            with self.subTest(route=route, stream=stream):
                self.assertEqual(calls, [("write_file", {"path": "out.txt", "content": value})])
                self.assertNotIn("</think>", text)


if __name__ == "__main__":
    unittest.main()
