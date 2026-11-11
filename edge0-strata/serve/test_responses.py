"""serve/test_responses.py - #451: the Responses API (POST /v1/responses) against the mock engine (no GPU, no pack).

    python -m unittest serve.test_responses -v
"""
from __future__ import annotations

import base64
import contextlib
import http.client
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import responses  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.responses import (ENCRYPTED_PREFIX, ResponsesError, collect, input_messages, prompt_made,  # noqa: E402
                             prompt_tools, request_tools, template_kwargs, text_format, thread_title_events)
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ChatTemplate(ROOT / "serve/chat_template.jinja")

CALL = ("Let me look.\n</think>\n\nChecking.\n\n<tool_call>\n<function=exec_command>\n<parameter=cmd>\ncat a.txt\n"
        "</parameter>\n</function>\n</tool_call>")
ANSWER = "Read it.\n</think>\n\nThe file says before."
TOOLS = [{"type": "function", "name": "exec_command", "description": "Runs a command.", "strict": False,
          "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}]


# ------------------------------------------------------------------------------------------------ parsing
class Parsing(unittest.TestCase):
    def test_a_string_input_is_one_user_message(self):
        self.assertEqual(input_messages({"input": "hi", "instructions": "Be brief."}),
                         [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}])

    def test_leading_developer_messages_join_the_system_message_later_ones_become_user(self):
        msgs = input_messages({"instructions": "I", "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "D1"},
                                                                 {"type": "input_text", "text": "D2"}]},
            {"role": "user", "content": "u"},
            {"type": "message", "role": "developer", "content": "late"}]})
        self.assertEqual(msgs, [{"role": "system", "content": "I\n\nD1D2"}, {"role": "user", "content": "u"},
                                {"role": "user", "content": "late"}])

    def test_codex_history_becomes_one_assistant_turn_per_answer(self):
        items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "edit it"}]},
                 {"type": "reasoning", "id": "rs_1", "summary": [],
                  "content": [{"type": "reasoning_text", "text": "look first"}]},
                 {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Looking."}]},
                 {"type": "function_call", "id": "fc_1", "call_id": "c1", "name": "exec_command",
                  "arguments": "{\"cmd\":\"cat a\"}"},
                 {"type": "function_call", "id": "fc_2", "call_id": "c2", "name": "spawn_agent",
                  "namespace": "multi_agent_v1", "arguments": "{}"},
                 # results in the order they finished, not the order of the calls
                 {"type": "function_call_output", "call_id": "c2", "output": "agent"},
                 {"type": "function_call_output", "call_id": "c1",
                  "output": [{"type": "input_text", "text": "before"}]},
                 {"type": "reasoning", "id": "rs_2", "summary": [{"type": "summary_text", "text": "s"}],
                  "content": [{"type": "reasoning_text", "text": "now write"}]},
                 {"type": "function_call", "call_id": "c3", "name": "exec_command", "arguments": "{\"cmd\":\"w\"}"},
                 {"type": "function_call_output", "call_id": "c3", "output": "ok"}]
        self.assertEqual(input_messages({"input": items}), [
            {"role": "user", "content": "edit it"},
            {"role": "assistant", "content": "Looking.", "reasoning_content": "look first", "tool_calls": [
                {"function": {"name": "exec_command", "arguments": {"cmd": "cat a"}}},
                {"function": {"name": "multi_agent_v1.spawn_agent", "arguments": {}}}]},
            {"role": "tool", "content": "before"}, {"role": "tool", "content": "agent"},
            {"role": "assistant", "content": "", "reasoning_content": "now write", "tool_calls": [
                {"function": {"name": "exec_command", "arguments": {"cmd": "w"}}}]},
            {"role": "tool", "content": "ok"}])

    def test_reasoning_comes_back_from_our_encrypted_content_and_foreign_tokens_are_ignored(self):
        enc = ENCRYPTED_PREFIX + base64.b64encode(json.dumps({"text": "hidden"}).encode()).decode()
        for item, want in (({"encrypted_content": enc, "summary": []}, "hidden"),
                           ({"encrypted_content": "gAAAAAB-someone-elses", "summary": []}, None)):
            msgs = input_messages({"input": [{"role": "user", "content": "q"}, {"type": "reasoning", **item},
                                             {"role": "assistant", "content": "a"}]})
            self.assertEqual(msgs[1].get("reasoning_content"), want)
        with self.assertRaises(ResponsesError) as e:
            input_messages({"input": [{"type": "reasoning", "encrypted_content": ENCRYPTED_PREFIX + "!!"}]})
        self.assertEqual(e.exception.param, "input[0].encrypted_content")

    def test_bad_arguments_from_a_cut_off_call_do_not_refuse_the_conversation(self):
        msgs = input_messages({"input": [{"role": "user", "content": "q"},
                                         {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{\"a"},
                                         {"type": "function_call_output", "call_id": "c", "output": "x"}]})
        self.assertEqual(msgs[1]["tool_calls"][0]["function"]["arguments"], {"arguments": "{\"a"})

    def test_unsupported_items_are_named(self):
        for items, param in (([{"type": "item_reference", "id": "msg_1"}], "input[0]"),
                             ([{"type": "web_search_call"}], "input[0].type"),
                             ([{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}],
                              "input[0].content[0].type"),
                             ([{"role": "robot", "content": "x"}], "input[0].role")):
            with self.assertRaises(ResponsesError) as e:
                input_messages({"input": items})
            self.assertEqual(e.exception.param, param)
        with self.assertRaises(ResponsesError):
            input_messages({})

    def test_tools_namespaces_custom_and_hosted(self):
        tools, names, skipped = request_tools({"tools": [
            *TOOLS, {"type": "namespace", "name": "mcp_fs", "description": "Files.",
                     "tools": [{"type": "function", "name": "read", "parameters": {"type": "object"}}]},
            {"type": "custom", "name": "apply_patch", "description": "Patch.",
             "format": {"type": "grammar", "syntax": "lark", "definition": "start: \"x\""}},
            {"type": "web_search"}]})
        self.assertEqual([t["name"] for t in tools], ["exec_command", "mcp_fs.read", "apply_patch"])
        self.assertEqual(tools[1]["description"], "Files.")
        self.assertIn("start: \"x\"", tools[2]["description"])
        self.assertEqual(names["mcp_fs.read"], ("mcp_fs", "read", "function"))
        self.assertEqual(names["apply_patch"], (None, "apply_patch", "custom"))
        self.assertEqual(skipped, ["web_search"])
        self.assertIsNone(request_tools({"tools": TOOLS, "tool_choice": "none"})[0])

    def test_additional_tools_input_items_are_accepted(self):
        """#782: Codex sends an `additional_tools` input item; it used to be refused with a 400."""
        extra = {"type": "additional_tools", "tools": [{"type": "function", "name": "later_tool",
                                                         "parameters": {"type": "object"}}]}
        req = {"input": [{"type": "message", "role": "user", "content": "hi"}, extra,
                         {"type": "additional_tools"}], "tools": TOOLS}
        self.assertEqual([m["role"] for m in input_messages(req)], ["user"])
        tools, names, _ = request_tools(req)
        self.assertEqual([t["name"] for t in tools], ["exec_command", "later_tool"])
        self.assertIn("later_tool", names)

    def test_effort_and_text_format(self):
        self.assertEqual(template_kwargs({"reasoning": {"effort": "minimal"}}, {}), {"enable_thinking": False})
        self.assertEqual(template_kwargs({"reasoning": {"effort": "high"}}, {}), {"reasoning_effort": "xhigh"})
        self.assertEqual(template_kwargs({"reasoning": {"summary": "auto"}}, {}), {})      # Codex: the default
        self.assertEqual(template_kwargs({}, {"reasoning_effort": "low"}), {"reasoning_effort": "low"})
        with self.assertRaises(ResponsesError):
            template_kwargs({"reasoning": {"effort": "huge"}}, {})
        self.assertEqual(text_format({"text": {"format": {"type": "json_schema", "name": "n", "schema": {}}}}),
                         {"type": "json_schema", "json_schema": {"name": "n", "schema": {}}})
        self.assertIsNone(text_format({"text": {"format": {"type": "text"}}}))


# ------------------------------------------------------------------------------------------------ Codex compaction
HISTORY = [{"role": "user", "content": "hi"},
           {"type": "function_call", "name": "exec_command", "arguments": "{}", "call_id": "c1"},
           {"type": "function_call_output", "call_id": "c1", "output": "ok"}]
COMPACT_PROMPT = "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary."   # Codex's default
OTHER_TOOLS = [{"type": "function", "name": "read_file", "description": "Reads a file.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}]
A, B = ("019a-a", "019a-a"), ("019a-b", "019a-b")    # (session_id, thread_id) of two root conversations


def codex_meta(kind, conversation=A):
    """client_metadata as Codex 0.160 sends it: the turn metadata is a JSON string."""
    session, thread = conversation
    return {"session_id": session, "thread_id": thread, "x-codex-turn-metadata": json.dumps(
        {"session_id": session, "thread_id": thread, "request_kind": kind})}


def codex_request(kind, conversation=A, tools=None, last=COMPACT_PROMPT, meta=None):
    """A Codex request; a compaction one is the conversation again with `tools: []` (codex-rs compact.rs).  Every
    conversation sends the same prompt_cache_key here: it is not what tells them apart."""
    req = {"instructions": "Be brief", "tools": tools or [], "tool_choice": "auto", "prompt_cache_key": "same",
           "parallel_tool_calls": bool(tools), "input": HISTORY + ([{"role": "user", "content": last}]
                                                                    if kind == "compaction" else [])}
    req["client_metadata"] = codex_meta(kind, conversation) if meta is None else meta
    return req


def tools_of(req):
    """(the request's tools, the prompt's tools), as _responses_prepare gets them (the prompt then goes out)."""
    tools = request_tools(req)[0]
    shown = prompt_tools(req, tools)
    prompt_made(req, shown)
    return tools, shown


class CodexCompaction(unittest.TestCase):
    def setUp(self):
        responses._kept_prompt_tools = (None, None)

    def test_a_compaction_prompt_starts_as_its_conversations_last_prompt(self):
        turn = codex_request("turn", tools=TOOLS)
        before = TEMPLATE.render(input_messages(turn), tools=tools_of(turn)[1], add_generation_prompt=False)
        req = codex_request("compaction")
        own, shown = tools_of(req)
        self.assertIsNone(own)                           # the request's tools: still none
        self.assertEqual(shown, request_tools(turn)[0])
        self.assertTrue(TEMPLATE.render(input_messages(req), tools=shown).startswith(before))
        self.assertFalse(TEMPLATE.render(input_messages(req), tools=own).startswith(before))   # what it was
        shown.clear()                                    # the kept list is nobody's to change
        self.assertEqual(tools_of(req)[1], request_tools(turn)[0])
        self.assertEqual(tools_of(req)[1], request_tools(turn)[0])   # a compaction leaves the entry as it was

    def test_another_conversation_never_gets_these_tools(self):
        tools_of(codex_request("turn", A, TOOLS))
        b = tools_of(codex_request("turn", B, OTHER_TOOLS))[1]
        self.assertIsNone(tools_of(codex_request("compaction", A))[1])    # one entry, B's: A reads again
        self.assertEqual(tools_of(codex_request("compaction", B))[1], b)
        for other in (("019a-b", "019a-sub"),            # a sub-agent: the root's session, its own thread
                      ("019a-fork", "019a-fork")):        # a fork, or an ephemeral fork (same prompt_cache_key)
            self.assertIsNone(tools_of(codex_request("compaction", other))[1])
        self.assertEqual(tools_of(codex_request("compaction", B))[1], b)  # and they changed nothing
        tools_of(codex_request("turn", A, TOOLS))
        self.assertIsNone(tools_of(codex_request("compaction", B))[1])    # A's turn came last: B reads again
        self.assertEqual(tools_of(codex_request("compaction", A))[1], request_tools({"tools": TOOLS})[0])

    def test_requests_that_are_not_a_codex_compaction_are_rendered_as_sent(self):
        tools_of(codex_request("turn", tools=TOOLS))
        self.assertEqual(tools_of({"input": "another chat"}), (None, None))
        self.assertEqual(tools_of({"input": COMPACT_PROMPT, "prompt_cache_key": "same"}), (None, None))
        self.assertEqual(tools_of(codex_request("compaction", meta={})), (None, None))
        self.assertEqual(tools_of(codex_request("compaction"))[1], request_tools({"tools": TOOLS})[0])
        self.assertEqual(tools_of(codex_request("turn")), (None, None))   # a turn of A without tools ...
        self.assertIsNone(tools_of(codex_request("compaction"))[1])        # ... is what its compaction follows

    def test_the_compact_prompt_in_a_message_is_not_a_compaction(self):
        tools_of(codex_request("turn", tools=TOOLS))
        quoted = {"input": HISTORY + [{"role": "user", "content": COMPACT_PROMPT}], "tools": []}
        self.assertEqual(tools_of(quoted), (None, None))  # the compact prompt's words, without Codex saying so
        turn = codex_request("turn", last=COMPACT_PROMPT)
        turn["input"] = HISTORY + [{"role": "user", "content": "Explain this: " + COMPACT_PROMPT}]
        self.assertEqual(tools_of(turn), (None, None))   # Codex says it is a turn: as sent

    def test_a_thread_title_turn_does_not_replace_the_kept_tools(self):
        tools_of(codex_request("turn", tools=TOOLS))
        kept = responses._kept_prompt_tools
        title = codex_request("turn", ("019a-title", "019a-title"))
        meta = json.loads(title["client_metadata"]["x-codex-turn-metadata"])
        meta["thread_source"] = "thread_title"
        title["client_metadata"]["x-codex-turn-metadata"] = json.dumps(meta)
        prompt_made(title, None)
        self.assertEqual(responses._kept_prompt_tools, kept)
        self.assertEqual(tools_of(codex_request("compaction"))[1], request_tools({"tools": TOOLS})[0])

    def test_a_custom_compact_prompt_is_a_compaction_when_codex_says_so(self):
        first = tools_of(codex_request("turn", tools=TOOLS))[1]
        self.assertEqual(tools_of(codex_request("compaction", last="Summarize it my way."))[1], first)

    def test_a_compaction_that_sends_tools_keeps_them(self):
        tools_of(codex_request("turn", tools=TOOLS))
        own, shown = tools_of(codex_request("compaction", tools=OTHER_TOOLS))
        self.assertEqual(shown, own)

    def test_without_its_conversation_nothing_is_kept_or_used(self):
        for meta in ({}, {"session_id": "019a-a", "thread_id": "019a-a"},     # only the flat keys
                     {"x-codex-turn-metadata": json.dumps({"request_kind": "turn"})},
                     {"x-codex-turn-metadata": json.dumps({"request_kind": "turn", "session_id": "019a-a"})},
                     {"x-codex-turn-metadata": json.dumps({"request_kind": "turn", "thread_id": "019a-a"})}):
            tools_of(codex_request("turn", tools=TOOLS, meta=meta))
            self.assertEqual(responses._kept_prompt_tools, (None, None))
        tools_of(codex_request("turn", tools=TOOLS))
        kept = responses._kept_prompt_tools
        compaction = {"x-codex-turn-metadata": json.dumps({"request_kind": "compaction"})}
        self.assertEqual(tools_of(codex_request("compaction", meta=compaction)), (None, None))
        self.assertEqual(responses._kept_prompt_tools, kept)

    def test_malformed_metadata_is_no_error_and_no_reuse(self):
        first = tools_of(codex_request("turn", tools=TOOLS))[1]
        for meta in ("x", ["x"], {"x-codex-turn-metadata": None}, {"x-codex-turn-metadata": "{not json"},
                     {"x-codex-turn-metadata": "[]"}, {"x-codex-turn-metadata": "\"compaction\""},
                     {"x-codex-turn-metadata": {"request_kind": "compaction", "session_id": "019a-a",
                                                "thread_id": "019a-a"}},          # an object: not what Codex sends
                     {"x-codex-turn-metadata": json.dumps({"request_kind": "compaction", "session_id": 1,
                                                           "thread_id": 1})},
                     {"x-codex-turn-metadata": json.dumps({"request_kind": "compaction", "session_id": "",
                                                           "thread_id": ""})}):
            self.assertEqual(tools_of(codex_request("compaction", meta=meta)), (None, None), meta)
        self.assertEqual(tools_of(codex_request("compaction"))[1], first)   # A's entry was not touched
        odd = {"x-codex-turn-metadata": json.dumps({"request_kind": ["compaction"], "session_id": "019a-a",
                                                    "thread_id": "019a-a"})}
        self.assertEqual(tools_of(codex_request("compaction", meta=odd)), (None, None))   # not a compaction: as sent


class NoLeakedSampler(unittest.TestCase):
    """Test isolation: a stopped server stops its hardware sampler.  Every Service these tests started used to leave one
    running (21 after this file), polling NVML and psutil every second, which skewed serve/test_parallel's timings."""

    def test_server_close_ends_the_sampler(self):
        import time
        tok = ByteTokenizer()
        httpd = serve(Service(MockEngine(tok, ANSWER, max_context=16384), tok, TEMPLATE), port=0)
        sampler = getattr(getattr(httpd, "svc", None), "telemetry", None)
        self.assertIsNotNone(sampler, "the server did not initialize telemetry")
        thread = getattr(sampler, "_thread", None)
        self.assertIsNotNone(thread, "the server did not start its hardware sampler thread")
        self.assertTrue(thread.is_alive(), "the hardware sampler thread died immediately")
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive(), "a stopped server left its hardware sampler running")


# ------------------------------------------------------------------------------------------------ over HTTP
class Server(unittest.TestCase):
    script = ANSWER

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, self.script, max_context=16384)
        self.svc = Service(self.engine, self.tok, TEMPLATE)
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, body, path="/v1/responses", headers=None):
        """(status, parsed body: a dict, or the list of SSE events of a stream)."""
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                c.request("POST", path, body=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json", **(headers or {})})
                r = c.getresponse()
                raw = r.read().decode()
            if r.getheader("Content-Type") == "text/event-stream":
                events = []
                for block in raw.split("\n\n"):
                    lines = block.strip().splitlines()
                    if not lines or lines[0].startswith(":"):
                        continue
                    name = lines[0][len("event: "):]
                    data = json.loads(lines[1][len("data: "):])
                    self.assertEqual(name, data["type"])
                    events.append(data)
                return r.status, events
            return r.status, json.loads(raw)
        finally:
            c.close()



class OverHttp(Server):
    def test_non_streaming_text_with_reasoning(self):
        code, r = self.post({"model": "m", "input": "hi", "store": False})
        self.assertEqual(code, 200)
        self.assertEqual((r["object"], r["status"], r["error"]), ("response", "completed", None))
        rs, msg = r["output"]
        self.assertEqual(rs["type"], "reasoning")
        self.assertEqual(rs["content"], [{"type": "reasoning_text", "text": "Read it.\n"}])
        self.assertEqual(rs["summary"], [])
        self.assertNotIn("encrypted_content", rs)
        self.assertEqual(msg["type"], "message")
        self.assertEqual(msg["content"][0]["text"], "The file says before.")
        u = r["usage"]
        self.assertEqual(u["output_tokens"], len(self.tok.encode(ANSWER)) + 1)        # + the end-of-turn token
        self.assertEqual(u["output_tokens_details"]["reasoning_tokens"], len("Read it.\n</think>"))
        self.assertEqual(u["total_tokens"], u["input_tokens"] + u["output_tokens"])
        self.assertEqual(r["store"], False)

    def test_stream_event_order(self):
        code, events = self.post({"model": "m", "input": "hi", "stream": True,
                                  "include": ["reasoning.encrypted_content"]})
        self.assertEqual(code, 200)
        types = [e["type"] for e in events]
        deltas_r = types.count("response.reasoning_text.delta")
        deltas_t = types.count("response.output_text.delta")
        self.assertGreater(deltas_r, 0)
        self.assertGreater(deltas_t, 0)
        self.assertEqual(types, ["response.created", "response.in_progress",
                                 "response.output_item.added", *["response.reasoning_text.delta"] * deltas_r,
                                 "response.reasoning_text.done", "response.output_item.done",
                                 "response.output_item.added", "response.content_part.added",
                                 *["response.output_text.delta"] * deltas_t,
                                 "response.output_text.done", "response.content_part.done",
                                 "response.output_item.done", "response.completed"])
        self.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))
        final = events[-1]["response"]
        self.assertEqual(final["status"], "completed")
        rs, msg = final["output"]
        self.assertEqual([e["item"]["id"] for e in events if e["type"] == "response.output_item.done"],
                         [rs["id"], msg["id"]])
        self.assertEqual("".join(e["delta"] for e in events if e["type"] == "response.output_text.delta"),
                         msg["content"][0]["text"])
        self.assertTrue(rs["encrypted_content"].startswith(ENCRYPTED_PREFIX))
        # the encrypted_content alone brings the thinking back into the prompt
        replay = input_messages({"input": [{"role": "user", "content": "hi"},
                                           {"type": "reasoning", "summary": [], "encrypted_content":
                                            rs["encrypted_content"]}, msg]})
        self.assertEqual(replay[1]["reasoning_content"], "Read it.\n")

    def test_max_output_tokens_ends_incomplete(self):
        code, r = self.post({"model": "m", "input": "hi", "max_output_tokens": 5})
        self.assertEqual(code, 200)
        self.assertEqual(r["status"], "incomplete")
        self.assertEqual(r["incomplete_details"], {"reason": "max_output_tokens"})
        self.assertEqual(r["output"][0]["status"], "incomplete")
        code, events = self.post({"model": "m", "input": "hi", "max_output_tokens": 5, "stream": True})
        self.assertEqual(events[-1]["type"], "response.incomplete")

    def test_max_output_tokens_zero_and_minus_one_mean_no_cap(self):
        # #4036: the documented "no cap" values; the answer is complete, not cut at the cap
        for n in (0, -1):
            code, r = self.post({"model": "m", "input": "hi", "max_output_tokens": n})
            self.assertEqual((code, r["status"]), (200, "completed"), (n, r))
            self.assertIsNone(r["incomplete_details"])
            code, events = self.post({"model": "m", "input": "hi", "max_output_tokens": n, "stream": True})
            self.assertEqual((code, events[-1]["type"]), (200, "response.completed"), n)

    def test_max_output_tokens_bad_values_are_400(self):
        for n in (-2, -100, 1.5, True, "5"):
            code, r = self.post({"model": "m", "input": "hi", "max_output_tokens": n})
            self.assertEqual(code, 400, (n, r))
            self.assertEqual(r["error"]["param"], "max_output_tokens", n)

    def test_effort_none_does_not_think(self):
        self.engine.scripts = [self.tok.encode("Hello.<|im_end|>", parse_special=True)]
        self.engine.script = self.engine.scripts[0]
        code, r = self.post({"model": "m", "input": "hi", "reasoning": {"effort": "none"}})
        self.assertEqual([o["type"] for o in r["output"]], ["message"])
        self.assertIn("<think>\n\n</think>", self.tok.decode(self.engine.last_prompt))

    def test_errors_use_the_responses_format(self):
        for body, param in (({"model": "m"}, "input"), ({"model": "m", "input": "x", "previous_response_id": "r"},
                                                         "previous_response_id"),
                            ({"model": "m", "input": "x", "reasoning": {"effort": "huge"}}, "reasoning.effort")):
            code, r = self.post(body)
            self.assertEqual(code, 400)
            self.assertEqual(set(r["error"]), {"message", "type", "param", "code"})
            self.assertEqual(r["error"]["param"], param)
        code, r = self.post({}, path="/v1/responses/resp_123/cancel")
        self.assertEqual(code, 404)
        self.assertEqual(r["error"]["code"], "not_found")
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        c.request("POST", "/v1/responses", body=b"{not json", headers={"Content-Type": "application/json"})
        r = c.getresponse()
        err = json.loads(r.read())["error"]
        c.close()
        self.assertEqual((r.status, err["type"], err["param"]), (400, "invalid_request_error", None))

    def test_too_long_for_the_context(self):
        code, r = self.post({"model": "m", "input": "x" * 20000})
        self.assertEqual(code, 400)
        self.assertEqual(r["error"]["code"], "context_length_exceeded")

    def test_api_key_and_foreign_pages(self):
        self.svc.api_key = "s3cret"
        self.assertEqual(self.post({"model": "m", "input": "hi"})[0], 401)
        self.assertEqual(self.post({"model": "m", "input": "hi"}, headers={"Authorization": "Bearer s3cret"})[0], 200)
        self.svc.api_key = None
        code, r = self.post({"model": "m", "input": "hi"}, headers={"Origin": "https://evil.example.com"})
        self.assertEqual(code, 403)

    def test_a_foreign_page_cannot_name_a_file_as_an_image(self):
        """#553: the chat routes refuse a file image from another origin; so does /v1/responses."""
        def body(src):
            return {"model": "m", "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "what is it?"}, {"type": "input_image", "image_url": src}]}]}
        self.svc.cors_origins = ["*"]                  # a page cors_origins lets in may send JSON, not name a file
        for src in (r"\\host\share\x.png", "/etc/passwd", "file:///C:/x.png"):
            with self.subTest(src=src):
                code, r = self.post(body(src), headers={"Origin": "https://evil.example.com"})
                self.assertEqual(code, 400, r)
                self.assertIn("another origin", r["error"]["message"])
        # data: and http(s) are the page's own to send: not refused by this check (no vision here, so a 400 of another kind)
        for src in ("data:image/png;base64,iVBORw0KGgo=", "https://example.com/x.png"):
            code, r = self.post(body(src), headers={"Origin": "https://evil.example.com"})
            self.assertNotIn("another origin", json.dumps(r))

    def test_json_schema_text_format(self):
        self.engine.scripts = [self.tok.encode("</think>\n\n{\"n\": 3}<|im_end|>", parse_special=True)]
        self.engine.script = self.engine.scripts[0]
        fmt = {"type": "json_schema", "name": "num", "strict": True,
               "schema": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}}
        code, r = self.post({"model": "m", "input": "a number", "text": {"format": fmt}})
        self.assertEqual(code, 200, r)
        self.assertEqual(r["output"][-1]["content"][0]["text"], "{\"n\":3}")
        code, events = self.post({"model": "m", "input": "a number", "text": {"format": fmt}, "stream": True})
        self.assertEqual([e["delta"] for e in events if e["type"] == "response.output_text.delta"], ["{\"n\":3}"])
        self.assertEqual(events[-1]["type"], "response.completed")
        # an answer that fails the schema - only jsonschema can tell: it is optional (serve/structured.py), and
        # without it a json_schema answer is checked to be one JSON object, which {"n": 3} is
        try:
            import jsonschema  # noqa: F401
        except ImportError:
            return
        bad = {**fmt, "schema": {**fmt["schema"], "properties": {"n": {"type": "string"}}}}
        code, r = self.post({"model": "m", "input": "a number", "text": {"format": bad}})
        self.assertEqual((code, r["error"]["code"]), (502, "structured_output_failed"))
        code, events = self.post({"model": "m", "input": "a number", "text": {"format": bad}, "stream": True})
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertEqual(events[-1]["response"]["error"]["code"], "structured_output_failed")

    def test_monitor_and_status(self):
        self.svc.api_monitor = True
        self.post({"model": "m", "input": "hi", "stream": True})
        rec = self.svc.request_records()[0]
        self.assertEqual(rec["path"], "/v1/responses")
        full = self.svc.request_records(rec["id"])
        self.assertEqual(full["output"], "The file says before.")
        self.assertEqual(full["reasoning"], "Read it.\n")
        self.assertIn("/v1/responses", self.svc.v1_status()["dialects"])


class ToolRoundTrip(Server):
    script = [CALL, ANSWER]

    def test_function_call_round_trip(self):
        user = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "what is in a.txt?"}]}
        code, events = self.post({"model": "m", "instructions": "You are a coding agent.", "input": [user],
                                  "tools": TOOLS, "store": False, "stream": True,
                                  "include": ["reasoning.encrypted_content"], "reasoning": {"summary": "auto"},
                                  "parallel_tool_calls": True, "tool_choice": "auto"})
        self.assertEqual(code, 200)
        first_prompt = self.tok.decode(self.engine.last_prompt)
        types = [e["type"] for e in events]
        n_args = types.count("response.function_call_arguments.delta")
        self.assertGreater(n_args, 0)
        call_part = types[types.index("response.output_text.done") + 2:]
        self.assertEqual(call_part, ["response.output_item.done", "response.output_item.added",
                                     *["response.function_call_arguments.delta"] * n_args,
                                     "response.function_call_arguments.done", "response.output_item.done",
                                     "response.completed"])
        output = events[-1]["response"]["output"]
        self.assertEqual([o["type"] for o in output], ["reasoning", "message", "function_call"])
        fc = output[2]
        self.assertEqual((fc["name"], json.loads(fc["arguments"]), fc["status"]),
                         ("exec_command", {"cmd": "cat a.txt"}, "completed"))
        self.assertTrue(fc["call_id"])
        added = next(e for e in events if e["type"] == "response.output_item.added"
                     and e["item"]["type"] == "function_call")
        self.assertEqual((added["item"]["call_id"], added["item"]["arguments"]), (fc["call_id"], ""))
        # Codex sends everything back, plus the result: the prompt continues exactly where the model stopped
        code, r = self.post({"model": "m", "instructions": "You are a coding agent.", "tools": TOOLS, "store": False,
                             "input": [user, *output, {"type": "function_call_output", "call_id": fc["call_id"],
                                                       "output": "before"}]})
        self.assertEqual(code, 200, r)
        second_prompt = self.tok.decode(self.engine.last_prompt)
        self.assertTrue(second_prompt.startswith(first_prompt + CALL + "<|im_end|>"), second_prompt[-600:])
        self.assertIn("<tool_response>\nbefore\n</tool_response>", second_prompt)
        self.assertEqual(r["output"][-1]["content"][0]["text"], "The file says before.")

    def test_quoted_incomplete_wrapper_is_not_a_completed_call(self):
        text = "Format: " + CALL[CALL.index("<tool_call>"):].removesuffix("</tool_call>") + "and then I explain it."
        self.engine.scripts = [self.tok.encode("</think>\n\n" + text + "<|im_end|>", parse_special=True)]
        self.engine.script = self.engine.scripts[0]
        for stream in (False, True):
            code, result = self.post({"model": "m", "input": "explain the format", "tools": TOOLS, "stream": stream})
            self.assertEqual(code, 200, result)
            final = result[-1]["response"] if stream else result
            calls = [item for item in final["output"] if item["type"] == "function_call"]
            self.assertFalse(any(item["status"] == "completed" for item in calls))
            for call in calls:
                with self.assertRaises(ValueError):
                    json.loads(call["arguments"])
            self.assertEqual("".join(item["content"][0]["text"] for item in final["output"]
                                     if item["type"] == "message"), text)

    def test_namespace_call_written_with_double_underscore(self):
        # Qwen writes Codex's MCP tools in their flat `mcp__server__tool` form; Codex only
        # runs the call when it comes back with its namespace.
        self.engine.scripts = [self.tok.encode(
            "</think>\n\n<tool_call>\n<function=mcp__websearch__web_search>\n<parameter=query>\nstrata\n"
            "</parameter>\n</function>\n</tool_call><|im_end|>", parse_special=True)]
        self.engine.script = self.engine.scripts[0]
        tools = [{"type": "namespace", "name": "mcp__websearch", "description": "Search.", "tools": [
            {"type": "function", "name": "web_search",
             "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}]}]
        code, r = self.post({"model": "m", "input": "go", "tools": tools})
        self.assertEqual(code, 200, r)
        call, = r["output"]
        self.assertEqual((call["namespace"], call["name"], json.loads(call["arguments"])),
                         ("mcp__websearch", "web_search", {"query": "strata"}))

        # Two canonical names share this alias. It must not invoke the first namespace.
        self.engine.scripts = [self.tok.encode(
            "</think>\n\n<tool_call>\n<function=a__b__c>\n</function>\n</tool_call><|im_end|>",
            parse_special=True)]
        self.engine.script = self.engine.scripts[0]
        tools = [{"type": "namespace", "name": "a", "tools": [{"type": "function", "name": "b__c"}]},
                 {"type": "namespace", "name": "a__b", "tools": [{"type": "function", "name": "c"}]}]
        for stream in (False, True):
            code, r = self.post({"model": "m", "input": "go", "tools": tools, "stream": stream})
            self.assertEqual(code, 200, r)
            output = r[-1]["response"]["output"] if stream else r["output"]
            # not an alias of either: with tools declared (the 0.2.2 rule, #3075) a call for an undeclared name stays text
            self.assertFalse([item for item in output if item.get("type") == "function_call"], output)
            self.assertIn("a__b__c", json.dumps(output))

        # A declared exact name wins over alias spellings and keeps its string schema.
        tools.append({"type": "function", "name": "a__b__c", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}}}})
        self.engine.scripts = [self.tok.encode(
            "</think>\n\n<tool_call>\n<function=a__b__c>\n<parameter=path>\n123\n</parameter>\n"
            "</function>\n</tool_call><|im_end|>", parse_special=True)]
        self.engine.script = self.engine.scripts[0]
        for stream in (False, True):
            code, r = self.post({"model": "m", "input": "go", "tools": tools, "stream": stream})
            self.assertEqual(code, 200, r)
            output = r[-1]["response"]["output"] if stream else r["output"]
            call, = output
            self.assertNotIn("namespace", call)
            self.assertEqual((call["name"], json.loads(call["arguments"])), ("a__b__c", {"path": "123"}))

    def test_namespace_and_custom_calls_come_back_under_their_own_names(self):
        self.engine.scripts = [self.tok.encode(
            "</think>\n\n<tool_call>\n<function=multi_agent_v1.spawn_agent>\n<parameter=message>\nhi\n</parameter>\n"
            "</function>\n</tool_call>\n<tool_call>\n<function=apply_patch>\n<parameter=input>\n*** Begin Patch\n"
            "</parameter>\n</function>\n</tool_call><|im_end|>", parse_special=True)]
        self.engine.script = self.engine.scripts[0]
        tools = [{"type": "namespace", "name": "multi_agent_v1", "description": "Agents.", "tools": [
            {"type": "function", "name": "spawn_agent", "strict": False,
             "parameters": {"type": "object", "properties": {"message": {"type": "string"}}}}]},
            {"type": "custom", "name": "apply_patch", "description": "Edit files."}]
        code, r = self.post({"model": "m", "input": "go", "tools": tools})
        self.assertEqual(code, 200, r)
        ns, custom = r["output"]
        self.assertEqual((ns["type"], ns["namespace"], ns["name"], json.loads(ns["arguments"])),
                         ("function_call", "multi_agent_v1", "spawn_agent", {"message": "hi"}))
        self.assertEqual((custom["type"], custom["name"], custom["input"]),
                         ("custom_tool_call", "apply_patch", "*** Begin Patch"))
        msgs = input_messages({"input": [{"role": "user", "content": "go"}, ns, custom,
                                         {"type": "function_call_output", "call_id": ns["call_id"], "output": "1"},
                                         {"type": "custom_tool_call_output", "call_id": custom["call_id"],
                                          "output": "2"}]})
        self.assertEqual([c["function"]["name"] for c in msgs[1]["tool_calls"]],
                         ["multi_agent_v1.spawn_agent", "apply_patch"])


class JsonFormatWithTools(Server):
    """#782: Codex's review request carries an `additional_tools` item, tools, and `text.format` json_schema together.
    A tool call passes through; the schema is checked on the answer that ends in text."""
    FMT = {"type": "json_schema", "name": "guardian_review", "strict": True,
           "schema": {"type": "object", "properties": {"outcome": {"type": "string"}, "rationale": {"type": "string"}},
                      "required": ["outcome", "rationale"], "additionalProperties": False}}

    def request(self, **extra):
        user = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "review this call"}]}
        more = {"type": "additional_tools", "role": "developer", "tools": [
            {"type": "function", "name": "later_tool", "parameters": {"type": "object"}}]}
        return {"model": "m", "instructions": "You are a reviewer.", "input": [more, user], "tools": TOOLS,
                "tool_choice": "auto", "parallel_tool_calls": False, "store": False,
                "text": {"format": self.FMT}, **extra}

    def script_with(self, text):
        self.engine.scripts = [self.tok.encode(text, parse_special=True)]
        self.engine.script = self.engine.scripts[0]

    def test_a_tool_call_passes_through(self):
        self.script_with(CALL + "<|im_end|>")
        for stream in (False, True):
            code, r = self.post(self.request(stream=stream))
            self.assertEqual(code, 200, r)
            final = r if not stream else r[-1]["response"]
            self.assertEqual(final["status"], "completed")
            self.assertEqual([o["type"] for o in final["output"]][-2:], ["message", "function_call"])
            call, = [o for o in final["output"] if o["type"] == "function_call"]
            self.assertEqual((call["type"], call["name"], json.loads(call["arguments"])),
                             ("function_call", "exec_command", {"cmd": "cat a.txt"}))
        # the model is told the schema is for its final answer, not for a turn that calls a tool
        self.assertIn("applies only to your final answer", self.tok.decode(self.engine.last_prompt))

    def test_the_final_text_answer_is_the_schema_object(self):
        self.script_with('</think>\n\n{"outcome": "allow", "rationale": "read only"}<|im_end|>')
        code, r = self.post(self.request())
        self.assertEqual(code, 200, r)
        self.assertEqual(json.loads(r["output"][-1]["content"][0]["text"]),
                         {"outcome": "allow", "rationale": "read only"})
        code, events = self.post(self.request(stream=True))
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(len([e for e in events if e["type"] == "response.output_text.delta"]), 1)

    def test_a_text_answer_that_breaks_the_schema_fails_as_without_tools(self):
        try:
            import jsonschema  # noqa: F401
        except ImportError:
            self.skipTest("the schema check needs jsonschema")
        self.script_with('</think>\n\n{"outcome": 3}<|im_end|>')
        code, r = self.post(self.request())
        self.assertEqual((code, r["error"]["code"]), (502, "structured_output_failed"))


class ThreadTitle(Server):
    def test_a_thread_title_is_answered_without_the_engine(self):
        self.svc.codex_thread_titles = True                      # opt-in: "codex_thread_titles": true in the config
        user = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "what is in a.txt?"}]}
        code, r = self.post({"model": "m", "instructions": "You are a coding agent.", "input": [user], "tools": TOOLS})
        self.assertEqual(code, 200, r)
        seen = list(self.engine.last_prompt)
        self.assertTrue(seen)
        meta = {"session_id": "019a-title", "thread_id": "019a-title", "request_kind": "turn",
                "thread_source": "thread_title"}
        code, r = self.post({
            "model": "m", "instructions": "You are Codex", "tools": [],
            "client_metadata": {"x-codex-turn-metadata": json.dumps(meta)},
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "Generate a concise title.\n\nUser prompt:\n只回四個字然後停：傾印測試"}]}]})
        self.assertEqual(code, 200, r)
        self.assertEqual(self.engine.last_prompt, seen)
        self.assertEqual(json.loads(r["output"][0]["content"][0]["text"]), {"title": "只回四個字然後停：傾印測試"})
        long = {"model": "m", "instructions": "You are Codex", "tools": [],
                "client_metadata": {"x-codex-turn-metadata": json.dumps(meta)},
                "input": "User prompt:\n" + ("甲" * 40) + "。"}
        self.assertEqual(json.loads(collect(thread_title_events(long, "m"))["output"][0]["content"][0]["text"]),
                         {"title": "甲" * 36})

    def test_off_by_default_the_engine_answers(self):
        meta = {"thread_source": "thread_title"}
        code, r = self.post({"model": "m", "instructions": "You are Codex", "tools": [],
                             "client_metadata": {"x-codex-turn-metadata": json.dumps(meta)},
                             "input": "User prompt:" + chr(10) + "hello"})
        self.assertEqual(code, 200, r)
        self.assertTrue(self.engine.last_prompt)                  # it ran on the engine
        self.assertNotIn('"title"', json.dumps(r["output"]))


class CodexCompactionOverHttp(Server):
    # the compaction answer is tool calls the conversation's tools would make a namespace call and a custom call
    script = [CALL, "</think>\n\n<tool_call>\n<function=multi_agent_v1.spawn_agent>\n<parameter=message>\nhi\n"
                    "</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=apply_patch>\n<parameter=input>\n"
                    "*** Begin Patch\n</parameter>\n</function>\n</tool_call>"]
    tools = TOOLS + [{"type": "namespace", "name": "multi_agent_v1", "description": "Agents.", "tools": [
        {"type": "function", "name": "spawn_agent", "parameters": {"type": "object", "properties": {
            "message": {"type": "string"}}}}]}, {"type": "custom", "name": "apply_patch", "description": "Edit files."}]

    def setUp(self):
        super().setUp()
        responses._kept_prompt_tools = (None, None)
        self.svc.codex_compaction_cache = True       # opt-in (#924)
        self.parser_tools = []
        run = self.svc.run

        def recorded(ids, thinking, tools, *rest):
            self.parser_tools.append(tools)
            return run(ids, thinking, tools, *rest)
        self.svc.run = recorded

    def compaction(self, meta, output="before", between=None, **extra):
        """Codex's turn, (`between`,) then its compaction request: (the compaction prompt, its status and response)."""
        self.engine.turns, self.engine.script = 0, self.engine.scripts[0]
        user = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "what is in a.txt?"}]}
        turn = {"model": "m", "instructions": "You are a coding agent.", "input": [user], "tools": self.tools,
                "include": ["reasoning.encrypted_content"], "client_metadata": codex_meta("turn")}
        code, r = self.post(turn)
        self.assertEqual(code, 200, r)
        self.first_prompt = self.tok.decode(self.engine.last_prompt)
        call = r["output"][-1]
        if between:
            between()
        code, r = self.post({**turn, "tools": [], "parallel_tool_calls": False, "client_metadata": meta, "input": [
            user, *r["output"], {"type": "function_call_output", "call_id": call["call_id"], "output": output},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": COMPACT_PROMPT}]}],
            **extra})
        return self.tok.decode(self.engine.last_prompt), code, r

    def test_the_kept_tools_shape_only_the_prompt(self):
        prompt, code, r = self.compaction(codex_meta("compaction"))
        self.assertEqual(code, 200, r)
        self.assertTrue(prompt.startswith(self.first_prompt + CALL + "<|im_end|>"))   # the cached conversation
        self.assertIn("multi_agent_v1.spawn_agent", prompt[:prompt.index("<|im_end|>")])
        self.assertEqual(self.parser_tools[-1], None)    # the parser gets the request's tools: none
        self.assertEqual(r["tools"], [])
        # the calls come back as for any request without tools: no namespace, no custom tool
        outputs = [(o["type"], o["name"], o.get("namespace")) for o in r["output"]]
        self.assertEqual(outputs, [("function_call", "multi_agent_v1.spawn_agent", None),
                                   ("function_call", "apply_patch", None)])
        # the same request without Codex's metadata (upstream's behaviour): the same answer, but its prompt starts
        # differently, so the engine reads it from the start
        plain, code, r2 = self.compaction({})
        self.assertEqual(code, 200, r2)
        self.assertFalse(plain.startswith(self.first_prompt))
        self.assertNotIn("multi_agent_v1.spawn_agent", plain[:plain.index("<|im_end|>")])
        self.assertEqual(self.parser_tools[-1], None)
        self.assertEqual([(o["type"], o["name"], o.get("namespace")) for o in r2["output"]], outputs)

    def test_a_compaction_too_long_with_the_kept_tools_is_rendered_as_sent(self):
        long = "before\n" * 300                         # a long tool output: the turn fits, its compaction barely
        plain, code, r = self.compaction({}, long, max_output_tokens=64)
        self.assertEqual(code, 200, r)
        fits = len(self.engine.last_prompt) + 64 + 8     # the request as sent, its output, CTX_SLACK
        shown, code, r = self.compaction(codex_meta("compaction"), long, max_output_tokens=64)
        self.assertGreater(len(self.engine.last_prompt) + 64 + 8, fits)   # with the kept tools it would not fit
        self.engine.max_context = fits
        again, code, r = self.compaction(codex_meta("compaction"), long, max_output_tokens=64)
        self.assertEqual(code, 200, r)                   # as without the kept tools, not a context error
        self.assertEqual(again, plain)
        self.assertEqual(self.parser_tools[-1], None)


    def test_a_request_that_fails_leaves_the_kept_tools_as_they_were(self):
        before, code, r = self.compaction(codex_meta("compaction"))
        self.assertEqual(code, 200, r)

        def fails():                                     # B's turn is too long: its prompt never reaches the engine
            code, r = self.post({"model": "m", "input": "x" * 20000, "tools": TOOLS,
                                 "client_metadata": codex_meta("turn", B)})
            self.assertEqual((code, r["error"]["code"]), (400, "context_length_exceeded"))
        again, code, r = self.compaction(codex_meta("compaction"), between=fails)
        self.assertEqual(code, 200, r)
        self.assertEqual(again, before)                  # so A's compaction still starts as A's turn


if __name__ == "__main__":
    unittest.main()
