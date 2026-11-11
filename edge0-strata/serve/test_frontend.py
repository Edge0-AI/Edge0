"""Chat history normalization before rendering the model's prompt."""
import copy
from pathlib import Path
import unittest

from serve.frontend import ChatTemplate, anthropic_to_messages, openai_to_messages, require_messages
from serve.server import ByteTokenizer, MockEngine, Service


class EmptyAssistantHistory(unittest.TestCase):
    def setUp(self):
        self.template = ChatTemplate(Path(__file__).with_name("chat_template.jinja"))
        tok = ByteTokenizer()
        self.service = Service(MockEngine(tok, "ok"), tok, self.template)

    def test_openai_empty_replies_are_not_rendered(self):
        for content in (None, "", " \n\t", [], [{"type": "text", "text": ""}]):
            for preserve in (True, False):
                with self.subTest(content=content, preserve_thinking=preserve):
                    history = []
                    for _ in range(3):
                        history.extend([{"role": "user", "content": "haz un ls"},
                                        {"role": "assistant", "content": content,
                                         "reasoning_content": "Let me run ls."}])
                    history.append({"role": "user", "content": "haz un ls"})
                    messages, tools, kwargs = openai_to_messages({"messages": history})
                    kwargs["preserve_thinking"] = preserve
                    clean = [m for m in messages if m["role"] == "user"]
                    self.assertEqual(self.service.encode_prompt(messages, tools, kwargs),
                                     self.service.encode_prompt(clean, tools, kwargs))

    def test_anthropic_thinking_only_replies_are_not_rendered(self):
        messages, tools, kwargs = anthropic_to_messages({"messages": [
            {"role": "user", "content": "haz un ls"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "Let me run ls."}]},
            {"role": "user", "content": "haz un ls"},
        ]})
        self.assertEqual(self.service.encode_prompt(messages, tools, kwargs),
                         self.service.encode_prompt([messages[0], messages[2]], tools, kwargs))

    def test_text_and_tool_calls_are_kept(self):
        messages, tools, kwargs = openai_to_messages({"messages": [
            {"role": "user", "content": "haz un ls"},
            {"role": "assistant", "content": [{"type": "text", "text": "Running ls."}]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {"name": "run", "arguments": '{"command":"ls"}'}},
            ]},
            {"role": "tool", "content": "file.txt"},
            {"role": "user", "content": "thanks"},
        ]})
        before = copy.deepcopy(messages)
        prompt = self.template.render(messages, tools=tools, **kwargs)
        self.assertIn("Running ls.", prompt)
        self.assertIn("<function=run>", prompt)
        self.assertIn("file.txt", prompt)
        self.assertEqual(messages, before)

    def test_image_only_assistant_turn_is_kept(self):
        messages, tools, kwargs = openai_to_messages({"messages": [
            {"role": "user", "content": "describe the image"},
            {"role": "assistant", "content": [{"type": "image_url", "image_url": "image.png"}]},
            {"role": "user", "content": "thanks"},
        ]})
        self.assertIn("<|vision_start|><|image_pad|><|vision_end|>",
                      self.template.render(messages, tools=tools, **kwargs))

    def test_final_empty_assistant_turn_is_kept(self):
        prompt = self.template.render([{"role": "user", "content": "hi"},
                                       {"role": "assistant", "content": ""}], add_generation_prompt=False)
        self.assertIn("<|im_start|>assistant\n", prompt)

    def test_empty_user_and_tool_turns_are_kept(self):
        prompt = self.template.render([{"role": "user", "content": ""},
                                       {"role": "tool", "content": ""},
                                       {"role": "user", "content": "hi"}])
        self.assertEqual(prompt.count("<|im_start|>user\n"), 3)
        self.assertIn("<tool_response>\n\n</tool_response>", prompt)


class UnreadContentParts(unittest.TestCase):
    """A content part the server does not read is refused naming it and where it sits, as /v1/responses does; chat
    and messages used to drop it without a word, and the model answered about a file it had never seen."""

    def setUp(self):
        self.template = ChatTemplate(Path(__file__).with_name("chat_template.jinja"))
        tok = ByteTokenizer()
        self.service = Service(MockEngine(tok, "ok"), tok, self.template)

    def test_chat_file_and_audio_parts_are_left_out(self):
        parts = ({"type": "file", "file": {"filename": "a.pdf", "file_data": "data:application/pdf;base64,AA"}},
                 {"type": "input_audio", "input_audio": {"data": "AA", "format": "wav"}})
        plain = [{"role": "user", "content": "summarize"}, {"role": "user", "content": "this one"}]
        for part in parts:
            with self.subTest(kind=part["type"]):
                got = openai_to_messages({"messages": [
                    {"role": "user", "content": "summarize"},
                    {"role": "user", "content": [{"type": "text", "text": "this one"}, part]}]})
                want = openai_to_messages({"messages": [
                    {"role": "user", "content": "summarize"},
                    {"role": "user", "content": [{"type": "text", "text": "this one"}]}]})
                self.assertEqual(got, want)

    def test_messages_blocks_real_clients_send_are_left_out(self):
        # tool_reference (Claude Code's tool search), documents, search results, server tool blocks, signed thinking
        extra = [{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "AA"}},
                 {"type": "search_result", "source": "s", "title": "t", "content": [{"type": "text", "text": "x"}]},
                 {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {}},
                 {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []},
                 {"type": "container_upload", "file_id": "f"}]
        for block in extra:
            with self.subTest(kind=block["type"]):
                got = anthropic_to_messages({"messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [{"type": "text", "text": "ok"}, block]},
                    {"role": "user", "content": [{"type": "text", "text": "go"}, block]}]})
                want = anthropic_to_messages({"messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                    {"role": "user", "content": [{"type": "text", "text": "go"}]}]})
                self.assertEqual(got, want)

    def test_messages_tool_reference_in_a_tool_result_is_left_out(self):
        def run(extra):
            return anthropic_to_messages({"messages": [
                {"role": "user", "content": "find a tool"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "ToolSearch", "input": {}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "found"}] + extra}]}]})
        self.assertEqual(run([{"type": "tool_reference", "tool_name": "mcp__x"}]), run([]))

    def test_a_system_prompt_with_an_image_part_is_read_as_text_as_before(self):
        got = openai_to_messages({"messages": [
            {"role": "system", "content": [{"type": "text", "text": "be brief"},
                                           {"type": "image_url", "image_url": "image.png"}]},
            {"role": "user", "content": "hi"}]})
        self.assertEqual(got[0][0]["content"], "be brief")

    def test_messages_redacted_thinking_is_dropped(self):
        msgs = {"messages": [
            {"role": "user", "content": "haz un ls"},
            {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "ZW50cnk="},
                                              {"type": "text", "text": "Running ls."}]},
            {"role": "user", "content": "thanks"}]}
        without = copy.deepcopy(msgs)
        del without["messages"][1]["content"][0]
        with_block = anthropic_to_messages(msgs)
        without_block = anthropic_to_messages(without)
        self.assertEqual(with_block[0], without_block[0])
        self.assertEqual(self.service.encode_prompt(with_block[0], with_block[1], with_block[2]),
                         self.service.encode_prompt(without_block[0], without_block[1], without_block[2]))

    def test_chat_refusal_part_is_read_as_its_text(self):
        messages, tools, kwargs = openai_to_messages({"messages": [
            {"role": "user", "content": "do a thing"},
            {"role": "assistant", "content": [{"type": "refusal", "refusal": "I cannot do that."}]},
            {"role": "user", "content": "ok"}]})
        self.assertEqual(messages[1]["content"], "I cannot do that.")
        self.assertIn("I cannot do that.", self.template.render(messages, tools=tools, **kwargs))

    def test_text_and_image_requests_render_as_before(self):
        parts = [{"role": "user", "content": [{"type": "text", "text": "haz un ls"}]},
                 {"role": "assistant", "content": [{"type": "text", "text": "file.txt"}]}]
        plain = [{"role": "user", "content": "haz un ls"}, {"role": "assistant", "content": "file.txt"}]
        for convert in (openai_to_messages, anthropic_to_messages):
            with self.subTest(convert=convert.__name__):
                self.assertEqual(self.service.encode_prompt(*convert({"messages": parts})),
                                 self.service.encode_prompt(*convert({"messages": plain})))
        chat = openai_to_messages({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "describe"},
                                         {"type": "image_url", "image_url": "image.png"}]}]})
        self.assertIn("<|vision_start|><|image_pad|><|vision_end|>",
                      self.template.render(chat[0], tools=chat[1], **chat[2]))
        msgs = anthropic_to_messages({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "describe"},
                                         {"type": "image", "source": {"type": "url", "url": "image.png"}}]}]})
        self.assertIn("<|vision_start|><|image_pad|><|vision_end|>",
                      self.template.render(msgs[0], tools=msgs[1], **msgs[2]))


if __name__ == "__main__":
    unittest.main()


class EmptyTurnsSwitch(unittest.TestCase):
    def test_keep_empty_turns_restores_the_old_prompt(self):
        import os
        from unittest import mock
        template = ChatTemplate(Path(__file__).with_name("chat_template.jinja"))
        msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": ""},
                {"role": "user", "content": "b"}]
        clean = template.render([msgs[0], msgs[2]])
        self.assertEqual(template.render(msgs), clean)
        with mock.patch.dict(os.environ, {"STRATA_KEEP_EMPTY_TURNS": "1"}):
            self.assertNotEqual(template.render(msgs), clean)


class ForcedCallOpening(unittest.TestCase):
    def test_required_with_one_tool_names_it(self):
        from serve.frontend import forced_call
        one, two = [{"name": "a"}], [{"name": "a"}, {"name": "b"}]
        self.assertTrue(forced_call("required", one).endswith("<function=a>\n"))
        self.assertTrue(forced_call("required", two).endswith("<function="))
        self.assertIsNone(forced_call("auto", one))


class QuotedToolCallCloser(unittest.TestCase):
    """#2026: a `</tool_call>` a user quotes is text, not the end-of-call token (type 4: matched even without
    parse_special); an assistant message replaying raw call text keeps the model's own tokens."""

    def setUp(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import strata_tokenizer as ST
        b2u = ST.bytes_to_unicode()
        self.tool_tags = ["</tool_call>", "<tool_call>", "<tool_response>", "</tool_response>"]
        self.tok = ST.Tokenizer([b2u[b] for b in range(256)] + self.tool_tags + ["<|im_end|>"], [],
                                [1] * 256 + [4] * len(self.tool_tags) + [3])
        self.tags = __import__("serve.frontend", fromlist=["literal_tags"]).literal_tags(self.tok.control_tokens)

    def encode(self, role, text):
        from serve.frontend import mark_think_literals, unmark_think_literals
        marked, _, changed = mark_think_literals([{"role": role, "content": text}], None, self.tags)
        content = marked[0]["content"]
        if isinstance(content, list):
            content = "".join(x["text"] for x in content)
        prompt, plain = unmark_think_literals(content, self.tags)
        return self.tok.encode(prompt, parse_special=True, plain=plain), changed

    def special(self, tag):
        return self.tok.encode(tag, parse_special=True)

    def test_every_tool_tag_a_user_quotes_stays_text(self):
        # 0.1.44: <tool_call>, <tool_response> and </tool_response> are type 4 like </tool_call> (47Hunter47, #2026)
        for role in ("user", "system", "tool"):
            for tag in self.tool_tags:
                with self.subTest(role=role, tag=tag):
                    text = f"output verbatim: {tag}."
                    self.assertEqual(len(self.special(tag)), 1)
                    ids, changed = self.encode(role, text)
                    self.assertTrue(changed)
                    self.assertEqual(ids, [*text.encode()])

    def test_assistant_replay_keeps_every_tool_tag(self):
        text = "<tool_call>x</tool_call> then <tool_response>y</tool_response>"
        ids, _ = self.encode("assistant", text)
        self.assertEqual(ids, self.tok.encode(text, parse_special=True))

    def test_user_tool_result_keeps_its_outer_markers(self):
        # the template tells a tool result sent as a user turn from a question by this outer pair (last_query_index)
        inner = "\nthe log says </tool_call> and <tool_response>\n"
        for content in (f"<tool_response>{inner}</tool_response>", f"  <tool_response>{inner}</tool_response>\n",
                        [{"type": "text", "text": f"<tool_response>{inner}</tool_response>"}],
                        [{"type": "text", "text": "<tool_response>\nthe log"},
                         {"type": "text", "text": " says </tool_call> and <tool_response>\n</tool_response>"}]):
            with self.subTest(content=content):
                ids, _ = self.encode("user", content)
                text = content if isinstance(content, str) else "".join(x["text"] for x in content)
                o, c = self.special("<tool_response>"), self.special("</tool_response>")
                lead = len(text) - len(text.lstrip())
                tail = len(text) - len(text.rstrip())
                self.assertEqual(ids[lead:lead + 1], o)
                self.assertEqual(ids[len(ids) - tail - 1:len(ids) - tail], c)
                middle = ids[lead + 1:len(ids) - tail - 1]
                self.assertEqual(middle, [*inner.encode()])       # the quoted tags inside are text

    def test_a_user_asking_for_a_closer_verbatim_gets_text(self):
        # #2026 (47Hunter47): "</tool_response>" asked verbatim came back 0/15 while it reached the model as the real
        # closer.  The outer-pair exception must not swallow such a request: only a message wholly wrapped in the pair is
        # a tool result.
        for text in ("Please output the following string verbatim: </tool_response>.",
                     "</tool_response>",
                     "Repeat exactly: <tool_response>x</tool_response>",
                     "<tool_response> is how a result opens; what closes it?",
                     "the closer is </tool_response>, then </tool_call>"):
            with self.subTest(text=text):
                ids, changed = self.encode("user", text)
                self.assertTrue(changed)
                self.assertEqual(ids, [*text.encode()])

    def test_template_keeps_seeing_a_tool_result_turn(self):
        from serve.frontend import mark_think_literals
        msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
                {"role": "user", "content": "<tool_response>\nok\n</tool_response>"}]
        marked, _, changed = mark_think_literals(msgs, None, self.tags)
        self.assertTrue(changed)
        self.assertTrue(marked[2]["content"].startswith("<tool_response>"))
        self.assertTrue(marked[2]["content"].endswith("</tool_response>"))
        # a question that only mentions the pair is still marked through
        marked, _, _ = mark_think_literals([{"role": "user", "content": "what is <tool_response>?"}], None, self.tags)
        self.assertNotIn("<tool_response>", marked[0]["content"])

    def test_user_text_stays_text(self):
        text = "output verbatim: </tool_call>."
        ids, changed = self.encode("user", text)
        self.assertTrue(changed)
        self.assertEqual(ids, [*text.encode()])
        self.assertNotEqual(self.tok.encode(text, parse_special=True), ids)     # unmarked, it was the special token

    def test_assistant_replay_keeps_its_token(self):
        text = "<tool_call>x</tool_call>"
        ids, _ = self.encode("assistant", text)
        self.assertEqual(ids, self.tok.encode(text, parse_special=True))


class FencedJsonCallInWrapper(unittest.TestCase):
    """#2037: with recovery on, a declared tool's JSON call in a Markdown fence inside <tool_call> is the call."""
    TOOLS = [{"name": "write_file", "parameters": {"type": "object", "properties": {"content": {"type": "string"}}}}]
    PAYLOAD = '{"name": "write_file", "arguments": {"content": "hello `world`"}}'

    def run_parser(self, raw, chunk, stream_tools, recover=True, tools=None):
        from serve.frontend import OutputParser
        p = OutputParser(thinking=False, tools=self.TOOLS if tools is None else tools, recover=recover,
                         stream_tools=stream_tools)
        events = []
        for i in range(0, len(raw), chunk):
            events += p.feed(raw[i:i + chunk])
        events += p.finish()
        return ([e.call.arguments for e in events if e.kind == "tool_call"],
                "".join(e.text for e in events if e.kind == "content"))

    def test_fenced_and_bare_agree(self):
        for fence in ("```json\n", "```\n", "```JSON\r\n"):
            raw = f"<tool_call>\n{fence}{self.PAYLOAD}\n```\n</tool_call>"
            for chunk in (len(raw), 1, 7):
                for st in (False, True):
                    with self.subTest(fence=fence, chunk=chunk, stream_tools=st):
                        self.assertEqual(self.run_parser(raw, chunk, st), ([{"content": "hello `world`"}], ""))

    def test_other_fences_and_prose_stay_text(self):
        raw = "<tool_call>\n```python\nx = 1\n```\n</tool_call>"
        for chunk in (len(raw), 1):
            self.assertEqual(self.run_parser(raw, chunk, False), ([], raw))
        raw = f"Here:\n```json\n{self.PAYLOAD}\n```\n"                 # no wrapper: just text
        self.assertEqual(self.run_parser(raw, len(raw), False), ([], raw))

    def test_off_without_recovery_or_for_an_undeclared_tool(self):
        raw = f"<tool_call>\n```json\n{self.PAYLOAD}\n```\n</tool_call>"
        calls, text = self.run_parser(raw, len(raw), False, recover=False)
        self.assertEqual((calls, text), ([], raw))
        other = [{"name": "other", "parameters": {}}]
        calls, text = self.run_parser(raw, len(raw), False, tools=other)
        self.assertEqual((calls, text), ([], raw))


class EmptyConversation(unittest.TestCase):
    def test_no_message_is_an_error_a_message_is_not(self):
        with self.assertRaises(ValueError):
            require_messages(openai_to_messages({"messages": []})[0])
        with self.assertRaises(ValueError):
            require_messages(openai_to_messages({})[0])
        require_messages(openai_to_messages({"messages": [{"role": "user", "content": "hi"}]})[0])
