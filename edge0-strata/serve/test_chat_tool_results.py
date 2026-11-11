"""Chat Completions tool results must follow call IDs before the template drops them."""
import copy
import http.client
import json
from pathlib import Path
import unittest

from serve.frontend import ChatTemplate, openai_to_messages
from serve.server import ByteTokenizer, MockEngine, Service, serve


TEMPLATE = ChatTemplate(Path(__file__).with_name("chat_template.jinja"))


def history():
    return [
        {"role": "user", "content": "Read a.txt and b.txt."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_a", "type": "function", "function": {
                "name": "read_file", "arguments": '{"path":"a.txt"}'}},
            {"id": "call_b", "type": "function", "function": {
                "name": "read_file", "arguments": '{"path":"b.txt"}'}}]},
        {"role": "tool", "tool_call_id": "call_b", "content": "RESULT_FROM_B"},
        {"role": "tool", "tool_call_id": "call_a", "content": "RESULT_FROM_A"},
    ]


def normalized(messages):
    return openai_to_messages({"messages": messages})[0]


def rendered(messages):
    return TEMPLATE.render(normalized(messages), enable_thinking=False)


class ChatToolResults(unittest.TestCase):
    def test_reversed_results_of_the_same_function_follow_call_ids(self):
        raw = history()
        before = copy.deepcopy(raw)
        ordered = raw[:2] + list(reversed(raw[2:]))
        self.assertEqual(rendered(raw), rendered(ordered))
        self.assertLess(rendered(raw).index("RESULT_FROM_A"), rendered(raw).index("RESULT_FROM_B"))
        self.assertEqual(raw, before)
        messages = normalized(raw)
        self.assertEqual([m["content"] for m in messages[2:]], ["RESULT_FROM_A", "RESULT_FROM_B"])
        self.assertTrue(all(set(m) == {"role", "content"} for m in messages[2:]))
        self.assertTrue(all(set(c) == {"function"} for c in messages[1]["tool_calls"]))

    def test_already_ordered_and_legacy_prompts_are_identical(self):
        ordered = history()
        ordered[2:] = reversed(ordered[2:])
        legacy = copy.deepcopy(ordered)
        for call in legacy[1]["tool_calls"]:
            call.pop("id")
        for result in legacy[2:]:
            result.pop("tool_call_id")
        self.assertEqual(normalized(ordered), normalized(legacy))
        self.assertEqual(rendered(ordered), rendered(legacy))

    def test_ambiguous_or_incomplete_ids_preserve_received_order(self):
        for case in ("missing_call", "missing_result", "duplicate_call", "duplicate_result", "unknown",
                     "list_call", "list_result", "empty_call", "empty_result", "partial"):
            raw = history()
            if case == "missing_call":
                raw[1]["tool_calls"][0].pop("id")
            elif case == "missing_result":
                raw[2].pop("tool_call_id")
            elif case == "duplicate_call":
                raw[1]["tool_calls"][1]["id"] = "call_a"
            elif case == "duplicate_result":
                raw[3]["tool_call_id"] = "call_b"
            elif case == "unknown":
                raw[2]["tool_call_id"] = "unknown"
            elif case == "list_call":
                raw[1]["tool_calls"][0]["id"] = []
            elif case == "list_result":
                raw[2]["tool_call_id"] = []
            elif case == "empty_call":
                raw[1]["tool_calls"][0]["id"] = ""
            elif case == "empty_result":
                raw[2]["tool_call_id"] = ""
            else:
                raw.pop()
            with self.subTest(case=case):
                self.assertEqual([m["content"] for m in normalized(raw)[2:]],
                                 [m["content"] for m in raw[2:]])

    def test_each_tool_run_is_scoped_to_its_assistant_turn(self):
        raw = history() + history()
        self.assertEqual([m["content"] for m in normalized(raw) if m["role"] == "tool"],
                         ["RESULT_FROM_A", "RESULT_FROM_B"] * 2)
        raw = history()
        raw.insert(2, {"role": "user", "content": "intervening message"})
        self.assertEqual([m["content"] for m in normalized(raw) if m["role"] == "tool"],
                         ["RESULT_FROM_B", "RESULT_FROM_A"])

    def test_json_encoded_history_and_calls(self):
        raw = history()
        raw[1]["tool_calls"] = json.dumps(raw[1]["tool_calls"])
        self.assertEqual(normalized(json.dumps(raw)), normalized(history()))

    def test_multimodal_result_moves_with_its_call(self):
        raw = history()
        raw[2]["content"] = [{"type": "text", "text": "RESULT_FROM_B"},
                             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
        messages = normalized(raw)
        self.assertEqual(messages[2]["content"], "RESULT_FROM_A")
        self.assertEqual(messages[3]["content"], [{"type": "text", "text": "RESULT_FROM_B"},
                                                {"type": "image", "source": "data:image/png;base64,AA=="}])

    def test_http_engine_receives_the_ordered_prompt(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "ok", max_context=4096)
        service = Service(engine, tok, TEMPLATE)
        httpd = serve(service, port=0)
        connection = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
        try:
            connection.request("POST", "/v1/chat/completions", json.dumps({
                "messages": history(), "max_tokens": 16, "reasoning_effort": "none"}),
                {"Content-Type": "application/json"})
            response = connection.getresponse()
            body = response.read()
            self.assertEqual(response.status, 200, body)
            prompt = tok.decode(engine.last_prompt)
            self.assertLess(prompt.index("RESULT_FROM_A"), prompt.index("RESULT_FROM_B"))
        finally:
            connection.close()
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
