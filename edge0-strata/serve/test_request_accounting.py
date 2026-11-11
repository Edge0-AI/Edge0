"""Request accounting across thinking-budget continuations, through the HTTP APIs (no GPU or model pack).

    python -m unittest serve.test_request_accounting -v
"""
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, Service, add_carry, carry_leg, request_timings, serve
from serve.test_server import ThinkingEngine


class ClockedThinkingEngine(ThinkingEngine):
    """The native DONE contract: closing a generation drains its statistics before the next one starts."""
    REUSED = 5

    def __init__(self, tok):
        super().__init__(tok)
        self.attempts = 0
        self.done = []
        self.fail_continuation = False
        self.cancel_continuation = False

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.attempts += 1
        if self.fail_continuation and self.attempts == 2:
            raise EngineDied("the continuation ended without a DONE line")
        first = self.attempts == 1
        generated = 0
        stopped = False
        try:
            for token in super().generate(ids, max_new, sampling, cancel, embeddings):
                generated += 1
                stopped = token in self.tok.encode("<|im_end|>", parse_special=True)
                yield token
                if self.cancel_continuation and not first and generated == 3:
                    cancel.set()
        finally:
            # The continuation reuses the original prompt plus the generated thinking.
            reused = self.REUSED if first else len(self.prompts[0]) + self.done[0]["generated"]
            self.last = dict(prompt_tokens=len(ids), generated=generated, reused=reused,
                             prompt_read=len(ids) - reused, finish="stop" if stopped else "cancel",
                             prompt_ms=40.0 if first else 7.0, decode_ms=200.0 if first else 80.0,
                             drafts_offered=18 if first else 11, drafts_accepted=8 if first else 7,
                             hits=9 if first else 7, lookups=10 if first else 8,
                             ram_blobs=1 if first else 3, file_blobs=2 if first else 4,
                             file_mb=1.5 if first else 2.5)
            self.done.append(dict(self.last))


class RequestAccounting(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = ClockedThinkingEngine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).with_name("chat_template.jinja")))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body else None,
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read().decode()
            return response.status, raw

    def chat(self, path="/v1/chat/completions", stream=False, budget=20):
        code, raw = self.request(path, {"model": "m", "messages": [{"role": "user", "content": "2+2?"}],
                                       "max_tokens": 400, "reasoning_budget_tokens": budget, "stream": stream})
        self.assertEqual(code, 200, raw)
        if stream:
            return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
        return json.loads(raw)

    def assert_original_input(self, usage):
        original = len(self.engine.prompts[0])
        self.assertEqual(self.engine.attempts, 2)
        self.assertGreater(self.engine.done[1]["reused"], original, "exercise reuse beyond the API input")
        self.assertEqual(usage["prompt_tokens"], original)
        self.assertEqual(usage["prompt_tokens_details"]["cached_tokens"], self.engine.REUSED)
        self.assertEqual(usage["total_tokens"], original + usage["completion_tokens"])

    def assert_request_work(self, timings, completion_tokens, finish="stop"):
        original = len(self.engine.prompts[0])
        self.assertEqual(len(self.engine.done), 2)
        native_generated = sum(done["generated"] for done in self.engine.done)
        self.assertEqual((timings["cache_n"], timings["prompt_n"]), (5, original - 5))
        self.assertEqual((timings["prompt_ms"], timings["predicted_ms"]), (47.0, 280.0))
        self.assertEqual(timings["predicted_n"], completion_tokens)
        self.assertGreater(completion_tokens, native_generated, "wrap-up tokens count as output, not native decode")
        self.assertEqual(timings["predicted_per_second"], round(native_generated / 0.280, 1))
        self.assertEqual(timings["predicted_per_token_ms"], round(280.0 / native_generated, 3))
        self.assertEqual((timings["draft_n"], timings["draft_n_accepted"]), (29, 15))
        code, raw = self.request("/metrics")
        self.assertEqual(code, 200)
        metrics = json.loads(raw)
        row, totals = metrics["requests"][0], metrics["totals"]
        self.assertEqual(row["finish"], finish)
        self.assertEqual((row["prompt_tokens"], row["reused"], row["prompt_read"]), (original, 5, original - 5))
        self.assertEqual(row["engine_generated"], native_generated)
        self.assertEqual(row["decode_tok_s"], timings["predicted_per_second"])
        self.assertEqual((row["prompt_ms"], row["decode_ms"]), (47.0, 280.0))
        self.assertEqual((row["ram_blobs"], row["file_blobs"], row["file_mb"]), (4, 6, 4.0))
        self.assertAlmostEqual(row["hit_rate"], 16 / 18, places=3)
        self.assertEqual((totals["requests"], totals["prompt_tokens"], totals["reused"]), (1, original, 5))
        self.assertEqual((totals["prompt_ms"], totals["decode_ms"]), (47.0, 280.0))
        self.assertEqual((totals["drafts_offered"], totals["drafts_accepted"]), (29, 15))
        code, raw = self.request("/v1/status")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(raw)["last_timings"]["draft_n"], 29)

    def test_openai_nonstream_keeps_original_cache_and_all_segment_work(self):
        response = self.chat()
        self.assertEqual(response["choices"][0]["message"]["content"], self.engine.ANSWER)
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assert_original_input(response["usage"])
        self.assert_request_work(response["timings"], response["usage"]["completion_tokens"])

    def test_openai_stream_keeps_original_cache_and_all_segment_work(self):
        events = self.chat(stream=True)
        final = events[-1]
        self.assertEqual(final["choices"][0]["finish_reason"], "stop")
        self.assert_original_input(final["usage"])
        self.assert_request_work(final["timings"], final["usage"]["completion_tokens"])

    def test_cancelled_continuation_keeps_original_input_and_counts_each_done_once(self):
        self.engine.cancel_continuation = True
        events = self.chat(stream=True)
        final = events[-1]
        content = "".join(event["choices"][0]["delta"].get("content", "") for event in events)
        self.assertEqual(content, "The")
        self.assertEqual(final["choices"][0]["finish_reason"], "stop")
        self.assertEqual([done["generated"] for done in self.engine.done], [20, 3])
        self.assert_original_input(final["usage"])
        self.assert_request_work(final["timings"], final["usage"]["completion_tokens"], finish="cancel")

    def test_anthropic_nonstream_classifies_only_original_input_as_cached(self):
        usage = self.chat("/v1/messages")["usage"]
        self.assertEqual(self.engine.attempts, 2)
        self.assertEqual(usage["cache_read_input_tokens"], 5)
        self.assertEqual(usage["input_tokens"] + usage["cache_creation_input_tokens"], len(self.engine.prompts[0]) - 5)

    def test_anthropic_stream_classifies_only_original_input_as_cached(self):
        events = self.chat("/v1/messages", stream=True)
        usage = next(event["usage"] for event in events if event["type"] == "message_delta")
        self.assertEqual(self.engine.attempts, 2)
        self.assertEqual(usage["cache_read_input_tokens"], 5)
        self.assertEqual(usage["input_tokens"] + usage["cache_creation_input_tokens"], len(self.engine.prompts[0]) - 5)

    def test_single_segment_keeps_its_own_usage_and_timings(self):
        response = self.chat(budget=10000)
        self.assertEqual(self.engine.attempts, 1)
        self.assertEqual(response["usage"]["prompt_tokens_details"]["cached_tokens"], 5)
        timings = response["timings"]
        self.assertEqual((timings["prompt_ms"], timings["predicted_ms"]), (40.0, 200.0))
        self.assertEqual((timings["draft_n"], timings["draft_n_accepted"]), (18, 8))

    def test_failed_continuation_does_not_count_the_first_done_twice(self):
        self.engine.fail_continuation = True
        events = self.chat(stream=True)
        self.assertTrue(any("error" in event for event in events))
        self.assertEqual(self.engine.attempts, 2)
        self.assertEqual(len(self.engine.done), 1)
        _, raw = self.request("/metrics")
        metrics = json.loads(raw)
        row = metrics["requests"][0]
        self.assertEqual(row["finish"], "error")
        self.assertEqual(row["engine_generated"], 20)
        self.assertEqual((row["prompt_ms"], row["decode_ms"]), (40.0, 200.0))
        self.assertEqual((metrics["totals"]["drafts_offered"], metrics["totals"]["drafts_accepted"]), (18, 8))


class SlotLegAccounting(unittest.TestCase):
    """#1869: a request that decoded in a batch slot and finished on the solo path keeps the slot leg's decode time and
    tokens (the solo DONE alone said 1 token in 18 ms); #1870: prompt_n is what the engine read."""

    def test_solo_leg_adds_to_the_slot_leg(self):
        carry = {}
        slot_leg = {"generated": 134, "decode_ms": 2900.0, "prompt_ms": 3794.7, "prompt_read": 2052, "finish": "cancel"}
        carry_leg(carry, slot_leg)
        solo_done = {"generated": 1, "decode_ms": 18.4, "prompt_ms": 2.0, "prompt_read": 1, "finish": "stop", "reused": 0}
        total = add_carry(carry, solo_done)
        self.assertEqual((total["generated"], total["decode_ms"]), (135, 2918.4))
        self.assertEqual((total["prompt_ms"], total["prompt_read"], total["finish"]), (3794.7, 2052, "stop"))
        timings = request_timings(2052, 135, total)
        self.assertEqual(timings["predicted_ms"], 2918.4)
        self.assertAlmostEqual(timings["predicted_per_second"], 46.3, places=1)

    def test_two_hand_backs_sum(self):
        carry = {}
        carry_leg(carry, {"generated": 10, "decode_ms": 100.0, "prompt_ms": 50.0})
        carry_leg(carry, {"generated": 20, "decode_ms": 300.0, "prompt_ms": 1.0})
        total = add_carry(carry, {"generated": 5, "decode_ms": 50.0, "prompt_ms": 1.0})
        self.assertEqual((total["generated"], total["decode_ms"], total["prompt_ms"]), (35, 450.0, 50.0))
        self.assertNotIn("prompt_read", total)

    def test_prompt_n_is_what_was_read(self):
        # the slot already held the whole prompt: 1 token read, the conversation cache off (reused 0)
        last = {"prompt_ms": 563.5, "decode_ms": 700.0, "generated": 30, "reused": 0, "prompt_read": 1, "finish": "stop"}
        t = request_timings(43086, 30, last)
        self.assertEqual((t["prompt_n"], t["cache_n"]), (1, 43085))
        self.assertEqual(t["prompt_n"] + t["cache_n"], 43086)
        self.assertLess(t["prompt_per_second"], 10)
        # a cold read, and an older engine without the field: unchanged
        cold = request_timings(2052, 10, {**last, "prompt_read": 2052})
        self.assertEqual((cold["prompt_n"], cold["cache_n"]), (2052, 0))
        old = request_timings(2052, 10, {k: v for k, v in last.items() if k != "prompt_read"})
        self.assertEqual((old["prompt_n"], old["cache_n"]), (2052, 0))
        # the conversation cache's own count still counts when larger
        both = request_timings(1000, 10, {**last, "reused": 900, "prompt_read": 100})
        self.assertEqual((both["prompt_n"], both["cache_n"]), (100, 900))


if __name__ == "__main__":
    unittest.main()
