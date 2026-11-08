"""Behavior checks for the example measurements; no weights or network needed."""

import json
from pathlib import Path
import tempfile
import unittest

from example_cases import CASES, score_tags
from example_server import collect_stream, metrics, payload, write_report
from server_test_utils import server_args


def chunks():
    events = [{"choices": [{"delta": {"role": "assistant"}}]},
              {"choices": [{"delta": {"content": "Hello"}}]},
              {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]},
              {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4},
               "timings": {"prompt_n": 10, "predicted_n": 4, "prompt_ms": 500,
                           "predicted_ms": 2000, "predicted_per_second": 1.5}}]
    return events


def report_rows():
    rows = []
    for mode in ("serial", "parallel"):
        for case in CASES:
            row = dict(case=case["id"], mode=mode, answer=f"Full {mode} {case['id']}",
                       prompt_tokens=10, generated_tokens=4, ttft_s=0.5, decode_tok_s=1.5,
                       server_wall_s=2.5, client_wall_s=2.6, finish_reason="stop",
                       before={"load": [10]}, after={"load": [12]})
            row["score"] = score_tags(case, "[]") if "labels" in case else None
            rows.append(row)
    return rows


class StreamTests(unittest.TestCase):
    def test_role_only_events_do_not_count_as_first_content(self):
        source = [b": heartbeat\n", b"\n"] + [
            ("data: " + json.dumps(event) + "\n").encode() for event in chunks()]
        result = collect_stream(source + [b"data: [DONE]\n"], iter([1.0, 1.5]).__next__, 0)
        self.assertEqual(result["answer"], "Hello world")
        self.assertEqual(result["ttft_s"], 1.0)
        self.assertEqual(result["timings"]["predicted_n"], 4)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["events"], chunks())

    def test_error_event_and_truncated_stream_are_failures(self):
        for source in ([b'data: {"error": {"message": "failed"}}\n'],
                       [b'data: {"choices": []}\n']):
            with self.subTest(source=source), self.assertRaises((AssertionError, RuntimeError)):
                collect_stream(source, lambda: 1, 0)

    def test_decode_excludes_prefill_token_and_server_wall_includes_prefill(self):
        result = dict(answer="ok", usage=chunks()[-1]["usage"], timings=chunks()[-1]["timings"])
        self.assertEqual(metrics(result)["decode_tok_s"], 1.5)
        self.assertEqual(metrics(result)["server_wall_s"], 2.5)
        result["timings"]["predicted_ms"] = 0
        with self.assertRaises(AssertionError):
            metrics(result)

    def test_fixed_request_and_requested_server_concurrency(self):
        for case in CASES:
            query = payload(case)
            self.assertEqual(query["max_tokens"], 400)
            self.assertEqual(query["seed"], 4242)
            self.assertEqual(query["temperature"], 0)
            self.assertFalse(query["cache_prompt"])
            self.assertFalse(query["chat_template_kwargs"]["enable_thinking"])
        for count in (1, 4):
            args = server_args("server", "model", 1234, 8, "lora", 512, count)
            self.assertEqual(args[args.index("-np") + 1], str(count))


class ScoringTests(unittest.TestCase):
    def test_exact_labels_and_partial_scores(self):
        for case in CASES[2:]:
            answers = [dict(id=i, **labels) for i, labels in enumerate(case["labels"], 1)]
            self.assertEqual(score_tags(case, json.dumps(answers))["exact"], 8)
            field = next(iter(case["labels"][0]))
            answers[0][field] = "wrong"
            score = score_tags(case, json.dumps(answers))
            self.assertEqual(score["exact"], 7)
            self.assertEqual(score["fields"][field], 7)

    def test_fenced_answers_are_scored_but_fail_the_json_only_contract(self):
        case = CASES[2]
        answers = [dict(id=i, **labels) for i, labels in enumerate(case["labels"], 1)]
        score = score_tags(case, "```json\n" + json.dumps(answers) + "\n```")
        self.assertEqual(score["exact"], 8)
        self.assertTrue(score["fence_removed"])
        self.assertFalse(score["valid"])

    def test_invalid_json_missing_duplicate_and_extra_ids_are_not_accepted(self):
        case = CASES[2]
        answers = [dict(id=i, **labels) for i, labels in enumerate(case["labels"], 1)]
        for text in ("```json\n[]\n```", "{}", json.dumps(answers[:-1]),
                     json.dumps(answers + [answers[0]]), json.dumps(answers + [dict(id=9)])):
            with self.subTest(text=text):
                score = score_tags(case, text)
                self.assertFalse(score["valid"])
                self.assertEqual(score["exact"], 0)

    def test_report_contains_each_prompt_both_full_answers_and_scores(self):
        rows = report_rows()
        result = dict(rows=rows, groups={mode: {"trace": {"steps": 1}, "wall_s": 5,
                      "aggregate_tok_s": 2.4} for mode in ("serial", "parallel")})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.md"
            write_report(path, result)
            report = path.read_text()
        for case in CASES:
            self.assertIn(case["prompt"], report)
        for row in rows:
            self.assertIn(row["answer"], report)
        self.assertIn("under load", report)
        self.assertIn("0/8", report)
        self.assertIn("2.40", report)


if __name__ == "__main__":
    unittest.main()
