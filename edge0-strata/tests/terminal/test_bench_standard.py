"""The standard benchmark (docs/BENCHMARK_PROTOCOL.md) against a scripted mock server: no engine, no model weights."""

from __future__ import annotations

import contextlib
import copy
import http.client
import io
import json
import os
import re
import shutil
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from strata_cli import bench, bench_facts, bench_standard as bs
from strata_cli.cli import main
from strata_control.profiles import ProfileError

COMMIT = "a" * 40
TEMPLATE = "{{ messages }} mock chat template"


def corpus_specs() -> dict[str, dict]:
    return {spec["id"]: spec for spec in bs.load_corpus().specs}


class MockStrata:
    """A server that speaks enough of Strata's API: /health, /props, /v1/status, /metrics and streamed chat completions."""

    def __init__(self, model_file: Path):
        self.specs = corpus_specs()
        self.model_file = model_file
        self.script: dict[str, dict] = {}
        self.bodies: list[dict] = []
        self.order: list[str] = []
        self.arrivals: list[float] = []
        self.active = 0
        self.max_active = 0
        self.dead = False
        self.lock = threading.Lock()
        self.requests_before = 0
        self.gpu_name = "Mock GPU 12 GB"
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, code, value):
                data = json.dumps(value).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if owner.dead and self.path == "/health":
                    return self._json(503, {"error": "down"})
                if self.path == "/health":
                    return self._json(200, {"service": "strata", "loaded": True})
                if self.path == "/props":
                    return self._json(200, {"model_alias": "qwen3.8-flash-next", "chat_template": TEMPLATE,
                                            "default_generation_settings": {"n_ctx": 131072, "params": {"temperature": 0.7}},
                                            "total_slots": 1, "build_info": "Strata 0.1.45.3",
                                            "model_path": str(owner.model_file), "is_sleeping": False})
                if self.path == "/v1/status":
                    return self._json(200, {"service": "strata", "model": "qwen3.8-flash-next", "engine": "0.1.45.3",
                                            "machine": {"gpu": {"name": owner.gpu_name}},
                                            "activity": {"requests": owner.requests_before}})
                if self.path == "/metrics":
                    return self._json(200, {"engine": {"version": "0.1.45.3"}, "hardware_static": {"gpu_name": owner.gpu_name}})
                self._json(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                rid = re.match(r"\[case ([^\]]+)\]", body["messages"][0]["content"]).group(1)
                with owner.lock:
                    owner.bodies.append(body)
                    owner.order.append(rid)
                    owner.arrivals.append(time.monotonic())
                    owner.active += 1
                    owner.max_active = max(owner.max_active, owner.active)
                try:
                    self.respond(rid, body)
                finally:
                    with owner.lock:
                        owner.active -= 1

            def respond(self, rid, body):
                spec = owner.specs[rid]
                plan = owner.script.get(rid, {})
                if plan.get("hold_for_arrivals"):
                    deadline = time.monotonic() + 3
                    while time.monotonic() < deadline and owner.active < plan["hold_for_arrivals"]:
                        time.sleep(0.01)
                if plan.get("die"):
                    owner.dead = True
                    self.connection.close()
                    return
                if plan.get("http"):
                    return self._json(plan["http"], {"error": {"message": "refused"}})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()

                def send(event):
                    self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                    self.wfile.flush()
                chunk = lambda delta, finish=None: {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}   # noqa: E731
                send(chunk({"role": "assistant", "content": ""}))
                self.wfile.write(b": keep-alive\n\n")
                send(chunk({"content": ""}))
                if plan.get("hang"):
                    time.sleep(plan["hang"])
                    return
                time.sleep(plan.get("first_delay", 0))
                send(chunk({"reasoning_content": "Thinking about it. "}))
                expected = spec.get("expected") or {}
                finish = plan.get("finish", "stop")
                if "tool_calls" in plan:
                    calls = plan["tool_calls"]
                    finish = plan.get("finish", "tool_calls" if calls else "stop")
                elif expected.get("kind") == "tool_call":
                    calls = [{"name": expected["name"], "arguments": json.dumps(expected["arguments"])}]
                    finish = plan.get("finish", "tool_calls")
                else:
                    calls = []
                if "content" in plan:
                    text = plan["content"]
                elif expected.get("kind") == "text":
                    text = expected["value"]
                elif expected.get("kind") == "json":
                    text = json.dumps(expected["value"], separators=(",", ":"))
                elif expected.get("kind") == "tool_call":
                    text = ""
                else:
                    text = "A hash table hashes the key."
                if text:
                    send(chunk({"content": text}))
                for index, call in enumerate(calls):
                    send(chunk({"tool_calls": [{"index": index, "id": f"call_{index}", "type": "function",
                                                "function": {"name": call["name"], "arguments": ""}}]}))
                    send(chunk({"tool_calls": [{"index": index, "function": {"arguments": call["arguments"]}}]}))
                final = chunk({}, finish)
                prompt = spec["input_tokens_frozen"]
                output = plan.get("completion_tokens", 50)
                if plan.get("usage", True):
                    final["usage"] = {"prompt_tokens": prompt, "completion_tokens": output, "total_tokens": prompt + output,
                                      "prompt_tokens_details": {"cached_tokens": plan.get("cached", 0)}}
                if plan.get("timings", True):
                    cached = plan.get("cached", 0)
                    read = prompt - cached
                    final["timings"] = {"cache_n": cached, "prompt_n": read, "prompt_ms": float(read), "prompt_per_second": 1000.0,
                                        "predicted_n": plan.get("predicted_n", output), "predicted_ms": 500.0,
                                        "predicted_per_second": plan.get("rate", 100.0)}
                send(final)
                if not plan.get("no_done"):
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class StandardBase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "Qwen3.8-Flash-Next-IQ2_XS-00001-of-00001.gguf"
        self.model.write_bytes(b"model bytes")
        self.server = MockStrata(self.model)
        self.addCleanup(self.server.close)
        for name, value in (("nvidia_gpus", ([], None, None)), ("other_gpus", ([], None)), ("cpu_name", lambda: "Mock CPU"),
                            ("ram_bytes", lambda: 64 * 2 ** 30), ("storage_of", lambda path: [{"kind": "nvme", "model": "Mock SSD", "bus": "nvme"}]),
                            ("git_commit", lambda folder=None: COMMIT)):
            patcher = mock.patch.object(bench_facts, name, **({"return_value": value} if isinstance(value, tuple) else {"side_effect": value}))
            patcher.start()
            self.addCleanup(patcher.stop)
        self.target = bs.Target(self.server.url, None)

    def run_standard(self, name="out", **kwargs):
        kwargs.setdefault("hash_model", True)
        doc, out = bs.run(self.target, self.root / name, **kwargs)
        return doc, out

    def record(self, doc, rid):
        return next(r for r in doc["requests"] if r["id"] == rid)

    def summary(self, doc, case, metric, phase="performance"):
        return next(s for s in doc["summaries"] if s["phase"] == phase and s["case_id"] == case and s["metric"] == metric)


class FixtureTests(unittest.TestCase):
    def test_bundle_follows_the_protocol_tables(self):
        corpus = bs.load_corpus()
        specs = {s["id"]: s for s in corpus.specs}
        self.assertEqual([s["id"] for s in corpus.specs][:10],
                         ["warmup", "performance-short-r1", "performance-prompt-4k-r1", "performance-prompt-16k-r1",
                          "performance-short-r2", "performance-prompt-4k-r2", "performance-prompt-16k-r2",
                          "performance-short-r3", "performance-prompt-4k-r3", "performance-prompt-16k-r3"])
        self.assertEqual([s["id"] for s in corpus.specs if s["phase"] == "sanity"],
                         [f"sanity-{n}" for n in ("I1", "I2", "I3", "I4", "R1", "R2", "R3", "R4", "T1", "T2", "T3", "T4")])
        self.assertEqual(len(corpus.specs), 34)
        self.assertEqual((specs["warmup"]["seed"], specs["warmup"]["max_output_tokens"]), (1, 128))
        for repetition in (1, 2, 3):
            for case in ("short", "prompt-4k", "prompt-16k"):
                spec = specs[f"performance-{case}-r{repetition}"]
                self.assertEqual((spec["seed"], spec["max_output_tokens"], spec["concurrency"]), (repetition, 512, 1))
        for spec in corpus.specs:
            if spec["phase"] == "sanity":
                self.assertEqual((spec["seed"], spec["max_output_tokens"]), (1, 2048))
            if spec["phase"] == "concurrency":
                self.assertEqual(spec["seed"], 4 * (spec["round"] - 1) + int(spec["id"][-1]))
                self.assertEqual((spec["max_output_tokens"], spec["concurrency"]), (512, 4))
            self.assertTrue(spec["messages"][0]["content"].startswith(f"[case {spec['id']}]\n"))
            self.assertEqual(len(spec["messages"]), 1)
        self.assertEqual(specs["performance-short-r1"]["messages"][0]["content"].split("\n", 1)[1],
                         "Explain how a hash table finds a value, including collisions and one small example.")

    def test_records_and_retrieval_placement(self):
        specs = corpus_specs()
        body = specs["performance-prompt-4k-r1"]["messages"][0]["content"].split("\n", 1)[1]
        lines = body.split("\n")
        self.assertEqual(lines[0], "record000001: alpha beta gamma delta epsilon zeta eta theta.")
        self.assertEqual(lines[-1], "Summarize the structure of these records and explain how to look up a record by its ID.")
        count = len(lines) - 1
        for rid, key, fraction in (("sanity-R1", "access_code = ORBIT-731", 0.10), ("sanity-R2", "storage_label = CEDAR-284", 0.50)):
            rows = specs[rid]["messages"][0]["content"].split("\n", 1)[1].split("\n")
            self.assertEqual(rows.index(key), int(fraction * count))          # inserted before record index floor(f * count)
            self.assertEqual(sum("record" not in r for r in rows), 2)         # the inserted line and the question
        long_count = len(specs["performance-prompt-16k-r1"]["messages"][0]["content"].split("\n")) - 2
        rows = specs["sanity-R4"]["messages"][0]["content"].split("\n", 1)[1].split("\n")
        self.assertEqual(rows.index("north = N-124"), int(0.10 * long_count))
        self.assertEqual(rows.index("south = S-875") - 1, int(0.90 * long_count))   # one earlier insert shifts it by one line
        self.assertEqual(specs["sanity-R4"]["expected"]["value"], {"north": "N-124", "south": "S-875"})
        for rid, name, props in (("sanity-T1", "lookup_record", {"id": "string"}),
                                 ("sanity-T3", "convert_units", {"value": "number", "from_unit": "string", "to_unit": "string"}),
                                 ("sanity-T4", "get_build_status", {"project": "string", "build": "integer"})):
            function = specs[rid]["tools"][0]["function"]
            self.assertEqual(function["name"], name)
            self.assertEqual({k: v["type"] for k, v in function["parameters"]["properties"].items()}, props)
            self.assertEqual(function["parameters"]["required"], list(props))
            self.assertIs(function["parameters"]["additionalProperties"], False)
            self.assertEqual(function["description"], "Synthetic benchmark lookup; no external action is performed.")
            self.assertEqual(specs[rid]["tool_choice"], "auto")
        self.assertIsNone(specs["sanity-I1"]["tools"])

    def test_prompt_lengths_are_within_two_percent_of_their_targets(self):
        for spec in bs.load_corpus().specs:
            if spec["target_input_tokens"]:
                self.assertLessEqual(abs(spec["input_tokens_frozen"] - spec["target_input_tokens"]) / spec["target_input_tokens"], 0.02)

    def test_a_changed_or_converted_file_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            copy_of = Path(temp) / "bundle"
            shutil.copytree(bs.corpus_dir(), copy_of)
            self.assertEqual(bs.load_corpus(copy_of).sha256, bs.load_corpus().sha256)
            target = copy_of / "requests" / "warmup.json"
            target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))
            with self.assertRaisesRegex(ProfileError, "fixture file changed: requests/warmup.json"):
                bs.load_corpus(copy_of)
            target.write_bytes(bs.load_corpus().manifest[:0] + (bs.corpus_dir() / "requests" / "warmup.json").read_bytes())
            (copy_of / "requests" / "extra.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ProfileError, "not in the manifest"):
                bs.load_corpus(copy_of)

    def test_gitattributes_keeps_fixture_bytes(self):
        text = (Path(__file__).resolve().parents[2] / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("bench/prompts/** -text", text)


class DocsAndPackagingTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[2]

    def test_the_protocol_names_the_frozen_bundle(self):
        text = (self.ROOT / "docs" / "BENCHMARK_PROTOCOL.md").read_text(encoding="utf-8")
        self.assertIn(bs.load_corpus().sha256, text)
        self.assertIn(bs.PROTOCOL_ID, text)
        for heading in ("## Run it", "## Submit and review", "## Activation checklist", "### Where the numbers come from"):
            self.assertIn(heading, text)
        self.assertNotIn("do not implement this protocol", text)

    def test_the_terminal_archive_carries_the_fixtures(self):
        import sys
        sys.path.insert(0, str(self.ROOT / "packaging"))
        import package_terminal
        self.assertIn("bench/prompts/standard-v1", package_terminal.PAYLOAD)

    def test_the_launchers_point_at_the_standard_benchmark(self):
        for name in ("BENCH.bat", "bench.sh"):
            self.assertIn("strata.py bench run", (self.ROOT / name).read_text(encoding="utf-8"))


class ScoringTests(unittest.TestCase):
    def test_strict_json(self):
        self.assertEqual(bs.strict_json(' {"a": 1}\n'), {"a": 1})
        for bad in ('{"a":1,"a":2}', "```json\n{}\n```", '{"a":1} extra', "NaN", "Infinity", "", '{"a":1}{"b":2}'):
            with self.assertRaises(ValueError, msg=bad):
                bs.strict_json(bad)

    def test_json_equality_checks_types(self):
        self.assertTrue(bs.same_json({"a": [1, 2], "b": True}, {"b": True, "a": [1, 2]}))
        self.assertFalse(bs.same_json({"ok": True}, {"ok": 1}))
        self.assertFalse(bs.same_json(3, 3.0))
        self.assertFalse(bs.same_json([1, 2], [2, 1]))
        self.assertFalse(bs.same_json({"a": 1}, {"a": 1, "b": 2}))

    def test_text_check_strips_only_ascii_whitespace(self):
        spec = {"expected": {"kind": "text", "value": "READY"}, "category": "instruction"}
        self.assertTrue(bs.evaluate(spec, "completed", " \nREADY\t", [])["passed"])
        self.assertFalse(bs.evaluate(spec, "completed", "READY.", [])["passed"])
        self.assertFalse(bs.evaluate(spec, "completed", "ready", [])["passed"])
        self.assertFalse(bs.evaluate(spec, "truncated", "READY", [])["passed"])

    def test_tool_call_needs_one_exact_structured_call(self):
        spec = {"expected": {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Zagreb", "unit": "celsius"}},
                "category": "tool_call"}
        good = {"name": "get_weather", "arguments": '{"unit":"celsius","city":"Zagreb"}'}
        self.assertTrue(bs.evaluate(spec, "completed", "", [good])["passed"])
        self.assertFalse(bs.evaluate(spec, "completed", "get_weather(...)", [])["passed"])
        self.assertFalse(bs.evaluate(spec, "completed", "", [good, good])["passed"])
        self.assertFalse(bs.evaluate(spec, "completed", "", [{**good, "name": "other"}])["passed"])
        self.assertFalse(bs.evaluate(spec, "completed", "", [{**good, "arguments": '{"city":"Zagreb"}'}])["passed"])
        self.assertFalse(bs.evaluate(spec, "completed", "", [{**good, "arguments": '{"city":"Zagreb","unit":"celsius","x":1}'}])["passed"])
        self.assertFalse(bs.evaluate(spec, "completed", "", [{**good, "arguments": '{"city":"a","city":"Zagreb","unit":"celsius"}'}])["passed"])


class RunTests(StandardBase):
    def test_a_full_run_follows_the_protocol(self):
        doc, out = self.run_standard()
        self.assertEqual(bs.validate(doc), [])
        run = doc["run"]
        self.assertEqual((doc["format"], doc["schema_version"], run["status"]), ("strata-standard-benchmark", 1, "complete"))
        self.assertEqual(doc["protocol"]["id"], "strata-qwen38-thinking-v1-draft.1")
        self.assertEqual(doc["protocol"]["corpus_sha256"], bs.load_corpus().sha256)
        self.assertEqual(doc["protocol"]["git_commit"], COMMIT)
        self.assertEqual(len(doc["requests"]), 22)
        self.assertEqual(self.server.order, [r["id"] for r in doc["requests"]])
        self.assertEqual(doc["request_counts"]["warmup"], {"planned": 1, "attempted": 1, "completed": 1, "truncated": 0, "failed": 0, "skipped": 0})
        self.assertEqual(doc["request_counts"]["performance"]["completed"], 9)
        self.assertEqual(doc["request_counts"]["concurrency"]["planned"], 0)
        self.assertEqual(doc["quality"], {"passed": 12, "total": 12, "categories": {
            "instruction": {"passed": 4, "total": 4}, "retrieval": {"passed": 4, "total": 4}, "tool_call": {"passed": 4, "total": 4}}})
        self.assertEqual(len(doc["summaries"]), 15)
        self.assertEqual(doc["concurrency_rounds"], [])
        self.assertEqual(self.summary(doc, "prompt-4k", "prompt_tps")["median"], 1000.0)
        self.assertEqual(self.summary(doc, "short", "decode_tps")["median"], 100.0)
        for case in ("short", "prompt-4k", "prompt-16k"):
            for metric in ("prompt_tps", "decode_tps", "ttft_s", "first_answer_s", "total_s"):
                row = self.summary(doc, case, metric)
                self.assertEqual(row["n"], 3)
                self.assertLessEqual(row["min"], row["median"])
                self.assertLessEqual(row["median"], row["max"])
        # the warm-up and the sanity cases do not enter the speed summaries
        self.assertNotIn("warmup", {i for s in doc["summaries"] for i in s["included_request_ids"]})
        self.assertFalse(any(i.startswith("sanity") for s in doc["summaries"] for i in s["included_request_ids"]))
        self.assertTrue(run["required_duration_s"] > 0)
        self.assertIsNone(run["optional_duration_s"])
        self.assertIn("optional_duration_s", "".join(doc["missing"]))

    def test_every_request_carries_the_required_settings(self):
        self.run_standard()
        self.assertEqual(len(self.server.bodies), 22)
        specs = corpus_specs()
        for body, rid in zip(self.server.bodies, self.server.order):
            spec = specs[rid]
            self.assertIs(body["stream"], True)
            self.assertEqual(body["stream_options"], {"include_usage": True})
            self.assertEqual((body["temperature"], body["top_p"], body["top_k"], body["min_p"]), (1.0, 0.95, 20, 0.0))
            self.assertEqual((body["presence_penalty"], body["repetition_penalty"]), (0.0, 1.0))
            self.assertEqual(body["reasoning_effort"], "xhigh")
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True, "preserve_thinking": True})
            self.assertEqual(body["reasoning_budget_tokens"], 0)
            self.assertEqual((body["seed"], body["max_tokens"]), (spec["seed"], spec["max_output_tokens"]))
            self.assertNotIn("stop", body)
            self.assertEqual(body["messages"], spec["messages"])
            self.assertEqual(body.get("tools"), spec["tools"])
            if spec["tools"]:
                self.assertEqual(body["tool_choice"], "auto")

    def test_the_report_files_and_their_hashes(self):
        doc, out = self.run_standard()
        for name in ("results.json", "README.md", "outputs.jsonl", "config.json", "provenance/chat_template.jinja",
                     "provenance/corpus-manifest.sha256", "provenance/corpus-build.json"):
            self.assertTrue((out / name).is_file(), name)
        saved = json.loads((out / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, doc)
        for artifact in doc["artifacts"]:
            self.assertEqual(bench_facts.sha256_file(out / artifact["path"]), artifact["sha256"])
            self.assertNotIn("..", artifact["path"])
        self.assertEqual({a["kind"] for a in doc["artifacts"]}, {"outputs", "config", "provenance"})
        lines = (out / "outputs.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(line)["request_id"] for line in lines], [r["id"] for r in doc["requests"]])
        first = json.loads(lines[0])
        self.assertEqual(first["dispatch_offset_s"], 0.0)
        self.assertEqual(first["first_delta"]["kind"], "reasoning")
        self.assertEqual(doc["configuration"]["template"], {"path": "provenance/chat_template.jinja",
                                                            "sha256": bench_facts.sha256_text(TEMPLATE)})
        self.assertEqual(doc["model"]["files"], [{"path": self.model.name, "sha256": bench_facts.sha256_file(self.model)}])
        self.assertEqual(doc["model"]["quantization"], "IQ2_XS")
        readme = (out / "README.md").read_text(encoding="utf-8")
        self.assertIn("Status: complete", readme)
        self.assertIn("12 of 12", readme)
        self.assertNotIn(str(self.root), readme)

    def test_every_null_has_a_reason_and_the_request_shape_is_complete(self):
        doc, _ = self.run_standard()
        self.assertEqual(set(doc["missing"]), set(bs.scan_nulls(doc)))
        self.assertIn("/hardware/gpus/0/power_limit_w", doc["missing"])
        self.assertIn("/protocol/model_card_revision", doc["missing"])
        self.assertIn("/requests/0/round", doc["missing"])
        record = self.record(doc, "performance-short-r1")
        self.assertEqual(set(record), {"id", "phase", "case_id", "repetition", "seed", "max_output_tokens", "concurrency", "round",
                                       "status", "finish_reason", "error", "tokens", "timings", "metrics", "sources",
                                       "first_delta_kind", "token_accounting_note", "evaluation"})
        self.assertEqual(record["tokens"], {"input_tokens": record["tokens"]["input_tokens"], "cached_tokens": 0,
                                            "read_tokens": record["tokens"]["input_tokens"], "output_tokens": 50, "decode_tokens": 50})
        self.assertEqual(record["timings"]["decode_s"], 0.5)
        self.assertEqual(record["first_delta_kind"], "reasoning")
        self.assertIsNone(record["token_accounting_note"])

    def test_the_run_names_its_default_folder_after_the_gpu(self):
        self.server.gpu_name = "NVIDIA GeForce RTX 3060"
        with mock.patch.object(bench_facts, "nvidia_gpus", return_value=([{
                "device_index": 0, "name": "NVIDIA GeForce RTX 3060", "memory_bytes": 12 * 2 ** 30, "memory_kind": "dedicated",
                "power_limit_w": 170.0, "driver": "560.1"}], "560.1", "12.6")):
            doc, out = bs.run(self.target, None, out_root=self.root / "results", hash_model=True)
        self.assertRegex(out.name, r"^\d{4}-\d\d-\d\d-standard-rtx-3060-[0-9a-f]{8}$")
        self.assertEqual(doc["hardware"]["gpus"][0]["power_limit_w"], 170.0)
        self.assertEqual(doc["software"]["runtime"], [{"name": "cuda", "version": "12.6"}])

    def test_early_eos_and_a_reached_cap_stay_in_the_speed_summaries(self):
        self.server.script["performance-short-r1"] = {"completion_tokens": 5, "predicted_n": 5, "rate": 10.0}
        self.server.script["performance-short-r2"] = {"finish": "length", "completion_tokens": 512, "predicted_n": 512, "rate": 1024.0}
        doc, _ = self.run_standard()
        self.assertEqual(self.record(doc, "performance-short-r2")["status"], "truncated")
        self.assertEqual(self.record(doc, "performance-short-r2")["finish_reason"], "length")
        row = self.summary(doc, "short", "decode_tps")
        self.assertEqual((row["n"], row["min"], row["max"]), (3, 10.0, 1024.0))
        self.assertEqual(doc["request_counts"]["performance"]["truncated"], 1)

    def test_a_failed_request_is_excluded_and_listed_but_the_run_goes_on(self):
        self.server.script["performance-prompt-4k-r2"] = {"http": 500}
        doc, out = self.run_standard()
        self.assertEqual(doc["run"]["status"], "complete")
        record = self.record(doc, "performance-prompt-4k-r2")
        self.assertEqual((record["status"], record["error"]["kind"]), ("failed", "http_500"))
        row = self.summary(doc, "prompt-4k", "decode_tps")
        self.assertEqual(row["n"], 2)
        self.assertEqual(row["excluded"], [{"request_id": "performance-prompt-4k-r2", "reason": "failed: http_500"}])
        self.assertEqual(doc["request_counts"]["performance"]["failed"], 1)
        self.assertEqual(len(self.server.order), 22)
        self.assertIn("performance-prompt-4k-r2 failed", (out / "README.md").read_text(encoding="utf-8"))
        self.assertEqual(bs.validate(doc), [])

    def test_a_failed_warmup_skips_everything_else(self):
        self.server.script["warmup"] = {"http": 500}
        doc, _ = self.run_standard()
        self.assertEqual(doc["run"]["status"], "partial")
        self.assertEqual(len(self.server.order), 1)
        self.assertEqual(len(doc["requests"]), 22)
        self.assertEqual(doc["request_counts"]["performance"]["skipped"], 9)
        self.assertEqual(doc["request_counts"]["sanity"], {"planned": 12, "attempted": 0, "completed": 0, "truncated": 0, "failed": 0, "skipped": 12})
        self.assertEqual(doc["quality"]["passed"], 0)
        skipped = self.record(doc, "sanity-T4")
        self.assertEqual((skipped["status"], skipped["error"]["kind"]), ("skipped", "skipped"))
        self.assertFalse(skipped["evaluation"]["passed"])
        self.assertTrue(all(s["n"] == 0 and s["median"] is None for s in doc["summaries"]))
        self.assertEqual(bs.validate(doc), [])

    def test_a_dead_server_marks_the_rest_skipped(self):
        self.server.script["performance-prompt-4k-r1"] = {"die": True}
        doc, _ = self.run_standard()
        self.assertEqual(doc["run"]["status"], "partial")
        self.assertEqual(self.record(doc, "performance-prompt-4k-r1")["status"], "failed")
        self.assertEqual(self.record(doc, "performance-prompt-16k-r1")["status"], "skipped")
        self.assertIn("server stopped answering", self.record(doc, "sanity-I1")["error"]["message"])
        self.assertEqual(len(doc["requests"]), 22)
        self.assertEqual(bs.validate(doc), [])

    def test_a_request_that_never_finishes_times_out(self):
        self.server.script["performance-short-r1"] = {"hang": 3}
        started = time.monotonic()
        doc, _ = self.run_standard(timeout_s=0.6)
        record = self.record(doc, "performance-short-r1")
        self.assertEqual((record["status"], record["error"]["kind"]), ("failed", "timeout"))
        self.assertLess(record["timings"]["total_s"], 2.0)
        self.assertGreaterEqual(record["timings"]["total_s"], 0.5)
        self.assertEqual(doc["run"]["status"], "complete")
        self.assertLess(time.monotonic() - started, 30)

    def test_a_stream_without_done_or_finish_reason_is_a_failure(self):
        self.server.script["performance-short-r1"] = {"no_done": True}
        self.server.script["performance-short-r2"] = {"finish": None}
        doc, _ = self.run_standard()
        self.assertEqual(self.record(doc, "performance-short-r1")["error"]["kind"], "malformed_stream")
        self.assertEqual(self.record(doc, "performance-short-r2")["error"]["kind"], "malformed_stream")
        self.assertEqual(self.summary(doc, "short", "decode_tps")["n"], 1)

    def test_empty_deltas_and_keepalives_do_not_count_as_the_first_token(self):
        self.server.script["performance-short-r1"] = {"first_delay": 0.4}
        doc, out = self.run_standard()
        record = self.record(doc, "performance-short-r1")
        self.assertGreaterEqual(record["timings"]["ttft_s"], 0.35)
        self.assertEqual(record["first_delta_kind"], "reasoning")
        self.assertGreaterEqual(record["timings"]["first_answer_s"], record["timings"]["ttft_s"])
        tool = self.record(doc, "sanity-T1")
        self.assertIsNotNone(tool["timings"]["first_answer_s"])          # a tool-call delta is an answer
        self.assertIsNone(self.record(doc, "sanity-T1")["error"])

    def test_missing_usage_and_timings_leave_the_rates_null_with_reasons(self):
        self.server.script["performance-short-r1"] = {"usage": False, "timings": False}
        doc, _ = self.run_standard()
        record = self.record(doc, "performance-short-r1")
        self.assertEqual(record["status"], "completed")
        self.assertIsNone(record["metrics"]["prompt_tps"])
        self.assertIsNone(record["metrics"]["decode_tps"])
        self.assertIsNone(record["tokens"]["output_tokens"])
        self.assertIsNotNone(record["timings"]["ttft_s"])
        index = doc["requests"].index(record)
        self.assertIn(f"/requests/{index}/metrics/decode_tps", doc["missing"])
        row = self.summary(doc, "short", "decode_tps")
        self.assertEqual(row["n"], 2)
        self.assertIn("decode_tps is not available", row["excluded"][0]["reason"])
        self.assertEqual(bs.validate(doc), [])

    def test_a_decode_count_that_disagrees_with_the_rate_is_explained(self):
        # the engine's rate and time imply 50 tokens (100/s over 0.5 s) while predicted_n says 60 and the API says 61
        self.server.script["performance-short-r1"] = {"predicted_n": 60, "completion_tokens": 61}
        doc, _ = self.run_standard()
        record = self.record(doc, "performance-short-r1")
        self.assertEqual(record["tokens"]["decode_tokens"], 50)
        self.assertEqual(record["tokens"]["output_tokens"], 61)
        self.assertIn("implied", record["token_accounting_note"])
        self.assertEqual(record["metrics"]["decode_tps"], 100.0)
        self.assertIn("derived", record["sources"]["decode_tokens"])

    def test_wrong_answers_are_kept_and_scored_without_changing_the_run_status(self):
        specs = corpus_specs()
        script = self.server.script
        script["sanity-I1"] = {"content": '```json\n{"ok":true,"count":3}\n```'}
        script["sanity-I2"] = {"content": "Here you go: [-4,1,2,9]"}
        script["sanity-I3"] = {"content": '{"project":"Orion","project":"Orion","build":17}'}
        script["sanity-R1"] = {"finish": "length", "content": "ORB", "completion_tokens": 2048, "predicted_n": 2048}
        script["sanity-R2"] = {"content": "cedar-284"}
        script["sanity-T1"] = {"tool_calls": [], "content": "lookup_record({\"id\":\"A-17\"})"}
        script["sanity-T2"] = {"tool_calls": [{"name": "get_weather", "arguments": '{"city":"Zagreb","unit":"celsius"}'}] * 2}
        script["sanity-T3"] = {"tool_calls": [{"name": "convert_units", "arguments": '{"value":3.0,"from_unit":"km","to_unit":"m"}'}]}
        doc, _ = self.run_standard()
        verdicts = {r["case_id"]: r["evaluation"]["passed"] for r in doc["requests"] if r["phase"] == "sanity"}
        self.assertEqual(verdicts, {"I1": False, "I2": False, "I3": False, "I4": True, "R1": False, "R2": False, "R3": True, "R4": True,
                                    "T1": False, "T2": False, "T3": False, "T4": True})
        self.assertEqual(doc["quality"]["passed"], 4)
        self.assertEqual(doc["quality"]["categories"]["instruction"]["passed"], 1)
        self.assertEqual(self.record(doc, "sanity-R1")["status"], "truncated")
        self.assertIn("output cap", self.record(doc, "sanity-R1")["evaluation"]["reason"])
        self.assertEqual(doc["run"]["status"], "complete")
        self.assertEqual(specs["sanity-I1"]["expected"]["value"], {"ok": True, "count": 3})
        text = bs.format_plain(doc)
        self.assertIn("Sanity checks: 4 of 12 passed", text)
        self.assertIn("I1:", text)

    def test_unexpected_reuse_of_the_user_payload_is_excluded_and_nonconforming(self):
        spec = corpus_specs()["performance-prompt-4k-r2"]
        self.server.script["performance-prompt-4k-r2"] = {"cached": spec["input_tokens_frozen"] - 10}
        # the shared template prefix alone is allowed
        self.server.script["performance-prompt-4k-r1"] = {"cached": 40}
        doc, _ = self.run_standard()
        self.assertEqual(doc["run"]["status"], "nonconforming")
        row = self.summary(doc, "prompt-4k", "prompt_tps")
        self.assertEqual(row["n"], 2)
        self.assertIn("unexpected cache reuse", row["excluded"][0]["reason"])
        self.assertTrue(any("unexpected reuse of the user payload in performance-prompt-4k-r2" in r
                            for r in doc["extensions"]["status_reasons"]))
        self.assertEqual(self.record(doc, "performance-prompt-4k-r1")["tokens"]["cached_tokens"], 40)
        self.assertEqual(bs.validate(doc), [])

    def test_a_prompt_length_off_its_target_is_nonconforming(self):
        original = bs.load_corpus()
        specs = copy.deepcopy(original.specs)
        for spec in specs:
            if spec["id"] == "performance-prompt-4k-r1":
                spec["target_input_tokens"] = 3000
        patched = bs.Corpus(original.sha256, specs, original.build, original.manifest, original.directory)
        doc, _ = self.run_standard(corpus=patched)
        self.assertEqual(doc["run"]["status"], "nonconforming")
        self.assertTrue(any("outside 2%" in r for r in doc["extensions"]["status_reasons"]))

    def test_a_refused_setting_is_nonconforming(self):
        self.server.script["warmup"] = {"http": 400}
        doc, _ = self.run_standard()
        self.assertEqual(doc["run"]["status"], "nonconforming")
        self.assertTrue(any("refused the protocol's required settings" in r for r in doc["extensions"]["status_reasons"]))
        self.assertEqual(self.record(doc, "sanity-I1")["status"], "skipped")

    def test_missing_identity_makes_the_run_nonconforming(self):
        with mock.patch.object(bench_facts, "git_commit", return_value=None):
            doc, _ = self.run_standard("nocommit")
        self.assertEqual(doc["run"]["status"], "nonconforming")
        self.assertIn("/software/strata_commit", doc["missing"])
        self.assertIsNone(doc["protocol"]["git_commit"])
        doc2, _ = self.run_standard("nohash", hash_model=False)
        self.assertEqual(doc2["run"]["status"], "nonconforming")
        self.assertIsNone(doc2["model"]["files"])
        self.assertIn("/model/files", doc2["missing"])

    def test_a_remote_server_is_nonconforming(self):
        with mock.patch.object(bs.Target, "loopback", new_callable=mock.PropertyMock, return_value=False):
            doc, _ = self.run_standard()
        self.assertEqual(doc["run"]["status"], "nonconforming")
        self.assertTrue(any("not on loopback" in r for r in doc["extensions"]["status_reasons"]))

    def test_cancelling_keeps_the_finished_requests_and_skips_the_rest(self):
        real = bs.stream_chat
        calls = []

        def interrupted(target, spec, model, **kwargs):
            calls.append(spec["id"])
            if len(calls) == 6:
                class Boom:
                    sock = None

                    def request(self, *args, **kw):
                        raise KeyboardInterrupt

                    def close(self):
                        pass
                with mock.patch.object(bs.Target, "connection", return_value=Boom()):
                    return real(target, spec, model, **kwargs)
            return real(target, spec, model, **kwargs)
        with mock.patch.object(bs, "stream_chat", interrupted):
            doc, _ = self.run_standard()
        self.assertEqual(doc["run"]["status"], "partial")
        self.assertEqual(self.record(doc, "performance-prompt-4k-r2")["error"]["kind"], "cancelled")
        self.assertEqual(self.record(doc, "performance-prompt-4k-r2")["status"], "failed")
        self.assertEqual(self.record(doc, "performance-prompt-16k-r3")["status"], "skipped")
        self.assertEqual(self.record(doc, "sanity-T4")["error"]["message"], "the operator cancelled the run")
        self.assertEqual(self.summary(doc, "short", "decode_tps")["n"], 3 - 1)
        self.assertEqual(bs.validate(doc), [])

    def test_the_optional_concurrency_stage_releases_four_together(self):
        for r in (1, 2, 3):
            for c in (1, 2, 3, 4):
                self.server.script[f"concurrency-r{r}-c{c}"] = {"hold_for_arrivals": 4}
        doc, _ = self.run_standard(optional_concurrency=True)
        self.assertEqual(doc["run"]["status"], "complete")
        self.assertEqual(len(doc["requests"]), 34)
        self.assertEqual(self.server.max_active, 4)
        self.assertEqual(doc["request_counts"]["concurrency"]["planned"], 12)
        self.assertEqual(len(doc["concurrency_rounds"]), 3)
        for round_ in doc["concurrency_rounds"]:
            self.assertEqual(round_["output_tokens"], 200)
            self.assertEqual(round_["failed"], 0)
            self.assertAlmostEqual(round_["aggregate_tps"], 200 / round_["wall_s"], places=3)
            self.assertEqual(len(round_["request_ids"]), 4)
        rows = [s for s in doc["summaries"] if s["phase"] == "concurrency"]
        self.assertEqual(len(rows), 5)
        self.assertEqual({(s["case_id"], s["concurrency"]) for s in rows}, {("short-burst", 4)})
        self.assertEqual(rows[0]["n"], 12)
        self.assertIsNotNone(doc["run"]["optional_duration_s"])
        for round_no in (1, 2, 3):                      # a round finishes before the next one is released

            self.assertEqual({n for n in self.server.order[22 + 4 * (round_no - 1): 22 + 4 * round_no]},
                             {f"concurrency-r{round_no}-c{c}" for c in (1, 2, 3, 4)})
        self.assertEqual(bs.validate(doc), [])

    def test_a_failed_concurrent_request_keeps_its_time_in_the_round(self):
        self.server.script["concurrency-r1-c2"] = {"http": 500}
        doc, _ = self.run_standard(optional_concurrency=True)
        first = doc["concurrency_rounds"][0]
        self.assertEqual((first["failed"], first["output_tokens"]), (1, 150))
        self.assertEqual(doc["concurrency_rounds"][1]["failed"], 0)
        self.assertEqual(doc["request_counts"]["concurrency"]["failed"], 1)

    def test_a_round_total_is_null_when_a_counted_request_has_no_token_count(self):
        self.server.script["concurrency-r2-c3"] = {"usage": False}
        doc, _ = self.run_standard(optional_concurrency=True)
        second = doc["concurrency_rounds"][1]
        self.assertIsNone(second["output_tokens"])
        self.assertIsNone(second["aggregate_tps"])
        self.assertIsNotNone(doc["concurrency_rounds"][0]["aggregate_tps"])
        self.assertEqual(bs.validate(doc), [])

    def test_the_plain_summary(self):
        doc, _ = self.run_standard()
        text = bs.format_plain(doc)
        self.assertIn("Status: complete. 21 of 21 required requests finished, 0 failed, 0 skipped.", text)
        self.assertIn("prompt-16k", text)
        self.assertIn("Sanity checks: 12 of 12 passed", text)
        self.assertIn("Prompt tok/s", text)
        self.assertIn("1000.0 (1000.0-1000.0) n=3", text)

    def test_the_summary_says_what_the_short_prompt_and_the_cap_mean(self):
        doc, _ = self.run_standard()
        for text in (bs.format_plain(doc), bs.report_markdown(doc)):
            self.assertIn("Capped", text)
            self.assertIn("not shown; read prompt speed from prompt-4k and prompt-16k", text)
            self.assertIn("time to the cap, not to an answer", text)
        short_row = next(line for line in bs.format_plain(doc).splitlines() if line.strip().startswith("short "))
        self.assertIn("n/a", short_row)
        self.assertNotIn("1000.0", short_row)
        self.assertIn("0/3", short_row)
        # the data keeps the short case's prompt rate: only the readable summaries leave it out
        self.assertIsNotNone(self.summary(doc, "short", "prompt_tps"))
        # a request that stopped at the cap is counted
        record = next(r for r in doc["requests"] if r["id"] == "performance-prompt-4k-r1")
        record["status"] = "truncated"
        row = next(line for line in bs.format_plain(doc).splitlines() if line.strip().startswith("prompt-4k "))
        self.assertTrue(row.rstrip().endswith("1/3"))

    def test_the_server_is_not_asked_for_anything_outside_the_protocol(self):
        self.run_standard()
        self.assertTrue(all(body["messages"] == corpus_specs()[rid]["messages"] for body, rid in zip(self.server.bodies, self.server.order)))


class ValidatorTests(StandardBase):
    def test_the_validator_refuses_broken_results(self):
        doc, _ = self.run_standard()
        self.assertEqual(bs.validate(doc), [])
        broken = copy.deepcopy(doc)
        broken["schema_version"] = 2
        self.assertTrue(bs.validate(broken))
        broken = copy.deepcopy(doc)
        broken["request_counts"]["performance"]["completed"] = 8
        self.assertTrue(any("identity" in p for p in bs.validate(broken)))
        broken = copy.deepcopy(doc)
        broken["missing"].pop(next(iter(broken["missing"])))
        self.assertTrue(any("missing must list" in p for p in bs.validate(broken)))
        broken = copy.deepcopy(doc)
        broken["summaries"][0]["median"] = 12345.0
        self.assertTrue(any("rebuilt" in p for p in bs.validate(broken)))
        broken = copy.deepcopy(doc)
        broken["requests"][1]["timings"]["ttft_s"] = float("nan")
        self.assertTrue(bs.validate(broken))
        broken = copy.deepcopy(doc)
        broken["surprise"] = 1
        self.assertTrue(any("extensions" in p for p in bs.validate(broken)))
        broken = copy.deepcopy(doc)
        broken["artifacts"][0]["path"] = "../escape"
        self.assertTrue(any("artifact" in p for p in bs.validate(broken)))

    def test_load_reads_a_saved_result_and_refuses_other_files(self):
        doc, out = self.run_standard()
        self.assertEqual(bs.load(out)["run"]["id"], doc["run"]["id"])
        self.assertIsNone(bs.load_if_standard(self.root / "nothing"))
        other = self.root / "other.json"
        other.write_text(json.dumps({"kind": "strata.serving.benchmark"}), encoding="utf-8")
        self.assertIsNone(bs.load_if_standard(other))
        bad = json.loads((out / "results.json").read_text(encoding="utf-8"))
        bad["schema_version"] = 9
        (self.root / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaises(ProfileError):
            bs.load_if_standard(self.root / "bad.json")


class CommandLineTests(StandardBase):
    def invoke(self, *args, env=None):
        output, errors = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, env or {}), contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = main(["bench", "run", *args])
        return code, output.getvalue(), errors.getvalue()

    def test_the_standard_suite_is_the_default_and_prints_a_plain_summary(self):
        out = self.root / "cli"
        code, text, errors = self.invoke("--url", self.server.url, "--out", str(out))
        self.assertEqual(code, 0, errors)
        self.assertIn("Strata standard benchmark (strata-qwen38-thinking-v1-draft.1)", text)
        self.assertIn("Sanity checks: 12 of 12 passed", text)
        self.assertIn("Report:", text)
        self.assertEqual(bs.load(out)["run"]["status"], "complete")
        self.assertIn("[1/22] warmup", errors)

    def test_json_output(self):
        code, text, _ = self.invoke("--url", self.server.url, "--out", str(self.root / "j"), "--output", "json")
        self.assertEqual(code, 0)
        value = json.loads(text)
        self.assertTrue(value["ok"])
        self.assertEqual(value["quality"]["passed"], 12)
        self.assertEqual(len(value["summaries"]), 15)

    def test_a_partial_run_exits_with_6(self):
        self.server.script["warmup"] = {"http": 500}
        code, text, _ = self.invoke("--url", self.server.url, "--out", str(self.root / "p"))
        self.assertEqual(code, 6)
        self.assertIn("Status: partial", text)
        self.assertIn("Why this is not a complete standard result", text)

    def test_serving_options_do_not_apply(self):
        for option in (("--concurrency", "2"), ("--repeat", "2"), ("--request-timeout", "30"), ("--condition", "baseline")):
            code, text, _ = self.invoke("--url", self.server.url, "--out", str(self.root / "x"), "--output", "json", *option)
            self.assertEqual(code, 2, option)
            self.assertIn("do not apply to the standard suite", text)
        self.assertFalse((self.root / "x").exists())

    def test_the_out_folder_must_be_new_and_the_default_goes_under_bench_results(self):
        (self.root / "taken").mkdir()
        code, text, _ = self.invoke("--url", self.server.url, "--out", str(self.root / "taken"), "--output", "json")
        self.assertEqual(code, 2)
        previous = Path.cwd()
        os.chdir(self.root)
        try:
            code, text, _ = self.invoke("--url", self.server.url)
        finally:
            os.chdir(previous)
        self.assertEqual(code, 0)
        self.assertTrue(any((self.root / "bench-results").glob("*-standard-*/results.json")))

    def test_the_serving_suites_still_need_out_and_keep_their_defaults(self):
        code, text, _ = self.invoke("--url", self.server.url, "--suite", "serving-smoke", "--output", "json")
        self.assertEqual(code, 2)
        self.assertIn("--out is required for the serving suites", text)

    def test_the_options_reach_the_report(self):
        out = self.root / "opts"
        code, _, errors = self.invoke("--url", self.server.url, "--out", str(out), "--track", "default", "--track-notes",
                                      "0.2.0 release, setup choices", "--model-revision", "r1", "--source-revision", "r2",
                                      "--tokenizer-revision", "r3", "--custom-pack", "none", "--background", "browser open",
                                      "--build-options", "cuda,sm86", "--optional-concurrency")
        self.assertEqual(code, 0, errors)
        doc = bs.load(out)
        self.assertEqual((doc["configuration"]["track"], doc["configuration"]["track_notes"]), ("default", "0.2.0 release, setup choices"))
        self.assertEqual((doc["model"]["revision"], doc["model"]["source_revision"], doc["model"]["tokenizer_revision"]), ("r1", "r2", "r3"))
        self.assertEqual(doc["hardware"]["background_workloads"], ["browser open"])
        self.assertEqual(doc["software"]["build_options"], ["cuda", "sm86"])
        self.assertTrue(doc["run"]["optional_concurrency"])
        self.assertIn("--track default", doc["run"]["command"])
        self.assertTrue(doc["run"]["command"].startswith("strata bench run"))

    def test_report_reads_a_standard_result_in_three_forms(self):
        out = self.root / "rep"
        self.assertEqual(self.invoke("--url", self.server.url, "--out", str(out))[0], 0)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["bench", "report", str(out)]), 0)
        self.assertIn("Sanity checks: 12 of 12 passed", output.getvalue())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["bench", "report", str(out), "--output", "markdown"]), 0)
        self.assertIn("# Strata standard benchmark", output.getvalue())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["bench", "report", str(out), "--output", "json"]), 0)
        self.assertEqual(json.loads(output.getvalue())["summary"]["status"], "complete")

    def test_a_profile_that_is_not_running_is_started_and_stopped(self):
        out = self.root / "prof"
        cfg = {"api_key": "", "model_name": "mock"}
        plan = {"effective": {"host": "127.0.0.1", "port": int(self.server.url.rsplit(":", 1)[1])}, "engine_argv": ["--kv", "int8", "--expert-cache", "1500"]}
        health = {"calls": 0}
        real_get = bs.Target.get

        def get(target, path, timeout=10):
            if path == "/health" and health["calls"] < 1:
                health["calls"] += 1
                return None                  # nothing answers before the engine is started
            return real_get(target, path, timeout)
        started = []
        with mock.patch("strata_cli.cli._source", return_value=(Path("p.json"), None, None, cfg)), \
                mock.patch("strata_cli.cli._launch_plan", return_value=(plan, [])), \
                mock.patch.object(bs.Target, "get", get), \
                mock.patch.object(bs, "start_engine", side_effect=lambda ref, target, log, say, wait: started.append(ref) or "proc") as start, \
                mock.patch.object(bs, "stop_engine") as stop:
            code, text, errors = self.invoke("--profile", "p.json", "--out", str(out))
        self.assertEqual(code, 0, errors)
        self.assertEqual(started, ["p.json"])
        stop.assert_called_once_with("proc")
        doc = bs.load(out)
        self.assertTrue(doc["extensions"]["started_engine"])
        self.assertEqual(doc["configuration"]["kv_cache"], {"--kv": "int8"})
        self.assertEqual(doc["configuration"]["expert_cache"], {"--expert-cache": "1500"})
        self.assertNotIn("/configuration/kv_cache", doc["missing"])

    def test_a_running_profile_is_used_and_left_running(self):
        plan = {"effective": {"host": "127.0.0.1", "port": int(self.server.url.rsplit(":", 1)[1])}, "engine_argv": []}
        with mock.patch("strata_cli.cli._source", return_value=(Path("p.json"), None, None, {})), \
                mock.patch("strata_cli.cli._launch_plan", return_value=(plan, [])), \
                mock.patch.object(bs, "start_engine") as start, mock.patch.object(bs, "stop_engine") as stop:
            code, text, errors = self.invoke("--profile", "p.json", "--out", str(self.root / "run"))
        self.assertEqual(code, 0, errors)
        start.assert_not_called()
        stop.assert_called_once_with(None)
        self.assertIn("already running", errors)
        self.assertFalse(bs.load(self.root / "run")["extensions"]["started_engine"])

    def test_no_start_refuses_to_launch(self):
        self.server.close()
        plan = {"effective": {"host": "127.0.0.1", "port": int(self.server.url.rsplit(":", 1)[1])}, "engine_argv": []}
        with mock.patch("strata_cli.cli._source", return_value=(Path("p.json"), None, None, {})), \
                mock.patch("strata_cli.cli._launch_plan", return_value=(plan, [])), \
                mock.patch.object(bs, "start_engine") as start:
            code, text, _ = self.invoke("--profile", "p.json", "--out", str(self.root / "ns"), "--no-start", "--output", "json")
        self.assertEqual(code, 5)
        start.assert_not_called()


class EngineProcessTests(unittest.TestCase):
    def test_start_engine_runs_serve_with_the_profile_and_stop_ends_only_that_process(self):
        class Fake:
            pid = 4242

            def poll(self):
                return None

            def wait(self, timeout=None):
                return 0
        seen = {}

        def popen(command, **kwargs):
            seen["command"] = command
            return Fake()
        target = mock.Mock()
        target.get.return_value = {"service": "strata", "loaded": True}
        with tempfile.TemporaryDirectory() as temp, mock.patch("subprocess.Popen", popen):
            proc = bs.start_engine("run.json", target, Path(temp) / "engine.log", lambda m: None, 5)
            self.assertEqual(seen["command"][1].replace("\\", "/").rsplit("/", 1)[-1], "strata.py")
            self.assertEqual(seen["command"][2:5], ["serve", "--profile", "run.json"])
            with mock.patch("subprocess.run") as run, mock.patch("os.killpg", create=True) as killpg:
                bs.stop_engine(proc)
            if os.name == "nt":
                self.assertEqual(run.call_args[0][0][:4], ["taskkill", "/PID", "4242", "/T"])
            else:
                killpg.assert_called()

    # #3438: the stop used to ignore taskkill's result and the wait timeout, so a stop that failed still ended as a normal finish
    def _stuck_proc(self):
        class Stuck:
            pid = 777

            def poll(self):
                return None          # never exits

            def wait(self, timeout=None):
                raise bs.subprocess.TimeoutExpired("x", timeout)

            def terminate(self):
                pass

            def kill(self):
                pass
        return Stuck()

    def test_stop_engine_says_so_when_the_process_survives(self):
        proc = self._stuck_proc()
        done = bs.subprocess.CompletedProcess([], 1, b"", b"ERROR: Access is denied.")
        with mock.patch("subprocess.run", return_value=done), mock.patch("os.killpg", create=True):
            problem = bs.stop_engine(proc)
        self.assertIsNotNone(problem)
        self.assertIn("777", problem)
        self.assertIn("still running", problem)
        if os.name == "nt":
            self.assertIn("taskkill", problem)
            self.assertIn("Access is denied", problem)

    @unittest.skipUnless(os.name == "nt", "taskkill is the Windows stop")
    def test_stop_engine_reports_a_taskkill_that_times_out(self):
        proc = self._stuck_proc()
        with mock.patch("subprocess.run", side_effect=bs.subprocess.TimeoutExpired("taskkill", 60)):
            problem = bs.stop_engine(proc)
        self.assertIn("did not finish in 60 s", problem)

    def test_stop_engine_is_quiet_when_the_process_ends(self):
        class Ends:
            pid = 5
            alive = True

            def poll(self):
                return None if self.alive else 0

            def wait(self, timeout=None):
                self.alive = False
                return 0

            def terminate(self):
                self.alive = False
        proc = Ends()
        done = bs.subprocess.CompletedProcess([], 0, b"", b"")

        def kill(*args, **kwargs):
            proc.alive = False
            return done
        with mock.patch("subprocess.run", side_effect=kill), mock.patch("os.killpg", create=True, side_effect=lambda *a: setattr(proc, "alive", False)):
            self.assertIsNone(bs.stop_engine(proc))
        self.assertIsNone(bs.stop_engine(None))
        # a process that exited by itself needs no kill at all
        gone = mock.Mock()
        gone.poll.return_value = 0
        self.assertIsNone(bs.stop_engine(gone))

    def test_a_run_whose_engine_cannot_be_stopped_is_marked_failed_and_loud(self):
        # cli: the run finished, the stop failed -> exit code 6, an ERROR line on stderr, a note in the folder, ok false in the json
        plan = {"effective": {"host": "127.0.0.1", "port": 1}, "engine_argv": [], "config": {}}
        fake_doc = {"run": {"status": "complete"}, "quality": {}, "summaries": []}
        out_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, out_dir, True)
        (out_dir / "results.json").write_text("{}", encoding="utf-8")
        (out_dir / "README.md").write_text("x", encoding="utf-8")
        with mock.patch("strata_cli.cli._source", return_value=(Path("p.json"), None, None, {})), \
                mock.patch("strata_cli.cli._launch_plan", return_value=(plan, [])), \
                mock.patch.object(bs, "start_engine", return_value=object()), \
                mock.patch.object(bs, "run", return_value=(fake_doc, out_dir)), \
                mock.patch.object(bs, "stop_engine", return_value="process 9 is still running 30 s after it was told to stop"), \
                mock.patch("strata_cli.client.server_url", side_effect=lambda u: u):
            err = io.StringIO()
            out = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
                try:
                    code = main(["bench", "run", "--suite", "standard-v1", "--profile", "p.json", "--out", str(out_dir), "--output", "json"])
                except SystemExit as stop:
                    code = stop.code
        self.assertEqual(code, 6)
        self.assertIn("could not be stopped", err.getvalue())
        self.assertIn("FAILED", err.getvalue())
        self.assertTrue((out_dir / "ENGINE-NOT-STOPPED.txt").is_file())

    def test_an_engine_that_exits_before_it_is_ready_is_an_error(self):
        class Dead:
            pid = 1

            def poll(self):
                return 3

            def wait(self, timeout=None):
                return 3
        target = mock.Mock()
        target.get.return_value = None
        with tempfile.TemporaryDirectory() as temp, mock.patch("subprocess.Popen", lambda *a, **k: Dead()):
            with self.assertRaisesRegex(ProfileError, "did not become ready"):
                bs.start_engine("run.json", target, Path(temp) / "engine.log", lambda m: None, 5)


class RealServerTests(unittest.TestCase):
    """The same request bodies against the real API server with its mock engine: every protocol setting is accepted."""

    def test_the_real_server_accepts_the_protocol_requests(self):
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service, serve
        with tempfile.TemporaryDirectory() as temp:
            tok = ByteTokenizer()
            template = ChatTemplate(Path(__file__).resolve().parents[2] / "serve/chat_template.jinja")
            service = Service(MockEngine(tok, "A hash table hashes the key.", max_context=400000), tok, template)
            service.api_key = "inference-test-secret"
            httpd = serve(service, port=0)
            self.addCleanup(httpd.server_close)
            self.addCleanup(httpd.shutdown)
            target = bs.Target(f"http://127.0.0.1:{httpd.server_address[1]}", "inference-test-secret")
            patches = [mock.patch.object(bench_facts, "git_commit", return_value=COMMIT)]
            for patch in patches:
                patch.start()
                self.addCleanup(patch.stop)
            doc, out = bs.run(target, Path(temp) / "real", hash_model=True)
        self.assertEqual(bs.validate(doc), [])
        failed = [(r["id"], r["error"]) for r in doc["requests"] if r["status"] == "failed"]
        self.assertEqual(failed, [])
        self.assertTrue(all(r["status"] == "completed" for r in doc["requests"]))
        self.assertEqual(doc["configuration"]["template"]["sha256"], bench_facts.sha256_text(template.source))
        self.assertIsNotNone(doc["configuration"]["context_tokens"])
        # the mock engine has no clock: the rates stay null and say why instead of being estimated
        record = next(r for r in doc["requests"] if r["id"] == "performance-short-r1")
        self.assertIsNone(record["metrics"]["decode_tps"])
        self.assertIsNotNone(record["tokens"]["input_tokens"])
        self.assertIsNotNone(record["timings"]["ttft_s"])
        self.assertEqual(doc["run"]["status"], "nonconforming")          # no model file to hash, so the identity is incomplete


class RealRunFindingsTests(StandardBase):
    """What the first run on release engines (RTX 3060, 2x RTX A4000, 0.1.46) got wrong."""

    LVM = {"blockdevices": [{"name": "ubuntu--vg-data", "type": "lvm", "rota": False, "model": None, "tran": None, "children": [
        {"name": "nvme0n1p3", "type": "part", "rota": False, "model": None, "tran": "nvme", "children": [
            {"name": "nvme0n1", "type": "disk", "rota": False, "model": "Lexar SSD NM620 512GB", "tran": "nvme"}]}]}]}

    def test_the_drive_under_an_lvm_volume_is_found(self):
        self.assertEqual(bench_facts.lsblk_disk(self.LVM), [{"kind": "nvme", "model": "Lexar SSD NM620 512GB", "bus": "nvme"}])

    def test_a_partition_on_a_disk_and_a_missing_disk(self):
        plain = {"blockdevices": [{"name": "sda1", "type": "part", "children": [
            {"name": "sda", "type": "disk", "rota": True, "model": "Spinner", "tran": "sata"}]}]}
        self.assertEqual(bench_facts.lsblk_disk(plain), [{"kind": "hdd", "model": "Spinner", "bus": "sata"}])
        self.assertIsNone(bench_facts.lsblk_disk({"blockdevices": [{"name": "loop0", "type": "loop"}]}))

    def test_the_runtime_is_the_one_the_engine_was_built_with(self):
        engine = self.root / "engine"
        engine.mkdir()
        exe = engine / "strata"
        exe.write_bytes(b"")
        self.assertIsNone(bench_facts.engine_build_runtime(str(exe)))
        (engine / "engine-component.json").write_text(json.dumps({"build_id": "a7bdbfdabc9a+cuda12.9"}), encoding="utf-8")
        self.assertEqual(bench_facts.engine_build_runtime(str(exe)), {"name": "cuda", "version": "12.9"})
        (engine / "engine-component.json").write_text(json.dumps({"build_id": "abc+hip6.4"}), encoding="utf-8")
        self.assertEqual(bench_facts.engine_build_runtime(str(exe)), {"name": "rocm", "version": "6.4"})
        why = bench_facts.Why()
        block = bench_facts.software(why, {"driver": "580.1", "cuda": "13.0"}, engine_version="0.1.46", build_options=None,
                                     engine_build={"name": "cuda", "version": "12.9"})
        self.assertEqual(block["runtime"], [{"name": "cuda", "version": "12.9"}, {"name": "cuda-driver-supports", "version": "13.0"}])

    def test_the_engine_log_comes_from_the_run_configs_log_file(self):
        console = self.root / "engine.log"
        console.write_bytes(b"")
        run_log = self.root / "strata-model.log"
        run_log.write_bytes(b"old line from an earlier start\n")
        offset = run_log.stat().st_size
        with run_log.open("ab") as stream:
            stream.write(b"strata serve: expert cache 1500 slots\n")
        kept = bs.engine_log_since(console, run_log, offset)
        self.assertEqual(kept.read_bytes(), b"strata serve: expert cache 1500 slots\n")
        self.assertEqual(bs.engine_log_since(console, None, 0), console)
        self.assertEqual(bs.engine_log_since(console, self.root / "missing.log", 0), console)
        self.assertEqual(bs.engine_log_since(console, run_log, run_log.stat().st_size), console)

    def test_a_freshly_loaded_server_is_reported_as_fresh(self):
        doc, out = self.run_standard()
        doc["extensions"]["server_requests_before_run"] = 0
        self.assertIn("freshly loaded engine", bs.report_markdown(doc))
        self.assertNotIn("could not tell", bs.report_markdown(doc))
        doc["extensions"]["server_requests_before_run"] = None
        self.assertIn("could not tell", bs.report_markdown(doc))
        doc["extensions"]["server_requests_before_run"] = 3
        self.assertIn("answered 3 request(s)", bs.report_markdown(doc))


class NvidiaBannerTests(unittest.TestCase):
    def test_the_cuda_version_is_read_from_old_and_new_driver_banners(self):
        query = "0, NVIDIA GeForce RTX 3060, 12288, 170.00, 615.71.09"
        for banner, version in (("| NVIDIA-SMI 580.1  Driver Version: 580.1     CUDA Version: 13.0 |", "13.0"),
                                ("| NVIDIA-SMI 615.71  KMD Version: 615.71.09     CUDA UMD Version: 13.4 |", "13.4")):
            def fake(cmd, banner=banner):
                return query if any(a.startswith("--query-gpu") for a in cmd) else banner
            with mock.patch.object(bench_facts, "run_cmd", side_effect=fake):
                cards, driver, cuda = bench_facts.nvidia_gpus()
            self.assertEqual((driver, cuda), ("615.71.09", version))


class ArchiveCommitTests(unittest.TestCase):
    """The release archive carried no COMMIT file, so every run on it was nonconforming."""

    ROOT = Path(__file__).resolve().parents[2]

    def setUp(self):
        import sys
        sys.path.insert(0, str(self.ROOT / "packaging"))
        import package_terminal
        self.pt = package_terminal

    def test_an_explicit_commit_or_the_variable_wins_over_git(self):
        sha = "ab" * 20
        self.assertEqual(self.pt.build_commit(sha.upper()), sha)
        with mock.patch.dict(os.environ, {"STRATA_COMMIT": "cd" * 20}):
            self.assertEqual(self.pt.build_commit(None), "cd" * 20)

    def test_a_tree_copy_without_git_has_no_commit(self):
        with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(self.pt.subprocess, "run") as run:
            os.environ.pop("STRATA_COMMIT", None)
            run.return_value = mock.Mock(stdout="fatal: not a git repository\n")
            self.assertIsNone(self.pt.build_commit("not a sha"))

    def test_a_distributable_build_refuses_to_run_without_a_commit(self):
        with tempfile.TemporaryDirectory() as out, mock.patch.object(self.pt, "build_commit", return_value=None), mock.patch("sys.argv", ["package_terminal.py", "--out", out]):
            with self.assertRaises(SystemExit) as raised:
                self.pt.main()
        self.assertIn("COMMIT", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
