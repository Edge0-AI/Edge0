"""`strata bench run` for the standard benchmark protocol (docs/BENCHMARK_PROTOCOL.md).

The runner sends the frozen requests of bench/prompts/standard-v1 to a Strata server through streaming chat completions, in the
protocol's order, and writes a `strata-standard-benchmark` result that other people's results can be compared with:

    warm-up (1, excluded)  ->  performance (short, prompt-4k, prompt-16k, three times)  ->  sanity (I1-I4, R1-R4, T1-T4)
    ->  optional concurrency (four short requests released together, three rounds)

Speeds are the engine's own timings (the `timings` of the last stream chunk); first-token times are the client's monotonic clock. A value
that cannot be read stays `None` with a reason in `missing`; nothing is estimated from chunks, characters or words. Prompts, answers and
timings are saved without home folders, host names, private addresses or keys.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from strata_cli import bench_facts as facts
from strata_cli.bench import _private_mkdir, _write_new
from strata_control.profiles import ProfileError

FORMAT = "strata-standard-benchmark"
SCHEMA_VERSION = 1
PROTOCOL_ID = "strata-qwen38-thinking-v1-draft.1"
PROFILE = "qwen3.8-flash-next-thinking"
SUITE = "standard-v1"
#: The model-card revision the profile was defined from. The owner has not pinned one yet, so every result says so.
MODEL_CARD_REVISION: str | None = None
MODEL_CARD_URL = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next"

REQUEST_TIMEOUT_S = 1800.0
#: Prompt tokens the cache may report beyond the longest prefix a request shares with the rest of the bundle before the
#: user payload counts as reused. The slack covers token merges at the boundary.
REUSE_SLACK_TOKENS = 8
LENGTH_TOLERANCE = 0.02

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
            "repetition_penalty": 1.0, "reasoning_effort": "xhigh"}
TEMPLATE_KWARGS = {"enable_thinking": True, "preserve_thinking": True}
PHASES = ("warmup", "performance", "sanity", "concurrency")
PLANNED = {"warmup": 1, "performance": 9, "sanity": 12, "concurrency": 12}
PERFORMANCE_CASES = ("short", "prompt-4k", "prompt-16k")
METRICS = (("prompt_tps", "tokens/s"), ("decode_tps", "tokens/s"), ("ttft_s", "s"),
           ("first_answer_s", "s"), ("total_s", "s"))
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class CorpusError(ProfileError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def _count(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        if isinstance(value, float) and value.is_integer() and value >= 0 and math.isfinite(value):
            return int(value)
        return None
    return value


# ------------------------------------------------------------------------------------------------------------ the fixtures
class Corpus:
    def __init__(self, sha256: str, specs: list[dict], build: dict, manifest: bytes, directory: Path):
        self.sha256, self.specs, self.build, self.manifest, self.directory = sha256, specs, build, manifest, directory


def corpus_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "bench" / "prompts" / SUITE


def load_corpus(directory: Path | None = None) -> Corpus:
    """Read the fixture bundle and check every file against its manifest. Any difference is an error: the run would not be the protocol's."""
    root = Path(directory) if directory else corpus_dir()
    try:
        manifest = (root / "MANIFEST.sha256").read_bytes()
        lines = manifest.decode("utf-8").splitlines()
    except (OSError, UnicodeError):
        raise CorpusError("the standard benchmark fixtures are missing (bench/prompts/standard-v1/MANIFEST.sha256)", 5) from None
    entries: dict[str, str] = {}
    for line in lines:
        digest, _, name = line.partition("  ")
        if not HEX64.match(digest) or not name or name.startswith(("/", "\\")) or ".." in Path(name).parts:
            raise CorpusError("the fixture manifest is malformed", 5)
        entries[name] = digest
    for name, digest in entries.items():
        try:
            data = (root / name).read_bytes()
        except OSError:
            raise CorpusError(f"fixture file missing: {name}", 5) from None
        if hashlib.sha256(data).hexdigest() != digest:
            raise CorpusError(f"fixture file changed: {name} (line endings converted? bench/prompts is hashed byte for byte)", 5)
    extra = {p.relative_to(root).as_posix() for p in (root / "requests").glob("*.json")} - set(entries)
    if extra:
        raise CorpusError(f"fixture file not in the manifest: {sorted(extra)[0]}", 5)
    try:
        build = json.loads((root / "build.json").read_text(encoding="utf-8"))
        specs = [json.loads((root / "requests" / f"{row['id']}.json").read_text(encoding="utf-8")) for row in build["requests"]]
    except (OSError, ValueError, KeyError, TypeError):
        raise CorpusError("the fixture bundle is unreadable", 5) from None
    counts = {p: sum(1 for s in specs if s.get("phase") == p) for p in PHASES}
    if counts != PLANNED or len({s["id"] for s in specs}) != len(specs) or build.get("protocol_id") != PROTOCOL_ID:
        raise CorpusError("the fixture bundle does not match this protocol", 5)
    return Corpus(hashlib.sha256(manifest).hexdigest(), specs, build, manifest, root)


# ------------------------------------------------------------------------------------------------------------ HTTP
class Target:
    def __init__(self, root: str, key: str | None):
        parts = urllib.parse.urlsplit(root)
        self.scheme, self.host = parts.scheme, parts.hostname or ""
        self.port = parts.port or (443 if parts.scheme == "https" else 80)
        self.key = key
        self.root = root.rstrip("/")

    @property
    def loopback(self) -> bool:
        return self.host in facts.LOOPBACK

    def connection(self, timeout: float) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=timeout)

    def headers(self, extra: dict | None = None) -> dict:
        out = {"User-Agent": "Strata-Bench/1", "Accept": "application/json", "Connection": "close", **(extra or {})}
        if self.key:
            out["Authorization"] = "Bearer " + self.key
        return out

    def get(self, path: str, timeout: float = 10) -> dict | None:
        conn = self.connection(timeout)
        try:
            conn.request("GET", path, headers=self.headers())
            response = conn.getresponse()
            body = response.read(8 << 20)
            if response.status != 200:
                return None
            value = json.loads(body)
            return value if isinstance(value, dict) else None
        except (OSError, http.client.HTTPException, ValueError):
            return None
        finally:
            conn.close()


def payload_of(spec: dict, model: str | None) -> dict:
    """The request body: the protocol's sampling and thinking settings, every one explicit, plus the fixture's messages and tools."""
    body = {"model": model or "strata", "messages": spec["messages"], "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": spec["max_output_tokens"], "seed": spec["seed"],
            **SAMPLING, "reasoning_budget_tokens": 0, "chat_template_kwargs": dict(TEMPLATE_KWARGS)}
    if spec.get("tools"):
        body["tools"] = spec["tools"]
        body["tool_choice"] = spec.get("tool_choice") or "auto"
    return body


def stream_chat(target: Target, spec: dict, model: str | None, *, timeout_s: float,
                gate: threading.Event | None = None) -> dict:
    """Send one streamed request and keep what the client saw, with monotonic offsets from the moment of dispatch."""
    body = json.dumps(payload_of(spec, model)).encode("utf-8")
    raw: dict = {"http_status": None, "error": None, "finish_reason": None, "usage": None, "timings": None,
                 "events": [], "reasoning": [], "content": [], "tool_calls": {}, "done": False,
                 "timed_out": False, "interrupted": False}
    if gate is not None:
        gate.wait()
    raw["t0"] = start = time.monotonic_ns()
    conn = target.connection(timeout_s)

    def expire():
        raw["timed_out"] = True
        sock = getattr(conn, "sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
    timer = threading.Timer(timeout_s, expire)
    timer.daemon = True
    timer.start()

    def offset() -> float:
        return round((time.monotonic_ns() - start) / 1e9, 6)

    def mark(kind: str) -> None:
        raw["events"].append([offset(), kind])

    try:
        conn.request("POST", "/v1/chat/completions", body,
                     target.headers({"Content-Type": "application/json", "Accept": "text/event-stream"}))
        response = conn.getresponse()
        raw["http_status"] = response.status
        if response.status != 200:
            text = response.read(4096).decode("utf-8", "replace")
            raw["error"] = {"kind": f"http_{response.status}", "message": text.strip()[:300] or f"HTTP {response.status}"}
        else:
            while True:
                line = response.readline(1 << 20)
                if not line:
                    break
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    raw["done"] = True
                    break
                try:
                    event = json.loads(data)
                except ValueError:
                    raw["error"] = {"kind": "malformed_stream", "message": "a stream event was not JSON"}
                    break
                if not isinstance(event, dict):
                    raw["error"] = {"kind": "malformed_stream", "message": "a stream event was not an object"}
                    break
                if "error" in event:
                    err = event["error"]
                    kind = (err.get("code") or err.get("type")) if isinstance(err, dict) else None
                    raw["error"] = {"kind": str(kind or "stream_error")[:60],
                                    "message": str(err.get("message") if isinstance(err, dict) else err)[:300]}
                    break
                if isinstance(event.get("usage"), dict):
                    raw["usage"] = event["usage"]
                if isinstance(event.get("timings"), dict):
                    raw["timings"] = event["timings"]
                for choice in event.get("choices") or []:
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
                    reasoning, text = delta.get("reasoning_content"), delta.get("content")
                    if isinstance(reasoning, str) and reasoning:
                        raw["reasoning"].append(reasoning)
                        mark("reasoning")
                    if isinstance(text, str) and text:
                        raw["content"].append(text)
                        mark("answer")
                    calls = delta.get("tool_calls")
                    if isinstance(calls, list):
                        qualifies = False
                        for call in calls:
                            if not isinstance(call, dict):
                                continue
                            slot = raw["tool_calls"].setdefault(call.get("index", len(raw["tool_calls"])),
                                                                {"id": "", "name": "", "arguments": ""})
                            function = call.get("function") if isinstance(call.get("function"), dict) else {}
                            for field, value in (("id", call.get("id")), ("name", function.get("name")),
                                                 ("arguments", function.get("arguments"))):
                                if isinstance(value, str) and value:
                                    slot[field] += value
                                    qualifies = True
                        if qualifies:
                            mark("tool_call")
                    if choice.get("finish_reason"):
                        raw["finish_reason"] = str(choice["finish_reason"])
            if raw["error"] is None:
                if not raw["done"]:
                    raw["error"] = {"kind": "malformed_stream", "message": "the stream ended without [DONE]"}
                elif raw["finish_reason"] is None:
                    raw["error"] = {"kind": "malformed_stream", "message": "the stream carried no finish_reason"}
    except KeyboardInterrupt:
        raw["interrupted"] = True
        raw["error"] = {"kind": "cancelled", "message": "cancelled by the operator"}
    except (OSError, http.client.HTTPException, ValueError) as exc:
        if raw["timed_out"]:
            raw["error"] = {"kind": "timeout", "message": f"no complete answer within {timeout_s:g} seconds"}
        else:
            raw["error"] = {"kind": "connection_" + type(exc).__name__, "message": str(exc)[:200] or type(exc).__name__}
    finally:
        timer.cancel()
        try:
            conn.close()
        except OSError:
            pass
    if raw["timed_out"] and raw["error"] is None:
        raw["error"] = {"kind": "timeout", "message": f"no complete answer within {timeout_s:g} seconds"}
    raw["t_end"] = time.monotonic_ns()
    return raw


# ------------------------------------------------------------------------------------------------------------ scoring
def strict_json(text: str):
    """One complete JSON value: no duplicate keys, no NaN or Infinity, no prose or fence around it."""
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"duplicate key {key!r}")
            out[key] = value
        return out

    def constant(name):
        raise ValueError(f"{name} is not JSON")
    return json.loads(text.strip(" \t\r\n\x0b\x0c"), object_pairs_hook=pairs, parse_constant=constant)


def same_json(a, b) -> bool:
    """Equal values of equal JSON types (true is not 1, 3 is not 3.0); object key order does not matter."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_json(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_json(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def evaluate(spec: dict, status: str, answer: str, calls: list[dict]) -> dict:
    expected = spec["expected"]
    category = spec["category"]

    def verdict(passed: bool, reason: str) -> dict:
        return {"category": category, "passed": passed, "reason": reason}
    if status != "completed":
        return verdict(False, {"truncated": "the output cap was reached before the answer was complete",
                               "failed": "the request failed", "skipped": "the request was skipped"}[status])
    kind = expected["kind"]
    if kind == "tool_call":
        if len(calls) != 1:
            return verdict(False, f"expected one structured tool call, got {len(calls)}")
        call = calls[0]
        if call["name"] != expected["name"]:
            return verdict(False, f"wrong function {call['name']!r}")
        try:
            arguments = strict_json(call["arguments"] or "{}")
        except ValueError as exc:
            return verdict(False, f"the tool arguments are not strict JSON ({exc})")
        if not isinstance(arguments, dict) or not same_json(arguments, expected["arguments"]):
            return verdict(False, "the tool arguments differ from the expected values")
        return verdict(True, "the expected tool call")
    if kind == "text":
        text = answer.strip(" \t\r\n\x0b\x0c")
        return verdict(text == expected["value"],
                       "the exact expected text" if text == expected["value"] else "the answer text differs from the expected text")
    try:
        value = strict_json(answer)
    except ValueError as exc:
        return verdict(False, f"the answer is not one strict JSON value ({exc})")
    return verdict(same_json(value, expected["value"]),
                   "the expected JSON value" if same_json(value, expected["value"]) else "the JSON value differs from the expected value")


# ------------------------------------------------------------------------------------------------------------ one request
def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def record_of(spec: dict, raw: dict, meta: dict, why: facts.Why | None = None) -> tuple[dict, dict, dict]:
    """-> (the report's request record, evidence for outputs.jsonl, extras for the report's `extensions`)."""
    usage = raw["usage"] if isinstance(raw.get("usage"), dict) else {}
    timings = raw["timings"] if isinstance(raw.get("timings"), dict) else {}
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    error = raw["error"]
    finish = raw["finish_reason"]
    if error is not None:
        status = "failed"
    elif finish == "length":
        status = "truncated"
    elif finish in ("stop", "tool_calls"):
        status = "completed"
    else:
        status = "failed"
        error = {"kind": "unexpected_finish_reason", "message": f"finish_reason {finish!r}"}
    t0 = raw["t0"]
    events = raw["events"]
    first = events[0] if events else None
    answer = next((e for e in events if e[1] in ("answer", "tool_call")), None)

    input_tokens, output_tokens = _count(usage.get("prompt_tokens")), _count(usage.get("completion_tokens"))
    cached, cached_source = _count(details.get("cached_tokens")), "usage.prompt_tokens_details.cached_tokens"
    if cached is None:
        cached, cached_source = _count(timings.get("cache_n")), "timings.cache_n"
    read, read_source = _count(timings.get("prompt_n")), "timings.prompt_n"
    if read is None and input_tokens is not None and cached is not None and cached <= input_tokens:
        read, read_source = input_tokens - cached, "derived: usage.prompt_tokens - cached tokens"
    prompt_s = _number(timings.get("prompt_ms"))
    prompt_s = prompt_s / 1000 if prompt_s is not None else None
    decode_s = _number(timings.get("predicted_ms"))
    decode_s = decode_s / 1000 if decode_s is not None else None
    rate, predicted_n = _number(timings.get("predicted_per_second")), _count(timings.get("predicted_n"))
    decode_tokens, decode_source, note = None, None, None
    if rate is not None and decode_s:
        implied = round(rate * decode_s)
        # the engine rounds its rate to 0.1 tokens/s, so the implied count is good to about 0.05 * decode_s tokens
        if predicted_n is not None and abs(predicted_n - implied) <= 0.05 * decode_s + 1:
            decode_tokens, decode_source = predicted_n, "timings.predicted_n (agrees with timings.predicted_per_second x predicted_ms)"
        else:
            decode_tokens = implied
            decode_source = "derived: timings.predicted_per_second x timings.predicted_ms / 1000 (the count the engine's rate used)"
            if predicted_n is not None:
                note = (f"timings.predicted_n is {predicted_n} but the engine's decode rate and time imply {implied} tokens; "
                        "the implied count is used because it is the one the rate was computed from")
    elif predicted_n is not None:
        decode_tokens, decode_source = predicted_n, "timings.predicted_n (no decode rate to check it against)"
    if note is None and output_tokens is not None and decode_tokens is not None and output_tokens != decode_tokens:
        note = (f"API usage.completion_tokens is {output_tokens}; the engine's decode count is {decode_tokens}. The engine count "
                "feeds decode_tps; the API count feeds the concurrent aggregate")
    prompt_tps = read / prompt_s if read and prompt_s else None
    decode_tps = decode_tokens / decode_s if decode_tokens and decode_s else None

    text = "".join(raw["content"])
    calls = [raw["tool_calls"][k] for k in sorted(raw["tool_calls"], key=lambda k: (not isinstance(k, int), k))]
    ttft, first_answer = (first[0] if first else None), (answer[0] if answer else None)
    total = round((raw["t_end"] - t0) / 1e9, 6)
    kind_names = {"reasoning": "reasoning", "answer": "answer", "tool_call": "tool_call"}
    record = {
        "id": spec["id"], "phase": spec["phase"], "case_id": spec["case_id"], "repetition": spec["repetition"],
        "seed": spec["seed"], "max_output_tokens": spec["max_output_tokens"], "concurrency": spec["concurrency"],
        "round": spec["round"], "status": status, "finish_reason": finish,
        "error": None if status in ("completed", "truncated") else error,
        "tokens": {"input_tokens": input_tokens, "cached_tokens": cached, "read_tokens": read,
                   "output_tokens": output_tokens, "decode_tokens": decode_tokens},
        "timings": {"prompt_s": _round(prompt_s), "decode_s": _round(decode_s), "ttft_s": _round(ttft),
                    "first_answer_s": _round(first_answer), "total_s": total},
        "metrics": {"prompt_tps": _round(prompt_tps), "decode_tps": _round(decode_tps)},
        "sources": {"input_tokens": "usage.prompt_tokens of the final stream chunk" if input_tokens is not None else None,
                    "cached_tokens": cached_source if cached is not None else None,
                    "read_tokens": read_source if read is not None else None,
                    "output_tokens": "usage.completion_tokens of the final stream chunk" if output_tokens is not None else None,
                    "decode_tokens": decode_source,
                    "engine_timings": ("timings.prompt_ms, timings.predicted_ms, timings.predicted_per_second of the final stream "
                                       "chunk (raw values in outputs.jsonl)") if timings else None,
                    "client_timings": ("client monotonic clock from dispatch; event offsets in outputs.jsonl")},
        "first_delta_kind": kind_names.get(first[1]) if first else None,
        "token_accounting_note": note, "evaluation": None}
    if spec["phase"] == "sanity":
        record["evaluation"] = evaluate(spec, status, text, [c for c in calls if c["name"] or c["arguments"]])
    reuse_limit = meta["max_shared_prefix_tokens"] + REUSE_SLACK_TOKENS
    extras = {"max_shared_prefix_tokens": meta["max_shared_prefix_tokens"],
              "frozen_input_tokens": meta["input_tokens"],
              "unexpected_payload_reuse": bool(cached is not None and cached > reuse_limit),
              "length_outside_tolerance": bool(spec.get("target_input_tokens") and input_tokens is not None and
                                               abs(input_tokens - spec["target_input_tokens"]) >
                                               LENGTH_TOLERANCE * spec["target_input_tokens"])}
    evidence = {"request_id": spec["id"], "phase": spec["phase"], "case_id": spec["case_id"],
                "request": {k: v for k, v in payload_of(spec, None).items() if k not in ("messages", "tools", "model")},
                "messages": spec["messages"], "tools": spec.get("tools"),
                "final_answer": text, "reasoning": "".join(raw["reasoning"]),
                "tool_calls": [{"id": c["id"], "name": c["name"], "arguments": c["arguments"]} for c in calls],
                "finish_reason": finish, "usage": raw.get("usage"), "timings": raw.get("timings"), "error": error,
                "http_status": raw["http_status"], "dispatch_offset_s": 0.0, "end_offset_s": total,
                "first_delta": {"offset_s": first[0], "kind": first[1]} if first else None,
                "first_answer_offset_s": first_answer, "client_events": events}
    return record, evidence, extras


def skipped_record(spec: dict, reason: str) -> tuple[dict, dict, dict]:
    record = {"id": spec["id"], "phase": spec["phase"], "case_id": spec["case_id"], "repetition": spec["repetition"],
              "seed": spec["seed"], "max_output_tokens": spec["max_output_tokens"], "concurrency": spec["concurrency"],
              "round": spec["round"], "status": "skipped", "finish_reason": None,
              "error": {"kind": "skipped", "message": reason},
              "tokens": {k: None for k in ("input_tokens", "cached_tokens", "read_tokens", "output_tokens", "decode_tokens")},
              "timings": {k: None for k in ("prompt_s", "decode_s", "ttft_s", "first_answer_s", "total_s")},
              "metrics": {"prompt_tps": None, "decode_tps": None},
              "sources": {k: None for k in ("input_tokens", "cached_tokens", "read_tokens", "output_tokens", "decode_tokens",
                                            "engine_timings", "client_timings")},
              "first_delta_kind": None, "token_accounting_note": None, "evaluation": None}
    if spec["phase"] == "sanity":
        record["evaluation"] = {"category": spec["category"], "passed": False, "reason": "the request was skipped: " + reason}
    return record, {"request_id": spec["id"], "skipped": reason}, {"unexpected_payload_reuse": False,
                                                                     "length_outside_tolerance": False}


# ------------------------------------------------------------------------------------------------------------ the run
class Collector:
    """Runs the scheduled requests in order and keeps their records."""

    def __init__(self, target: Target, corpus: Corpus, model: str | None, log, timeout_s: float):
        self.target, self.corpus, self.model, self.log, self.timeout_s = target, corpus, model, log, timeout_s
        self.meta = {row["id"]: row for row in corpus.build["requests"]}
        self.results: dict[str, tuple[dict, dict, dict]] = {}
        self.abort: str | None = None
        self.number = 0
        self.total = sum(PLANNED[p] for p in ("warmup", "performance", "sanity"))

    def finish(self, spec: dict, raw: dict) -> dict:
        result = record_of(spec, raw, self.meta[spec["id"]])
        self.results[spec["id"]] = result
        return result[0]

    def alive(self) -> bool:
        health = self.target.get("/health", timeout=5)
        return bool(health and health.get("service") == "strata")

    def one(self, spec: dict) -> None:
        if self.abort:
            self.results[spec["id"]] = skipped_record(spec, self.abort)
            return
        self.number += 1
        self.log(f"[{self.number}/{self.total}] {spec['id']} (up to {spec['max_output_tokens']} tokens)")
        raw = stream_chat(self.target, spec, self.model, timeout_s=self.timeout_s)
        record = self.finish(spec, raw)
        timing = record["timings"]
        speed = record["metrics"]
        self.log(f"    {record['status']}: prompt {speed['prompt_tps'] or 'n/a'} tok/s, decode {speed['decode_tps'] or 'n/a'} tok/s, "
                 f"first token {timing['ttft_s'] if timing['ttft_s'] is not None else 'n/a'} s"
                 + (f"; {record['error']['kind']}" if record["error"] else ""))
        if raw["interrupted"]:
            self.abort = "the operator cancelled the run"
        elif record["status"] == "failed":
            if spec["phase"] == "warmup":
                self.abort = f"the warm-up request failed ({record['error']['kind']}); the measured stages were not run"
            elif not self.alive():
                self.abort = f"the server stopped answering after {spec['id']} ({record['error']['kind']})"

    def concurrent_round(self, specs: list[dict]) -> dict:
        gate = threading.Event()
        raws: dict[str, dict] = {}

        def work(spec):
            raws[spec["id"]] = stream_chat(self.target, spec, self.model, timeout_s=self.timeout_s, gate=gate)
        with ThreadPoolExecutor(max_workers=len(specs)) as pool:
            futures = [pool.submit(work, spec) for spec in specs]
            time.sleep(0.05)
            gate.set()
            for future in futures:
                future.result()
        records = [self.finish(spec, raws[spec["id"]]) for spec in specs]
        start = min(raws[s["id"]]["t0"] for s in specs)
        end = max(raws[s["id"]]["t_end"] for s in specs)
        wall = round((end - start) / 1e9, 6)
        counted = [r for r in records if r["status"] in ("completed", "truncated")]
        outputs = [r["tokens"]["output_tokens"] for r in counted]
        total = sum(outputs) if outputs and all(o is not None for o in outputs) else None
        failed = sum(r["status"] == "failed" for r in records)
        self.log(f"    round {specs[0]['round']}: {len(counted)} completed or capped, {failed} failed, {wall} s")
        if any(raws[s["id"]]["interrupted"] for s in specs):
            self.abort = "the operator cancelled the run"
        return {"round": specs[0]["round"], "request_ids": [s["id"] for s in specs], "wall_s": wall,
                "output_tokens": total, "aggregate_tps": _round(total / wall) if total is not None and wall > 0 else None,
                "failed": failed}


def summaries_of(records: list[dict], extras: dict, phase: str, cases: tuple[str, ...], concurrency: int) -> list[dict]:
    rows = []
    for case in cases:
        group = [r for r in records if r["phase"] == phase and r["case_id"] == case]
        for metric, unit in METRICS:
            included, excluded, values = [], [], []
            for r in group:
                reason = None
                if r["status"] == "skipped":
                    reason = "skipped: " + r["error"]["message"]
                elif r["status"] == "failed":
                    reason = "failed: " + r["error"]["kind"]
                elif extras[r["id"]]["unexpected_payload_reuse"]:
                    reason = (f"unexpected cache reuse: {r['tokens']['cached_tokens']} cached tokens exceed the "
                              f"{extras[r['id']]['max_shared_prefix_tokens']}-token shared template prefix")
                else:
                    value = r["metrics"].get(metric) if metric in ("prompt_tps", "decode_tps") else r["timings"].get(metric)
                    if value is None:
                        reason = f"{metric} is not available for this request"
                    else:
                        values.append(value)
                        included.append(r["id"])
                if reason:
                    excluded.append({"request_id": r["id"], "reason": reason})
            rows.append({"phase": phase, "case_id": case, "concurrency": concurrency, "metric": metric, "unit": unit,
                         "median": _round(statistics.median(values)) if values else None,
                         "min": _round(min(values)) if values else None, "max": _round(max(values)) if values else None,
                         "n": len(values), "included_request_ids": included, "excluded": excluded})
    return rows


def counts_of(records: list[dict], phase: str, planned: int) -> dict:
    rows = [r for r in records if r["phase"] == phase]
    tally = {s: sum(r["status"] == s for r in rows) for s in ("completed", "truncated", "failed", "skipped")}
    attempted = tally["completed"] + tally["truncated"] + tally["failed"]
    return {"planned": planned, "attempted": attempted, **tally}


def quality_of(records: list[dict]) -> dict:
    sanity = [r for r in records if r["phase"] == "sanity"]
    cats = {name: {"passed": sum(r["evaluation"]["passed"] for r in sanity if r["evaluation"]["category"] == name),
                   "total": 4} for name in ("instruction", "retrieval", "tool_call")}
    return {"passed": sum(c["passed"] for c in cats.values()), "total": 12, "categories": cats}


def scan_nulls(value, path: str = ""):
    if value is None:
        yield path
    elif isinstance(value, dict):
        for key, item in value.items():
            if path == "" and key in ("missing", "extensions"):
                continue
            yield from scan_nulls(item, path + facts.pointer(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from scan_nulls(item, path + facts.pointer(index))


def default_reason(doc: dict, pointer: str) -> str:
    parts = [p.replace("~1", "/").replace("~0", "~") for p in pointer.split("/")[1:]]
    if parts[0] == "requests" and len(parts) >= 3:
        record = doc["requests"][int(parts[1])]
        field = ".".join(parts[2:])
        if record["status"] == "skipped":
            return "not measured: the request was skipped (" + record["error"]["message"] + ")"
        if field == "error":
            return "none: the request ended normally"
        if field == "round":
            return "not applicable outside the concurrency stage"
        if field == "evaluation":
            return "not applicable outside the sanity stage"
        if field == "token_accounting_note":
            return "none: the API and engine counts agree or only one exists"
        if field == "finish_reason":
            return "the server sent no finish_reason"
        if field == "first_delta_kind":
            return "no nonempty reasoning, answer or tool-call delta arrived"
        if field in ("timings.ttft_s", "timings.first_answer_s"):
            return ("no nonempty delta arrived" if field == "timings.ttft_s"
                    else "no answer text or tool call arrived (reasoning alone does not count)")
        if field.startswith("metrics."):
            return ("the rate needs a nonzero freshly read count (or generated count) and an engine duration, and one is missing or zero")
        if field.startswith("sources."):
            return "no source: the value was not reported"
        if record["status"] == "failed":
            return f"the request failed ({record['error']['kind']}) before this value was reported"
        return "the server's response did not include this value (usage or timings)"
    if parts[0] == "summaries" and len(parts) >= 3:
        row = doc["summaries"][int(parts[1])]
        return f"no valid samples of {row['metric']} for {row['case_id']} ({len(row['excluded'])} excluded)"
    if parts[0] == "concurrency_rounds":
        return "an included request has no output token count, or the round wall time is zero"
    if parts[0] == "run" and parts[1] == "optional_duration_s":
        return "the optional concurrency stage was not selected or did not run"
    if parts[0] == "protocol" and parts[1] == "model_card_revision":
        return "the profile's model-card revision is not pinned yet (protocol draft)"
    return "not available"


def _start_wait(target: Target, proc, timeout_s: float, log, log_path: Path | None) -> bool:
    began, announced = time.monotonic(), 0.0
    while time.monotonic() - began < timeout_s:
        if proc is not None and proc.poll() is not None:
            return False
        health = target.get("/health", timeout=5)
        if health and health.get("service") == "strata" and health.get("loaded"):
            return True
        if time.monotonic() - announced > 30:
            log(f"waiting for the model to load ({int(time.monotonic() - began)} s) ...")
            announced = time.monotonic()
        time.sleep(2)
    return False


def start_engine(ref: str, target: Target, log_path: Path, log, timeout_s: float):
    """Start `strata serve --profile REF` as a child of this run. -> the process."""
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "strata.py"), "serve", "--profile", ref, "--quiet"]
    handle = open(log_path, "wb")
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(command, cwd=str(root), stdout=handle, stderr=subprocess.STDOUT, **kwargs)
    proc._strata_log = handle                              # closed by stop_engine
    log("starting the engine (the model load is not part of the benchmark time) ...")
    if not _start_wait(target, proc, timeout_s, log, log_path):
        stuck = stop_engine(proc)
        tail = ""
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-600:]
        except OSError:
            pass
        raise ProfileError("the engine did not become ready" + (": " + tail.strip().splitlines()[-1] if tail.strip() else "") +
                           ("; and it could not be stopped (" + stuck + "): it may still hold the port" if stuck else ""), 6)
    return proc


def engine_log_since(console_log: Path, run_log: Path | None, offset: int) -> Path:
    """The log to keep with a result when the runner started the engine. `strata serve --quiet` prints nothing, so the
    console capture is empty; the engine's own lines are in the run config's `log` file, from `offset` (its size before
    the start). Falls back to the console capture when there is no such file or nothing new in it."""
    try:
        if run_log is not None and run_log.is_file() and run_log.stat().st_size > offset:
            with run_log.open("rb") as stream:
                stream.seek(offset)
                data = stream.read()
            if data.strip():
                merged = console_log.with_name("engine-run.log")
                merged.write_bytes(data)
                return merged
    except OSError:
        pass
    return console_log


def stop_engine(proc) -> str | None:
    """Stop only the process this run started. -> None when it is gone, else a plain sentence on why it may still be running
    (a failed taskkill / signal, or a process that outlives the wait): the caller must not report a normal finish then, because
    a process that is still alive can hold the server's port and the GPU (#3438)."""
    if proc is None:
        return None
    problem = None
    try:
        if proc.poll() is None:
            if os.name == "nt":
                try:
                    done = subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=60)
                    if done.returncode != 0 and proc.poll() is None:
                        said = (done.stderr or done.stdout or b"").decode("utf-8", errors="replace").strip()
                        problem = f"taskkill for process {proc.pid} failed (exit {done.returncode}){': ' + said[:200] if said else ''}"
                except subprocess.TimeoutExpired:
                    problem = f"taskkill for process {proc.pid} did not finish in 60 s"
                except OSError as error:
                    problem = f"taskkill could not run for process {proc.pid}: {error}"
            else:
                import signal
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except OSError:
                    proc.terminate()
                try:
                    proc.wait(90)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except OSError:
                        proc.kill()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                pass
            if proc.poll() is None:
                problem = (problem + "; " if problem else "") + f"process {proc.pid} is still running 30 s after it was told to stop"
    finally:
        handle = getattr(proc, "_strata_log", None)
        if handle is not None:
            handle.close()
    return problem


def slug(text: str, limit: int = 40) -> str:
    cleaned = re.sub(r"(?i)\b(nvidia|geforce|amd|radeon|intel|graphics|corporation)\b", " ", text or "")
    return re.sub(r"[^a-z0-9]+", "-", cleaned.lower()).strip("-")[:limit].strip("-") or "unknown"


def run(target: Target, out: Path | None, *, out_root: Path | None = None, optional_concurrency: bool = False,
        corpus: Corpus | None = None, declared: dict | None = None, plan: dict | None = None,
        engine_log: Path | None = None, engine_run_log: tuple[Path | None, int] | None = None, command: str = "strata bench run --suite standard-v1",
        log=lambda message: None, timeout_s: float = REQUEST_TIMEOUT_S, started_engine: bool = False,
        hash_model: bool = True) -> tuple[dict, Path]:
    """Run the protocol against `target` and write the report into the new directory `out` (or a new folder named
    <date>-standard-<gpu>-<run id> under `out_root`). -> (the result, the folder)."""
    declared = {"track": "custom", "track_notes": "no notes given by the operator", **(declared or {})}
    if declared["track"] not in ("default", "custom"):
        raise ProfileError("--track must be default or custom", 2)
    corpus = corpus or load_corpus()
    if out is not None and (out.exists() or out.is_symlink()):
        raise ProfileError("benchmark output directory already exists", 2)
    if out is None and out_root is None:
        raise ProfileError("benchmark needs an output directory", 2)
    health = target.get("/health", timeout=10)
    if not health or health.get("service") != "strata":
        raise ProfileError("no Strata server answers at that address", 5)
    if health.get("loaded") is not True:
        raise ProfileError("the server has no model loaded; the benchmark needs a loaded model", 5)
    props = target.get("/props", timeout=15)
    status = target.get("/v1/status", timeout=15) or {}
    if props is None:
        raise ProfileError("the server did not answer /props (an API key may be needed: STRATA_API_KEY)", 5)
    redact = facts.redactor(target.key or "")
    why = facts.Why()
    reasons_nonconforming: list[str] = []
    if not target.loopback:
        reasons_nonconforming.append("the server is not on loopback: standard latency is measured over loopback")

    served_model = props.get("model_alias") or status.get("model")
    prior = ((status.get("activity") or {}).get("requests"))
    # the server's own view of its hardware and build
    machine = target.get("/metrics", timeout=20) or {}
    static = machine.get("hardware_static") or {}
    served_gpu = ((status.get("machine") or {}).get("gpu") or {}).get("name") or static.get("gpu_name")
    engine_version = ((machine.get("engine") or {}).get("version") or status.get("engine")
                      or (str(props.get("build_info", "")).replace("Strata ", "") or None))

    plan = plan or {}
    options = facts.parse_options(plan["engine_argv"]) if plan.get("engine_argv") else None
    model_paths = [str(props.get("model_path") or "")] + [str(v) for v in (plan.get("engine_argv") or [])]
    files = facts.model_files_from([p for p in model_paths if p]) if hash_model else []
    model_dir = files[0].parent if files else None
    selected = facts.gpu_selection(plan.get("config"), plan.get("engine_argv"))
    template_text = props.get("chat_template") if isinstance(props.get("chat_template"), str) else None

    hardware_block, machine_facts = facts.hardware(why, selected=selected, served_gpu=served_gpu, model_dir=model_dir,
                                                   declared_background=list(declared.get("background") or []))
    software_block = facts.software(why, machine_facts, engine_version=engine_version,
                                    build_options=declared.get("build_options"),
                                    engine_build=facts.engine_build_runtime(plan.get("engine_exe")))
    model_block = facts.model(why, props=props, status=status, files=files, declared=declared, log=log)
    template_ref = {"path": "provenance/chat_template.jinja", "sha256": facts.sha256_text(template_text)} if template_text else None
    configuration_block, other_options = facts.configuration(
        why, options=options, env=(plan.get("config") or {}).get("env") if isinstance(plan.get("config"), dict) else None,
        props=props, declared=declared,
        sampling={"sampling": SAMPLING, "chat_template_kwargs": TEMPLATE_KWARGS}, template_ref=template_ref)

    if software_block["strata_commit"] is None:
        reasons_nonconforming.append("the Strata commit is unknown (not a git checkout and no COMMIT file)")
    if not model_block["files"]:
        reasons_nonconforming.append("no model file could be hashed")
    if template_ref is None:
        reasons_nonconforming.append("the effective chat template could not be read from the server")
    if props.get("is_sleeping"):
        reasons_nonconforming.append("the server reported its model as sleeping")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    if out is None:
        names = "-plus-".join(slug(g["name"], 30) for g in hardware_block["gpus"]) or "cpu"
        out = out_root / (datetime.now(timezone.utc).strftime("%Y-%m-%d") + "-standard-" + names[:60].strip("-") + "-" + run_id[-8:])
        out_root.mkdir(parents=True, exist_ok=True)
    _private_mkdir(out, parents=False, exist_ok=False)
    started_at = _now()
    collector = Collector(target, corpus, served_model, log, timeout_s)
    by_phase = {p: [s for s in corpus.specs if s["phase"] == p] for p in PHASES}
    required_start = time.monotonic()
    interrupted = False
    try:
        for phase in ("warmup", "performance", "sanity"):
            for spec in by_phase[phase]:
                collector.one(spec)
        required_duration = round(time.monotonic() - required_start, 3)
        optional_duration = None
        rounds: list[dict] = []
        if optional_concurrency:
            optional_start = time.monotonic()
            specs = by_phase["concurrency"]
            for round_no in (1, 2, 3):
                group = [s for s in specs if s["round"] == round_no]
                if collector.abort:
                    for spec in group:
                        collector.results[spec["id"]] = skipped_record(spec, collector.abort)
                    rounds.append({"round": round_no, "request_ids": [s["id"] for s in group], "wall_s": None,
                                   "output_tokens": None, "aggregate_tps": None, "failed": 0})
                    continue
                log(f"concurrency round {round_no}/3: four requests released together")
                rounds.append(collector.concurrent_round(group))
                if not collector.abort and any(collector.results[s["id"]][0]["status"] == "failed" for s in group) \
                        and not collector.alive():
                    collector.abort = "the server stopped answering during the concurrency stage"
            optional_duration = round(time.monotonic() - optional_start, 3)
    except KeyboardInterrupt:
        interrupted = True
        required_duration = round(time.monotonic() - required_start, 3)
        optional_duration, rounds = None, []
        collector.abort = "the operator cancelled the run"
        for spec in corpus.specs:
            if spec["id"] not in collector.results and (spec["phase"] != "concurrency" or optional_concurrency):
                collector.results[spec["id"]] = skipped_record(spec, collector.abort)
    for spec in corpus.specs:                              # a stage that never started is skipped, not missing
        if spec["id"] not in collector.results and (spec["phase"] != "concurrency" or optional_concurrency):
            collector.results[spec["id"]] = skipped_record(spec, collector.abort or "not run")

    selected_specs = [s for s in corpus.specs if s["id"] in collector.results]
    records = [collector.results[s["id"]][0] for s in selected_specs]
    evidence = [collector.results[s["id"]][1] for s in selected_specs]
    extras = {s["id"]: collector.results[s["id"]][2] for s in selected_specs}

    for r in records:
        if r["phase"] == "performance" and extras[r["id"]]["unexpected_payload_reuse"]:
            reasons_nonconforming.append(f"unexpected reuse of the user payload in {r['id']}")
        if extras[r["id"]]["length_outside_tolerance"]:
            reasons_nonconforming.append(f"{r['id']} measured {r['tokens']['input_tokens']} prompt tokens, outside 2% of its target")
        if r["status"] == "failed" and r["error"]["kind"] in ("http_400", "http_422"):
            reasons_nonconforming.append(f"the server refused the protocol's required settings in {r['id']} ({r['error']['kind']})")
    skipped_required = any(r["status"] == "skipped" for r in records if r["phase"] != "concurrency")
    skipped_optional = any(r["status"] == "skipped" for r in records if r["phase"] == "concurrency")
    run_status = ("nonconforming" if reasons_nonconforming else
                  "partial" if skipped_required or interrupted else "complete")
    reasons_nonconforming = list(dict.fromkeys(reasons_nonconforming))

    # the artifacts first, so the report can hash their final bytes
    artifacts: list[dict] = []

    def put(kind: str, relative: str, data: bytes) -> None:
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_new(path, data)
        artifacts.append({"kind": kind, "path": relative, "sha256": hashlib.sha256(data).hexdigest()})
    lines = [json.dumps(facts.redact_object(row, redact), ensure_ascii=False, allow_nan=False) for row in evidence]
    put("outputs", "outputs.jsonl", ("\n".join(lines) + "\n").encode("utf-8"))
    config_doc = facts.redact_object({"engine_options": options, "other_engine_options": other_options,
                                      "server_defaults": (props.get("default_generation_settings") or {}).get("params"),
                                      "total_slots": props.get("total_slots"), "context": configuration_block["context_tokens"],
                                      "source": "effective engine arguments of the profile" if options is not None
                                      else "the server was already running; its engine options are not exposed"}, redact)
    put("config", "config.json", (json.dumps(config_doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    if engine_log is not None and engine_run_log is not None:
        engine_log = engine_log_since(engine_log, *engine_run_log)
    if engine_log is not None and engine_log.is_file():
        text = engine_log.read_text(encoding="utf-8", errors="replace")[-2_000_000:]
        put("engine_log", "engine.log", redact(text).encode("utf-8"))
    if template_text:
        put("provenance", "provenance/chat_template.jinja", template_text.encode("utf-8"))
    put("provenance", "provenance/corpus-manifest.sha256", corpus.manifest)
    put("provenance", "provenance/corpus-build.json", (corpus.directory / "build.json").read_bytes())

    summaries = summaries_of(records, extras, "performance", PERFORMANCE_CASES, 1)
    if optional_concurrency:
        summaries += summaries_of(records, extras, "concurrency", ("short-burst",), 4)
    doc = {
        "format": FORMAT, "schema_version": SCHEMA_VERSION,
        "protocol": {"id": PROTOCOL_ID, "git_commit": software_block["runner_commit"], "profile": PROFILE,
                     "corpus_sha256": corpus.sha256, "model_card_revision": MODEL_CARD_REVISION},
        "run": {"id": run_id, "started_at": started_at, "status": run_status, "required_duration_s": required_duration,
                "optional_duration_s": optional_duration, "optional_concurrency": bool(optional_concurrency),
                "command": command},
        "hardware": hardware_block, "software": software_block, "model": model_block,
        "configuration": configuration_block,
        "request_counts": {p: counts_of(records, p, PLANNED[p] if p != "concurrency" or optional_concurrency else 0)
                           for p in PHASES},
        "requests": records, "summaries": summaries, "concurrency_rounds": rounds if optional_concurrency else [],
        "quality": quality_of(records), "artifacts": artifacts, "missing": {},
        "extensions": {"suite": SUITE, "runner": "strata_cli.bench_standard",
                       "status_reasons": reasons_nonconforming or (
                           [collector.abort] if collector.abort else []),
                       "started_engine": started_engine,
                       "server_requests_before_run": prior if isinstance(prior, int) else None,
                       "other_engine_options": other_options,
                       "request_checks": {rid: v for rid, v in extras.items()},
                       "optional_stage_skipped": skipped_optional,
                       "api_key_sent": bool(target.key)}}
    if doc["run"]["optional_duration_s"] is None:
        why.add("/run/optional_duration_s", "the optional concurrency stage was not selected" if not optional_concurrency
                else "the optional concurrency stage did not run")
    for pointer_text in scan_nulls(doc):
        doc["missing"][pointer_text] = why.get(pointer_text) or default_reason(doc, pointer_text)
    problems = validate(doc)
    if problems:
        raise ProfileError("the benchmark report failed its own validation: " + problems[0], 6)
    encoded = (json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _write_new(out / "results.json", encoded)
    _write_new(out / "README.md", report_markdown(doc, redact).encode("utf-8"))
    return doc, out


# ------------------------------------------------------------------------------------------------------------ validation
def validate(doc: dict) -> list[str]:
    """A strict check of the result contract: shape, types, count identities, null reasons and summaries rebuilt from the requests."""
    bad: list[str] = []

    def need(condition: bool, message: str) -> None:
        if not condition:
            bad.append(message)

    if not isinstance(doc, dict):
        return ["the result is not an object"]
    need(doc.get("format") == FORMAT and doc.get("schema_version") == SCHEMA_VERSION, "unknown format or schema version")
    required = ("protocol", "run", "hardware", "software", "model", "configuration", "request_counts", "requests", "summaries",
                "concurrency_rounds", "quality", "artifacts", "missing")
    for key in required:
        need(key in doc, f"missing top-level field {key}")
    if bad:
        return bad
    allowed = {"format", "schema_version", *required, "extensions"}
    need(set(doc) <= allowed, "unknown top-level fields must be under extensions")
    protocol, run_block = doc["protocol"], doc["run"]
    need(all(k in protocol for k in ("id", "git_commit", "profile", "corpus_sha256", "model_card_revision")), "protocol fields")
    need(isinstance(protocol.get("corpus_sha256"), str) and bool(HEX64.match(protocol["corpus_sha256"])), "corpus_sha256 is not a SHA-256")
    for field in ("id", "started_at", "status", "required_duration_s", "optional_duration_s", "optional_concurrency", "command"):
        need(field in run_block, f"run.{field} missing")
    need(run_block.get("status") in ("complete", "partial", "nonconforming"), "run.status")
    need(bool(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", str(run_block.get("started_at")))), "run.started_at is not UTC ISO 8601")
    for field in ("gpus", "cpu", "ram_bytes", "storage", "background_workloads"):
        need(field in doc["hardware"], f"hardware.{field} missing")
    for gpu in doc["hardware"].get("gpus") or []:
        need(gpu.get("memory_kind") in ("dedicated", "shared"), "gpu memory_kind")
        need(isinstance(gpu.get("device_index"), int), "gpu device_index")
    for field in ("os", "drivers", "runtime", "strata_commit", "engine_version", "build_kind", "runner_commit", "build_options"):
        need(field in doc["software"], f"software.{field} missing")
    need(doc["software"].get("build_kind") in ("release", "source"), "software.build_kind")
    for field in ("id", "revision", "source_revision", "quantization", "files", "tokenizer_revision", "custom_pack"):
        need(field in doc["model"], f"model.{field} missing")
    for entry in doc["model"].get("files") or []:
        need(bool(HEX64.match(str(entry.get("sha256")))) and "/" not in str(entry.get("path")) and "\\" not in str(entry.get("path")),
             "model file entry")
    cfg = doc["configuration"]
    for field in ("track", "track_notes", "context_tokens", "server_capacity", "kv_cache", "expert_cache", "offloading", "prefill",
                  "mtp", "calibration", "experimental_options", "template", "sampling", "chat_template_kwargs"):
        need(field in cfg, f"configuration.{field} missing")
    need(cfg.get("track") in ("default", "custom"), "configuration.track")
    need(all(k in cfg.get("sampling", {}) for k in SAMPLING), "configuration.sampling")

    records = doc["requests"]
    ids = [r.get("id") for r in records]
    need(len(set(ids)) == len(ids), "duplicate request ids")
    need(len(records) in (22, 34), "the requests array has neither 22 nor 34 records")
    sanity_ids = {f"sanity-{n}" for n in ("I1", "I2", "I3", "I4", "R1", "R2", "R3", "R4", "T1", "T2", "T3", "T4")}
    need(sanity_ids <= set(ids), "a sanity case is missing")
    for r in records:
        need(r.get("status") in ("completed", "truncated", "failed", "skipped"), f"{r.get('id')}: status")
        need(r.get("phase") in PHASES, f"{r.get('id')}: phase")
        if r.get("status") in ("completed", "truncated"):
            need(r.get("error") is None, f"{r.get('id')}: error on a finished request")
        else:
            need(isinstance(r.get("error"), dict) and "kind" in r["error"] and "message" in r["error"], f"{r.get('id')}: error object")
        for group, keys in (("tokens", ("input_tokens", "cached_tokens", "read_tokens", "output_tokens", "decode_tokens")),
                            ("timings", ("prompt_s", "decode_s", "ttft_s", "first_answer_s", "total_s")),
                            ("metrics", ("prompt_tps", "decode_tps")),
                            ("sources", ("input_tokens", "cached_tokens", "read_tokens", "output_tokens", "decode_tokens",
                                         "engine_timings", "client_timings"))):
            need(isinstance(r.get(group), dict) and all(k in r[group] for k in keys), f"{r.get('id')}: {group}")
        for k, v in (r.get("tokens") or {}).items():
            need(v is None or (isinstance(v, int) and not isinstance(v, bool) and v >= 0), f"{r.get('id')}: tokens.{k}")
        for group in ("timings", "metrics"):
            for k, v in (r.get(group) or {}).items():
                need(v is None or (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0),
                     f"{r.get('id')}: {group}.{k}")
        if r.get("phase") == "sanity":
            need(isinstance(r.get("evaluation"), dict) and isinstance(r["evaluation"].get("passed"), bool), f"{r.get('id')}: evaluation")
        else:
            need(r.get("evaluation") is None, f"{r.get('id')}: evaluation outside sanity")
        need((r.get("round") is not None) == (r.get("phase") == "concurrency"), f"{r.get('id')}: round")
    counts = doc["request_counts"]
    for phase in PHASES:
        rows = [r for r in records if r.get("phase") == phase]
        c = counts.get(phase) or {}
        need(c.get("attempted") == c.get("completed", 0) + c.get("truncated", 0) + c.get("failed", 0), f"{phase}: attempted identity")
        need(c.get("planned") == c.get("attempted", 0) + c.get("skipped", 0), f"{phase}: planned identity")
        need(len(rows) == c.get("planned"), f"{phase}: record count")
    need(doc["quality"] == quality_of(records) if len(records) >= 22 else False, "quality does not match the sanity verdicts")
    need(isinstance(run_block.get("optional_concurrency"), bool), "run.optional_concurrency")
    need((counts["concurrency"]["planned"] == 12) == bool(run_block.get("optional_concurrency")), "concurrency count")

    extras = {k: v for k, v in (doc.get("extensions") or {}).get("request_checks", {}).items()}
    if extras:
        rebuilt = summaries_of(records, extras, "performance", PERFORMANCE_CASES, 1)
        if run_block.get("optional_concurrency"):
            rebuilt += summaries_of(records, extras, "concurrency", ("short-burst",), 4)
        need(rebuilt == doc["summaries"], "summaries cannot be rebuilt from the request records")
    for row in doc["summaries"]:
        need(row["n"] == len(row["included_request_ids"]), "summary n")
        if row["n"] == 0:
            need(row["median"] is None and row["min"] is None and row["max"] is None, "empty summary must be null")
    for artifact in doc["artifacts"]:
        need(artifact.get("kind") in ("config", "engine_log", "outputs", "provenance") and bool(HEX64.match(str(artifact.get("sha256"))))
             and not str(artifact.get("path")).startswith(("/", "\\")) and ".." not in Path(str(artifact.get("path"))).parts
             and not re.match(r"^[A-Za-z]:", str(artifact.get("path"))), "artifact entry")
    nulls = set(scan_nulls(doc))
    need(nulls == set(doc["missing"]), "missing must list exactly the null values: "
         f"{sorted(nulls ^ set(doc['missing']))[:3]}")
    need(all(isinstance(v, str) and v for v in doc["missing"].values()), "every missing entry needs a reason")
    try:
        json.dumps(doc, allow_nan=False)
    except ValueError:
        bad.append("the result contains NaN or infinity")
    return bad


# ------------------------------------------------------------------------------------------------------------ reading it out
def _span(row: dict | None, digits: int = 1) -> str:
    if not row or row["n"] == 0:
        return "not measured"
    return f"{row['median']:.{digits}f} ({row['min']:.{digits}f}-{row['max']:.{digits}f}) n={row['n']}"


def _find(doc: dict, phase: str, case: str, metric: str) -> dict | None:
    return next((s for s in doc["summaries"] if s["phase"] == phase and s["case_id"] == case and s["metric"] == metric), None)


def _minutes(seconds: float | None) -> str:
    if seconds is None:
        return "not measured"
    return f"{int(seconds // 60)} min {int(seconds % 60)} s"


SHORT_PROMPT_NOTE = ("short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not "
                     "shown; read prompt speed from prompt-4k and prompt-16k")
CAP_NOTE = ("Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the "
            "time to the cap, not to an answer, and it has no first-answer time when it was still reasoning")


def capped(doc: dict, phase: str, case: str) -> str:
    """'k/n': how many of the case's measured requests stopped at the output cap."""
    rows = [r for r in doc["requests"] if r["phase"] == phase and r["case_id"] == case]
    return f"{sum(1 for r in rows if r['status'] == 'truncated')}/{len(rows)}"


def prompt_cell(doc: dict, phase: str, case: str, digits: int = 1) -> str:
    return "n/a" if case == "short" else _span(_find(doc, phase, case, "prompt_tps"), digits)


def table_lines(doc: dict, phase: str, cases: tuple[str, ...]) -> list[str]:
    head = ["Case", "Prompt tok/s", "Decode tok/s", "TTFT s", "Total s", "Capped"]
    rows = [head]
    for case in cases:
        rows.append([case, prompt_cell(doc, phase, case), _span(_find(doc, phase, case, "decode_tps")),
                     _span(_find(doc, phase, case, "ttft_s"), 2), _span(_find(doc, phase, case, "total_s"), 1),
                     capped(doc, phase, case)])
    widths = [max(len(r[i]) for r in rows) for i in range(len(head))]
    return (["  " + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows]
            + ["  " + SHORT_PROMPT_NOTE + ".", "  " + CAP_NOTE + "."])


def format_plain(doc: dict) -> str:
    """The summary printed at the end of a run: what ran, how fast, what passed, what to be careful about."""
    hw, sw, run_block = doc["hardware"], doc["software"], doc["run"]
    gpu = ", ".join(g["name"] for g in hw["gpus"]) or "no GPU found"
    counts = doc["request_counts"]
    done = sum(counts[p]["completed"] + counts[p]["truncated"] for p in ("performance", "sanity"))
    quality = doc["quality"]
    lines = [f"Strata standard benchmark ({doc['protocol']['id']})",
             f"Status: {run_block['status']}. {done} of 21 required requests finished, "
             f"{sum(counts[p]['failed'] for p in PHASES)} failed, {sum(counts[p]['skipped'] for p in PHASES)} skipped. "
             f"Time after the model loaded: {_minutes(run_block['required_duration_s'])}.",
             f"Machine: {gpu}; {hw['cpu'] or 'CPU not read'}; "
             f"{round(hw['ram_bytes'] / 2 ** 30) if hw['ram_bytes'] else 'unknown'} GB RAM; {sw['os']['name']} {sw['os']['version']}.",
             f"Model: {doc['model']['id'] or 'unknown'}, {doc['model']['quantization'] or 'quantization not read'}, "
             f"context {doc['configuration']['context_tokens'] or 'unknown'}; engine {sw['engine_version'] or 'unknown'}; "
             f"track {doc['configuration']['track']}.", "",
             "Speed (median, min-max, n; the engine's own prompt and decode timings, the client's first-token time):"]
    lines += table_lines(doc, "performance", PERFORMANCE_CASES)
    lines += ["", f"Sanity checks: {quality['passed']} of 12 passed "
              f"(instruction {quality['categories']['instruction']['passed']}/4, retrieval "
              f"{quality['categories']['retrieval']['passed']}/4, tool calls {quality['categories']['tool_call']['passed']}/4)."]
    failed = [r["evaluation"] and f"{r['case_id']}: {r['evaluation']['reason']}" for r in doc["requests"]
              if r["phase"] == "sanity" and not r["evaluation"]["passed"]]
    lines += ["  Not passed: " + "; ".join(failed)] if failed else []
    if doc["run"]["optional_concurrency"]:
        rounds = [r["aggregate_tps"] for r in doc["concurrency_rounds"] if r["aggregate_tps"] is not None]
        lines += ["", "Four requests at once (server capacity "
                  f"{doc['configuration']['server_capacity'] or 'unknown'}): "
                  + (", ".join(f"{v:.1f}" for v in rounds) + " tok/s in total per round" if rounds else "no round finished")]
    reasons = doc["extensions"]["status_reasons"]
    if run_block["status"] != "complete" and reasons:
        lines += ["", "Why this is not a complete standard result:"] + [f"  - {r}" for r in reasons]
    failures = [f"{r['id']}: {r['error']['kind']}" for r in doc["requests"] if r["status"] == "failed"]
    if failures:
        lines += ["", "Failed requests: " + "; ".join(failures)]
    return "\n".join(lines)


def report_markdown(doc: dict, redact=lambda text: text) -> str:
    hw, sw, run_block, model, cfg = doc["hardware"], doc["software"], doc["run"], doc["model"], doc["configuration"]
    counts = doc["request_counts"]
    quality = doc["quality"]
    gpu = "; ".join(("%s (%.1f GiB %s)" % (g["name"], g["memory_bytes"] / 2 ** 30, g["memory_kind"])) if g["memory_bytes"]
                    else g["name"] for g in hw["gpus"]) or "not measured"
    nm = "not measured"
    ram = "%.1f GiB" % (hw["ram_bytes"] / 2 ** 30) if hw["ram_bytes"] else nm
    storage = ", ".join("%s %s (%s)" % (s["kind"], s["model"], s["bus"]) for s in hw["storage"]) if hw["storage"] else nm
    drivers = ", ".join("%s %s" % (d["name"], d["version"]) for d in sw["drivers"]) if sw["drivers"] else nm
    runtime = ", ".join("%s %s" % (d["name"], d["version"]) for d in sw["runtime"]) if sw["runtime"] else nm
    lines = [
        "# Strata standard benchmark: %s" % (", ".join(g["name"] for g in hw["gpus"]) or "unknown GPU"), "",
        "Protocol `%s` (draft), run `%s`, started %s. **Status: %s.**" % (doc["protocol"]["id"], run_block["id"],
                                                                         run_block["started_at"], run_block["status"]), "",
        "Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage "
        "and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).", ""]
    reasons = doc["extensions"]["status_reasons"]
    if run_block["status"] != "complete":
        lines += ["This run is not a complete standard result:", ""] + ["- " + r for r in reasons] + [""]
    lines += [
        "## Hardware and software", "",
        "- GPU: " + gpu,
        "- CPU: %s; RAM: %s" % (hw["cpu"] or nm, ram),
        "- Storage of the model files: " + storage,
        "- OS: %s %s; drivers: %s; runtime: %s" % (sw["os"]["name"], sw["os"]["version"], drivers, runtime),
        "- Strata commit: `%s`; engine %s; %s build" % (sw["strata_commit"] or nm, sw["engine_version"] or nm, sw["build_kind"]),
        "- Background work: " + ("; ".join(hw["background_workloads"]) if hw["background_workloads"] else nm), "",
        "## Model and configuration", "",
        f"- Model: {model['id'] or nm}; quantization {model['quantization'] or nm}; revision {model['revision'] or nm}; "
        f"source revision {model['source_revision'] or nm}",
        f"- Files: " + ("; ".join(f"`{f['path']}` `{f['sha256'][:16]}...`" for f in model['files']) if model['files'] else nm),
        f"- Track: **{cfg['track']}** ({cfg['track_notes']})",
        f"- Context {cfg['context_tokens'] or nm} tokens; server capacity {cfg['server_capacity'] or nm} slot(s)",
        f"- Sampling: " + ", ".join(f"{k} {v}" for k, v in cfg["sampling"].items()) + "; thinking on, preserved",
        f"- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`", "",
        "## Method", "",
        f"The frozen requests of `bench/prompts/standard-v1` (corpus `{doc['protocol']['corpus_sha256'][:16]}...`) were sent over "
        "loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, "
        "and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's "
        "monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. "
        f"Time after the model loaded: {_minutes(run_block['required_duration_s'])}.", "",
        "## Results", ""]
    lines += ["| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for case in PERFORMANCE_CASES:
        lines.append(f"| {case} | " + prompt_cell(doc, "performance", case) + " | " + " | ".join(
            _span(_find(doc, "performance", case, m), d) for m, d in (("decode_tps", 1), ("ttft_s", 2),
                                                                      ("first_answer_s", 2), ("total_s", 1)))
            + f" | {capped(doc, 'performance', case)} |")
    lines += ["", "Cells are median (min-max) n=samples. " + SHORT_PROMPT_NOTE[0].upper() + SHORT_PROMPT_NOTE[1:] + ". " + CAP_NOTE + ".", "",
              "| Phase | Planned | Completed | Truncated | Failed | Skipped |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for phase in PHASES:
        c = counts[phase]
        lines.append(f"| {phase} | {c['planned']} | {c['completed']} | {c['truncated']} | {c['failed']} | {c['skipped']} |")
    lines += ["", f"Sanity checks: **{quality['passed']} of 12**.", "", "| Case | Result | Reason |", "| --- | --- | --- |"]
    for r in doc["requests"]:
        if r["phase"] == "sanity":
            lines.append(f"| {r['case_id']} | {'pass' if r['evaluation']['passed'] else 'fail'} | {r['evaluation']['reason']} |")
    if run_block["optional_concurrency"]:
        lines += ["", "Concurrency rounds (four requests released together):", "",
                  "| Round | Wall s | Output tokens | Aggregate tok/s | Failed |", "| ---: | ---: | ---: | ---: | ---: |"]
        for r in doc["concurrency_rounds"]:
            lines.append(f"| {r['round']} | {r['wall_s']} | {r['output_tokens']} | {r['aggregate_tps']} | {r['failed']} |")
    failures = [r for r in doc["requests"] if r["status"] == "failed"]
    lines += ["", "## Failures, exclusions and limitations", ""]
    lines += [f"- {r['id']} failed: {r['error']['kind']} ({r['error']['message']})" for r in failures] or ["- No request failed."]
    excluded = {(e["request_id"], e["reason"]) for s in doc["summaries"] for e in s["excluded"]}
    lines += [f"- Excluded from speed summaries: {rid}: {reason}" for rid, reason in sorted(excluded)]
    lines += ["- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert "
              "cache stays warm and evolves. These are not cold-model measurements.",
              "- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.",
              "- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or "
              "the quantization repository's revision; see `missing` in results.json for every value it could not read.",
              f"- The server had answered {doc['extensions']['server_requests_before_run']} request(s) before this run."
              if doc["extensions"]["server_requests_before_run"] else
              ("- The server had answered no request before this run (a freshly loaded engine)."
               if doc["extensions"]["server_requests_before_run"] == 0 else
               "- The runner could not tell whether the server was freshly loaded."), ""]
    return redact("\n".join(lines))


def load(path: Path) -> dict:
    """Read a saved standard result and validate it. A result in another format or schema version is refused, never merged."""
    target = path / "results.json" if path.is_dir() else path
    try:
        if target.stat().st_size > 64 << 20:
            raise ProfileError("benchmark result is larger than 64 MiB", 5)
        doc = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise ProfileError("cannot read the benchmark result", 5) from None
    problems = validate(doc)
    if problems:
        raise ProfileError("not a valid standard benchmark result: " + problems[0], 5)
    return doc


def load_if_standard(path: Path) -> dict | None:
    """The validated result at `path` when it is a standard benchmark result; None when it is another kind of report."""
    target = path / "results.json" if path.is_dir() else path
    try:
        head = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    if not isinstance(head, dict) or head.get("format") != FORMAT:
        return None
    return load(path)
