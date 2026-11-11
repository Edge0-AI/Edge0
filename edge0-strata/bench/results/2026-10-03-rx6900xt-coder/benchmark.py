#!/usr/bin/env python3
"""Measure Strata streaming latency and engine throughput using only Python's stdlib.

Run against an already-started local server. This script does not install or start
Strata, clear server state, or contact any host other than the configured URL.
"""
import argparse
import hashlib
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_URL = "http://127.0.0.1:8080"
DEFAULT_CASES = "short,4k,32k"
ENGINE_LINE = re.compile(
    r"prompt (?P<prompt>\d+) tokens = (?P<reused>\d+) reused \+ (?P<read>\d+)(?: of \d+)? read in "
    r"(?P<read_ms>[\d.]+) ms \((?P<prefill>[\d.]+) tok/s\), (?P<generated>\d+) generated in "
    r"(?P<decode_ms>[\d.]+) ms \((?P<decode>[\d.]+) tok/s\)"
)


def json_request(url, payload=None, headers=None, timeout=30):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def auth_headers(api_key, content_type=False):
    headers = {"Content-Type": "application/json"} if content_type else {}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    return headers


def read_log_from(path, offset):
    if path is None:
        return ""
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            end = stream.tell()
            stream.seek(min(offset, end))
            return stream.read().decode("utf-8", "replace")
    except OSError as exc:
        return f"[log read error: {exc}]"


def request_body(content, maximum=256):
    return {"model": "strata", "messages": [{"role": "user", "content": content}],
            "temperature": 0, "reasoning_effort": "none", "max_tokens": maximum,
            "stream": True, "stream_options": {"include_usage": True}}


def short_prompt(run_id):
    return (f"Benchmark nonce: {run_id}. Write a complete Python module for a robust in-memory task queue. "
            "Include typed dataclasses, validation, stable priority ordering, enqueue/dequeue/peek/remove operations, "
            "clear exceptions, and a detailed unittest suite with edge cases. Explain behavior in docstrings and "
            "comments. Output the complete code only; make it production-quality and detailed.")


def sized_prompt(target_tokens, run_id, count_tokens):
    """Build a deterministic code-review prompt at the target rendered length."""
    prefix = (f"Benchmark nonce: {run_id}. Treat this as an independent request.\n"
              "Review this synthetic Python module and explain its behavior, edge cases, tests, and complexity.\n")
    suffix = "\n\nGive a concise code review with concrete observations.\n"
    def build(lines):
        filler = "".join(
            f"def transform_{i:05d}(value: int, offset: int = {i}) -> int:\n"
            f"    return (value * {(i % 97) + 1} + offset + {i}) % 100003\n\n"
            for i in range(lines)
        )
        return prefix + filler + suffix

    # The count endpoint renders and tokenizes the prompt without running the
    # model. Binary search the generated lines; the actual OpenAI request usage
    # is retained too because the server reports that authoritative count.
    lo, hi = 0, max(8, int(target_tokens / 12))
    while count_tokens(build(hi)) < target_tokens:
        hi *= 2
    while lo < hi:
        middle = (lo + hi + 1) // 2
        if count_tokens(build(middle)) <= target_tokens:
            lo = middle
        else:
            hi = middle - 1
    return build(lo)


def history_record(metrics, started_wall):
    for record in metrics.get("requests", []):
        if record.get("time", 0) >= started_wall - 0.1:
            return record
    return None


def summarize(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return {"median": None, "min": None, "max": None}
    return {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Strata base URL (default: %(default)s)")
    parser.add_argument("--cases", default=DEFAULT_CASES, help="comma-separated subset of short,4k,32k")
    parser.add_argument("--runs", type=int, default=3, help="measured runs per case (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent,
                        help="directory for raw.jsonl, summary.json, and log snippets")
    parser.add_argument("--log", type=Path, help="optional Strata engine log to capture per-run appended lines")
    parser.add_argument("--api-key", default="", help="API key, if the local server requires one")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    cases = [name.strip().lower() for name in args.cases.split(",") if name.strip()]
    unknown = sorted(set(cases) - {"short", "4k", "32k"})
    if not cases or unknown:
        parser.error("--cases must be a non-empty comma-separated subset of short,4k,32k")
    args.out.mkdir(parents=True, exist_ok=True)
    base = args.url.rstrip("/")
    headers = auth_headers(args.api_key, content_type=True)
    get_headers = auth_headers(args.api_key)

    initial = json_request(base + "/health", headers=get_headers)
    (args.out / "initial-health.json").write_text(json.dumps(initial, indent=2) + "\n", encoding="utf-8")
    records = []
    warmup_records = []
    raw_path = args.out / "raw.jsonl"
    # Reset this harness's output for a new invocation; server/model state stays intact.
    raw_path.write_text("", encoding="utf-8")

    def perform(label, content, maximum=256, warmup=False, expected_prompt_tokens=None):
        payload = request_body(content, maximum)
        wire = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            metrics_before = json_request(base + "/metrics", headers=get_headers)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            metrics_before = {"error": str(exc)}
        started_wall = time.time()
        log_offset = 0
        if args.log:
            try:
                log_offset = args.log.stat().st_size
            except OSError:
                pass
        started = time.perf_counter()
        first_text_at = None
        chunks, text_parts, reasoning_parts = [], [], []
        finish_reason = None
        stream_error = None
        usage = None
        request = urllib.request.Request(base + "/v1/chat/completions", data=wire, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        break
                    try:
                        chunk = json.loads(data.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    chunks.append(chunk)
                    if chunk.get("error"):
                        stream_error = chunk["error"]
                    if "usage" in chunk:
                        usage = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta") or {}
                        text = delta.get("content") or ""
                        reasoning = delta.get("reasoning_content") or ""
                        if text or reasoning:
                            if first_text_at is None:
                                first_text_at = time.perf_counter() - started
                            text_parts.append(text)
                            reasoning_parts.append(reasoning)
                        finish_reason = choice.get("finish_reason") or finish_reason
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            stream_error = {"message": str(exc), "type": "client_error"}
        elapsed = time.perf_counter() - started
        metrics, metric_error = {}, None
        try:
            metrics = json_request(base + "/metrics", headers=get_headers)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            metric_error = str(exc)
        engine = history_record(metrics, started_wall)
        log_text = read_log_from(args.log, log_offset)
        log_lines = [line.rstrip() for line in log_text.splitlines() if line.strip()]
        parsed_log = []
        for line in log_lines:
            match = ENGINE_LINE.search(line)
            if match:
                parsed_log.append({key: (float(value) if "." in value else int(value))
                                   for key, value in match.groupdict().items()})
        stream_timings = next((chunk["timings"] for chunk in reversed(chunks) if chunk.get("timings")), None)
        if engine:
            fresh = max(0, (engine.get("prompt_tokens") or 0) - (engine.get("reused") or 0))
            engine_timings = {
                "prompt_tokens": engine.get("prompt_tokens"), "fresh_prompt_tokens": fresh,
                "reused_tokens": engine.get("reused"), "prompt_ms": engine.get("prompt_ms"),
                "prefill_tok_s": (fresh / (engine["prompt_ms"] / 1000)
                                  if engine.get("prompt_ms") else None),
                "generated_tokens": engine.get("engine_generated"), "decode_ms": engine.get("decode_ms"),
                "decode_tok_s": engine.get("decode_tok_s"),
                "drafts_offered": engine.get("drafts_offered"),
                "drafts_accepted": engine.get("drafts_accepted"),
            }
        else:
            engine_timings = None
        row = {
            "label": label, "warmup": warmup, "target_prompt_tokens": expected_prompt_tokens,
            "request_sha256": hashlib.sha256(wire).hexdigest(),
            "request_body": payload,
            "client_ttft_s": first_text_at, "client_total_latency_s": elapsed,
            "finish_reason": finish_reason, "usage": usage, "engine": engine,
            "metrics_before": metrics_before, "metrics_after": metrics,
            "engine_timings": stream_timings or engine_timings,
            "prefix_reused_tokens": (engine or {}).get("reused"),
            "api_cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
            "output_text": "".join(text_parts), "reasoning_text": "".join(reasoning_parts),
            "raw_stream_chunks": chunks, "stream_error": stream_error,
            "metrics_error": metric_error, "engine_log_lines": log_lines[-30:],
            "parsed_engine_log_timings": parsed_log,
        }
        if warmup:
            warmup_records.append(row)
            (args.out / "warmup.json").write_text(
                json.dumps(warmup_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            records.append(row)
            with raw_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if args.log:
                safe_label = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
                (args.out / f"{safe_label}.log-snippet.txt").write_text(
                    "\n".join(log_lines) + ("\n" if log_lines else ""), encoding="utf-8")
        print(json.dumps({"run": label, "ttft_s": first_text_at, "total_s": elapsed,
                          "prompt_tokens": (engine or {}).get("prompt_tokens"),
                          "reused": (engine or {}).get("reused"), "engine_timings": row["engine_timings"],
                          "error": stream_error}, ensure_ascii=False), flush=True)
        if stream_error:
            raise RuntimeError(f"{label} failed: {stream_error}")
        if not text_parts:
            raise RuntimeError(f"{label} produced no nonempty content delta")
        return row

    # Separate warm-up from the measured sample set.
    perform("warmup", "Benchmark warm-up: explain in detail how a Python heap maintains priority order.",
            maximum=256, warmup=True)
    (args.out / "warmup.json").write_text(json.dumps(warmup_records, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    for case in cases:
        for run in range(1, args.runs + 1):
            nonce = f"rx6900xt-coder-{case}-trial-{run}"
            if case == "short":
                content, maximum = short_prompt(nonce), 256
                label = f"short-run-{run}"
            else:
                target = 4096 if case == "4k" else 32768
                def count_tokens(text):
                    result = json_request(base + "/v1/messages/count_tokens",
                                          {"model": "strata", "messages": [{"role": "user", "content": text}]},
                                          headers=headers, timeout=180)
                    return int(result["input_tokens"])
                content, maximum = sized_prompt(target, nonce, count_tokens), 256
                expected_tokens = count_tokens(content)
                label = f"{case}-run-{run}"
            if case == "short":
                expected_tokens = None
            perform(label, content, maximum=maximum, expected_prompt_tokens=expected_tokens)

    grouped = {}
    for case in cases:
        subset = [row for row in records if row["label"].startswith(case + "-")]
        def metric(row, key):
            engine = row.get("engine") or {}
            if key == "prompt_tok_s":
                n = engine.get("prompt_tokens", 0) - (engine.get("reused") or 0)
                ms = engine.get("prompt_ms")
                return n / (ms / 1000) if ms else None
            if key == "decode_tok_s":
                n = engine.get("engine_generated") or 0
                ms = engine.get("decode_ms")
                return n / (ms / 1000) if ms else None
            return row.get(key)
        grouped[case] = {
            "runs": len(subset),
            "prompt_tokens": summarize([(r.get("engine") or {}).get("prompt_tokens") for r in subset]),
            "prefix_reused_tokens": summarize([r.get("prefix_reused_tokens") for r in subset]),
            "completion_tokens": summarize([(r.get("usage") or {}).get("completion_tokens") for r in subset]),
            "client_ttft_s": summarize([metric(r, "client_ttft_s") for r in subset]),
            "client_total_latency_s": summarize([metric(r, "client_total_latency_s") for r in subset]),
            "prompt_tok_s": summarize([metric(r, "prompt_tok_s") for r in subset]),
            "decode_tok_s": summarize([metric(r, "decode_tok_s") for r in subset]),
            "finish_reasons": [r.get("finish_reason") for r in subset],
        }
    summary = {"url": base, "cases": cases, "runs_per_case": args.runs,
               "request_settings": {"max_tokens": 256, "temperature": 0, "reasoning_effort": "none",
                                    "stream": True, "warmup_excluded": True},
               "cache_note": "No server cache-clear API is used. Every measured prompt starts with a unique nonce; "
                             "actual prefix reuse is taken from before/after engine history and API usage. The target "
                             "size uses /v1/messages/count_tokens; completion usage is also retained for the actual "
                             "OpenAI request.",
               "case_results": grouped, "raw_jsonl": str(raw_path)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
