#!/usr/bin/env python3
"""Run one cached-prefix continuation after the fresh 32K benchmark suite.

Run once immediately after the measured 32k-run-3 result is present, before
another request changes the engine's active conversation prefix.
"""
import argparse
import copy
import json
import time
import urllib.request
from pathlib import Path


def get_json(url, headers):
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url, payload, headers, timeout=1800):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def latest_engine_record(metrics, request_started):
    for record in metrics.get("requests", []):
        if record.get("time", 0) >= request_started - 0.1:
            return record
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--raw", type=Path, default=Path(__file__).resolve().parent /
                        "workers-31-long" / "raw.jsonl")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent /
                        "workers-31-long" / "followup.json")
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()

    source = None
    with args.raw.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") == "32k-run-3" and not row.get("warmup"):
                source = row
    if source is None:
        raise SystemExit(f"No completed 32k-run-3 record found in {args.raw}; no request was sent.")
    original = source.get("request_body") or {}
    original_messages = original.get("messages") or []
    if len(original_messages) != 1 or original_messages[0].get("role") != "user":
        raise SystemExit("The saved 32k-run-3 request does not contain the expected single user message.")
    assistant_text = source.get("output_text") or ""
    if not assistant_text:
        raise SystemExit("The saved 32k-run-3 response is empty; no request was sent.")

    payload = copy.deepcopy(original)
    payload["messages"] = [
        original_messages[0],
        {"role": "assistant", "content": assistant_text},
        {"role": "user", "content": "Continue the review with concrete tests."},
    ]
    payload["max_tokens"] = 256
    payload["temperature"] = 0
    payload["reasoning_effort"] = "none"
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}

    base = args.url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    get_headers = {}
    if args.api_key:
        headers["Authorization"] = "Bearer " + args.api_key
        get_headers["Authorization"] = "Bearer " + args.api_key

    before = get_json(base + "/metrics", get_headers)
    started_wall = time.time()
    started = time.perf_counter()
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
    )
    chunks, text_parts, reasoning_parts = [], [], []
    usage, finish_reason, first_text, stream_error = None, None, None, None
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
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or ""
                if content or reasoning:
                    if first_text is None:
                        first_text = time.perf_counter() - started
                    text_parts.append(content)
                    reasoning_parts.append(reasoning)
                finish_reason = choice.get("finish_reason") or finish_reason
    total = time.perf_counter() - started
    after = get_json(base + "/metrics", get_headers)
    engine = latest_engine_record(after, started_wall)
    timings_chunk = next((chunk["timings"] for chunk in reversed(chunks) if chunk.get("timings")), None)
    engine_timings = None
    if engine:
        fresh = max(0, (engine.get("prompt_tokens") or 0) - (engine.get("reused") or 0))
        engine_timings = {
            "prompt_tokens": engine.get("prompt_tokens"),
            "fresh_prompt_tokens": fresh,
            "reused_tokens": engine.get("reused"),
            "prompt_ms": engine.get("prompt_ms"),
            "prefill_tok_s": fresh / (engine["prompt_ms"] / 1000) if engine.get("prompt_ms") else None,
            "generated_tokens": engine.get("engine_generated"),
            "decode_ms": engine.get("decode_ms"),
            "decode_tok_s": engine.get("decode_tok_s"),
        }
    result = {
        "source_run": "32k-run-3",
        "source_request_sha256": source.get("request_sha256"),
        "url": base,
        "request_body": payload,
        "client_ttft_s": first_text,
        "client_total_latency_s": total,
        "finish_reason": finish_reason,
        "usage": usage,
        "prefix_reused_tokens": (engine or {}).get("reused"),
        "api_cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
        "engine": engine,
        "engine_timings": timings_chunk or engine_timings,
        "output_text": "".join(text_parts),
        "reasoning_text": "".join(reasoning_parts),
        "raw_stream_chunks": chunks,
        "stream_error": stream_error,
        "metrics_before": before,
        "metrics_after": after,
        "cache_note": "This is a continuation of the saved 32k-run-3 conversation. Reuse is reported from the "
                      "engine history/API usage, not inferred from the repeated messages.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "ttft_s": first_text, "total_s": total,
                      "prompt_tokens": (engine or {}).get("prompt_tokens"),
                      "reused_tokens": result["prefix_reused_tokens"], "decode_tok_s":
                      (engine or {}).get("decode_tok_s"), "finish_reason": finish_reason,
                      "error": stream_error}, ensure_ascii=False))
    if stream_error:
        raise SystemExit(f"The stream reported an error; details saved in {args.out}")


if __name__ == "__main__":
    main()
