#!/usr/bin/env python3
"""Strata benchmark harness with pluggable model name.

For each configured --max-context, finds a chat content that lands the prompt
near 92 % of capacity, runs:
  - 1 cold-prefill probe (max_tokens=1, non-stream) for the real prefill
    throughput number (the number the README's 2,000+ t/s headlines measure)
  - 3 SSE stream /v1/chat/completions calls (run once each, with `reasoning_effort=minimal`)
  - no extra reference non-stream call

Why two prefill numbers:
  - SSE's `timings.prompt_per_second` is "time to FIRST decode chunk after
    prefilled chunk" (first-chunk-completion throughput). With --prefill auto
    doing 8192-token chunks, the engine starts streaming before the entire
    prompt is in. So SSE reports a smaller denominator than the cold prefill.
  - Cold-prefill non-stream (max_tokens=1) reports end-of-prompt throughput.
    Useful for apples-to-apples comparison with the README's stated numbers.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional
import requests

TARGET_FRACTION = 0.92
CHAR_PER_TOKEN = 4.0
PEER_WINDOW = 32

FILLER = (
    "Strata models a pagoda of experts and policies; during a long context "
    "session the persistent graph speculatively drafts the next token while "
    "the verifier follows. The verifier commits the draft when the "
    "distributions agree, else resamples and re-aligns. On a workstation with "
    "a single 16 GB card, the expert cache holds about a thousand experts "
    "and the README warns that context past 65 k degrades without KV-cache "
    "compression. The MTP draft layer is a five-token head tuned from the "
    "1.3 B Qwen3 base and imported unchanged. "
)


def render_prompt(target_chars: int) -> str:
    out, cur = [], 0
    while cur < target_chars:
        out.append(FILLER)
        cur += len(FILLER)
    return "".join(out)[:target_chars]


@dataclass
class Run:
    seq: int
    target_ctx: int
    target_fill_chars: int
    prompt_tokens_actual: int
    completion_tokens: int
    content_chunks: int
    reasoning_chunks: int
    ttft_ms: float
    decode_avg_tps: float
    decode_peak_tps: float
    eng_predicted_per_second: float
    eng_predicted_per_token_ms: float
    eng_prompt_per_second: float
    sse_prompt_per_second: float
    eng_draft_n: int
    eng_draft_n_accepted: int
    eng_predicted_ms: float
    eng_total_request_ms: float
    finish_reason: str
    elapsed_ms: float


@dataclass
class ColdPrefill:
    prompt_tokens_actual: int
    prompt_total_ms: float
    prompt_per_second: float
    completion_tokens: int


def lookup_v1_models(base_url: str, model_name: str) -> str:
    r = requests.get(f"{base_url}/v1/models", timeout=20)
    r.raise_for_status()
    data = r.json().get("data", [])
    for m in data:
        if m["id"] == model_name:
            return model_name
    if data:
        print(f"  WARN: {model_name} not in /v1/models; server reports: {data[0]['id']}")
    return data[0]["id"] if data else model_name


def calibration(base_url: str, fill_chars: int, model_name: str) -> int:
    body = {
        "model": model_name,
        "stream": False,
        "reasoning_effort": "minimal",
        "messages": [{"role": "user", "content": render_prompt(fill_chars)}],
        "max_tokens": 4,
        "temperature": 0,
    }
    r = requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=900)
    r.raise_for_status()
    return r.json()["usage"]["prompt_tokens"]


def find_sweet_fill_chars(base_url: str, target_ctx: int, model_name: str):
    goal = int(target_ctx * TARGET_FRACTION)
    lo = int(target_ctx * 0.50 / CHAR_PER_TOKEN)
    hi = int(target_ctx * 0.97 / CHAR_PER_TOKEN)
    n_lo = calibration(base_url, lo, model_name)
    n_hi = calibration(base_url, hi, model_name)
    if n_hi == n_lo:
        return hi, n_hi
    slope = (n_hi - n_lo) / (hi - lo)
    fill = max(1, int(lo + (goal - n_lo) / slope))
    best_c, best_n = fill, calibration(base_url, fill, model_name)
    for d in (-2048, -1024, 0, +1024, +2048):
        c = max(1, fill + d)
        if c < lo or c > hi:
            continue
        n = calibration(base_url, c, model_name)
        if abs(n - goal) < abs(best_n - goal):
            best_c, best_n = c, n
    return best_c, best_n


def force_cold_prefill_state(base_url: str) -> None:
    # Upstream (>= 6f32ec07) requires Content-Type: application/json on POST
    # /unload; the server returns HTTP 415 without it. Older (rc/0.1.30) was
    # tolerant. Send an explicit JSON header.
    try:
        r = requests.post(
            f"{base_url}/unload",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        r.raise_for_status()
    except Exception as e:
        print(f"  warn: /unload failed: {e!r}")
    try:
        for _ in range(60):
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                j = r.json()
                if j.get("loaded"):
                    break
            time.sleep(1)
    except Exception:
        pass


def cold_prefill(base_url: str, fill_chars: int, model_name: str,
                 target_ctx: int) -> ColdPrefill:
    force_cold_prefill_state(base_url)
    body = {
        "model": model_name,
        "stream": False,
        "reasoning_effort": "minimal",
        "messages": [{"role": "user", "content": render_prompt(fill_chars)}],
        "max_tokens": 1,
        "temperature": 0,
    }
    r = requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=1800)
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage", {}) or {}
    timings = data.get("timings", {}) or {}
    prompt_n = timings.get("prompt_n") or usage.get("prompt_tokens", 0)
    prompt_ms = timings.get("prompt_ms", 0.0)
    prompt_per_s = timings.get("prompt_per_second", float(prompt_n) * 1000.0 / max(1.0, prompt_ms)) if prompt_ms else 0.0
    return ColdPrefill(
        prompt_tokens_actual=prompt_n,
        prompt_total_ms=round(prompt_ms, 1),
        prompt_per_second=round(prompt_per_s, 1),
        completion_tokens=timings.get("predicted_n") or usage.get("completion_tokens", 0),
    )


def stream_run(base_url: str, fill_chars: int, seq: int, target_ctx: int,
               model_name: str) -> Run:
    body = {
        "model": model_name,
        "stream": True,
        "reasoning_effort": "minimal",
        "messages": [{"role": "user", "content": render_prompt(fill_chars)}],
        "max_tokens": 192,
        "temperature": 0,
    }
    t_req = time.perf_counter()
    content_ts: List[float] = []
    n_content, n_reasoning = 0, 0
    final_event = None
    finish_reason = None

    with requests.post(f"{base_url}/v1/chat/completions", json=body,
                       stream=True, timeout=1800) as r:
        r.raise_for_status()
        # first chunk timestamp — proxy for server "received by client" latency
        first_remote = None
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            payload = raw[5:].strip()
            if payload == "[DONE]": break
            try:
                c = json.loads(payload)
            except json.JSONDecodeError:
                continue
            t = time.perf_counter()
            delta = (c["choices"][0].get("delta", {}) if c.get("choices") else {})
            if "content" in delta and delta["content"]:
                content_ts.append(t)
                n_content += 1
            if "reasoning_content" in delta and delta["reasoning_content"]:
                n_reasoning += 1
            if c.get("choices") and c["choices"][0].get("finish_reason"):
                finish_reason = c["choices"][0]["finish_reason"]
            if c.get("timings"):
                final_event = c
        last_remote = time.perf_counter()

    usage = (final_event or {}).get("usage", {}) or {}
    timings = (final_event or {}).get("timings", {}) or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    ttft_ms = ((content_ts[0] - t_req) * 1000) if content_ts else 0.0
    decode_avg_tps, decode_peak_tps = 0.0, 0.0
    if content_ts and completion_tokens > 0:
        decode_dt = max(0.001, content_ts[-1] - content_ts[0])
        decode_avg_tps = completion_tokens / decode_dt
        if len(content_ts) > PEER_WINDOW:
            for i in range(len(content_ts) - PEER_WINDOW):
                w = content_ts[i + PEER_WINDOW - 1] - content_ts[i]
                if w > 0:
                    rate = PEER_WINDOW / w
                    if rate > decode_peak_tps:
                        decode_peak_tps = rate
    elapsed = ((content_ts[-1] - t_req) * 1000) if content_ts else 0.0
    total_request_ms = (last_remote - t_req) * 1000

    return Run(
        seq=seq,
        target_ctx=target_ctx,
        target_fill_chars=fill_chars,
        prompt_tokens_actual=prompt_tokens,
        completion_tokens=completion_tokens,
        content_chunks=n_content,
        reasoning_chunks=n_reasoning,
        ttft_ms=round(ttft_ms, 2),
        decode_avg_tps=round(decode_avg_tps, 2),
        decode_peak_tps=round(decode_peak_tps, 2),
        eng_predicted_per_second=round(timings.get("predicted_per_second", 0.0), 2),
        eng_predicted_per_token_ms=round(timings.get("predicted_per_token_ms", 0.0), 2),
        eng_prompt_per_second=round(timings.get("prompt_per_second", 0.0), 2),
        sse_prompt_per_second=round(timings.get("prompt_per_second", 0.0), 2),
        eng_draft_n=timings.get("draft_n", 0),
        eng_draft_n_accepted=timings.get("draft_n_accepted", 0),
        eng_predicted_ms=round(timings.get("predicted_ms", 0.0), 2),
        eng_total_request_ms=round(total_request_ms, 1),
        finish_reason=str(finish_reason or "stop"),
        elapsed_ms=round(elapsed, 1),
    )


def measure_window(base_url: str, target_ctx: int, model_name: str) -> dict:
    print(f"\n=== {model_name} @ ctx {target_ctx} ===")
    lookup_v1_models(base_url, model_name)
    fill_chars, prompt_tokens_actual = find_sweet_fill_chars(base_url, target_ctx, model_name)
    print(f"  calibration: fill_chars={fill_chars}, prompt_tokens={prompt_tokens_actual}")

    print(f"  cold-prefill probe (max_tokens=1, non-stream) ...")
    cp = cold_prefill(base_url, fill_chars, model_name, target_ctx)
    print(f"  coldfill  prompt_n={cp.prompt_tokens_actual:5d}  "
          f"prompt_total_ms={cp.prompt_total_ms:8.1f}  prompt_per_second={cp.prompt_per_second:7.1f} t/s")

    runs: List[Run] = []
    for i in range(1, 4):
        time.sleep(1.5)
        r = stream_run(base_url, fill_chars, seq=i, target_ctx=target_ctx,
                       model_name=model_name)
        runs.append(r)
        print(
            f"  seq={r.seq} prompt_tok={r.prompt_tokens_actual:5d} "
            f"completion_tok={r.completion_tokens:3d} ttft={r.ttft_ms:7.0f}ms "
            f"avg={r.decode_avg_tps:6.2f} t/s peak={r.decode_peak_tps:6.2f} t/s "
            f"(eng_pred={r.eng_predicted_per_second:6.2f}, "
            f"sse-prompt-rpt={r.sse_prompt_per_second:6.1f}, "
            f"draft+={r.eng_draft_n_accepted}/{r.eng_draft_n}) "
            f"finish={r.finish_reason}"
        )

    avgs = [r.decode_avg_tps for r in runs if r.decode_avg_tps]
    peaks = [r.decode_peak_tps for r in runs if r.decode_peak_tps]
    engs = [r.eng_predicted_per_second for r in runs if r.eng_predicted_per_second]
    return {
        "target_ctx": target_ctx,
        "model_name": model_name,
        "target_prompt_tokens": prompt_tokens_actual,
        "fill_chars": fill_chars,
        "cold_prefill": asdict(cp),
        "runs": [asdict(r) for r in runs],
        "avg_decode_avg_tps": round(statistics.mean(avgs), 2) if avgs else 0.0,
        "avg_peak_decode_tps": round(statistics.mean(peaks), 2) if peaks else 0.0,
        "peak_decode_tps_overall": round(max(peaks), 2) if peaks else 0.0,
        "engine_predicted_avg_tps": round(statistics.mean(engs), 2) if engs else 0.0,
        "engine_predicted_per_run": engs,
        "ttft_ms_per_run": [r.ttft_ms for r in runs],
        "engine_prefill_per_run": [r.eng_prompt_per_second for r in runs],
        "finish_reasons": [r.finish_reason for r in runs],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", required=True)
    ap.add_argument("--contexts", type=int, nargs="+", default=[32768])
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    out = {
        "engine": "Strata",
        "model_name": args.model,
        "host": "Shin-BlackMamba",
        "started_at_utc": int(time.time()),
        "contexts": args.contexts,
        "note": args.note,
        "results": [],
    }
    for ctx in args.contexts:
        out["results"].append(measure_window(args.base_url, ctx, args.model))

    out["finished_at_utc"] = int(time.time())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("\nWrote", args.out)


if __name__ == "__main__":
    main()
