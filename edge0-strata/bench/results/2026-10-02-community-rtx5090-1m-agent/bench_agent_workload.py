#!/usr/bin/env python3
"""Community benchmark: Strata at 1M context under an agent-like workload.

Follows docs/COMMUNITY_BENCHMARKS.md conventions:
- prompt and decode throughput come from the server's own `timings` block,
  never from dividing tokens by wall time;
- time to first token (TTFT) is measured at the client on the first
  non-empty streamed delta (reasoning or answer text — flagged per run);
- each size runs three fresh prompts plus one immediate repeat (warm prefix
  reuse), plus one greedy no-thinking run;
- actual prompt token counts are reported from the server, since the
  synthetic prompt length is only approximate.

Writes one JSON record per run to the file given as argv[1]
(default: agent_bench.jsonl). Contains no credentials.

Requires a running server on 127.0.0.1:8080 (edit SERVER_URL if needed).
"""

import json
import sys
import time
import urllib.request

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.8-flash-next-iq3_s"

# A paragraph of plain English; repeated to build synthetic prompts.
PARAGRAPH = (
    "The lighthouse keeper recorded the weather each hour: "
    "wind from the northwest, sea moderate, visibility good. "
)
# Rough paragraphs-per-token for this text; the real count is reported by
# the server in `prompt_n`, so this only sizes the request.
PARAGRAPHS_PER_TOKEN = 22

TASK_SUFFIX = "\n\nSummarize the pattern in these records in two sentences."


def build_prompt(target_tokens: int) -> str:
    paragraphs = target_tokens // PARAGRAPHS_PER_TOKEN + 2
    return PARAGRAPH * paragraphs + TASK_SUFFIX


def _first_delta(chunk: dict) -> dict:
    """The streamed delta of a chunk, defensively typed (malformed chunks
    simply yield {}, which the caller treats as 'no token yet')."""
    choices = chunk.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        delta = choices[0].get("delta")
        return delta if isinstance(delta, dict) else {}
    return {}


def run_once(prompt: str, *, max_tokens: int, temperature: float | None,
             top_p: float | None, thinking: bool) -> dict:
    """Stream one chat completion; return the server's timings plus client TTFT."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if not thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    request = urllib.request.Request(
        SERVER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    started = time.monotonic()
    ttft_s = None
    last_chunk: dict = {}
    with urllib.request.urlopen(request, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[len(b"data: "):].strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            if isinstance(chunk, dict):
                last_chunk = chunk
            if ttft_s is None:
                delta = _first_delta(chunk if isinstance(chunk, dict) else last_chunk)
                if delta.get("content") or delta.get("reasoning_content"):
                    ttft_s = time.monotonic() - started

    if not last_chunk:
        raise RuntimeError("server streamed no completion chunks")
    timings = last_chunk.get("timings") or {}

    record = {
        "t_wall_s": round(time.monotonic() - started, 3),
        "ttft_s": round(ttft_s, 3) if ttft_s is not None else None,
        "thinking": thinking,
        "temperature": temperature,
        "prompt_n": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_n": timings.get("predicted_n"),
        "predicted_ms": timings.get("predicted_ms"),
        "cache_n": timings.get("cache_n"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_per_second": timings.get("predicted_per_second"),
    }
    return record


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "agent_bench.jsonl"
    sizes = (4096, 32768, 131072, 524288)
    with open(out_path, "w", encoding="utf-8") as out:
        for size in sizes:
            prompt = build_prompt(size)
            for run_index in range(1, 4):
                record = run_once(prompt, max_tokens=256,
                                  temperature=1.0, top_p=0.95, thinking=True)
                record["tag"] = f"think_{size}_run{run_index}"
                out.write(json.dumps(record) + "\n")
                out.flush()
            # Immediate repeat: exercises prefix reuse (see cache_n / prompt_n).
            record = run_once(prompt, max_tokens=256,
                              temperature=1.0, top_p=0.95, thinking=True)
            record["tag"] = f"think_{size}_warm_repeat"
            out.write(json.dumps(record) + "\n")
            out.flush()
            # Instruct mode: greedy, thinking suppressed at the template.
            record = run_once(prompt, max_tokens=256,
                              temperature=0.0, top_p=None, thinking=False)
            record["tag"] = f"instruct_{size}_greedy"
            out.write(json.dumps(record) + "\n")
            out.flush()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
