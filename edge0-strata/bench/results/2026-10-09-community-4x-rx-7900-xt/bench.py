"""Strata community bench driver: sends prompts.jsonl (from gen.py) one at a time, records engine timings.

usage: bench.py prompts.jsonl OUT.jsonl
Streaming chat completions, temperature 0 (greedy), reasoning_effort none, 256-token cap. After each
request the engine's own timings are read from /v1/status (last_timings), with RAM and VRAM use.
"""

import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8086"
MODEL = "strata-ista-iq3s"
CAP = 256


def status() -> dict:
    with urllib.request.urlopen(URL + "/v1/status", timeout=30) as r:
        return json.load(r)


def request(content: str) -> dict:
    body = {
        "model": MODEL,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "reasoning_effort": "none",
        "max_tokens": CAP,
        "messages": [{"role": "user", "content": content}],
    }
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0_wall, t0 = time.time(), time.perf_counter()
    t_first = None
    usage, finish, reasoning_chars, answer_chars = {}, None, 0, 0
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                a, r = d.get("content") or "", d.get("reasoning_content") or ""
                if (a or r) and t_first is None:
                    t_first = time.perf_counter()
                answer_chars += len(a)
                reasoning_chars += len(r)
                finish = ch.get("finish_reason") or finish
    t1 = time.perf_counter()
    st = status()
    lt = st.get("last_timings") or {}
    machine = st.get("machine") or {}
    return {
        "t_start": round(t0_wall, 3), "t_end": round(time.time(), 3),
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": finish, "answer_chars": answer_chars, "reasoning_chars": reasoning_chars,
        "ttft_s": round(t_first - t0, 3) if t_first else None, "total_s": round(t1 - t0, 3),
        "engine": {k: lt.get(k) for k in ("cache_n", "prompt_n", "prompt_ms", "prompt_per_second",
                                          "predicted_n", "predicted_ms", "predicted_per_second",
                                          "draft_n", "draft_n_accepted")},
        "ram_used_gib": machine.get("ram", {}).get("used_gib"),
        "gpu_used_mib": machine.get("gpu", {}).get("used_mib"),
    }


prompts = [json.loads(line) for line in open(sys.argv[1])]
with open(sys.argv[2], "a") as out:
    for p in prompts:
        rec = {k: p[k] for k in ("label", "target", "counted", "offset", "content_tokens")}
        rec.update(request(p["content"]))
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + "\n")
print("BENCH_DONE", flush=True)
