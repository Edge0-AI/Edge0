#!/usr/bin/env python3
"""bench_report.py — community-report battery for the V100+P100 helper rig.

Fresh salted prompts (no prefix reuse), output cap 256 tokens, greedy,
reasoning_effort none (matches other community reports), streaming TTFT.
Sizes: ~550 / ~13.5K / ~33K x3 runs; ~90K x1 (single run, marked).
Saves per-run results to /root/benchpr-results.json and prints them.
"""
import json, sys, time, urllib.request

BASE = "http://127.0.0.1:8082"
OUT = "/root/benchpr-results.json"
unit = "The following is a technical report excerpt that must be summarized. "

TASK = ("\n\nTask: using general knowledge (the excerpt above is filler), explain in "
        "about 220 words how TCP congestion control works, covering slow start, congestion "
        "avoidance, fast retransmit and fast recovery, with rough numbers. Plain text.")

def run(tag, rep, n=256, timeout=2400):
    salt = f"[{tag}-{time.time_ns()}] "
    prompt = salt + unit * rep + TASK
    body = json.dumps({
        "model": "q",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n, "temperature": 0, "stream": True,
        "reasoning_effort": "none",
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    content = ""
    timings = None
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            ch = d.get("choices") or []
            if ch:
                delta = ch[0].get("delta") or {}
                txt = delta.get("content")
                if txt:
                    if ttft is None:
                        ttft = time.time() - t0
                    content += txt
            if d.get("timings"):
                timings = d["timings"]
            if d.get("usage"):
                usage = d["usage"]
    wall = time.time() - t0
    res = {
        "tag": tag, "rep": rep,
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "wall_s": round(wall, 2),
        "ttft_is_first_content_delta": True,
        "timings": timings, "usage": usage,
        "out_chars": len(content),
        "out_head": content[:120],
    }
    print(json.dumps({k: res[k] for k in ("tag", "ttft_s", "wall_s")}
                     | {"pn": (timings or {}).get("prompt_n"),
                        "reused": (timings or {}).get("cache_n"),
                        "gen": (timings or {}).get("predicted_n"),
                        "ptps": round((timings or {}).get("prompt_per_second", 0), 1),
                        "dtps": round((timings or {}).get("predicted_per_second", 0), 1)}), flush=True)
    return res

def main():
    print(f"=== BENCH REPORT START {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    results = []

    # warm-up (not part of results)
    w = run("warmup", 60, n=64)
    print("warmup done", flush=True)
    results.append(dict(w, tag="warmup", counted=False))

    sizes = [("tokens550", 60, 3), ("tokens13500", 1490, 3), ("tokens33000", 3660, 3), ("tokens90000", 9970, 1)]
    for name, rep, reps in sizes:
        for i in range(reps):
            r = run(f"{name}-run{i+1}", rep)
            r["counted"] = True
            results.append(r)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print(f"=== BENCH REPORT DONE {time.strftime('%Y-%m-%d %H:%M:%S')} -> {OUT} ===", flush=True)

if __name__ == "__main__":
    main()
