#!/usr/bin/env python3
"""Strata community benchmark runner: warm-up + N measured runs per prompt, server timings, one JSON per config.

Follows docs/COMMUNITY_BENCHMARKS.md: prompt and decode throughput from the engine's own timings, actual token
counts, reused (cache_n) vs fresh prompt tokens, every run kept (median and range computed from them).
"""
import argparse, hashlib, json, statistics, time, urllib.request
from pathlib import Path

def call(base, prompt, max_tokens):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "seed": 42,
            "max_tokens": max_tokens, "cache_prompt": False}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    wall = time.perf_counter() - t0
    t = d.get("timings", {})
    text = d["choices"][0]["message"].get("content") or ""
    return {"wall_s": round(wall, 3), "timings": t, "usage": d.get("usage"),
            "finish_reason": d["choices"][0].get("finish_reason"),
            "content_sha256": hashlib.sha256(text.encode()).hexdigest()}

def summary(runs, key):
    v = [r["timings"].get(key) for r in runs if r["timings"].get(key) is not None]
    return {"median": statistics.median(v), "min": min(v), "max": max(v)} if v else None

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--prompts", default=str(Path(__file__).parent / "prompts"))
    p.add_argument("--sizes", default="short,medium,long")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    props = json.load(urllib.request.urlopen(a.base + "/props", timeout=30))
    res = {"label": a.label, "build_info": props.get("build_info"), "model_path": props.get("model_path"),
           "n_ctx": props.get("default_generation_settings", {}).get("n_ctx"), "max_tokens": a.max_tokens,
           "sampling": "temperature 0, seed 42, cache_prompt false", "sizes": {}}
    for size in a.sizes.split(","):
        prompt = (Path(a.prompts) / f"{size}.txt").read_text(encoding="utf-8")
        # The server reuses a matching prefix even with cache_prompt false, so every request starts with a unique
        # line: each run then reads the whole prompt fresh (cache_n stays at the chat template's few tokens).
        tag = lambda kind, i: f"[{a.label} {size} {kind} {i} {time.time_ns()}]\n"
        warm = [call(a.base, tag("warmup", i) + prompt, a.max_tokens) for i in range(a.warmup)]
        runs = []
        for i in range(a.runs):
            r = call(a.base, tag("run", i + 1) + prompt, a.max_tokens)
            r["run"] = i + 1
            runs.append(r)
            t = r["timings"]
            print(f"{a.label} {size} run {i+1}: prompt {t.get('prompt_n')} (+{t.get('cache_n')} reused) "
                  f"{t.get('prompt_per_second')} tok/s, decode {t.get('predicted_n')} tok "
                  f"{t.get('predicted_per_second')} tok/s, wall {r['wall_s']} s", flush=True)
        res["sizes"][size] = {"warmup": warm, "runs": runs,
                              "prompt_tps": summary(runs, "prompt_per_second"),
                              "decode_tps": summary(runs, "predicted_per_second"),
                              "prompt_ms": summary(runs, "prompt_ms")}
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("written", a.out)

if __name__ == "__main__":
    main()
