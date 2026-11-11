#!/usr/bin/env python3
"""benchmark.py - this report's client: one ~1,000-token warm-up (not counted), then three runs each at
4,096 / 32,768 / 128,000 prompt tokens (prompts.json, built by make_prompts.py), one request at a time,
streamed, temperature 0, 256-token cap, the server's other defaults. Per request: time to the first streamed token
(reasoning or answer text, whichever comes first; keep-alives and empty deltas ignored), total wall time, the server's
`timings` (llama.cpp's names, from the engine's own clock: cache_n reused, prompt_n read, prompt_ms, predicted_n,
predicted_ms, drafts) and memory right after it (MemAvailable, the engine's RSS, each GPU's used VRAM) and what the engine read from the
drive during it (/proc/PID/io read_bytes). Linux only (it reads /proc).

    python benchmark.py --base http://127.0.0.1:8081 --label NAME --out NAME.json"""
import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8081")
ap.add_argument("--prompts", default="prompts.json")
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--engine-pattern", default="strata --serve --pack")
ap.add_argument("--sizes", default="", help="only these sizes (warmup,4096,32768,128000); default all")
ap.add_argument("--reps", default="", help="only these reps, e.g. 0,1")
a = ap.parse_args()


def engine_pid():
    """the engine process: matches the pattern and its executable is named strata"""
    import os
    for pid in subprocess.run(["pgrep", "-f", a.engine_pattern], capture_output=True, text=True).stdout.split():
        try:
            if os.path.basename(os.readlink(f"/proc/{pid}/exe")) == "strata":
                return pid
        except OSError:
            pass
    return None


def memory():
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        if k in ("MemAvailable", "SwapFree", "SwapTotal"):
            m[k] = int(v.split()[0]) // 1024
    pid = engine_pid()
    rss = 0
    if pid:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS"):
                rss = int(line.split()[1]) // 1024
    g = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.split("\n")
    return {"mem_available_mib": m.get("MemAvailable"), "swap_used_mib": m.get("SwapTotal", 0) - m.get("SwapFree", 0),
            "engine_rss_mib": rss, "gpu_used_mib": {x.split(",")[0].strip(): int(x.split(",")[1]) for x in g if x.strip()}}


def engine_read_bytes():
    """bytes the engine process fetched from storage so far (/proc/PID/io read_bytes: page-cache hits not counted)"""
    pid = engine_pid()
    try:
        for line in open(f"/proc/{pid}/io"):
            if line.startswith("read_bytes"):
                return int(line.split()[1])
    except (OSError, TypeError):
        return None


def ask(msgs):
    body = json.dumps({"model": "strata", "messages": msgs, "max_tokens": a.max_tokens, "temperature": 0,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    r = urllib.request.Request(a.base + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    first, finish, usage, timings, n_content, n_reason, first_kind = None, None, None, None, 0, 0, None
    with urllib.request.urlopen(r, timeout=1800) as f:
        for raw in f:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            usage = d.get("usage") or usage
            timings = d.get("timings") or timings
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                txt, rsn = delta.get("content") or "", delta.get("reasoning_content") or ""
                if (txt or rsn) and first is None:
                    first, first_kind = time.time() - t0, "reasoning" if rsn else "answer"
                n_content += len(txt)
                n_reason += len(rsn)
                finish = c.get("finish_reason") or finish
    return {"ttft_s": round(first, 3) if first is not None else None, "first_token_kind": first_kind,
            "total_s": round(time.time() - t0, 3), "finish": finish, "usage": usage, "timings": timings,
            "answer_chars": n_content, "reasoning_chars": n_reason}


P = json.loads(Path(a.prompts).read_text())
runs = []
warm = None
for p in P:
    if a.sizes and p["size"] not in a.sizes.split(","):
        continue
    if a.reps and p["size"] != "warmup" and str(p["rep"]) not in a.reps.split(","):
        continue
    rb0 = engine_read_bytes()
    rec = {"label": a.label, "size": p["size"], "rep": p["rep"], "target_tokens": p["target"],
           "started": time.strftime("%Y-%m-%dT%H:%M:%S"), **ask(p["messages"])}
    rb1 = engine_read_bytes()
    rec["drive_read_gb"] = round((rb1 - rb0) / 1e9, 2) if rb0 is not None and rb1 is not None else None
    rec["memory_after"] = memory()
    t = rec["timings"] or {}
    print(f"{a.label} {p['size']:>7} r{p['rep']}: prompt {t.get('prompt_n')} read + {t.get('cache_n')} reused in "
          f"{t.get('prompt_ms')} ms ({t.get('prompt_per_second')} tok/s), {t.get('predicted_n')} generated at "
          f"{t.get('predicted_per_second')} tok/s, TTFT {rec['ttft_s']} s ({rec['first_token_kind']}), finish {rec['finish']}, "
          f"drive {rec['drive_read_gb']} GB, MemAvailable {rec['memory_after']['mem_available_mib']} MiB",
          flush=True)
    if p["size"] != "warmup":
        runs.append(rec)
    else:
        warm = rec
Path(a.out).write_text(json.dumps({"label": a.label, "warmup": warm, "runs": runs}, indent=1))
print("wrote", a.out)
