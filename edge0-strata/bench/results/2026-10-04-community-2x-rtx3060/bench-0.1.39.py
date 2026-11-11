#!/usr/bin/env python3
"""
Strata community benchmark runner — 2x RTX 3060, Qwen3.8-Flash-Next IQ3_S.

Protocol follows docs/COMMUNITY_BENCHMARKS.md:
  * three prompt sizes (~4k / ~32k / ~128k tokens), RUNS measured runs each;
  * non-streaming requests for the main numbers, because the engine returns exact
    per-request `timings` (prompt_n / cache_n / prompt_ms / predicted_n / predicted_ms /
    draft_n) — the streaming path does NOT emit usage or timings on this version;
  * a separate streaming run per size to measure time-to-first-token, because TTFT can
    only be observed client-side;
  * fixed output cap (256 tokens), greedy (temperature 0);
  * every run carries a distinct leading tag, so a prompt prefix is never reused and
    `cache_n` stays 0 — freshly processed prompt tokens, as the guide asks.

Environment variables (all optional):
  STRATA_URL       default http://127.0.0.1:8081
  STRATA_KEY       default "local"
  STRATA_MODEL     default qwen3.8-flash-next-iq3_s
  STRATA_RUNS      default 3
  STRATA_MAXTOK    default 256
  STRATA_TARGETS   default 4096,32768,131072
  STRATA_TTFT      default 1 (also run one streaming TTFT run per size)

Outputs (next to this script):
  results.json     meta + every run + summary
  raw/*.json       the complete response of every non-streaming run
  samples.jsonl    GPU/RAM samples taken every 2 s while the bench runs
"""
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

URL = os.environ.get("STRATA_URL", "http://127.0.0.1:8081")
API_KEY = os.environ.get("STRATA_KEY", "local")
MODEL = os.environ.get("STRATA_MODEL", "qwen3.8-flash-next-iq3_s")
OUTDIR = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(OUTDIR, "raw")
RUNS = int(os.environ.get("STRATA_RUNS", "3"))
MAXTOK = int(os.environ.get("STRATA_MAXTOK", "256"))
TARGETS = [int(x) for x in os.environ.get("STRATA_TARGETS", "4096,32768,131072").split(",")]
DO_TTFT = os.environ.get("STRATA_TTFT", "1") == "1"

# Fixed filler text; the exact prompts are reproducible from this string plus the
# repetition count recorded in each run (see `filler_repeats` / `prompt_sha256`).
FILLER = (
    "A mixture-of-experts layer replaces one large feed-forward network with several smaller "
    "expert networks plus a routing function. For every token the router selects a small subset "
    "of experts, so the total parameter count can grow while the compute per token stays roughly "
    "constant. The trade-off is memory: every expert that might be selected has to be resident "
    "somewhere, and on a machine whose VRAM is smaller than the model, the experts that are not "
    "cached are streamed from host memory or from disk. This makes decode throughput sensitive to "
    "the expert-cache hit rate, the PCIe link, and the layout of the cache itself. "
)
TASK = "\n\nTask: summarise the text above in exactly one sentence."


def build_prompt(tag, target_tokens):
    """English filler is ~4 chars/token; aim a little above the target and let the
    engine report the exact prompt token count."""
    want = int(target_tokens * 3.9)
    parts, n = [], 0
    while n < want:
        parts.append(FILLER)
        n += len(FILLER)
    body = "".join(parts)
    return f"[run {tag}] " + body + TASK, len(parts)


def gpu_mem():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        return {int(a): int(b) for a, b in (l.split(",") for l in out.strip().splitlines())}
    except Exception:
        return {}


def ram():
    info = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0])
    mib = 1048576.0
    return {"total_gb": round(info["MemTotal"] / mib, 2),
            "used_gb": round((info["MemTotal"] - info["MemAvailable"]) / mib, 2),
            "available_gb": round(info["MemAvailable"] / mib, 2)}


_samples = []
_sampling = threading.Event()


def _sampler():
    while not _sampling.is_set():
        _samples.append({"t": round(time.time(), 3), "gpu_mib": gpu_mem(), "ram": ram()})
        _sampling.wait(2.0)


def post(path, payload, timeout=3600):
    req = urllib.request.Request(
        URL + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def run_nonstream(tag, target, repeat):
    prompt, reps = build_prompt(tag, target)
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": MAXTOK, "temperature": 0, "stream": False}
    t0 = time.perf_counter()
    with post("/v1/chat/completions", payload) as resp:
        raw = resp.read().decode("utf-8", "replace")
    wall = time.perf_counter() - t0
    obj = json.loads(raw)
    with open(os.path.join(RAW, f"{tag}.json"), "w") as fh:
        json.dump({"request": {k: v for k, v in payload.items() if k != "messages"},
                   "prompt_chars": len(prompt), "filler_repeats": reps,
                   "response": obj}, fh, indent=1)
    usage, tim = obj.get("usage") or {}, obj.get("timings") or {}
    ch = (obj.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return {
        "run": tag, "mode": "nonstream", "target_prompt_tokens": target,
        "filler_repeats": reps, "prompt_chars": len(prompt),
        "wall_seconds": round(wall, 3),
        "prompt_tokens_reported": usage.get("prompt_tokens"),
        "completion_tokens_reported": usage.get("completion_tokens"),
        "finish_reason": ch.get("finish_reason"),
        "content_chars": len(msg.get("content") or ""),
        "reasoning_chars": len(msg.get("reasoning_content") or ""),
        "timings": tim,
        "prompt_tok_s": tim.get("prompt_per_second"),
        "decode_tok_s": tim.get("predicted_per_second"),
        "cache_n": tim.get("cache_n"),
        "draft_n": tim.get("draft_n"),
    }


def run_stream(tag, target):
    prompt, reps = build_prompt(tag, target)
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": MAXTOK, "temperature": 0, "stream": True}
    t0 = time.perf_counter()
    ttft = None
    n_content = n_reason = 0
    with post("/v1/chat/completions", payload) as resp:
        for rawline in resp:
            line = rawline.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue                      # keep-alive comments
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            delta = ((obj.get("choices") or [{}])[0].get("delta")) or {}
            if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                ttft = time.perf_counter() - t0
            if delta.get("content"):
                n_content += 1
            if delta.get("reasoning_content"):
                n_reason += 1
    wall = time.perf_counter() - t0
    return {"run": tag, "mode": "stream", "target_prompt_tokens": target,
            "filler_repeats": reps, "prompt_chars": len(prompt),
            "ttft_seconds": round(ttft, 3) if ttft is not None else None,
            "wall_seconds": round(wall, 3),
            "content_chunks": n_content, "reasoning_chunks": n_reason}


def meta():
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip()
        except Exception:
            return ""
    gpu = sh(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,power.limit,"
              "power.max_limit,pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,"
              "pcie.link.width.current", "--format=csv,noheader"])
    return {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "engine": {"url": URL, "model": MODEL, "health": sh(["curl", "-s", "-m", "5",
                   "--noproxy", "*", URL + "/health"])},
        "host": {"uname": sh(["uname", "-a"]),
                 "os": sh(["bash", "-c", ". /etc/os-release && echo $PRETTY_NAME"]),
                 "cpu_model": sh(["bash", "-c", "lscpu | grep -m1 'Model name' | cut -d: -f2- | xargs"]),
                 "cpu_sockets_cores_threads": sh(["bash", "-c",
                    "lscpu | grep -E '^(Socket|Core|Thread)\\(s\\)' | tr -s ' ' | tr '\\n' ';'"]),
                 "ram": ram(), "gpu": gpu, "gpu6": gpu_mem()},
        "protocol": {"targets": TARGETS, "runs_per_target": RUNS, "max_tokens": MAXTOK,
                     "temperature": 0, "streaming_ttft_runs": DO_TTFT,
                     "filler_sha256": __import__("hashlib").sha256(FILLER.encode()).hexdigest(),
                     "filler_chars": len(FILLER)},
    }


def main():
    os.makedirs(RAW, exist_ok=True)
    print("== health ==")
    try:
        with urllib.request.urlopen(URL + "/health", timeout=10) as r:
            print(r.read().decode())
    except Exception as e:
        print("health failed:", e)
        return 1
    m = meta()
    print(json.dumps(m["host"], indent=1)[:1200])

    t = threading.Thread(target=_sampler, daemon=True)
    t.start()
    runs = []
    for target in TARGETS:
        for i in range(1, RUNS + 1):
            tag = f"nonstream-{target}-r{i}"
            print(f"[{time.strftime('%H:%M:%S')}] {tag} ...", flush=True)
            try:
                rec = run_nonstream(tag, target, None)
            except Exception as e:
                rec = {"run": tag, "mode": "nonstream", "target_prompt_tokens": target,
                       "error": f"{type(e).__name__}: {e}"}
            rec["gpu_mib_after"] = gpu_mem()
            rec["ram_after"] = ram()
            runs.append(rec)
            print("   ->", {k: rec.get(k) for k in
                            ("prompt_tokens_reported", "completion_tokens_reported",
                             "prompt_tok_s", "decode_tok_s", "wall_seconds", "cache_n")}, flush=True)
    if DO_TTFT:
        for target in TARGETS:
            tag = f"stream-{target}-r1"
            print(f"[{time.strftime('%H:%M:%S')}] {tag} (TTFT) ...", flush=True)
            try:
                rec = run_stream(tag, target)
            except Exception as e:
                rec = {"run": tag, "mode": "stream", "target_prompt_tokens": target,
                       "error": f"{type(e).__name__}: {e}"}
            runs.append(rec)
            print("   -> ttft =", rec.get("ttft_seconds"), "wall =", rec.get("wall_seconds"), flush=True)
    _sampling.set()

    # summary: median + range per target (non-streaming runs only)
    summary = {}
    for target in TARGETS:
        rs = [r for r in runs if r.get("mode") == "nonstream"
              and r.get("target_prompt_tokens") == target and "error" not in r]
        if not rs:
            continue
        def med(key):
            vals = [r[key] for r in rs if isinstance(r.get(key), (int, float))]
            return {"median": round(statistics.median(vals), 3),
                    "min": round(min(vals), 3), "max": round(max(vals), 3),
                    "n": len(vals)} if vals else None
        summary[str(target)] = {
            "runs": len(rs),
            "prompt_tokens": med("prompt_tokens_reported"),
            "completion_tokens": med("completion_tokens_reported"),
            "prompt_tok_s": med("prompt_tok_s"),
            "decode_tok_s": med("decode_tok_s"),
            "wall_seconds": med("wall_seconds"),
            "cache_n": med("cache_n"),
        }
    out = {"meta": m, "runs": runs, "summary": summary,
           "gpu_peak_mib": {str(k): max((s["gpu_mib"].get(k, 0) for s in _samples), default=None)
                            for k in (0, 1)},
           "ram_peak_used_gb": max((s["ram"]["used_gb"] for s in _samples), default=None)}
    with open(os.path.join(OUTDIR, "results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(OUTDIR, "samples.jsonl"), "w") as fh:
        for s in _samples:
            fh.write(json.dumps(s) + "\n")
    print("\n== summary ==")
    print(json.dumps(summary, indent=1))
    print("gpu_peak_mib:", out["gpu_peak_mib"], "ram_peak_used_gb:", out["ram_peak_used_gb"])
    print("wrote results.json / samples.jsonl / raw/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
