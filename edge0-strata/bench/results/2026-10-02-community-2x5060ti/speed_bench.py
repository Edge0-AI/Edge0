"""Strata speed test following docs/COMMUNITY_BENCHMARKS.md.

Three conditions, 3 measured runs each after one unmeasured warm-up, output capped at 256 tokens:
  short       104-token prompt, fully processed (a unique first line defeats prefix reuse)
  long-cold   ~21K-token prompt, fully processed (unique first line)
  long-reuse  the same ~21K prefix as the previous request, new question at the end
Throughput comes from the engine's own timings (prompt_ms / predicted_ms); TTFT and total
latency are measured at the client. The engine's log lines for each request are kept.

Usage: python speed_bench.py <folder of .ts files used as filler text> <path to the strata serve log>
"""
import json, pathlib, subprocess, sys, time, urllib.request, uuid

URL = "http://127.0.0.1:8080/v1/chat/completions"
if len(sys.argv) != 3:
    sys.exit(__doc__)
SRC = pathlib.Path(sys.argv[1])  # a folder of .ts files, used as filler text
LOG = pathlib.Path(sys.argv[2])  # the log file named in the run configuration
OUT = pathlib.Path(__file__).with_name("runs.json")
MAX_TOKENS = 256


def filler(chars):
    parts, n = [], 0
    for p in sorted(SRC.rglob("*.ts")):
        t = p.read_text(encoding="utf-8", errors="replace")
        parts.append(f"// ===== {p.name} =====\n{t}")
        n += len(t)
        if n > chars:
            break
    return "\n".join(parts)


def vram():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.split("\n")
    return {f"gpu{l.split(',')[0].strip()}_mib": int(l.split(",")[1]) for l in q if l.strip()}


def call(label, messages):
    log_pos = LOG.stat().st_size
    body = {"model": "x", "messages": messages, "max_tokens": MAX_TOKENS, "stream": True, "temperature": 0.6,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; first_kind = None; text = []; reasoning = []; timings = usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            timings = ev.get("timings") or timings
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                rc = d.get("reasoning_content") or d.get("reasoning")
                if (d.get("content") or rc) and first is None:
                    first, first_kind = time.perf_counter(), ("reasoning" if rc else "answer")
                if rc: reasoning.append(rc)
                if d.get("content"): text.append(d["content"])
    t1 = time.perf_counter()
    time.sleep(1.0)
    with LOG.open("rb") as f:
        f.seek(log_pos)
        log_lines = [l for l in f.read().decode("utf-8", "replace").splitlines() if l.startswith("strata serve:")]
    t = timings or {}
    row = {
        "label": label,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "reused_tokens": t.get("cache_n"), "read_tokens": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
        "prompt_tok_s": t.get("prompt_per_second"),
        "generated_tokens": t.get("predicted_n"), "decode_ms": t.get("predicted_ms"),
        "decode_tok_s": t.get("predicted_per_second"),
        "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted"),
        "ttft_s": round((first or t1) - t0, 3), "first_token_kind": first_kind,
        "total_latency_s": round(t1 - t0, 3),
        "vram_after": vram(),
        "engine_log": log_lines,
        "reasoning_chars": len("".join(reasoning)), "answer": "".join(text),
    }
    print(f"{label:14s} read {row['read_tokens']:>6} reused {row['reused_tokens']:>6} | prompt {row['prompt_tok_s']:7.1f} t/s"
          f" | decode {row['generated_tokens']} tok {row['decode_tok_s']:5.1f} t/s | TTFT {row['ttft_s']:6.2f}s total {row['total_latency_s']:6.2f}s",
          flush=True)
    return row


def main():
    ask = "Write a TypeScript function that parses a unified diff into hunks. Code only."
    big = filler(80_000)
    q = "Above is part of a codebase. In 5 bullet points, name the main modules and what each does."
    runs = []
    call("warm-up", [{"role": "user", "content": f"run {uuid.uuid4()}\n" + ask}])
    for i in range(3):
        runs.append(call(f"short-{i+1}", [{"role": "user", "content": f"run {uuid.uuid4()}\n" + ask}]))
    for i in range(3):
        prefix = f"run {uuid.uuid4()}\n" + big
        runs.append(call(f"long-cold-{i+1}", [{"role": "user", "content": prefix + "\n\n" + q}]))
        runs.append(call(f"long-reuse-{i+1}", [{"role": "user", "content": prefix + "\n\nList the 3 largest files you saw."}]))
    OUT.write_text(json.dumps({"date": time.strftime("%Y-%m-%d %H:%M"), "max_tokens": MAX_TOKENS,
                               "temperature": 0.6, "runs": runs}, indent=1), encoding="utf-8")

    def summary(prefix, key):
        v = sorted(r[key] for r in runs if r["label"].startswith(prefix) and r[key] is not None)
        return f"median {v[len(v)//2]:.1f} (range {v[0]:.1f}-{v[-1]:.1f}, n={len(v)})" if v else "n/a"
    for c in ("short", "long-cold", "long-reuse"):
        print(f"\n{c}: prompt t/s {summary(c, 'prompt_tok_s')} | decode t/s {summary(c, 'decode_tok_s')}"
              f" | TTFT s {summary(c, 'ttft_s')} | total s {summary(c, 'total_latency_s')}")


if __name__ == "__main__":
    main()
