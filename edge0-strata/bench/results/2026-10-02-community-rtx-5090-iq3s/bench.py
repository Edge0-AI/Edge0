"""Speed sweep against the live :8080, in the docs/COMMUNITY_BENCHMARKS.md format.

Prompts are built from this repository's text the way tools/needle_bench.py does, with a fresh
slice per run so no prompt is reused (cache_n must stay 0; a run that shows reuse is reported).
Greedy, thinking off, 256 output tokens, 1 warm-up + 3 measured runs per cell. Throughput comes
from the engine's own clock (the response's `timings`, llama.cpp names); TTFT is measured at the
client on one streaming run per cell (the server is a FIFO, so TTFT can include queue time).
Decode runs under 75% of the cell median are flagged as possible WDDM stalls (PR #279).

    python bench/results/2026-10-02-community-rtx-5090-iq3s/bench.py
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # the Strata runtime folder
sys.path.insert(0, str(ROOT / "tools"))
from needle_bench import haystack, CHARS_PER_TOKEN  # noqa: E402

URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.8-flash-next-iq3_s"
LOG = ROOT / "strata-iq3_s.log"
LENGTHS = [1024, 4096, 32768, 131072]
MAX_NEW = 256
OUT = Path(__file__).with_name("runs.json")

TAIL = re.compile(r"(decode expert cache hit rate: [\d.]+%[^\n]*|KV streaming: [\d.]+%[^\n]*)")


def post(payload: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def make_prompt(n_tokens: int, off: int) -> str:
    n_chars = int(n_tokens * CHARS_PER_TOKEN)
    text = haystack(n_chars + off + 4096)[off:off + n_chars]
    return (text + "\n\nQuestion: In at least 150 words, summarize what kinds of files the text "
                    "above is made of and what they are for. Name three of them.\n")


def log_tails(n_lines: int = 400) -> dict:
    try:
        lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-n_lines:]
    except OSError:
        return {}
    hits = [m for l in lines for m in TAIL.findall(l)]
    out = {}
    for h in hits[::-1]:
        if h.startswith("decode") and "hit" not in out:
            out["hit"] = h
        if h.startswith("KV") and "kv" not in out:
            out["kv"] = h
        if len(out) == 2:
            break
    return out


def one_run(n_tokens: int, off: int) -> dict:
    body = {"model": MODEL, "max_tokens": MAX_NEW, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": make_prompt(n_tokens, off)}]}
    t0 = time.time()
    r = post(body)
    t = r.get("timings") or {}
    u = r.get("usage") or {}
    return {"prompt_target": n_tokens, "prompt_tokens": u.get("prompt_tokens"),
            "cache_n": t.get("cache_n"), "generated": t.get("predicted_n") or u.get("completion_tokens"),
            "prompt_tok_s": t.get("prompt_per_second"), "decode_tok_s": t.get("predicted_per_second"),
            "draft_n": t.get("draft_n"), "draft_ok": t.get("draft_n_accepted"),
            "total_s": round(time.time() - t0, 2), "wall_s": round(t.get("prompt_ms", 0) / 1000 + t.get("predicted_ms", 0) / 1000, 2),
            **log_tails()}


def ttft_run(n_tokens: int, off: int) -> float | None:
    body = {"model": MODEL, "max_tokens": MAX_NEW, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": make_prompt(n_tokens, off)}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[6:])["choices"][0].get("delta") or {}
            except (KeyError, ValueError):
                continue
            if d.get("content") or d.get("reasoning_content"):
                return round(time.time() - t0, 2)
    return None


class Vram(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop_at, self.mins = time.time() + 100000, []

    def run(self) -> None:
        while time.time() < self.stop_at:
            try:
                s = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.free",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                self.mins.append(int(s.stdout.strip()))
            except Exception:
                pass
            time.sleep(5)


def main() -> int:
    vram = Vram()
    vram.start()
    rows = []
    for L in LENGTHS:
        for i in range(4):                                   # run 0 = warm-up, not counted
            row = one_run(L, i * 1_000_003 + L)
            row["length"], row["run"], row["warmup"] = L, i, i == 0
            rows.append(row)
            print(f"{L:>7} run {i}: prompt {row['prompt_tok_s']} tok/s, decode {row['decode_tok_s']} tok/s, "
                  f"cache_n {row['cache_n']}, gen {row['generated']}, {row['total_s']} s", flush=True)
        rows.append({"length": L, "ttft_s": ttft_run(L, 4 * 1_000_003 + L), "stream_ttft": True})
        print(f"{L:>7} TTFT {rows[-1]['ttft_s']} s", flush=True)
    vram.stop_at = time.time()
    time.sleep(0.2)

    print("\n| Prompt | prompt tok/s (3 runs) | decode tok/s (3 runs) | TTFT s | draft accept |")
    print("| --- | --- | --- | --- | --- |")
    for L in LENGTHS:
        m = [r for r in rows if r.get("length") == L and not r.get("warmup") and not r.get("stream_ttft")]
        p = [r["prompt_tok_s"] for r in m if r["prompt_tok_s"]]
        d = [r["decode_tok_s"] for r in m if r["decode_tok_s"]]
        tt = next((r["ttft_s"] for r in rows if r.get("length") == L and r.get("stream_ttft")), None)
        dr = sum(r["draft_ok"] or 0 for r in m) / max(1, sum(r["draft_n"] or 0 for r in m))
        med = statistics.median(d) if d else 0
        stall = [r["decode_tok_s"] for r in m if r["decode_tok_s"] and r["decode_tok_s"] < 0.75 * med]
        print(f"| {L:,} | {statistics.median(p):,.0f} ({min(p):,.0f}-{max(p):,.0f}) | "
              f"{med:,.1f} ({min(d):,.1f}-{max(d):,.1f}) | {tt} | {dr:.0%} |"
              + (f"  STALL: {stall}" if stall else ""))
    reused = [r for r in rows if not r.get("warmup") and not r.get("stream_ttft") and r.get("cache_n")]
    print(f"\nruns with prompt reuse (cache_n>0): {len(reused)}; VRAM free min during sweep: {min(vram.mins)} MiB")
    OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
