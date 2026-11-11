#!/usr/bin/env python3
"""Strata community benchmark harness (RX 6800M / gfx1031).

Reports what the project's template asks for, using the server's own `timings`
object -- the same source the existing community reports use.

Design notes
  * Prompts are built from this repository's own source files so anyone can repeat
    them (the 2x Arc Pro B60 report does the same). Actual prompt token counts come
    from the response (`timings.prompt_n`), never from an estimate.
  * One request at a time. A warm-up request runs first and is not counted: the
    first request after a load is cold (empty expert cache) and is far slower.
  * Each configuration is repeated RUNS times; the summary keeps every run and
    reports the median and the range, per the template.
  * Recall checks put a needle at a chosen depth and verify the answer.

Usage
  python strata-bench.py --sizes 512,2048 --runs 3 --tag default
  python strata-bench.py --sizes 8192 --runs 3 --tag f16   # engine started with STRATA_HIP_PROMPT_F16=1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(r"F:\Strata")
URL = "http://127.0.0.1:8080/v1/chat/completions"
OUT = Path(r"F:\harness\strata-bench")
MAX_TOKENS = 256
NEEDLE = "XYLOPHONE-7742"

# Filler sources: the repository's own files, read in a fixed order so the prompt
# is reproducible. CHARS_PER_TOKEN is only used to size the build; the real count
# comes back from the server.
SOURCES = [
    REPO / "tools" / "iq_pack.py",
    REPO / "tools" / "make_profile.py",
    REPO / "tools" / "bench_prefill.py",
    REPO / "setup.py",
    REPO / "serve" / "server.py",
]
CHARS_PER_TOKEN = 3.6          # measured after the first run; only a sizing hint


def filler_text() -> str:
    parts = []
    for p in SOURCES:
        try:
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n\n# ---- next file ----\n\n".join(parts)


FILLER = filler_text()


def build_prompt(approx_tokens: int, needle_depth: float | None = None,
                 nonce: str | None = None) -> str:
    """A block of source plus an instruction.

    `nonce` is prefixed so every measured run has a different token prefix:
    Strata reuses a cached conversation prefix, so without it runs 2..N only read
    a handful of new tokens and their prompt speed is not comparable to run 1.
    The template's "Reused tokens" column exists for exactly this.
    """
    head = f"# benchmark run {nonce}\n" if nonce else ""
    if approx_tokens <= 64:
        return head + "Write a Python function that returns the n-th Fibonacci number."
    body = FILLER[: int(approx_tokens * CHARS_PER_TOKEN)]
    if needle_depth is not None:
        pos = int(len(body) * needle_depth)
        pos = body.rfind("\n", 0, pos) or pos
        body = body[:pos] + f"\n# RECORD-KEY: {NEEDLE}\n" + body[pos:]
        task = "What is the value of RECORD-KEY in the code above? Answer with the value only."
    else:
        task = ("Summarize what the code above does and name the three most important functions, "
                "in about 200 words.")
    return head + body + "\n\n# ---- task ----\n" + task


def call(prompt: str, max_tokens: int = MAX_TOKENS):
    body = {
        "model": "qwen3.8-flash-next-abliterated-q2_0",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,                                    # greedy, as in the Arc report
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer local"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}",
                "wall_s": time.time() - t0}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "wall_s": time.time() - t0}
    wall = time.time() - t0
    t = d.get("timings") or {}
    msg = d["choices"][0]["message"]
    return {
        "wall_s": round(wall, 2),
        "prompt_n": t.get("prompt_n"),
        "cache_n": t.get("cache_n"),
        "prompt_ms": t.get("prompt_ms"),
        "prompt_tok_s": t.get("prompt_per_second"),
        "predicted_n": t.get("predicted_n"),
        "predicted_ms": t.get("predicted_ms"),
        "decode_tok_s": t.get("predicted_per_second"),
        "draft_n": t.get("draft_n"),
        "draft_accepted": t.get("draft_n_accepted"),
        "finish": d["choices"][0].get("finish_reason"),
        "answer_head": (msg.get("content") or "").strip()[:160],
    }


def med_range(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.1f} (single run)"
    return f"{statistics.median(vals):.1f} ({min(vals):.1f}-{max(vals):.1f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512,2048")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--recall", action="store_true", help="也做 recall 校验（在 2048 提示里插针）")
    ap.add_argument("--skip-warmup", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    print(f"tag={args.tag}  sizes={sizes}  runs={args.runs}  max_tokens={MAX_TOKENS}")
    print(f"filler={len(FILLER)} chars from {len(SOURCES)} repo files\n")

    if not args.skip_warmup:
        print("warm-up (not counted) ...", flush=True)
        w = call(build_prompt(64), max_tokens=16)
        print(f"  warm-up: {w.get('prompt_n')} prompt tokens, "
              f"{w.get('prompt_tok_s')} prompt tok/s, {w.get('decode_tok_s')} decode tok/s, "
              f"wall {w.get('wall_s')}s\n")

    out = {"tag": args.tag, "max_tokens": MAX_TOKENS, "runs_per_config": args.runs,
           "warmup": not args.skip_warmup, "configs": []}

    for size in sizes:
        rows = []
        for i in range(1, args.runs + 1):
            # 每次一个不一样的 nonce：否则服务端复用前缀缓存，第 2..N 次只读几个新 token
            prompt = build_prompt(size, nonce=f"{args.tag}-{size}-{i}-{random.randrange(10**9)}")
            r = call(prompt)
            r["run"] = i
            rows.append(r)
            if "error" in r:
                print(f"  [{size}] run {i}: ERROR {r['error'][:110]}", flush=True)
            else:
                print(f"  [{size}] run {i}: prompt {r['prompt_n']} tok @ {r['prompt_tok_s']:.1f} tok/s, "
                      f"gen {r['predicted_n']} tok @ {r['decode_tok_s']:.1f} tok/s, "
                      f"TTFT {r['prompt_ms']/1000:.1f}s, wall {r['wall_s']}s", flush=True)
        cfg = {
            "requested_tokens": size,
            "prompt_n": rows[0].get("prompt_n") if rows else None,
            "cache_n": [r.get("cache_n") for r in rows],
            "generated_n": [r.get("predicted_n") for r in rows],
            "runs": len(rows),
            "prompt_tok_s_median_range": med_range([r.get("prompt_tok_s") for r in rows]),
            "decode_tok_s_median_range": med_range([r.get("decode_tok_s") for r in rows]),
            "ttft_s_median_range": med_range([(r.get("prompt_ms") or 0) / 1000 for r in rows]),
            "wall_s": [r.get("wall_s") for r in rows],
            "rows": rows,
        }
        out["configs"].append(cfg)
        print(f"  ⇒ {size}: prompt {cfg['prompt_n']} tok | prompt tok/s {cfg['prompt_tok_s_median_range']} "
              f"| decode tok/s {cfg['decode_tok_s_median_range']} | TTFT {cfg['ttft_s_median_range']}\n",
              flush=True)

    if args.recall:
        prompt = build_prompt(2048, needle_depth=0.5, nonce=f"recall-{random.randrange(10**9)}")
        r = call(prompt, max_tokens=32)
        ok = NEEDLE.lower() in (r.get("answer_head") or "").lower()
        r["needle"] = NEEDLE
        r["needle_found"] = ok
        out["recall"] = r
        print(f"recall @50% of ~2048: prompt {r.get('prompt_n')} tok -> "
              f"{'FOUND ✅' if ok else 'MISSED ❌'}  answer={r.get('answer_head')!r}")

    path = OUT / f"strata-bench-{args.tag}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
