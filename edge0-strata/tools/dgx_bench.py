#!/usr/bin/env python3
"""dgx_bench - the DGX Spark benchmark family for Strata (Qwen3.8-Flash-Next first).

    python tools/dgx_bench.py --config strata-iq3_s.json --tag base
    python tools/dgx_bench.py --config strata-iq3_s.json --tag gu-swiglu --env STRATA_GU_SWIGLU=1
    python tools/dgx_bench.py --config strata-iq3_s.json --tag base --cells 4096:0:256,4096:0.7:256 --rounds 3

It starts the server on the model's own config (serve/server.py, the path setup writes), waits for it, sends the cells
through the real HTTP API, restarts the server between arms, and writes every request plus a summary.

A cell is PROMPT_TOKENS:TEMPERATURE:MAX_NEW.  The default family is the one the DGX Spark guide's contract names:
512 / 4096 / 16384 / 32768 fresh prompt tokens at temperature 0 and 0.7, with a 256-token generation cap, plus one
1024-token decode cell.  Every request carries a unique nonce as its first token sequence, so the conversation cache
cannot serve the prompt (verified: timings.cache_n must be 0, or the row is marked reused and excluded from the summary).

Per request it records the client's own time to first content token (TTFT), the wall time, and the server's timings
(prompt_n, prompt_ms, prompt_per_second, predicted_n, predicted_ms, predicted_per_second, draft counts) read from
GET /v1/status right after the streamed answer.  Medians over rounds; nothing is retried or filtered except requests
that failed (they are kept and reported as failures).

Standard library only, like tools/strata_bench.py and tools/ab_engine.py.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CELLS = "512:0:256,4096:0:256,4096:0.7:256,16384:0:256,16384:0.7:256,32768:0:256,32768:0.7:256,4096:0:1024"
WORDS = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember ledger quartz "
         "meadow anchor violet thunder cedar marble piston ribbon cobalt falcon glacier hollow ivory juniper kernel "
         "lattice mirror nectar oasis pepper quarry raven saddle tundra umber vessel willow xenon yellow zephyr "
         "basalt cinder dune elm fjord granite heather inlet jetty kiln lagoon mesa niche opal prairie quill").split()


def build_prompt(target_tokens: int, nonce: str) -> str:
    """A prompt of about target_tokens tokens: a unique nonce line, a word list, and a task that keeps the answer going."""
    rng = random.Random(hash((target_tokens, nonce)) & 0xFFFFFFFF)
    # ~1.35 tokens per word for this vocabulary in Qwen's tokenizer
    words = max(8, int(target_tokens / 1.35))
    body = " ".join(rng.choice(WORDS) for _ in range(words))
    return (f"Request nonce {nonce}. Read the word list below carefully.\n\n{body}\n\n"
            f"Write a long, detailed analysis of the word list above: its sounds, its themes and its structure. "
            f"Keep writing for at least 400 words; do not stop early and do not ask questions.")


def post_stream(port: int, prompt: str, max_tokens: int, temperature: float, timeout: float = 3600) -> dict:
    """One streamed chat completion; returns the client's times and the token count."""
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": temperature, "stream": True,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    n = 0
    text_len = 0
    finish = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                piece = d.get("content") or d.get("reasoning_content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    text_len += len(piece)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
            if obj.get("usage"):
                n = int(obj["usage"].get("completion_tokens") or n)
    wall = time.time() - t0
    return {"ttft_s": ttft, "wall_s": round(wall, 3), "stream_tokens": n, "chars": text_len, "finish": finish}


def last_timings(port: int) -> dict:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/status", timeout=10) as r:
            st = json.loads(r.read())
        return st.get("last_timings") or {}
    except Exception:
        return {}


def wait_ready(port: int, proc: subprocess.Popen, timeout: float = 3600) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError("server not ready")


def calibrate(port: int, target: int, max_tokens: int, temperature: float, log) -> str:
    """Build a prompt whose prompt_n is within ~2% of target (two probes at most)."""
    nonce = f"{random.getrandbits(48):012x}"
    prompt = build_prompt(target, nonce)
    for attempt in range(3):
        r = post_stream(port, prompt, 8, 0.0)
        t = last_timings(port)
        got = int(t.get("prompt_n") or 0)
        log(f"    calibrate {target}: prompt_n {got}")
        if got and abs(got - target) <= max(16, target // 50):
            return prompt
        if not got:
            break
        words = len(prompt.split()) * target / got
        rng = random.Random(target + attempt)
        body = " ".join(rng.choice(WORDS) for _ in range(max(8, int(words))))
        prompt = (f"Request nonce {nonce}. Read the word list below carefully.\n\n{body}\n\n"
                  f"Write a long, detailed analysis of the word list above: its sounds, its themes and its structure. "
                  f"Keep writing for at least 400 words; do not stop early and do not ask questions.")
    return prompt


def run_arm(cfg_path: Path, tag: str, cells, rounds: int, port: int, env: dict, out: Path, warm: int,
            exe: str | None = None, xargs: list[str] | None = None) -> list[dict]:
    base = json.loads(cfg_path.read_text())
    cfg = dict(base)
    if exe:
        cfg["exe"] = exe
    if xargs:
        cfg["args"] = list(cfg.get("args") or []) + list(xargs)
    cfg["port"] = port
    cfg["log"] = str(out / f"{tag}.engine.log")
    cpath = out / f"{tag}.config.json"
    cpath.write_text(json.dumps(cfg, indent=1))
    rows: list[dict] = []
    meta = {"tag": tag, "config": str(cfg_path), "cells": cells, "rounds": rounds, "env": env,
            "engine": cfg.get("exe"), "args": cfg.get("args"), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / f"{tag}.meta.json").write_text(json.dumps(meta, indent=1))
    logf = open(out / f"{tag}.jsonl", "a", encoding="utf-8")
    penv = dict(os.environ, **env)
    for rnd in range(rounds):
        srvlog = open(out / f"{tag}.server.log", "a", encoding="utf-8")
        proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
                                 str(cpath), "--port", str(port)], cwd=str(ROOT), env=penv,
                                stdout=srvlog, stderr=srvlog)
        try:
            ready = wait_ready(port, proc)
            print(f"round {rnd} {tag}: ready in {ready:.0f} s", flush=True)
            nonce = f"{random.getrandbits(48):012x}"
            post_stream(port, f"Request nonce {nonce}. Say hello in one short sentence, then list five colors.", warm, 0.0)
            # a 4K prompt first: with --mmap-experts the first request pages the experts in (a one-time cold cost of
            # tens of seconds); every measured cell then runs on the same warm state the fork's baselines had
            nonce = f"{random.getrandbits(48):012x}"
            post_stream(port, build_prompt(4096, nonce), 8, 0.0)
            prompts = {}
            for (length, temp, maxnew) in cells:
                key = (length, maxnew)
                if key not in prompts:
                    prompts[key] = calibrate(port, length, maxnew, 0.0, print)
                nonce = f"{random.getrandbits(48):012x}"
                prompt = prompts[key]
                prompt = prompt.replace("Request nonce ", "Request nonce ", 1)
                # a fresh nonce: rewrite the first line so no prefix can be reused
                prompt = prompt.split("\n", 1)
                prompt = f"Request nonce {nonce}. Read the word list below carefully.\n" + prompt[1]
                r = post_stream(port, prompt, maxnew, temp)
                t = last_timings(port)
                row = {"round": rnd, "arm": tag, "len": length, "temp": temp, "max_new": maxnew, **r, **t}
                rows.append(row)
                logf.write(json.dumps(row) + "\n")
                logf.flush()
                print(f"  {length:6d} T={temp} n={maxnew:4d}: prompt_n {t.get('prompt_n')} "
                      f"prompt {t.get('prompt_per_second')} tok/s decode {t.get('predicted_per_second')} tok/s "
                      f"n {t.get('predicted_n')} ttft {r['ttft_s'] and round(r['ttft_s'], 2)} wall {r['wall_s']}",
                      flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(60)
            except subprocess.TimeoutExpired:
                proc.kill()
            time.sleep(5)
    logf.close()
    return rows


def summarize(rows: list[dict], cells) -> str:
    lines = ["| cell | n | prompt tok/s | decode tok/s | ttft s | wall s |", "|---|---|---|---|---|---|"]
    for (length, temp, maxnew) in cells:
        sel = [r for r in rows if r["len"] == length and r["temp"] == temp and r["max_new"] == maxnew
               and not r.get("cache_n")]
        if not sel:
            lines.append(f"| {length} T={temp} n={maxnew} | 0 | - | - | - | - |")
            continue
        med = lambda k: statistics.median([r[k] for r in sel if r.get(k)]) if any(r.get(k) for r in sel) else None
        p, d, ttft, wall = med("prompt_per_second"), med("predicted_per_second"), med("ttft_s"), med("wall_s")
        fmt = lambda v, f="{:.1f}": "-" if v is None else f.format(v)
        lines.append(f"| {length} T={temp} n={maxnew} | {len(sel)} | {fmt(p)} | {fmt(d)} | {fmt(ttft, '{:.2f}')} | "
                     f"{fmt(wall, '{:.1f}')} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--cells", default=DEFAULT_CELLS)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--env", action="append", default=[], help="NAME=VALUE for the server process (repeatable)")
    ap.add_argument("--exe", default="", help="engine binary to run instead of the config's")
    ap.add_argument("--xargs", default="", help="extra engine arguments, ';'-separated")
    ap.add_argument("--warm", type=int, default=64)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    cells = []
    for s in a.cells.split(","):
        l, t, n = s.split(":")
        cells.append((int(l), float(t), int(n)))
    out = Path(a.out) if a.out else ROOT / "bench" / "results" / time.strftime("dgx-%Y-%m-%d-%H%M") 
    out.mkdir(parents=True, exist_ok=True)
    env = {}
    for e in a.env:
        k, v = e.split("=", 1)
        env[k] = v
    print(f"out {out}; cells {cells}; rounds {a.rounds}", flush=True)
    rows = run_arm(Path(a.config), a.tag, cells, a.rounds, a.port, env, out, a.warm,
                   exe=(a.exe or None), xargs=(a.xargs.split(";") if a.xargs else None))
    summary = summarize(rows, cells)
    (out / f"{a.tag}.summary.md").write_text(f"# {a.tag}\n\n{summary}\n")
    print("\n" + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
