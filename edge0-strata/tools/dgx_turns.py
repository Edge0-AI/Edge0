#!/usr/bin/env python3
"""Multi-turn (agent-style) comparison: one conversation, five turns of ~1,200 new tokens each, cache reuse measured.

    python tools/dgx_turns.py CONFIG TAG OUTDIR [--turns 5] [--tokens 1200] [--port 8091]

Per turn it records the engine's prompt_n / cache_n, prompt tok/s and TTFT, so the cache path is compared, not just the
fresh-prompt path the other tools measure.
"""
import argparse
import json
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORDS = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember ledger quartz "
         "meadow anchor violet thunder cedar marble piston ribbon cobalt falcon glacier hollow ivory juniper kernel "
         "lattice mirror nectar oasis pepper quarry raven saddle tundra umber vessel willow xenon yellow zephyr").split()


def blob(n, seed):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n))


def chat(port, messages, max_tokens=64):
    body = json.dumps({"messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": False,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("tag")
    ap.add_argument("outdir")
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=1200)
    ap.add_argument("--port", type=int, default=8091)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text())
    cfg["port"] = a.port
    cfg["log"] = str(out / f"{a.tag}.engine.log")
    cpath = out / f"{a.tag}.config.json"
    cpath.write_text(json.dumps(cfg, indent=1))
    log = open(out / f"{a.tag}.jsonl", "a", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
                             str(cpath), "--port", str(a.port)], cwd=str(ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.time()
        while time.time() - t0 < 3600:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(3)
        print(f"{a.tag}: ready in {time.time() - t0:.0f} s", flush=True)
        msgs = [{"role": "user", "content": "Read this note and answer with one short sentence.\n" + blob(a.tokens, 1)}]
        outj, wall = chat(a.port, msgs)
        msgs.append({"role": "assistant", "content": outj["choices"][0]["message"].get("content", "")})
        for turn in range(a.turns):
            msgs.append({"role": "user", "content": "More notes.\n" + blob(a.tokens, 100 + turn) +
                                                   "\nSummarise the last note in one sentence."})
            outj, wall = chat(a.port, msgs)
            t = outj.get("timings") or {}
            msgs.append({"role": "assistant", "content": outj["choices"][0]["message"].get("content", "")})
            row = {"round": turn, "arm": a.tag, "len": 0, "temp": 0.0, "max_new": 64, "wall_s": round(wall, 2),
                   "ttft_s": (t.get("prompt_ms") or 0) / 1000.0 or None, **t}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"  turn {turn}: prompt_n {t.get('prompt_n')} cache_n {t.get('cache_n')} "
                  f"prompt {t.get('prompt_per_second')} tok/s decode {t.get('predicted_per_second')} tok/s "
                  f"ttft {row['ttft_s']}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(90)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


if __name__ == "__main__":
    main()
