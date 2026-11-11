#!/usr/bin/env python3
"""Long-context recall check: hide one fact in a long prompt and ask for it back (the guide's "long-context retrieval" cell).

    python tools/dgx_needle.py --config strata-iq3s-final.json --tokens 12000

Starts the model's server, sends one request, and reports whether the answer contains the fact.  The fact is placed at a
random position in the prompt and never appears in the question, so a correct answer means the context was read.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORDS = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember ledger quartz "
         "meadow anchor violet thunder cedar marble piston ribbon cobalt falcon glacier hollow ivory juniper kernel "
         "lattice mirror nectar oasis pepper quarry raven saddle tundra umber vessel willow xenon yellow zephyr").split()


def build(tokens: int, needle: str, rng: random.Random) -> str:
    words = max(8, int(tokens / 1.35))
    body = [rng.choice(WORDS) for _ in range(words)]
    at = rng.randrange(len(body) // 4, 3 * len(body) // 4)
    body[at:at] = [needle]
    return ("Read the following notes carefully, then answer the question at the end.\n\n" + " ".join(body) +
            "\n\nQuestion: what is the access code for the harbor gate? Answer with the code only.")


def post(port: int, prompt: str, max_tokens: int = 32) -> dict:
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code}: {e.read()[:400]!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokens", type=int, default=12000)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    cfg["port"] = a.port
    cfg["log"] = "/tmp/dgx-needle.log"
    cpath = Path("/tmp/dgx-needle.json")
    cpath.write_text(json.dumps(cfg, indent=1))
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
                             str(cpath), "--port", str(a.port)], cwd=str(ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok = 0
    try:
        for _ in range(600):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(2)
        post(a.port, "Say hi.", 8)
        for i in range(a.runs):
            rng = random.Random(1000 + i)
            code = f"{rng.randrange(100000, 999999)}"
            needle = f"The access code for the harbor gate is {code}."
            r = post(a.port, build(a.tokens, needle, rng))
            answer = r["choices"][0]["message"].get("content", "")
            t = r.get("timings") or {}
            hit = code in answer
            ok += hit
            print(f"run {i}: prompt_n {t.get('prompt_n')} at {t.get('prompt_per_second')} tok/s, "
                  f"answer {answer[:40]!r} -> {'HIT' if hit else 'MISS'} (code {code})")
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
    print(f"{ok} of {a.runs} recalled")
    return 0 if ok == a.runs else 1


if __name__ == "__main__":
    sys.exit(main())
