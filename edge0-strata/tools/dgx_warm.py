#!/usr/bin/env python3
"""One server, several repeats per cell: the warm-state runner (the per-start page cache and the tens-of-GB expert
prefetch are the main variance sources on a unified-memory box, so repeats inside one server are the comparable ones).

    python tools/dgx_warm.py CONFIG TAG OUTDIR [--cells 4096,16384,32768] [--repeats 3] [--port 8091] [--new 256]

Writes <OUTDIR>/<TAG>.jsonl rows in tools/dgx_bench.py's schema (round, arm, len, temp, max_new, prompt_n,
prompt_per_second, predicted_n, predicted_per_second, ttft_s, cache_n) so tools/dgx_compare.py reads them.
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
         "meadow anchor violet thunder cedar marble piston ribbon cobalt falcon glacier hollow ivory juniper kernel").split()


def prompt_for(tokens, nonce):
    rng = random.Random(tokens)
    body = " ".join(rng.choice(WORDS) for _ in range(max(8, int(tokens / 1.35))))
    return (f"Request nonce {nonce}. Read the word list below carefully.\n\n{body}\n\n"
            "Write a long, detailed analysis of the word list above. Keep writing for at least 400 words.")


def post(port, prompt, max_new):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_new, "temperature": 0,
                       "stream": False, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    return out, time.time() - t0


def status(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/status", timeout=10) as r:
            return json.loads(r.read()).get("last_timings") or {}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("tag")
    ap.add_argument("outdir")
    ap.add_argument("--cells", default="512,2048,4096")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--new", type=int, default=256)
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
            if proc.poll() is not None:
                raise SystemExit(f"server exited with {proc.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(3)
        print(f"{a.tag}: ready in {time.time() - t0:.0f} s", flush=True)
        post(a.port, prompt_for(256, "warm"), 8)
        for rep in range(a.repeats):
            for length in [int(x) for x in a.cells.split(",")]:
                nonce = f"{random.getrandbits(48):012x}"
                outj, wall = post(a.port, prompt_for(length, nonce), a.new)
                t = status(a.port)
                row = {"round": rep, "arm": a.tag, "len": length, "temp": 0.0, "max_new": a.new,
                       "wall_s": round(wall, 2), "ttft_s": (t.get("prompt_ms") or 0) / 1000.0 or None,
                       "cache_n": t.get("cache_n", 0), **t}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"  {length:5d} prompt_n {t.get('prompt_n')} prompt {t.get('prompt_per_second')} tok/s "
                      f"decode {t.get('predicted_per_second')} tok/s", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(90)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


if __name__ == "__main__":
    main()
