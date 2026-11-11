#!/usr/bin/env python3
"""Aggregate output: N concurrent clients against one server, measuring tokens per second summed over the clients. Run the
same N against a second configuration to compare.

    python tools/dgx_concurrent.py CONFIG OUT.json --clients 8 [--clients 16] [--prompt 2000] [--max-new 128]
                                   [--parallel 4] [--distinct]

--parallel N runs the server with "parallel": N (the engine's batch slots: up to N requests decode together; the run config of setup has
none, one request at a time). --distinct gives every client its own prompt with its own hidden code and checks the answer: every request
must finish without an error, and its answer must name ITS code and no other client's (a batch that mixed two requests' state would fail
that), so the aggregate figure comes with a correctness check. Without --distinct all clients send the same prompt, as the
2026-10-26 handover's numbers were measured.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORDS = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember ledger quartz "
         "meadow anchor violet thunder cedar marble piston ribbon cobalt falcon glacier hollow ivory juniper kernel").split()


def make_prompt(tokens: int) -> str:
    body = []
    n = int(tokens / 1.35)
    for i in range(n):
        body.append(WORDS[i % len(WORDS)])
    return "Summarise the following log in two sentences.\n\n" + " ".join(body)


def distinct_prompt(tokens: int, idx: int) -> tuple[str, str]:
    """(prompt, code): about `tokens` tokens of a log that is different for every client, with one code in the middle."""
    code = f"{(idx * 7919 + 1357) % 9000 + 1000}-{WORDS[(idx * 5) % len(WORDS)]}"
    n = int(tokens / 1.35)
    body = [WORDS[(i * (idx % 7 + 1) + idx) % len(WORDS)] for i in range(n)]
    body.insert(n // 2, f"(the access code of this log is {code})")
    return ("Read the following log. First give the access code of this log, then describe in about 80 words what the log looks like.\n\n" + " ".join(body), code)


def serve(cfg_path: str, port: int, parallel: int | None = None):
    cfg = json.loads(Path(cfg_path).read_text())
    if parallel:
        cfg["parallel"] = parallel
    cfg["port"] = port
    cfg["log"] = f"/tmp/dgx-conc-{port}.log"
    tmp = Path(f"/tmp/dgx-conc-{port}.json")
    tmp.write_text(json.dumps(cfg, indent=1))
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
                             str(tmp), "--port", str(port)], cwd=str(ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < 1800:
        if proc.poll() is not None:
            raise SystemExit(f"server exited {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return proc
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit("server did not come up")


def one(port: int, prompt: str, max_new: int, out: list, idx: int):
    t0 = time.time()
    try:
        out[idx] = _one(port, prompt, max_new)
    except Exception as e:                                   # a request that errors or times out is a finding, not a crash of the run
        out[idx] = {"error": f"{type(e).__name__}: {e}", "latency": time.time() - t0, "n": 0, "text": "", "finish": None}


def _one(port: int, prompt: str, max_new: int) -> dict:
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_new, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        res = json.loads(r.read())
    choice = (res.get("choices") or [{}])[0]
    return {"latency": time.time() - t0, "n": res.get("timings", {}).get("predicted_n", 0),
            "text": (choice.get("message") or {}).get("content") or "", "finish": choice.get("finish_reason"), "error": None}


def coherent(text: str) -> bool:
    """A readable answer: not empty, not one repeated token or a wall of the same few words."""
    words = text.split()
    return len(words) >= 3 and len(set(words)) / len(words) > 0.35


def run(cfg_path: str, port: int, clients: int, prompt: str, max_new: int, parallel: int | None = None,
        distinct: int | None = None) -> dict:
    proc = serve(cfg_path, port, parallel)
    try:
        with urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}).encode(),
                headers={"Content-Type": "application/json"}), timeout=1800) as r:
            r.read()
        out: list = [None] * clients
        prompts = [distinct_prompt(distinct, i) for i in range(clients)] if distinct else [(prompt, None)] * clients
        t0 = time.time()
        threads = [threading.Thread(target=one, args=(port, prompts[i][0], max_new, out, i)) for i in range(clients)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - t0
        tokens = sum(r["n"] for r in out if r["n"])
        times = [r["latency"] for r in out if not r["error"]]
        row = {"clients": clients, "parallel": parallel, "wall_s": round(wall, 2), "tokens": tokens,
               "aggregate_tok_s": round(tokens / wall, 2) if wall > 0 else None,
               "median_latency_s": round(statistics.median(times), 2) if times else None,
               "errors": [r["error"] for r in out if r["error"]], "finish": sorted({str(r["finish"]) for r in out}),
               "all_coherent": all(coherent(r["text"]) for r in out if not r["error"])}
        if distinct:
            right = [(prompts[i][1] in r["text"]) and not any(prompts[j][1] in r["text"] for j in range(clients) if j != i)
                     for i, r in enumerate(out) if not r["error"]]
            row["own_code_named"] = f"{sum(right)} of {clients}"
            row["answers"] = [r["text"][:100] for r in out]
        return row
    finally:
        proc.terminate()
        try:
            proc.wait(180)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--clients", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--prompt", type=int, default=2000)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--parallel", type=int, default=None, help='run the server with "parallel": N (batch slots)')
    ap.add_argument("--distinct", action="store_true", help="a prompt of its own with a hidden code for every client, and check the answers")
    a = ap.parse_args()
    prompt = make_prompt(a.prompt)
    rows = [run(a.config, a.port, c, prompt, a.max_new, a.parallel, a.prompt if a.distinct else None) for c in a.clients]
    Path(a.out).write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
