"""bench-local/par.py - several requests at once against the running server: what each one gets and what they get
together.  The server's "parallel" setting is whatever the running config has (ab.py's v139-par* arms set it).

  python par.py 1 2 4          (requests at once, one round each; a line per round, results appended to par.jsonl)
"""
from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from pathlib import Path

import requests

HERE = Path(__file__).parent
BASE = "http://127.0.0.1:8080"
TOPICS = ["how a hash map works: hashing, collisions, resizing, complexity",
          "how TCP establishes and closes a connection, and what each flag means",
          "processes and threads: memory, scheduling, communication, failure isolation",
          "how a B-tree index speeds up a database query, and what it costs on writes",
          "how garbage collection works: reference counting, mark and sweep, generations",
          "how TLS sets up an encrypted connection: certificates, key exchange, session keys",
          "how virtual memory works: pages, page tables, the TLB, swapping",
          "how a compiler turns source into machine code: parsing, IR, optimisation, codegen"]


def one(i, out, t0):
    """One streamed request: the time to its first token and its own decode rate (the server's timings)."""
    body = {"model": "strata", "max_tokens": 300, "temperature": 0, "reasoning_effort": "none", "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": f"[{uuid.uuid4().hex[:8]}] Explain in detail {TOPICS[i % 8]}. "
                                                     "About 220 words."}]}
    rec = {"i": i}
    try:
        with requests.post(BASE + "/v1/chat/completions", json=body, stream=True, timeout=1800) as r:
            for raw in r.iter_lines():
                if not raw.startswith(b"data: ") or raw == b"data: [DONE]":
                    continue
                ev = json.loads(raw[6:])
                delta = (ev.get("choices") or [{}])[0].get("delta") or {}
                if "first_s" not in rec and (delta.get("content") or delta.get("reasoning_content")):
                    rec["first_s"] = round(time.time() - t0, 2)
                if ev.get("timings"):
                    rec["tok_s"] = ev["timings"].get("predicted_per_second")
                    rec["out_n"] = ev["timings"].get("predicted_n")
                if ev.get("usage") and "out_n" not in rec:
                    rec["out_n"] = ev["usage"].get("completion_tokens")
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
    rec["done_s"] = round(time.time() - t0, 2)
    out[i] = rec


def main():
    label = json.loads(requests.get(BASE + "/v1/status", timeout=10).text).get("concurrency", {})
    for k in [int(x) for x in sys.argv[1:]]:
        out, t0 = [None] * k, time.time()
        threads = [threading.Thread(target=one, args=(i, out, t0)) for i in range(k)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - t0
        toks = sum(r.get("out_n") or 0 for r in out)
        rec = {"time": time.strftime("%H:%M:%S"), "requests": k, "server": label, "wall_s": round(wall, 1),
               "total_tok_s": round(toks / wall, 1), "runs": out}
        firsts = sorted(r.get("first_s") or -1 for r in out)
        print(f"PAR {k} at once (server {label.get('serving', '?')}): together {rec['total_tok_s']} tok/s, all done in "
              f"{rec['wall_s']} s | first token after {firsts} s | each "
              f"{[r.get('tok_s') for r in out]} tok/s | errors {sum(1 for r in out if r.get('error'))}", flush=True)
        with (HERE / "par.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
