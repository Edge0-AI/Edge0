#!/usr/bin/env python3
"""Parallel-Last: python3 parlast.py OUT.json LEVELS(1,2,4) ROUNDS PROMPT_TOK OUT_TOK. Streamt, misst TTFT, Dauer, Token je Strom und gesamt."""
import json, sys, time, threading, pathlib, urllib.request, statistics
OUT = pathlib.Path(sys.argv[1]); LEVELS = [int(x) for x in sys.argv[2].split(",")]
ROUNDS = int(sys.argv[3]); PTOK = int(sys.argv[4]); OTOK = int(sys.argv[5])
ROOT = pathlib.Path.home() / "ai/Strata-0.1.36"
files = []
for d, g in [("docs", "*.md"), ("src", "*.cpp"), ("src", "*.cu"), ("include", "*.hpp"), ("serve", "*.py"), ("tools", "*.py"), ("third_party/llama.cpp/docs", "*.md")]:
    files += sorted((ROOT / d).rglob(g))
corp = "".join(f"\n### {f.name}\n" + f.read_text(errors="ignore") for f in files)
SEQ = [int(time.time()) % 1000000]
def prompt():
    SEQ[0] += 1
    n = int(PTOK * 3.2); off = (SEQ[0] * 37000) % max(1, len(corp) - n)
    return f"[nonce {SEQ[0]}]\nHere is source code:\n" + corp[off:off + n] + "\n\nWrite a short summary of what this code does, then propose two refactorings with example code."
def one(res, i):
    body = {"model": "swift-1.5-iq3_xxs", "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt()}], "max_tokens": OTOK,
            "temperature": 0.7, "top_p": 0.95, "top_k": 20}
    r = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = None; chunks = 0; usage = None; last = t0
    try:
        with urllib.request.urlopen(r, timeout=1800) as f:
            for line in f:
                line = line.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"): continue
                j = json.loads(line[5:])
                if j.get("usage"): usage = j["usage"].get("completion_tokens")
                ch = j.get("choices") or []
                if ch and (ch[0].get("delta", {}).get("content") or ch[0].get("delta", {}).get("reasoning_content") or ch[0].get("delta", {}).get("reasoning")):
                    chunks += 1; last = time.time()
                    if first is None: first = last
    except Exception as e:
        res[i] = {"error": repr(e)}; return
    toks = usage or chunks
    res[i] = {"ttft": round((first or last) - t0, 2), "total_s": round(last - t0, 2), "tokens": toks, "chunks": chunks,
              "decode_tps": round(toks / max(0.01, last - (first or t0)), 1) if first else None}
out = []
one({}, 0) if False else None
for c in LEVELS:
    for rd in range(ROUNDS):
        res = [None] * c; th = [threading.Thread(target=one, args=(res, i)) for i in range(c)]
        t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]; wall = time.time() - t0
        tot = sum((r or {}).get("tokens", 0) or 0 for r in res)
        row = {"streams": c, "round": rd + 1, "wall_s": round(wall, 1), "total_tokens": tot, "total_tps": round(tot / wall, 1), "per_stream": res}
        out.append(row); print(json.dumps(row), flush=True)
json.dump(out, open(OUT, "w"), indent=1)
