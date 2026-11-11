#!/usr/bin/env python3
"""Live check (0.1.39 rerun of 2026-10-03/strata-512k-live) of the running Strata 512K server (127.0.0.1:8080): long-prompt speed and recall at
32K/120K/240K/480K, then short follow-ups for stability (upstream #606), with RAM sampled. Reuses
scripts/strata/ctx_sweep.py's haystack and chat helpers against the live port."""
import json, sys, threading, time
sys.path.insert(0, "~/bench-scripts")
import ctx_sweep as cs
cs.PORT = 8080
out = "./results.jsonl"
mem = {"min_avail": 99.0, "max_swap": 0.0}
def sample():
    while True:
        m = dict(l.split(":")[0:2] for l in open("/proc/meminfo"))
        a = int(m["MemAvailable"].split()[0]) / 1048576
        sw = (int(m["SwapTotal"].split()[0]) - int(m["SwapFree"].split()[0])) / 1048576
        mem["min_avail"] = min(mem["min_avail"], a); mem["max_swap"] = max(mem["max_swap"], sw)
        time.sleep(1)
threading.Thread(target=sample, daemon=True).start()
def log(rec):
    rec.update({k: round(v, 2) for k, v in mem.items()})
    print(json.dumps(rec), flush=True)
    open(out, "a").write(json.dumps(rec) + "\n")
for n in (32000, 120000, 240000, 480000):
    text, secret = cs.haystack(n, seed=11)
    q = "\n\nWhat is the vault rotation code? Answer with the number only."
    t0 = time.time(); r = cs.chat(text + q, 48)
    r2 = cs.chat(text + q + "\n\nThen write a 200-word summary of what these logs show.", 300)
    log({"target": n, "prompt_tokens": r["prompt_tokens"], "read_secs": r["secs"], "prefill_tps": r["prefill_tps"],
         "recall": secret in r["text"], "answer": r["text"][:40], "decode_at_depth": r2["decode_tps"],
         "summary_head": r2["text"][:120]})
for i, p in enumerate(["What is 17 * 23? Answer with the number only.",
                       "Name the capital of France in one word.",
                       "Write a Python one-liner that reverses a string s."]):
    r = cs.chat(p, 80)
    log({"followup": i, "text": r["text"][:100], "decode_tps": r["decode_tps"]})
log({"done": True})
