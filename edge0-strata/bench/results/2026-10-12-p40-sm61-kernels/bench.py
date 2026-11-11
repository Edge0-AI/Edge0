#!/usr/bin/env python3
"""bench.py LOGFILE LABEL [--runs 3] - fixed prompts straight to the engine (BENCH_PORT, default 8081): thinking off,
temperature 0, a random first line so nothing is reused from the prompt / conversation cache; then the speeds the
engine itself logged.  short = ~565-token prompt + 200 tokens out; long = ~7,047-token prompt + 60 tokens out.
Prints every request and the medians: prompt tok/s and decode tok/s per case."""
import json, os, random, re, statistics, sys, time, urllib.request
LOG, LABEL = sys.argv[1], sys.argv[2]; RUNS = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 3
WORDS = ("the engine reads a prompt and writes an answer while the cache keeps the experts that were used most often "
         "in the last rounds so a second request with a different first line has to be read again from the start").split()
def text(n, seed):
    r = random.Random(seed); return " ".join(r.choice(WORDS) for _ in range(n))
def ask(prompt, max_tokens):
    body = {"model": "qwen3.8-flash-next-iq3_s", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
            "reasoning_effort": "none", "stream": False}
    req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % os.environ.get("BENCH_PORT", "8081"), json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time(); out = json.load(urllib.request.urlopen(req, timeout=1800)); return time.time() - t, out
PAT = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d.]+) tok/s\)")
res = {}
for case, n, mt in (("short", 520, 200), ("long", 7000, 60)):
    for i in range(RUNS):
        nonce = "%016x" % random.getrandbits(64); before = sum(1 for _ in open(LOG, errors="replace"))
        wall, out = ask(f"Run {nonce}. Continue this text with a long plain description of a harbour town, no lists.\n\n" + text(n, case), mt)
        time.sleep(1.5); lines = open(LOG, errors="replace").read().split("\n")[before:]; m = [PAT.search(l) for l in lines]; m = [x for x in m if x]
        if not m: print(case, i, "no log line", wall); continue
        g = m[-1].groups(); res.setdefault(case, []).append((float(g[4]), float(g[7]), int(g[0]), int(g[2]), int(g[5]), wall))
        print(f"{LABEL} {case} run {i}: prompt {g[0]} tok ({g[2]} read) at {g[4]} tok/s | {g[5]} generated at {g[7]} tok/s | wall {wall:.1f}s", flush=True)
for case, v in res.items():
    print(f"== {LABEL} {case}: prompt median {statistics.median(x[0] for x in v):.0f} tok/s | decode median {statistics.median(x[1] for x in v):.1f} tok/s | n={len(v)}")
