#!/usr/bin/env python3
"""equal_check.py OUT.json — fixed prompts at temperature 0 straight to the dev engine (BENCH_PORT), the answers saved for a
byte comparison between two engines. Run it first after a fresh start so both engines have the same history."""
import json, os, random, sys, urllib.request
OUT = sys.argv[1]; PORT = os.environ.get("BENCH_PORT", "8082")
WORDS = ("the engine reads a prompt and writes an answer while the cache keeps the experts that were used most often "
         "in the last rounds so a second request with a different first line has to be read again from the start").split()
def text(n, seed):
    r = random.Random(seed); return " ".join(r.choice(WORDS) for _ in range(n))
PROMPTS = [
    ("code", "Write a Python function that merges two sorted lists into one sorted list without using sort(). Then explain its complexity.", 160),
    ("prose", "Describe a small harbour town at dawn in plain prose, no lists.", 160),
    ("math", "A train leaves at 9:40 and travels 150 km at 75 km/h, waits 25 minutes, then travels 90 km at 60 km/h. When does it arrive? Show the steps.", 160),
    ("json", "Return a JSON object with the keys name, year and tags describing the first Moon landing. JSON only.", 100),
    ("italian", "Spiega in italiano, in poche frasi, la differenza tra IVA a debito e IVA a credito.", 140),
    ("long", "Summarise the following text in three sentences.\n\n" + text(2600, "eq-long"), 120),
    ("needle", text(1500, "eq-a") + "\n\nThe access code for the archive room is 7391-KESTREL.\n\n" + text(1500, "eq-b") + "\n\nWhat is the access code for the archive room? Answer with the code only.", 40),
]
res = {}
for name, prompt, mt in PROMPTS:
    body = {"model": "qwen3.8-flash-next-iq3_s", "messages": [{"role": "user", "content": prompt}], "max_tokens": mt, "temperature": 0,
            "reasoning_effort": "none", "stream": False}
    req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % PORT, json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=1800))
    res[name] = out["choices"][0]["message"].get("content") or ""
    print(f"eq {name}: {len(res[name])} chars: {res[name][:70]!r}", flush=True)
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
