#!/usr/bin/env python3
"""cmp_texts.py A.json B.json - compare the answers (par_load --keep-text) of two ledgers request by request (same nonce)."""
import json, sys
a, b = (json.load(open(x)) for x in sys.argv[1:3])
ra = {(r["workload"], r.get("level"), r["idx"]): r for r in a["requests"]}
rb = {(r["workload"], r.get("level"), r["idx"]): r for r in b["requests"]}
same = diff = 0
for k, r in sorted(ra.items(), key=str):
    o = rb.get(k)
    if o is None or "text" not in r or "text" not in o:
        continue
    if r["text"] == o["text"]:
        same += 1
    else:
        diff += 1
        n = next((i for i, (x, y) in enumerate(zip(r["text"], o["text"])) if x != y), min(len(r["text"]), len(o["text"])))
        print("DIFF", k, "first difference at char", n, "|", repr(r["text"][max(0, n - 20):n + 30]), "vs", repr(o["text"][max(0, n - 20):n + 30]))
print("identical", same, "different", diff)
sys.exit(1 if diff else 0)
