#!/usr/bin/env python3
"""Recall checks: put a needle at several depths and verify the answer.

Reuses the prompt builder from strata-bench.py so the prompts are identical in shape.
"""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

spec = importlib.util.spec_from_file_location("sb", r"F:\harness\strata-bench.py")
sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb)

CHECKS = [
    (2048, 0.25),
    (2048, 0.50),
    (2048, 0.75),
    (8192, 0.25),
    (8192, 0.50),
    (8192, 0.75),
]

rows = []
print(f"{'target':>7}{'depth':>7}{'prompt_n':>10}{'prompt_tok/s':>14}{'found':>8}   answer")
print("-" * 90)
for size, depth in CHECKS:
    prompt = sb.build_prompt(size, needle_depth=depth, nonce=f"recall-{random.randrange(10**9)}")
    r = sb.call(prompt, max_tokens=32)
    ans = (r.get("answer_head") or "").strip()
    ok = sb.NEEDLE.lower() in ans.lower()
    r.update(target=size, depth=depth, needle=sb.NEEDLE, found=ok)
    rows.append(r)
    print(f"{size:>7}{depth:>7.2f}{r.get('prompt_n'):>10}{r.get('prompt_tok_s'):>14.1f}{('YES' if ok else 'NO'):>8}   {ans[:40]!r}")

out = Path(r"F:\harness\strata-bench\strata-bench-recall.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"needle": sb.NEEDLE, "checks": rows}, ensure_ascii=False, indent=1),
               encoding="utf-8")
print(f"\n{sum(1 for r in rows if r['found'])}/{len(rows)} found  ->  {out}")
