#!/usr/bin/env python3
"""Quality check for a numerically different path (the fused IQ prompt kernels): run the same greedy prompts through
two configs and compare the answers.

    python quality_fused.py strata-iq3s-nofused.json strata-iq3s-best.json out.json
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
    "Explain in two short paragraphs how a heat pump moves heat from outside to inside a house.",
    "A train leaves at 09:15 and travels 210 km at 84 km/h. At what time does it arrive? Show the arithmetic.",
    "Summarize the main trade-offs between 4-bit and 8-bit weight quantization for LLM inference.",
    "List five concrete steps to debug a CUDA kernel that returns wrong results, one sentence each.",
]


def post(port, prompt, max_tokens=160):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"].get("content", ""), out.get("timings") or {}


def run(cfg, port, tag):
    cpath = Path(f"/tmp/quality-{tag}.json")
    c = json.loads(Path(cfg).read_text())
    c["port"] = port
    c["log"] = f"/tmp/quality-{tag}.log"
    cpath.write_text(json.dumps(c, indent=1))
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
                             str(cpath), "--port", str(port)], cwd=str(ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(600):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(2)
        post(port, "Say hi.", 8)
        out = []
        for p in PROMPTS:
            txt, t = post(port, p)
            out.append(txt)
            print(f"  {tag}: {t.get('prompt_per_second')} tok/s prompt, {t.get('predicted_per_second')} tok/s decode")
        return out
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(5)


a = run(sys.argv[1], 8097, "a")
b = run(sys.argv[2], 8097, "b")
res = []
same = 0
for i, (x, y) in enumerate(zip(a, b)):
    eq = x == y
    same += eq
    pre = 0
    while pre < min(len(x), len(y)) and x[pre] == y[pre]:
        pre += 1
    res.append({"prompt": PROMPTS[i], "identical": eq, "common_prefix_chars": pre,
                "len_a": len(x), "len_b": len(y), "a": x, "b": y})
    print(f"prompt {i}: {'IDENTICAL' if eq else 'differs'} (prefix {pre} chars, {len(x)} vs {len(y)} chars)")
Path(sys.argv[3]).write_text(json.dumps({"identical": same, "of": len(PROMPTS), "items": res}, indent=1))
print(f"{same} of {len(PROMPTS)} answers identical")
