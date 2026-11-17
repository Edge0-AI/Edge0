#!/usr/bin/env python3
"""A wider identity gate than tools/dgx_quality.py's five: 24 prompts across code, prose, math, multilingual and tool-shaped
asks, greedy, comparing the fused prompt path against MMQ byte for byte.

    python quality24.py CONFIG_A CONFIG_B OUT.json [--port 8091]
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [
    "Write a Python function that returns the longest palindromic substring of a string, with a docstring.",
    "In C, what does `static inline` mean at file scope and when should it be used? Two sentences.",
    "Explain the difference between a mutex and a semaphore to a junior developer, in under 100 words.",
    "Translate 'The server is down and we are investigating' into German, French and Spanish, one line each.",
    "Summarise the plot of Pride and Prejudice in exactly three sentences.",
    "What is the time complexity of building a binary heap, and why?",
    "Write a SQL query that returns the five customers with the highest total order value.",
    "List four failure modes of a distributed lock and one mitigation each, as a table.",
    "给一个刚学编程的人解释什么是递归，100字以内。",
    "Erkläre den Unterschied zwischen TCP und UDP in drei Sätzen.",
    "Écris une fonction JavaScript qui inverse une chaîne sans utiliser reverse().",
    "What happens in a Linux system when you run `rm -rf /` as root? Be technical and brief.",
    "Give me a shell one-liner that finds the ten largest files under /var/log.",
    "Why does a transformer need positional encodings? Answer in two sentences.",
    "Write a haiku about a GPU fan spinning up.",
    "What is the difference between `git rebase` and `git merge` for a shared branch? Two sentences.",
    "Explain Bloom filters and give one production use case, under 80 words.",
    "Sort these numbers ascending and show the result: 42, 7, 19, 3, 88, 15.",
    "What does the error `CUDA error: an illegal memory access was encountered` usually mean?",
    "Describe how to measure the power draw of a single process on Linux.",
    "Write a regex that matches ISO-8601 dates and explain each part.",
    "What is the difference between a process and a thread in one sentence?",
    "Give me a JSON object with three fields describing a book, and no prose.",
    "Explain why 0.1 + 0.2 != 0.3 in floating point, briefly.",
]


def wait(port, proc):
    t0 = time.time()
    while time.time() - t0 < 900:
        if proc.poll() is not None:
            raise SystemExit(f"server exited {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit("server did not come up")


def ask(port, prompt, max_new=96):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_new, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"].get("content", "")


def run_config(cfg_path, port, tag):
    cfg = json.loads(Path(cfg_path).read_text())
    cfg["port"] = port
    cfg["log"] = f"/tmp/quality24-{tag}.log"
    tmp = Path(f"/tmp/quality24-{tag}.json")
    tmp.write_text(json.dumps(cfg, indent=1))
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
                             str(tmp), "--port", str(port)], cwd=str(ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    answers = []
    try:
        wait(port, proc)
        ask(port, "Say hi.", 8)
        for p in PROMPTS:
            answers.append(ask(port, p))
    finally:
        proc.terminate()
        try:
            proc.wait(120)
        except subprocess.TimeoutExpired:
            proc.kill()
    return answers


def main():
    a_cfg, b_cfg, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    port = int(sys.argv[4]) if len(sys.argv) > 4 else 8091
    a = run_config(a_cfg, port, "a")
    b = run_config(b_cfg, port, "b")
    same = sum(1 for x, y in zip(a, b) if x == y)
    rows = [{"prompt": p, "a": x, "b": y, "identical": x == y} for p, x, y in zip(PROMPTS, a, b)]
    Path(out_json).write_text(json.dumps({"identical": same, "total": len(PROMPTS), "rows": rows}, indent=1))
    print(f"{same} of {len(PROMPTS)} answers identical")
    for i, r in enumerate(rows):
        if not r["identical"]:
            print(f"  differs at prompt {i}: {r['prompt'][:60]!r}")
            print(f"    A: {r['a'][:90]!r}")
            print(f"    B: {r['b'][:90]!r}")


if __name__ == "__main__":
    main()
