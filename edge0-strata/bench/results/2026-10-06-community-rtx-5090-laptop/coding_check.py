#!/usr/bin/env python3
"""Small verifiable coding check: the model writes a function, a fixed test suite runs it locally.

    python coding_check.py --url http://127.0.0.1:18096 --out coding-check.json
"""
import argparse
import json
import re
import time
import urllib.request

TASK = ("Write a Python function `roman_to_int(s: str) -> int` that converts a Roman numeral (I, V, X, L, C, D, M, "
        "including subtractive forms like IV, IX, XL, XC, CD, CM) to an integer, and raises ValueError for an empty "
        "string or any character that is not a Roman numeral. Reply with one Python code block only.")
TESTS = [("III", 3), ("IV", 4), ("IX", 9), ("LVIII", 58), ("MCMXCIV", 1994), ("MMXXVI", 2026), ("CDXLIV", 444)]
ERRORS = ["", "ABC", "MMX1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18096")
    ap.add_argument("--out", default="coding-check.json")
    ap.add_argument("--effort", default="low")
    a = ap.parse_args()
    body = {"model": "strata", "messages": [{"role": "user", "content": TASK}], "temperature": 0,
            "reasoning_effort": a.effort, "max_tokens": 4096}
    t0 = time.perf_counter()
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        answer = json.loads(r.read())
    elapsed = time.perf_counter() - t0
    text = answer["choices"][0]["message"].get("content") or ""
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    code = m.group(1) if m else text
    results, passed = [], 0
    namespace = {}
    try:
        exec(compile(code, "<model>", "exec"), namespace)
        f = namespace["roman_to_int"]
        for s, want in TESTS:
            got = f(s)
            ok = got == want
            passed += ok
            results.append({"input": s, "expected": want, "got": got, "ok": ok})
        for s in ERRORS:
            try:
                f(s)
                results.append({"input": s, "expected": "ValueError", "got": "no error", "ok": False})
            except ValueError:
                passed += 1
                results.append({"input": s, "expected": "ValueError", "got": "ValueError", "ok": True})
    except Exception as e:
        results.append({"error": repr(e)})
    total = len(TESTS) + len(ERRORS)
    out = {"task": TASK, "reasoning_effort": a.effort, "elapsed_s": round(elapsed, 2), "usage": answer.get("usage"),
           "passed": passed, "total": total, "results": results, "code": code}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"coding check: {passed}/{total} tests passed in {elapsed:.1f} s")


if __name__ == "__main__":
    main()
