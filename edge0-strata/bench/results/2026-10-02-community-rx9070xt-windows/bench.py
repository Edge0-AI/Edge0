#!/usr/bin/env python3
"""Measure any OpenAI-compatible server the same way: decode speed and prompt (prefill) speed.

Standard library only. Every request starts with a fresh nonce so no server can answer from a prompt cache.

  python3 bench/bench.py --url http://127.0.0.1:8080/v1 --label strata-iq3s --out bench/results/<dir>
  python3 bench/bench.py --url http://127.0.0.1:8811/v1 --label mimo-stock --extra '{"chat_template_kwargs":{"enable_thinking":false}}'

Tests:
  decode     short prompt, --decode-tokens tokens out (default 256): tokens/s after the first token
  prefill-N  a ~N-token code prompt, 1 token out: prompt tokens / time to first token
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
import urllib.request

DECODE_PROMPT = ("Write a Python function that parses an Apache access log line into a dict with ip, time, method, "
                 "path, status and size. Then explain each regular expression group in two sentences.")


def code_corpus(n_tokens, seed=1234):
    """Deterministic synthetic Python code, roughly n_tokens long (about 3.2 characters per token)."""
    rng = random.Random(seed)
    words = ["user", "token", "session", "query", "cache", "request", "payload", "config", "record", "result",
             "handler", "buffer", "index", "limit", "offset", "account", "secret", "path", "status", "retry"]
    out, size, target = [], 0, int(n_tokens * 3.2)
    i = 0
    while size < target:
        a, b, c = rng.sample(words, 3)
        block = (f"def {a}_{b}_{i}({c}, {a}=None):\n"
                 f"    \"\"\"Return the {b} for {c}.\"\"\"\n"
                 f"    if {a} is None:\n"
                 f"        {a} = load_{a}({c})\n"
                 f"    for k in range({rng.randint(2, 64)}):\n"
                 f"        {c} = update_{b}({c}, k, {a})\n"
                 f"    return {{'{b}': {c}, 'n': {rng.randint(0, 9999)}}}\n\n")
        out.append(block)
        size += len(block)
        i += 1
    return "".join(out)


def load_extra(value):
    if value.startswith("@"):
        with open(value[1:], encoding="utf-8") as f:
            return json.load(f)
    return json.loads(value)


def post_stream(url, key, body, timeout):
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    t_first, n_chunks, usage = None, 0, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    if t_first is None:
                        t_first = time.perf_counter()
                    n_chunks += 1
    t_end = time.perf_counter()
    return t0, t_first, t_end, n_chunks, usage


def one(args, model, prompt, max_tokens):
    nonce = f"[run {time.time_ns()}] "
    body = {"model": model, "messages": [{"role": "user", "content": nonce + prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    body.update(args.extra)
    t0, t_first, t_end, n_chunks, usage = post_stream(args.url, args.key, body, args.timeout)
    if t_first is None:
        if max_tokens > 1:
            raise RuntimeError("no tokens came back")
        # prefill test: the one token may be a template token with no visible text; the request still ends right
        # after the prompt is read
        t_first = t_end
    out_tokens = (usage or {}).get("completion_tokens") or n_chunks
    prompt_tokens = (usage or {}).get("prompt_tokens")
    decode_s = t_end - t_first
    return {"ttft_s": t_first - t0, "total_s": t_end - t0, "prompt_tokens": prompt_tokens,
            "completion_tokens": out_tokens,
            "decode_tps": (out_tokens - 1) / decode_s if out_tokens > 1 and decode_s > 0 else None,
            "prefill_tps": prompt_tokens / (t_first - t0) if prompt_tokens else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="base URL ending in /v1")
    p.add_argument("--key", default=os.environ.get("BENCH_API_KEY", "none"))
    p.add_argument("--model", default=None, help="default: first id from /v1/models")
    p.add_argument("--label", required=True)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--decode-tokens", type=int, default=256)
    p.add_argument("--prefill", default="4096,32768", help="prompt sizes in tokens, comma separated; '' for none")
    p.add_argument("--extra", type=load_extra, default={},
                   help="JSON merged into every request body, or @file.json (avoids shell quoting, e.g. on Windows)")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--out", default=None, help="directory for <label>.json")
    p.add_argument("--skip-decode", action="store_true", help="only the prefill tests")
    args = p.parse_args()

    model = args.model
    if not model:
        req = urllib.request.Request(args.url + "/models", headers={"Authorization": f"Bearer {args.key}"})
        model = json.load(urllib.request.urlopen(req, timeout=60))["data"][0]["id"]

    tests = [] if args.skip_decode else [("decode", DECODE_PROMPT, args.decode_tokens)]
    tests += [(f"prefill-{n}", "Summarize what this code does.\n\n" + code_corpus(n), 1)
              for n in (int(x) for x in args.prefill.split(",") if x)]

    print(f"# {args.label}: {model} at {args.url}", file=sys.stderr)
    one(args, model, "Say hi.", 8)  # warm-up, not counted
    result = {"label": args.label, "model": model, "url": args.url, "extra": args.extra, "runs": args.runs,
              "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tests": {}}
    for name, prompt, max_tokens in tests:
        rows = []
        for i in range(args.runs):
            r = one(args, model, prompt, max_tokens)
            rows.append(r)
            print(f"  {name} run {i + 1}: " + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                                        for k, v in r.items()), file=sys.stderr)
        key = "decode_tps" if name == "decode" else "prefill_tps"
        vals = [r[key] for r in rows if r[key]]
        result["tests"][name] = {"median_" + key: statistics.median(vals) if vals else None, "runs": rows}
        print(f"{args.label}\t{name}\t{key}={result['tests'][name]['median_' + key]:.1f}" if vals else
              f"{args.label}\t{name}\tno value")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"{args.label}.json"), "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
