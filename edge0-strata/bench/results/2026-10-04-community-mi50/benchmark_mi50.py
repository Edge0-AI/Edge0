#!/usr/bin/env python3
"""Serial, fresh-prompt Strata benchmark — adapted for the MI50 (gfx906) container.

Based on bench/results/2026-09-30-community-rtx-5090/benchmark.py. Runs INSIDE the
strata container (imports strata_tokenizer and serve.frontend from /opt/strata).
Added: --api-key (Strata's server requires an Authorization header).
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/opt/strata"))
    parser.add_argument("--pack", type=Path, default=Path("/data/packs/iq2_xs"))
    parser.add_argument("--url", default="http://127.0.0.1:8085")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--out", type=Path, default=Path("/work/bench/results"))
    parser.add_argument("--targets", default="4096,32768")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(args.root), str(args.root / "tools")]
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate, openai_to_messages

    auth = {"Content-Type": "application/json"}
    if args.api_key:
        auth["Authorization"] = "Bearer " + args.api_key

    def get(url):
        request = urllib.request.Request(url, headers=auth)
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    directory = args.pack / "tokenizer"
    vocab = json.loads((directory / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for token, number in vocab.items():
        tokens[number] = token
    tokenizer = Tokenizer(tokens, (directory / "merges.txt").read_text().splitlines(),
                          json.loads((directory / "token_type.json").read_text()))
    template = ChatTemplate(directory / "chat_template.jinja")
    initial = get(args.url + "/v1/status")
    (args.out / "initial-status.json").write_text(json.dumps(initial, indent=2) + "\n")
    rows = []

    def request(content, maximum=256):
        return {"model": "strata", "messages": [{"role": "user", "content": content}],
                "temperature": 0, "reasoning_effort": "none", "max_tokens": maximum,
                "stream": True, "stream_options": {"include_usage": True}}

    def count(req):
        messages, tools, kwargs = openai_to_messages(req)
        return len(tokenizer.encode(template.render(messages, tools, **kwargs), parse_special=True))

    def perform(label, req, expected=None):
        body = json.dumps(req).encode()
        (args.out / (label + "-request.json")).write_bytes(body + b"\n")
        print("start", label, "tokens", expected, flush=True)
        chunks, texts, first, finish = [], [], None, None
        started = time.perf_counter()
        wire = urllib.request.Request(args.url + "/v1/chat/completions", data=body, headers=auth)
        with urllib.request.urlopen(wire, timeout=1800) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                chunk = json.loads(payload)
                chunks.append(chunk)
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    text = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                    if text:
                        first = first if first is not None else time.perf_counter() - started
                        texts.append(text)
                    finish = choice.get("finish_reason") or finish
        elapsed = time.perf_counter() - started
        metrics = get(args.url + "/metrics")
        engine = metrics["requests"][0]
        usage = next((chunk["usage"] for chunk in reversed(chunks) if "usage" in chunk), {})
        row = {"label": label, "expected_prompt_tokens": expected,
               "request_sha256": hashlib.sha256(body).hexdigest(),
               "client_ttft_s": first, "client_elapsed_s": elapsed, "finish_reason": finish,
               "usage": usage, "engine": engine, "text": "".join(texts)}
        if expected is not None and engine["prompt_tokens"] != expected:
            raise RuntimeError(f"token count mismatch: {expected} vs {engine['prompt_tokens']}")
        if not texts:
            raise RuntimeError(f"no text received for {label}")
        (args.out / (label + "-raw.json")).write_text(json.dumps({"chunks": chunks, "row": row}, indent=2) + "\n")
        rows.append(row)
        (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        print("done", label, json.dumps({"ttft_s": first, "elapsed_s": elapsed, "engine": engine}), flush=True)
        return row

    perform("warmup", request("Reply with exactly the word READY.", 16))
    filler = "\n".join(f"def task_{i:05d}(value: int) -> int: return (value * {(i % 97) + 1} + {i}) % 100003"
                       for i in range(12000))
    ending = ("\n\nWrite a detailed explanation of the code above. Discuss deterministic integer transforms, "
              "modulo arithmetic, testing, naming, complexity, and maintainability. Write at least 600 words.")
    for target in map(int, args.targets.split(",")):
        for run in range(1, args.runs + 1):
            prefix = f"Benchmark nonce: series-{target}-trial-{run}.\nReview this synthetic Python module:\n"
            lo, hi = 0, len(filler)
            while lo < hi:
                middle = (lo + hi + 1) // 2
                if count(request(prefix + filler[:middle] + ending)) <= target:
                    lo = middle
                else:
                    hi = middle - 1
            req = request(prefix + filler[:lo] + ending)
            actual = count(req)
            if actual < target - 20 or actual > target:
                raise RuntimeError(f"could not construct target {target}: {actual}")
            perform(f"tokens-{target}-run-{run}", req, actual)
    measured = [row for row in rows if row["label"] != "warmup"]
    groups = {}
    for target in map(int, args.targets.split(",")):
        subset = [r for r in measured if r["label"].startswith(f"tokens-{target}-")]
        values = {
            "prompt_tokens": [r["engine"]["prompt_tokens"] for r in subset],
            "cached_tokens": [r["engine"].get("reused", 0) for r in subset],
            "output_tokens": [r["engine"]["engine_generated"] for r in subset],
            "client_ttft_s": [r["client_ttft_s"] for r in subset],
            "client_elapsed_s": [r["client_elapsed_s"] for r in subset],
            "prefill_tok_s": [(r["engine"]["prompt_tokens"] - (r["engine"].get("reused") or 0)) / (r["engine"]["prompt_ms"] / 1000) for r in subset],
            "decode_tok_s": [r["engine"]["engine_generated"] / (r["engine"]["decode_ms"] / 1000) for r in subset],
        }
        groups[str(target)] = {key: {"median": statistics.median(v), "min": min(v), "max": max(v)}
                              for key, v in values.items()}
    (args.out / "summary.json").write_text(json.dumps(groups, indent=2) + "\n")
    print("summary", json.dumps(groups), flush=True)


if __name__ == "__main__":
    main()
