#!/usr/bin/env python3
"""Serial fresh-prompt benchmark for the installed IQ3_S server.

Same shape as the 2026-10-04 IQ3_XXS community run, plus three prompts that
fill this host's 262144-token window. The full-window prompt is 261880 tokens
so a 256-token answer still fits: the server keeps 8 tokens of slack and
refuses a request that would pass the limit.
"""
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path


def get(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"D:\AI\Strata"))
    parser.add_argument("--pack", type=Path, default=Path(r"D:\AI\Strata-data\packs\iq3_s"))
    parser.add_argument("--url", default="http://127.0.0.1:5310")
    parser.add_argument("--out", type=Path, default=Path(r"D:\AI\Strata-data\community-bench-20261005-iq3s"))
    parser.add_argument("--targets", default="4096,32768,128000,261880")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default="qwen3.8-flash-next-iq3_s")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(args.root), str(args.root / "tools")]
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate, openai_to_messages

    directory = args.pack / "tokenizer"
    vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for token, number in vocab.items():
        tokens[number] = token
    tokenizer = Tokenizer(tokens, (directory / "merges.txt").read_text(encoding="utf-8").splitlines(),
                          json.loads((directory / "token_type.json").read_text(encoding="utf-8")))
    template = ChatTemplate(directory / "chat_template.jinja")
    initial = get(args.url + "/v1/status")
    (args.out / "initial-status.json").write_text(json.dumps(initial, indent=2) + "\n", encoding="utf-8")
    metrics0 = get(args.url + "/metrics")
    (args.out / "initial-metrics.json").write_text(json.dumps(metrics0, indent=2) + "\n", encoding="utf-8")
    rows = []
    stop = threading.Event()
    tel_path = args.out / "telemetry.csv"
    progress_path = args.out / "progress.txt"

    def progress(line):
        print(line, flush=True)
        with progress_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def sample():
        if not tel_path.exists():
            tel_path.write_text("unix,mem_used_mib,util_gpu,pcie_gen,pcie_width,power_w\n", encoding="utf-8")
        while not stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=memory.used,utilization.gpu,pcie.link.gen.current,pcie.link.width.current,power.draw",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=15).strip()
                with tel_path.open("a", encoding="utf-8") as fh:
                    fh.write(f"{time.time():.0f},{out}\n")
            except (OSError, subprocess.SubprocessError):
                pass
            stop.wait(5)

    threading.Thread(target=sample, daemon=True).start()

    def request(content, maximum=256):
        return {"model": args.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0, "reasoning_effort": "none", "max_tokens": maximum,
                "stream": True, "stream_options": {"include_usage": True}}

    def count(req):
        messages, tools, kwargs = openai_to_messages(req)
        return len(tokenizer.encode(template.render(messages, tools, **kwargs), parse_special=True))

    def perform(label, req, expected=None):
        body = json.dumps(req).encode()
        (args.out / (label + "-request.json")).write_bytes(body + b"\n")
        progress(f"start {label} tokens {expected}")
        chunks, texts, first, finish = [], [], None, None
        started = time.perf_counter()
        wire = urllib.request.Request(args.url + "/v1/chat/completions", data=body,
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(wire, timeout=3600) as response:
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
        if expected is not None and engine.get("prompt_tokens") != expected:
            raise RuntimeError(f"token count mismatch: {expected} vs {engine.get('prompt_tokens')}")
        if not texts:
            raise RuntimeError(f"no text received for {label}")
        (args.out / (label + "-raw.json")).write_text(
            json.dumps({"chunks": chunks, "row": row}, indent=2) + "\n", encoding="utf-8")
        rows.append(row)
        (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        progress("done " + label + " " + json.dumps({
            "ttft_s": first, "elapsed_s": elapsed,
            "prompt_tokens": engine.get("prompt_tokens"), "reused": engine.get("reused"),
            "prompt_ms": engine.get("prompt_ms"), "decode_ms": engine.get("decode_ms"),
            "engine_generated": engine.get("engine_generated"),
            "finish": finish}))
        return row

    try:
        perform("warmup", request("Reply with exactly the word READY.", 16))
        filler = "\n".join(
            f"def task_{i:05d}(value: int) -> int: return (value * {(i % 97) + 1} + {i}) % 100003"
            for i in range(80000))
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
                "cached_tokens": [r["engine"].get("reused") or 0 for r in subset],
                "output_tokens": [r["engine"]["engine_generated"] for r in subset],
                "client_ttft_s": [r["client_ttft_s"] for r in subset],
                "client_elapsed_s": [r["client_elapsed_s"] for r in subset],
                "prefill_tok_s": [
                    (r["engine"]["prompt_tokens"] - (r["engine"].get("reused") or 0))
                    / (r["engine"]["prompt_ms"] / 1000) for r in subset],
                "decode_tok_s": [
                    r["engine"]["engine_generated"] / (r["engine"]["decode_ms"] / 1000) for r in subset],
            }
            groups[str(target)] = {
                key: {"median": statistics.median(v), "min": min(v), "max": max(v)}
                for key, v in values.items()}
        (args.out / "summary.json").write_text(json.dumps(groups, indent=2) + "\n", encoding="utf-8")
        progress("summary " + json.dumps(groups))
    finally:
        stop.set()


if __name__ == "__main__":
    main()
