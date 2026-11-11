#!/usr/bin/env python3
"""Serial, fresh-prompt Strata benchmark (Windows-capable).

Derived from bench/results/2026-09-30-community-rtx-5090/benchmark.py: same synthetic prompts, nonces,
token targets, greedy decoding, reasoning off and 256-token output cap, so the numbers are comparable.
Changes: runs with Strata's own Windows venv; a failed or cancelled run is recorded with its error and
excluded from the statistics instead of stopping the suite; peak host RAM and GPU memory are sampled
every 0.5 s during each request.

    .venv\\Scripts\\python.exe benchmark.py --root . --pack <data-dir>\\packs\\q2_0 --url http://127.0.0.1:18096 --out results
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

import psutil


def get(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


class Peak:
    """Samples system-wide host RAM in use (total - available) and GPU memory used while a request runs."""

    def __init__(self, interval=0.5):
        self.interval, self.ram, self.vram, self._stop = interval, 0, 0, threading.Event()

    def _gpu_mib(self):
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                          text=True, timeout=5)
            return int(out.strip().splitlines()[0])
        except Exception:
            return 0

    def _run(self):
        while not self._stop.is_set():
            vm = psutil.virtual_memory()
            self.ram = max(self.ram, vm.total - vm.available)
            self.vram = max(self.vram, self._gpu_mib())
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def as_dict(self):
        return {"peak_host_ram_used_gib": round(self.ram / 2**30, 2), "peak_gpu_memory_used_mib": self.vram}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18096")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--targets", default="4096,32768,128000")
    parser.add_argument("--runs", type=int, default=3)
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
    rows = []

    def request(content, maximum=256):
        return {"model": "strata", "messages": [{"role": "user", "content": content}],
                "temperature": 0, "reasoning_effort": "none", "max_tokens": maximum,
                "stream": True, "stream_options": {"include_usage": True}}

    def count(req):
        messages, tools, kwargs = openai_to_messages(req)
        return len(tokenizer.encode(template.render(messages, tools, **kwargs), parse_special=True))

    def save():
        (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    def perform(label, req, expected=None):
        body = json.dumps(req).encode()
        (args.out / (label + "-request.json")).write_bytes(body + b"\n")
        print("start", label, "tokens", expected, flush=True)
        chunks, texts, first, finish = [], [], None, None
        started = time.perf_counter()
        try:
            with Peak() as peak:
                wire = urllib.request.Request(args.url + "/v1/chat/completions", data=body,
                                              headers={"Content-Type": "application/json"})
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
            engine = get(args.url + "/metrics")["requests"][0]
            usage = next((chunk["usage"] for chunk in reversed(chunks) if "usage" in chunk), {})
            row = {"label": label, "status": "ok", "expected_prompt_tokens": expected,
                   "request_sha256": hashlib.sha256(body).hexdigest(),
                   "client_ttft_s": first, "client_elapsed_s": elapsed, "finish_reason": finish,
                   "usage": usage, "engine": engine, "memory": peak.as_dict(), "text": "".join(texts)}
            if expected is not None and engine["prompt_tokens"] != expected:
                raise RuntimeError(f"token count mismatch: {expected} vs {engine['prompt_tokens']}")
            if not texts:
                raise RuntimeError(f"no text received for {label}")
        except Exception as error:  # recorded, excluded from the statistics
            row = {"label": label, "status": "failed", "error": repr(error), "expected_prompt_tokens": expected,
                   "request_sha256": hashlib.sha256(body).hexdigest(),
                   "client_elapsed_s": time.perf_counter() - started, "text": "".join(texts)}
        (args.out / (label + "-raw.json")).write_text(json.dumps({"chunks": chunks, "row": row}, indent=2) + "\n",
                                                      encoding="utf-8")
        rows.append(row)
        save()
        print("done", label, row["status"], json.dumps({k: row.get(k) for k in ("client_ttft_s", "client_elapsed_s", "memory")}),
              flush=True)
        return row

    perform("warmup", request("Reply with exactly the word READY.", 16))
    # Each nonce occurs before the filler so previous long prompts cannot be reused.
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
    measured = [row for row in rows if row["label"] != "warmup" and row["status"] == "ok"]
    groups = {}
    for target in map(int, args.targets.split(",")):
        subset = [r for r in measured if r["label"].startswith(f"tokens-{target}-")]
        failed = [r["label"] for r in rows if r["label"].startswith(f"tokens-{target}-") and r["status"] != "ok"]
        if not subset:
            groups[str(target)] = {"runs_ok": 0, "failed": failed}
            continue
        values = {
            "prompt_tokens": [r["engine"]["prompt_tokens"] for r in subset],
            "cached_tokens": [r["engine"].get("reused", 0) for r in subset],
            "output_tokens": [r["engine"]["engine_generated"] for r in subset],
            "client_ttft_s": [r["client_ttft_s"] for r in subset],
            "client_elapsed_s": [r["client_elapsed_s"] for r in subset],
            "prefill_tok_s": [(r["engine"]["prompt_tokens"] - (r["engine"].get("reused") or 0)) / (r["engine"]["prompt_ms"] / 1000) for r in subset],
            "decode_tok_s": [r["engine"]["engine_generated"] / (r["engine"]["decode_ms"] / 1000) for r in subset],
            "peak_host_ram_used_gib": [r["memory"]["peak_host_ram_used_gib"] for r in subset],
            "peak_gpu_memory_used_mib": [r["memory"]["peak_gpu_memory_used_mib"] for r in subset],
        }
        groups[str(target)] = {"runs_ok": len(subset), "failed": failed,
                               **{key: {"median": statistics.median(v), "min": min(v), "max": max(v)} for key, v in values.items()}}
    (args.out / "summary.json").write_text(json.dumps(groups, indent=2) + "\n", encoding="utf-8")
    print("summary", json.dumps(groups), flush=True)


if __name__ == "__main__":
    main()
