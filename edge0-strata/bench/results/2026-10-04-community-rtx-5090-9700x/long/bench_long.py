#!/usr/bin/env python3
"""One full-window prompt, then two waves of two overlapping 128000-token prompts.

The installed config is unchanged. Prompt tok/s is fresh tokens / prompt_ms.
Decode tok/s is engine_generated / decode_ms. A wave records /metrics slot
samples so overlap is counted from the server, not assumed.
"""
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(r"D:\AI\Strata")
PACK = Path(r"D:\AI\Strata-data\packs\iq3_xxs")
OUT = Path(r"D:\AI\Strata-data\community-bench-20261004-long")
URL = "http://127.0.0.1:5310"
# 262144 context, 8-token slack, 256-token answer.
FULL = 262144 - 8 - 256
LONG = 128000
MODEL = "qwen3.8-flash-next-iq3_xxs"


def get(path):
    with urllib.request.urlopen(URL + path, timeout=30) as response:
        return json.load(response)


def post(body):
    wire = urllib.request.Request(
        URL + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    chunks, texts, first, finish = [], [], None, None
    started = time.perf_counter()
    wall = time.time()
    try:
        response = urllib.request.urlopen(wire, timeout=1800)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:800]
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    with response:
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
    usage = next((chunk["usage"] for chunk in reversed(chunks) if "usage" in chunk), {})
    return {"client_ttft_s": first, "client_elapsed_s": elapsed, "client_start": wall,
            "finish_reason": finish, "usage": usage, "text": "".join(texts), "chunks": len(chunks)}


def request(content, maximum=256):
    return {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "temperature": 0, "reasoning_effort": "none", "max_tokens": maximum,
            "stream": True, "stream_options": {"include_usage": True}}


def progress(text):
    print(text, flush=True)
    path = OUT / "progress.txt"
    previous = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(previous + time.strftime("%H:%M:%S ") + text + "\n", encoding="utf-8")


def rates(engine):
    reused = engine.get("reused") or 0
    prompt_ms = engine.get("prompt_ms") or 0
    decode_ms = engine.get("decode_ms") or 0
    generated = engine.get("engine_generated") or 0
    fresh = (engine.get("prompt_tokens") or 0) - reused
    return {
        "prefill_tok_s": (fresh / (prompt_ms / 1000)) if prompt_ms else None,
        "decode_tok_s": (generated / (decode_ms / 1000)) if decode_ms and generated else None,
    }


def assign_engines(pairs):
    """pairs: (name, client_start, prompt_tokens). One history row each."""
    history = list(get("/metrics?requests=all")["requests"])
    assigned = {}
    for name, start, tokens in sorted(pairs, key=lambda item: item[1]):
        candidates = [row for row in history
                      if row.get("prompt_tokens") == tokens and abs((row.get("time") or 0) - start) <= 180]
        if not candidates:
            assigned[name] = None
            continue
        best = min(candidates, key=lambda row: abs((row.get("time") or 0) - start))
        history.remove(best)
        assigned[name] = best
    return assigned


def main():
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate, openai_to_messages

    directory = PACK / "tokenizer"
    vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for token, number in vocab.items():
        tokens[number] = token
    tokenizer = Tokenizer(
        tokens, (directory / "merges.txt").read_text(encoding="utf-8").splitlines(),
        json.loads((directory / "token_type.json").read_text(encoding="utf-8")))
    template = ChatTemplate(directory / "chat_template.jinja")

    def count(req):
        messages, tools, kwargs = openai_to_messages(req)
        return len(tokenizer.encode(template.render(messages, tools, **kwargs), parse_special=True))

    (OUT / "initial-status.json").write_text(
        json.dumps(get("/v1/status"), indent=2) + "\n", encoding="utf-8")
    (OUT / "initial-metrics.json").write_text(
        json.dumps(get("/metrics"), indent=2) + "\n", encoding="utf-8")

    ending = ("\n\nWrite a detailed explanation of the code above. Discuss deterministic integer transforms, "
              "modulo arithmetic, testing, naming, complexity, and maintainability. Write at least 600 words.")

    def build(label, target):
        prefix = f"Benchmark nonce: {label}.\nReview this synthetic Python module:\n"
        lines = 8000
        filler = ""
        while True:
            filler = "\n".join(
                f"def task_{i:05d}(value: int) -> int: return (value * {(i % 97) + 1} + {i}) % 100003"
                for i in range(lines))
            total = count(request(prefix + filler + ending))
            progress(f"size {label} lines={lines} tokens={total} target={target}")
            if total >= target or lines >= 40000:
                break
            lines += 4000
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
            raise RuntimeError(f"could not construct {label} target {target}: got {actual}")
        return req, actual

    def perform(label, req, expected):
        body = json.dumps(req).encode()
        (OUT / f"{label}-request.json").write_bytes(body + b"\n")
        progress(f"start {label} tokens {expected} bytes {len(body)}")
        client = post(body)
        if not client["text"]:
            raise RuntimeError(f"no text received for {label}")
        engine = assign_engines([(label, client["client_start"], expected)])[label]
        if engine is None:
            raise RuntimeError(f"no /metrics row for {label}")
        if engine.get("prompt_tokens") != expected:
            raise RuntimeError(f"token count mismatch for {label}: {expected} vs {engine.get('prompt_tokens')}")
        row = {"label": label, "expected_prompt_tokens": expected,
               "request_sha256": hashlib.sha256(body).hexdigest(),
               **{k: v for k, v in client.items() if k != "chunks"},
               "stream_chunks": client["chunks"], "engine": engine, **rates(engine)}
        (OUT / f"{label}-row.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        progress("done " + label + " " + json.dumps({
            "ttft_s": row["client_ttft_s"], "elapsed_s": row["client_elapsed_s"],
            "prefill_tok_s": row["prefill_tok_s"], "decode_tok_s": row["decode_tok_s"],
            "reused": engine.get("reused"), "finish": row["finish_reason"],
            "drafts": [engine.get("drafts_accepted"), engine.get("drafts_offered")],
            "hit_rate": engine.get("hit_rate"), "pcie_share": engine.get("pcie_share")}))
        return row

    rows = []
    progress("warmup")
    warm = perform("warmup", request("Reply with exactly the word READY.", 16), None) if False else None
    # Warmup has no expected token count. Run it without the equality check.
    body = json.dumps(request("Reply with exactly the word READY.", 16)).encode()
    client = post(body)
    progress(f"warmup ttft {client['client_ttft_s']} finish {client['finish_reason']}")
    (OUT / "warmup-row.json").write_text(json.dumps(client, indent=2) + "\n", encoding="utf-8")

    for run in (1, 2):
        req, actual = build(f"full-window-run-{run}", FULL)
        rows.append(perform(f"full-window-run-{run}", req, actual))
        (OUT / "results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    def wave(number):
        left_req, left_n = build(f"wave-{number}-a", LONG)
        right_req, right_n = build(f"wave-{number}-b", LONG)
        jobs = (("a", left_req, left_n), ("b", right_req, right_n))
        bodies = {}
        for name, req, _expected in jobs:
            bodies[name] = json.dumps(req).encode()
            (OUT / f"wave-{number}-{name}-request.json").write_bytes(bodies[name] + b"\n")
        samples = []
        stop = threading.Event()

        def sample():
            while not stop.is_set():
                try:
                    live = get("/metrics")["live"]
                    samples.append({"t": time.time(), "state": live.get("state"),
                                     "running": live.get("running"), "outside_slots": live.get("outside_slots"),
                                     "waiting": live.get("waiting"), "slots": live.get("slots")})
                except (OSError, ValueError, urllib.error.URLError):
                    pass
                stop.wait(2)

        sampler = threading.Thread(target=sample, daemon=True)
        barrier = threading.Barrier(2)
        found = {}
        errors = {}

        def worker(name, expected):
            try:
                barrier.wait(timeout=120)
                progress(f"start wave-{number}-{name} tokens {expected}")
                client = post(bodies[name])
                found[name] = (expected, client)
            except Exception as error:
                errors[name] = f"{type(error).__name__}: {error}"

        sampler.start()
        threads = [threading.Thread(target=worker, args=(name, expected)) for name, _req, expected in jobs]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        stop.set()
        sampler.join(timeout=5)
        (OUT / f"wave-{number}-slots.json").write_text(
            json.dumps(samples, indent=2) + "\n", encoding="utf-8")
        if errors:
            raise RuntimeError(f"wave {number} failed: {errors}")
        active = 0
        for sample_row in samples:
            busy = [slot for slot in (sample_row.get("slots") or []) if slot.get("state") != "idle"]
            active = max(active, len(busy))
        paired = assign_engines([
            (name, found[name][1]["client_start"], found[name][0]) for name, _req, _expected in jobs])
        for name, _req, expected in jobs:
            got, client = found[name]
            if not client["text"]:
                raise RuntimeError(f"wave {number} {name} returned no text")
            engine = paired[name]
            if engine is None:
                raise RuntimeError(f"wave {number} {name} has no /metrics row")
            row = {"label": f"wave-{number}-{name}", "expected_prompt_tokens": got,
                   "request_sha256": hashlib.sha256(bodies[name]).hexdigest(),
                   "overlap_samples": len(samples), "max_busy_slots_seen": active,
                   **{k: v for k, v in client.items() if k != "chunks"},
                   "stream_chunks": client["chunks"], "engine": engine, **rates(engine)}
            rows.append(row)
            progress("done " + row["label"] + " " + json.dumps({
                "ttft_s": row["client_ttft_s"], "elapsed_s": row["client_elapsed_s"],
                "prefill_tok_s": row["prefill_tok_s"], "decode_tok_s": row["decode_tok_s"],
                "reused": engine.get("reused"), "finish": row["finish_reason"],
                "drafts": [engine.get("drafts_accepted"), engine.get("drafts_offered")],
                "hit_rate": engine.get("hit_rate"), "max_busy_slots_seen": active}))
        (OUT / "results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    wave(1)
    wave(2)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    progress("complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        progress(f"ERROR {type(error).__name__}: {error}")
        raise
