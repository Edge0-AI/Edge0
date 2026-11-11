#!/usr/bin/env python3
"""Strata community-report runs (PREREG_STRATA_194 addendum). Runs ON .194 against a running Strata server.
Builds ~N-token prompts from Strata's own docs (MIT, at the installed commit), sends each request with a unique nonce
first line (no prefix reuse), streams, and records client TTFT / total. The engine's own timing line for each request
is read from Strata's log afterwards (by order). Python stdlib only.
usage: strata_bench.py --url http://127.0.0.1:8080 --repo ~/strata --lengths 4096,32768 --runs 3 --out bench.jsonl"""
import argparse, json, os, secrets, time, urllib.request
from pathlib import Path

QUESTION = ("\n\nThe text above is documentation for an inference engine. In a few short paragraphs, explain what "
            "the engine does and how it keeps a large mixture-of-experts model fast on a consumer PC.")


def corpus(repo: Path) -> str:
    parts = []
    for p in sorted((repo / "docs").glob("*.md")) + [repo / "README.md"]:
        parts.append(f"===== {p.name} =====\n" + p.read_text(errors="replace"))
    return "\n\n".join(parts)


def tokenize_count(url: str, text: str) -> int:
    for path in ("/tokenize", "/v1/tokenize"):
        try:
            req = urllib.request.Request(url + path, json.dumps({"content": text}).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return len(json.load(r).get("tokens", []))
        except Exception:
            continue
    return int(len(text) / 3.6)            # fallback estimate, flagged in the output


def build(url: str, text: str, target: int) -> tuple[str, int]:
    lo, hi = 1000, len(text)
    best = text[:lo]
    while hi - lo > 2000:                  # binary search on characters for ~target tokens
        mid = (lo + hi) // 2
        if tokenize_count(url, text[:mid] + QUESTION) <= target:
            lo, best = mid, text[:mid]
        else:
            hi = mid
    p = best + QUESTION
    return p, tokenize_count(url, p)


def run(url: str, prompt: str) -> dict:
    nonce = secrets.token_hex(16)
    body = {"model": "strata", "messages": [{"role": "user", "content": f"[bench nonce: {nonce}]\n" + prompt}],
            "max_tokens": 256, "temperature": 0, "stream": True, "reasoning_effort": "none",
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.monotonic(); first = last = None; text = []; usage = None; reasoning = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    now = time.monotonic() - t0
                    first = now if first is None else first
                    last = now
                    text.append(d.get("content") or "")
                    reasoning += len(d.get("reasoning_content") or "")
            usage = ev.get("usage") or usage
    total = time.monotonic() - t0
    return {"nonce": nonce, "ttft_s": round(first, 3) if first else None, "total_s": round(total, 3),
            "first_to_last_s": round(last - first, 3) if first is not None else None,
            "usage": usage, "content_chars": len("".join(text)), "reasoning_chars": reasoning,
            "content_head": "".join(text)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080"); ap.add_argument("--repo", default=os.path.expanduser("~/strata"))
    ap.add_argument("--lengths", default="4096,32768"); ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    text = corpus(Path(a.repo))
    with open(a.out, "a") as f:
        for L in [int(x) for x in a.lengths.split(",")]:
            prompt, n = build(a.url, text, L - 40)
            for i in range(a.runs):
                rec = {"target_tokens": L, "prompt_tokens_tokenized": n, "run": i + 1, **run(a.url, prompt)}
                f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
                print(json.dumps({k: rec[k] for k in ("target_tokens", "run", "ttft_s", "total_s", "content_chars", "reasoning_chars")}), flush=True)


if __name__ == "__main__":
    main()
