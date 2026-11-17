"""Decoding streams while long cold prompts join: how smooth the streams stay, and what the joining prompt pays.

Runs against a running `serve/server.py` (OpenAI API). N streams decode long greedy answers (thinking off); after a
delay one or two long cold prompts join. Every streamed token's arrival is timed on the client. Reported per scenario:

  joiner_ttft_s            the joining prompt's time to first token
  worst_max_gap_s          the longest time any decoding stream received no token during the join window
  mean_stalled_s           per stream, the sum of gaps above 0.5 s during the window (mean over the streams)
  stream_tokens_during_join   tokens the decoding streams received from the first joiner's arrival to the last
                           joiner's first token + 1 s
  aggregate_tok_s          the streams' tokens / the run's wall time (throughput of the whole run)

Two scenarios without decoding streams (#1504's test): `followup` reads a cold 45K prompt alone and sends a short
prompt 1 s later (followup_ttft_s: how long the short one waits behind the read; long_ttft_s: what the read pays);
`read_alone` is the same read without the follow-up (what the read costs by itself).

The prompts are synthetic: the joiners are text from this repository's own files (src, include, serve, tools, docs), so
they route to most experts the way real code and tool output do; a joiner never repeats an earlier prompt (the
conversation cache would turn it into a warm read). `--determinism` instead sends fixed greedy prompts and prints the
SHA-256 of each answer (to compare two builds' output tokens).

  python tools/piped_read_join_bench.py --port 8080 --out join.json
  python tools/piped_read_join_bench.py --port 8080 --scenarios two_joiners,followup --repeat 3
  python tools/piped_read_join_bench.py --port 8080 --determinism
"""
import argparse
import hashlib
import http.client
import json
import statistics
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODING = ["Write a Python function that parses an INI file into nested dicts, with tests. Explain each step.",
          "Implement a thread-safe LRU cache in C++ with a short explanation of the design and its complexity.",
          "Explain, with code, how to stream a large CSV file in Go and aggregate per-key statistics.",
          "Write a TypeScript debounce and throttle pair with tests and explain where each one fits."]
DETERMINISM = ["Count from 1 to 40, comma separated.", "List the planets of the solar system in order, one per line.",
               "Write the first 12 Fibonacci numbers.", "Explain in three sentences what a hash map is.",
               'Translate "good morning, how are you" into French, German and Spanish.',
               "Write a haiku about compilers."]
# name -> the joins (delay s, size in K tokens) beside decoding streams, or (size, follow-up delay s) with none decoding
SCENARIOS = {"cold32k": [(6, 32)], "cold64k": [(6, 64)], "two_joiners": [(6, 64), (12, 32)],
             "followup": (45, 1.0), "read_alone": (45, 0)}
_FILES = []


def corpus(chars, offset):
    """About `chars` characters of this repository's own files, from the `offset`-th file on (at most 60,000 each)."""
    if not _FILES:
        for sub, pat in (("src", "*.cpp"), ("src", "*.cu"), ("include", "*.hpp"), ("serve", "*.py"), ("tools", "*.py"),
                         ("docs", "*.md")):
            _FILES.extend(sorted((ROOT / sub).rglob(pat)))
    parts, n, i = [], 0, offset
    while n < chars:
        p = _FILES[i % len(_FILES)]
        i += 1
        text = p.read_text(encoding="utf-8", errors="replace")[:60000]
        parts.append("=== %s ===\n%s\n" % (p.relative_to(ROOT).as_posix(), text))
        n += len(parts[-1])
    return "".join(parts)[:chars]


def request(content, max_tokens, model):
    """One streamed greedy chat request with thinking off."""
    return json.dumps({"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0,
                       "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()


def stream(port, raw, timeout=1800):
    """-> (token arrival times, text, usage, the time the request was sent)"""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    ts, text, usage = [], [], None
    t0 = time.monotonic()
    c.request("POST", "/v1/chat/completions", raw, {"Content-Type": "application/json", "Connection": "close"})
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError("HTTP %d" % r.status)
    while True:
        line = r.readline(1 << 20)
        if not line:
            break
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        item = json.loads(data)
        if item.get("usage"):
            usage = item["usage"]
        for ch in item.get("choices") or []:
            d = (ch.get("delta") or {}).get("content")
            if d:
                ts.append(time.monotonic())
                text.append(d)
    c.close()
    return ts, "".join(text), usage or {}, t0


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round(p / 100 * (len(v) - 1)))))] if v else 0


def smooth(ts, lo, hi):
    """One stream's arrivals in [lo, hi]: tokens, the longest and p99 gap, and the time in gaps above 0.5 s."""
    inside = [t for t in ts if lo <= t <= hi]
    gaps = [b - a for a, b in zip(ts, ts[1:]) if b >= lo and a <= hi]
    return {"tokens": len(inside), "max_gap_s": round(max(gaps or [0]), 3), "p99_gap_s": round(pct(gaps, 99), 3),
            "stalled_s": round(sum(g for g in gaps if g > 0.5), 3)}


def scenario(port, model, name, n, joins, max_tokens, tag):
    """`n` streams decode; each join (delay s, size in K tokens) then sends a cold prompt of this repository's text."""
    events, joined = {}, {}

    def decode(i):
        content = "Stream %s-%s-%d. %s" % (tag, name, i, CODING[i % len(CODING)])
        events[i] = stream(port, request(content, max_tokens, model))[0]

    def join(k, delay, size):
        time.sleep(delay)
        text = corpus(int(size * 1024 * 3.3), 7 * size + k + 131 * len(name))
        content = "Join %s-%s-%d.\n%s\nName the file that defines the class above in one line." % (tag, name, k, text)
        ts, _, usage, t0 = stream(port, request(content, 64, model))
        joined[k] = {"sent": t0, "first": ts[0] if ts else None, "usage": usage}

    t0 = time.monotonic()
    threads = [threading.Thread(target=decode, args=(i,)) for i in range(n)]
    threads += [threading.Thread(target=join, args=(k, d, size)) for k, (d, size) in enumerate(joins)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lo = min(v["sent"] for v in joined.values())
    hi = max(v["first"] for v in joined.values()) + 1.0
    per = [smooth(ts, lo, hi) for _, ts in sorted(events.items())]
    ends = [ts[-1] for ts in events.values() if ts]
    joins_in_order = [v for _, v in sorted(joined.items())]
    return {"joiner_ttft_s": [round(v["first"] - v["sent"], 3) for v in joins_in_order],
            "joiner_prompt_tokens": [v["usage"].get("prompt_tokens") for v in joins_in_order],
            "joiner_cached_tokens": [(v["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
                                     for v in joins_in_order],
            "worst_max_gap_s": max(p["max_gap_s"] for p in per),
            "mean_stalled_s": round(statistics.mean(p["stalled_s"] for p in per), 3),
            "stream_tokens_during_join": sum(p["tokens"] for p in per),
            "aggregate_tok_s": round(sum(len(t) for t in events.values()) / (max(ends) - t0), 1),
            "streams": per}


def alone(port, model, name, size, follow, tag):
    """A cold prompt of `size` K tokens with nothing decoding; with `follow`, a short prompt `follow` s after it."""
    got = {}

    def long_read():
        text = corpus(int(size * 1024 * 3.3), 11 * size + 131 * len(name))
        content = "Read %s-%s.\n%s\nName the file that defines the class above in one line." % (tag, name, text)
        ts, _, usage, t0 = stream(port, request(content, 64, model))
        got["long"] = (t0, ts[0] if ts else None, usage)

    def short():
        time.sleep(follow)
        content = "Follow-up %s-%s: what is 17 x 23? Reply with the number only." % (tag, name)
        ts, _, usage, t0 = stream(port, request(content, 16, model))
        got["short"] = (t0, ts[0] if ts else None, usage)

    threads = [threading.Thread(target=long_read)] + ([threading.Thread(target=short)] if follow else [])
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out = {"long_ttft_s": round(got["long"][1] - got["long"][0], 3),
           "long_prompt_tokens": got["long"][2].get("prompt_tokens")}
    if follow:
        out["followup_ttft_s"] = round(got["short"][1] - got["short"][0], 3)
        out["followup_prompt_tokens"] = got["short"][2].get("prompt_tokens")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model", default=None, help="default: the server's own model id")
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--scenarios", default="cold32k,cold64k,two_joiners",
                    help="comma-separated: " + ", ".join(SCENARIOS))
    ap.add_argument("--repeat", type=int, default=1, help="runs of each scenario (each with its own cold prompts)")
    ap.add_argument("--out")
    ap.add_argument("--determinism", action="store_true")
    a = ap.parse_args()
    tag = time.strftime("%H%M%S")
    if a.model is None:
        c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=30)
        c.request("GET", "/v1/models")
        a.model = json.loads(c.getresponse().read())["data"][0]["id"]
        c.close()
    if a.determinism:
        rows = []
        for p in DETERMINISM:
            _, text, usage, _ = stream(a.port, request(p, 128, a.model))
            rows.append({"prompt_sha256": hashlib.sha256(p.encode()).hexdigest()[:12],
                         "answer_sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "tokens": usage.get("completion_tokens")})
        print(json.dumps(rows, indent=1))
        return
    out = {}
    for rep in range(a.repeat):
        for name in a.scenarios.split(","):
            spec, run_tag = SCENARIOS[name], "%s.%d" % (tag, rep)
            key = name if a.repeat == 1 else "%s#%d" % (name, rep + 1)
            if isinstance(spec, tuple):
                out[key] = alone(a.port, a.model, name, spec[0], spec[1], run_tag)
            else:
                out[key] = scenario(a.port, a.model, name, a.streams, spec, a.max_tokens, run_tag)
            print(json.dumps({"scenario": key, **{k: v for k, v in out[key].items() if k != "streams"}}), flush=True)
            time.sleep(3)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
