#!/usr/bin/env python3
"""Fresh-prompt read + decode speed, and follow-up turns on a long conversation, through a running Strata server.

Every measured prompt starts with its own nonce, so no earlier prompt's prefix can be reused: each fresh run reads its
whole prompt (the engine log's "N reused" confirms it). The prompt text is the repository's own files (the haystack
of tools/needle_bench.py: docs, sources), cut to an exact token count with the pack's tokenizer and the chat
template, followed by a request for ~600 words of plain prose; the output cap is 400 tokens, greedy, reasoning
off (reasoning_effort "none"). Prompts of 100K tokens or more carry a code word at 40 % depth; with --followup, the
last run of such a size is followed by a second user turn on the same conversation (the long prefix reused) that
asks for the code word and ~400 more words.

Before every request the script waits until GET /status says busy=false, queued=0 for 3 s. After it, the run is
discarded and repeated if GET /metrics counted more than one finished request or the engine log shows more than one
prompt line (someone else's request overlapped). Run with the install's Python (it needs `regex` for the tokenizer):

    STRATA_API_KEY=... python bench_server.py --targets 4096,32768,128000 --runs 3 --followup
    python bench_server.py --key-config <server config.json> --targets 250000 --runs 1 --followup
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

from common import LogTail, Server, api_key, parse_engine, scrub_lines

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from strata_tokenizer import Tokenizer  # noqa: E402
from serve.frontend import ChatTemplate, openai_to_messages  # noqa: E402
from needle_bench import haystack  # noqa: E402

ENDING = ("\n\nWrite a plain-prose overview of the material above in about 600 words, as continuous paragraphs "
          "without lists, headings or code.")
FOLLOWUP = ("What is the secret code word mentioned in the text above? Write the code word first, on its own line. "
            "Then, in about 400 words of plain prose, explain how the server described in these files handles a "
            "long prompt.")
WORDS = ["amber", "falcon", "quartz", "willow", "copper", "harbor", "saffron", "glacier", "orchid", "lantern"]


def load_tokenizer(directory: Path):
    vocab = json.loads((directory / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for token, number in vocab.items():
        tokens[number] = token
    tok = Tokenizer(tokens, (directory / "merges.txt").read_text().splitlines(),
                    json.loads((directory / "token_type.json").read_text()))
    return tok, ChatTemplate(directory / "chat_template.jinja")


def body(messages, max_tokens):
    return {"model": "strata", "messages": messages, "temperature": 0, "reasoning_effort": "none",
            "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True}}


def stream(server: Server, req: dict, timeout: float = 3600) -> dict:
    data = json.dumps(req).encode()
    first = finish = usage = timings = None
    texts, keepalives = [], 0
    t0 = time.perf_counter()
    wire = urllib.request.Request(server.url + "/v1/chat/completions", data=data, headers=server.headers())
    with urllib.request.urlopen(wire, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith(":"):
                keepalives += 1
                continue
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            usage = chunk.get("usage") or usage
            timings = chunk.get("timings") or timings
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                text = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if text:
                    if first is None:
                        first = time.perf_counter() - t0
                    texts.append(text)
                finish = choice.get("finish_reason") or finish
    return {"request_sha256": hashlib.sha256(data).hexdigest(), "client_ttft_s": first,
            "client_total_s": time.perf_counter() - t0, "finish_reason": finish, "usage": usage, "timings": timings,
            "keepalives": keepalives, "text": "".join(texts)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--key-config", help="server config JSON to read api_key from (else STRATA_API_KEY)")
    ap.add_argument("--tokenizer", required=True, help="the pack's tokenizer directory")
    ap.add_argument("--engine-log", required=True)
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--out", type=Path, default=HERE / "data")
    ap.add_argument("--targets", default="4096,32768,128000")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--followup", action="store_true", help="a follow-up turn after the last run of sizes >= 100K")
    ap.add_argument("--warmup", action="store_true", help="one short unmeasured request first")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    server = Server(a.url, api_key(a.key_config))
    tok, tpl = load_tokenizer(Path(a.tokenizer))
    elog, slog = LogTail(a.engine_log), LogTail(a.server_log)
    rnd = random.Random(a.seed)
    hay = haystack(1_100_000)

    def count(messages):
        m, tools, kw = openai_to_messages(body(messages, 1))
        return len(tok.encode(tpl.render(m, tools, **kw), parse_special=True))

    def measured(label, messages, meta):
        for attempt in range(1, 4):
            status = server.wait_idle()
            if not server.loaded():
                print("  the model is unloaded: this request would include loading; aborting", flush=True)
                raise SystemExit(2)
            before = server.totals()
            elog.mark(), slog.mark()
            t_start = time.time()
            print(f"start {label} (attempt {attempt}) {meta.get('expected_prompt_tokens')} tokens", flush=True)
            res = stream(server, body(messages, a.max_tokens))
            t_end = time.time()
            time.sleep(1.5)                                   # the engine prints its timing lines after DONE
            after = server.totals()
            elines, slines = elog.since(), slog.since()
            parsed = parse_engine(elines)
            metrics = server.get("/metrics")["requests"][0]
            foreign = (after - before != 1) or len(parsed) != 1
            row = {"label": label, "attempt": attempt, "epoch_start": t_start, "epoch_end": t_end, **meta, **res,
                   "status_before": status, "totals_before": before, "totals_after": after, "foreign_overlap": foreign,
                   "engine": parsed[0] if parsed else None, "metrics_record": metrics,
                   "engine_log": scrub_lines(elines), "server_log": scrub_lines(slines)}
            name = f"{label}" + ("" if not foreign else f"-discarded-{attempt}")
            (a.out / f"{name}.json").write_text(json.dumps(row, indent=1, ensure_ascii=False) + "\n")
            e = parsed[0] if parsed else {}
            print(f"done  {label}: {e.get('line', '(no engine line)')}  ttft {res['client_ttft_s']:.2f} s "
                  f"total {res['client_total_s']:.1f} s foreign={foreign}", flush=True)
            if not foreign:
                return row
            print("  foreign traffic overlapped this run: discarded, repeating", flush=True)
        raise SystemExit(f"{label}: three attempts overlapped with other requests")

    if a.warmup:
        server.wait_idle()
        r = stream(server, body([{"role": "user", "content": "Reply with exactly the word READY."}], 8))
        print("warm-up:", r["text"].strip(), flush=True)

    for target in map(int, a.targets.split(",")):
        for run in range(1, a.runs + 1):
            label = f"fresh-{target}-run{run}"
            nonce = f"Benchmark nonce {label}-{rnd.getrandbits(48):012x}.\n"
            head = nonce + "The text below is a collection of files from a software repository.\n"
            word = None
            if target >= 100_000:
                word = f"{rnd.choice(WORDS)}-{rnd.choice(WORDS)}-{rnd.randint(100, 999)}"

            def build(n):
                text = hay[:n]
                if word:
                    cut = text.rfind("\n", 0, int(n * 0.40)) + 1
                    text = text[:cut] + f"\nThe secret code word for this text is: {word}. Remember it.\n" + text[cut:]
                return [{"role": "user", "content": head + text + ENDING}]

            guess = int(target * 3.3)
            lo, hi = max(0, int(guess * 0.8)), min(len(hay), int(guess * 1.25))
            if count(build(lo)) > target:
                lo = 0
            if count(build(hi)) <= target:
                raise SystemExit(f"the haystack is too short for {target} tokens")
            while lo < hi:                                     # largest cut whose prompt fits in `target` tokens
                mid = (lo + hi + 1) // 2
                if count(build(mid)) <= target:
                    lo = mid
                else:
                    hi = mid - 1
            messages = build(lo)
            n = count(messages)
            meta = {"kind": "fresh", "target": target, "run": run, "expected_prompt_tokens": n, "needle": word,
                    "needle_depth_pct": 40 if word else None, "max_tokens": a.max_tokens}
            row = measured(label, messages, meta)
            if word:
                row["needle_found_in_first_answer"] = word in row["text"]
            if a.followup and word and run == a.runs:
                follow = messages + [{"role": "assistant", "content": row["text"]},
                                     {"role": "user", "content": FOLLOWUP}]
                fmeta = {"kind": "followup", "target": target, "run": 1, "expected_prompt_tokens": count(follow),
                         "needle": word, "needle_depth_pct": 40, "max_tokens": a.max_tokens,
                         "follows": label}
                frow = measured(f"followup-{target}", follow, fmeta)
                frow["needle_found"] = word in frow["text"][:200]
                (a.out / f"followup-{target}.json").write_text(json.dumps(frow, indent=1, ensure_ascii=False) + "\n")
                print(f"  code word {word}: {'FOUND' if frow['needle_found'] else 'missed'}", flush=True)


if __name__ == "__main__":
    main()
