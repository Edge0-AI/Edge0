#!/usr/bin/env python3
"""Strata benchmark harness (contribution plan, PR 1 step 1.2).

Follows the upstream method (bench/results/2026-09-28-speed-0114): one code-agent prompt per
length, greedy, 256 generated tokens, MTP as configured by setup. Every run gets a unique
prefix, so conversation checkpoints cannot skip the prompt. Numbers come from the engine log
lines the owner asks for in issue #74. /metrics and API timing are stored as cross-checks.

Stdlib only. Examples:
    ./strata-bench.py --label IQ3_XXS-1x3090 --log ./strata-iq3_xxs.log
  ./strata-bench.py --label Coder-2x3090-auto --contexts 16k,28k --mode needle
  ./strata-bench.py --url http://127.0.0.1:9292 --model flash-next-strata --label via-llama-swap
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

NOMINAL = {"1k": 1024, "2.7k": 2765, "4k": 4096, "8k": 8192, "16k": 16384, "28k": 28672,
           "32k": 32768, "64k": 65536, "128k": 131072, "262k": 262144}
CODE_EXT = {".py", ".cpp", ".cc", ".h", ".hpp", ".cu", ".cuh", ".js", ".md", ".txt", ".cmake"}
SKIP_DIRS = {".git", ".venv", "build", "engine", "third_party", "node_modules", "models",
             "packs", "mtp", "bench", "data", "__pycache__", "Strata-data"}
LOG_PATTERNS = [
    re.compile(r"decode expert cache hit rate", re.I),
    re.compile(r"\bPCIe\b"),
    re.compile(r"prompt .*tok/s.*generated .*tok/s", re.I),
    re.compile(r"reus|checkpoint", re.I),
    re.compile(r"layer split", re.I),
]
TOKS_PER_S = re.compile(r"([\d][\d,]*(?:\.\d+)?)\s*tok(?:ens)?/s", re.I)


def load_corpus(dirs):
    parts = []
    for d in dirs:
        root = Path(d)
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in CODE_EXT or not p.is_file():
                continue
            if any(s in SKIP_DIRS for s in p.relative_to(root).parts[:-1]):
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                parts.append(f"### {p.relative_to(root).as_posix()}\n```\n{text}\n```\n")
    if not parts:
        sys.exit(f"no source files found in {dirs}")
    return parts


def build_text(parts, n_chars):
    out, size, i, repeated = [], 0, 0, False
    while size < n_chars:
        if i and i % len(parts) == 0:
            repeated = True
        chunk = parts[i % len(parts)]
        out.append(chunk)
        size += len(chunk)
        i += 1
    return "".join(out)[:n_chars], repeated


def post_stream(url, payload, api_key, timeout):
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key or 'none'}"})
    t0 = time.perf_counter()
    t_first, text, reasoning, usage, chunks = None, [], [], {}, 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = delta.get("content") or ""
                rpiece = delta.get("reasoning_content") or ""
                if (piece or rpiece) and t_first is None:
                    t_first = time.perf_counter()
                if piece or rpiece:
                    chunks += 1
                text.append(piece)
                reasoning.append(rpiece)
    t_end = time.perf_counter()
    return {"t_start": t0, "t_first": t_first or t_end, "t_end": t_end, "text": "".join(text),
            "reasoning": "".join(reasoning), "usage": usage, "chunks": chunks}


def get_json(url, path, api_key, timeout=10):
    try:
        req = urllib.request.Request(f"{url}{path}", headers={"Authorization": f"Bearer {api_key or 'none'}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="ignore")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body
    except (urllib.error.URLError, TimeoutError) as e:
        return {"error": str(e)}


def wait_healthy(url, api_key, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        h = get_json(url, "/health", api_key, timeout=5)
        if not (isinstance(h, dict) and "error" in h):
            return True
        time.sleep(5)
    return False


def log_offset(log):
    return log.stat().st_size if log and log.exists() else 0


def log_lines_since(log, offset):
    if not log or not log.exists():
        return []
    time.sleep(0.5)
    with log.open("r", encoding="utf-8", errors="ignore") as f:
        f.seek(offset)
        new = f.read().splitlines()
    return [ln for ln in new if any(p.search(ln) for p in LOG_PATTERNS)]


def engine_numbers(lines):
    """Best effort: the last 'prompt ... tok/s ... generated ... tok/s' line -> (prompt, decode)."""
    for ln in reversed(lines):
        if LOG_PATTERNS[2].search(ln):
            vals = [float(v.replace(",", "")) for v in TOKS_PER_S.findall(ln)]
            if len(vals) >= 2:
                return vals[0], vals[1]
    return None, None


def target_tokens(key, max_tokens):
    n = NOMINAL[key]
    # leave room for the answer, like upstream's 259,943-token "262K" prompt.
    # NAT-local fix: at >=128k the tokenizer drifts over the cap (129,792 target
    # -> HTTP 400 on a 131,072-context server), so keep a 2048-token margin.
    margin = 2048 if n >= 131072 else 1024
    return n - max_tokens - margin if n >= 32768 else n


def make_messages(mode, parts, n_chars, run_id):
    body, repeated = build_text(parts, n_chars)
    system = f"[bench run {run_id}] You are a coding agent working in the repository shown below."
    if mode == "needle":
        word = uuid.uuid4().hex[:8].upper()
        mid = len(body) // 2
        body = f"{body[:mid]}\n# NOTE: the secret code word is {word}.\n{body[mid:]}"
        task = "What is the secret code word mentioned in the files above? Answer with the word only."
    else:
        word = None
        task = ("Task: explain what the last file above does, then propose one concrete improvement "
                "as a unified diff.")
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": f"Repository files:\n\n{body}\n\n{task}"}]
    return msgs, word, repeated


def run_once(a, parts, key, run, cpt):
    tgt = target_tokens(key, a.max_tokens)
    run_id = uuid.uuid4().hex[:12]
    msgs, word, repeated = make_messages(a.mode, parts, int(tgt * cpt), run_id)
    payload = {"model": a.model, "messages": msgs, "max_tokens": a.max_tokens, "stream": True,
               "stream_options": {"include_usage": True}, "temperature": 0,
               "reasoning_effort": a.reasoning_effort}
    off = log_offset(a.log)
    rec = {"label": a.label, "mode": a.mode, "ctx": key, "target_tokens": tgt, "run": run,
           "run_id": run_id, "corpus_repeated": repeated, "time": datetime.now().isoformat(timespec="seconds")}
    for attempt in range(4):
        try:
            r = post_stream(a.url, payload, a.api_key, a.timeout)
            break
        except urllib.error.HTTPError as e:
            # The server reports the exact prompt token count in the 400 body;
            # rescale the char budget and retry until it fits (tokenizer drift
            # per corpus slice is ~1%, so a single rescale can still overshoot).
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                pass
            mtok = re.search(r"prompt \((\d+) tokens\)", body)
            if e.code == 400 and mtok and attempt < 3:
                allowed = a.max_context - a.max_tokens - 64
                scale = allowed / int(mtok.group(1))
                if scale >= 1:
                    rec["error"] = f"HTTP 400: prompt {mtok.group(1)} tokens, scale {scale:.2f} >= 1"
                    return rec, cpt
                # rebuild messages with the scaled char budget
                scaled_chars = int(len(msgs[1]["content"]) * scale)
                msgs, word, repeated = make_messages(a.mode, parts, scaled_chars, run_id)
                payload["messages"] = msgs
                continue
            rec["error"] = f"HTTP {e.code}: {body[:200]}"
            return rec, cpt
        except Exception as e:  # noqa: BLE001 - record and continue with the next cell
            rec["error"] = str(e)[:300]
            return rec, cpt
    pt = r["usage"].get("prompt_tokens")
    ct = r["usage"].get("completion_tokens") or r["chunks"]
    ttft = r["t_first"] - r["t_start"]
    dec_t = r["t_end"] - r["t_first"]
    lines = log_lines_since(a.log, off)
    e_prompt, e_decode = engine_numbers(lines)
    rec.update({
        "prompt_tokens": pt, "completion_tokens": ct,
        "ttft_s": round(ttft, 3), "e2e_s": round(r["t_end"] - r["t_start"], 3),
        "api_prompt_tps": round(pt / ttft, 1) if pt and ttft > 0 else None,
        "api_decode_tps": round((ct - 1) / dec_t, 1) if ct and ct > 1 and dec_t > 0 else None,
        "engine_prompt_tps": e_prompt, "engine_decode_tps": e_decode,
        "log_lines": lines, "answer_head": r["text"][:300],
    })
    if word:
        rec["needle_found"] = word in r["text"].upper() or word in r["reasoning"].upper()
    if a.mode == "two-turn":
        follow = msgs + [{"role": "assistant", "content": r["text"]},
                         {"role": "user", "content": "Summarize your answer in one sentence."}]
        off2 = log_offset(a.log)
        r2 = post_stream(a.url, {**payload, "messages": follow, "max_tokens": 64}, a.api_key, a.timeout)
        rec["turn2_ttft_s"] = round(r2["t_first"] - r2["t_start"], 3)
        rec["turn2_prompt_tokens"] = r2["usage"].get("prompt_tokens")
        rec["turn2_log_lines"] = log_lines_since(a.log, off2)
    if pt:
        cpt = (int(tgt * cpt) + 400) / pt  # refine chars per token for the next cells
    return rec, cpt


def write_md(path, recs, label):
    def med(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.median(vals), 1) if vals else None

    rows = ["| ctx | prompt tokens | runs | TTFT s (med) | prompt tok/s engine / API (med) "
            "| decode tok/s engine / API (run 1) | decode tok/s engine / API (med) | needle |",
            "|---|---|---|---|---|---|---|---|"]
    for key in dict.fromkeys(r["ctx"] for r in recs):
        rs = [r for r in recs if r["ctx"] == key and "error" not in r]
        errs = sum(1 for r in recs if r["ctx"] == key and "error" in r)
        if not rs:
            rows.append(f"| {key} | - | 0 ({errs} errors) | | | | | |")
            continue
        first = rs[0]
        needle = "" if rs[0].get("needle_found") is None else f"{sum(r['needle_found'] for r in rs)}/{len(rs)}"
        rows.append(
            f"| {key} | {rs[0]['prompt_tokens']} | {len(rs)}{f' ({errs} err)' if errs else ''} "
            f"| {med(r['ttft_s'] for r in rs)} "
            f"| {med(r['engine_prompt_tps'] for r in rs)} / {med(r['api_prompt_tps'] for r in rs)} "
            f"| {first['engine_decode_tps']} / {first['api_decode_tps']} "
            f"| {med(r['engine_decode_tps'] for r in rs)} / {med(r['api_decode_tps'] for r in rs)} | {needle} |")
    path.write_text(f"# {label}\n\n" + "\n".join(rows) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", default="strata", help="model name sent (llama-swap: the model id)")
    ap.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY", ""))
    ap.add_argument("--label", required=True, help="e.g. IQ3_XXS-1x3090-calibrated")
    ap.add_argument("--contexts", default="1k,4k,32k,128k", help=f"comma list of {','.join(NOMINAL)}")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--mode", choices=["speed", "needle", "two-turn"], default="speed")
    ap.add_argument("--source-dir", action="append", default=None,
                    help="code corpus for the prompts (default: current directory, the Strata repo)")
    ap.add_argument("--log", type=Path, default=None,
                    help="engine log, e.g. ./strata-iq3_xxs.log")
    ap.add_argument("--out", type=Path, default=Path(f"./benchmark-output/{datetime.now():%Y-%m-%d}"))
    ap.add_argument("--timeout", type=float, default=1800, help="per request, seconds (262K prefill takes minutes)")
    ap.add_argument("--chars-per-token", type=float, default=3.4, help="start value, refined after each run")
    ap.add_argument("--max-context", type=int, default=131072, help="server context cap for the 400-rescale retry")
    a = ap.parse_args()

    keys = [k.strip().lower() for k in a.contexts.split(",") if k.strip()]
    bad = [k for k in keys if k not in NOMINAL]
    if bad:
        sys.exit(f"unknown context(s): {bad}")
    parts = load_corpus(a.source_dir or ["."])
    if a.log and not a.log.exists():
        print(f"warning: log {a.log} not found; engine numbers will be empty", file=sys.stderr)
    if not wait_healthy(a.url, a.api_key, 900):
        sys.exit(f"{a.url}/health not reachable")

    out = a.out / a.label
    (out / "data").mkdir(parents=True, exist_ok=True)
    recs, cpt = [], a.chars_per_token
    for key in keys:
        for run in range(1, a.repeats + 1):
            rec, cpt = run_once(a, parts, key, run, cpt)
            rec["metrics_after"] = get_json(a.url, "/metrics", a.api_key)
            recs.append(rec)
            (out / "data" / f"{a.mode}-{key}-{run}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
            status = rec.get("error") or (f"prompt {rec['prompt_tokens']} tok, TTFT {rec['ttft_s']} s, "
                                          f"decode engine {rec['engine_decode_tps']} / API {rec['api_decode_tps']} tok/s")
            print(f"[{a.label}] {a.mode} {key} run {run}: {status}", flush=True)

    matrix_path = out / "matrix.json"
    merged = {}
    if matrix_path.exists():
        for record in json.loads(matrix_path.read_text(encoding="utf-8")):
            merged[(record["mode"], record["ctx"], record["run"])] = record
    for record in recs:
        key = (record["mode"], record["ctx"], record["run"])
        merged[key] = {k: v for k, v in record.items() if k != "metrics_after"}
    combined = list(merged.values())
    matrix_path.write_text(json.dumps(combined, indent=1), encoding="utf-8")
    write_md(out / "matrix.md", combined, f"{a.label} ({a.mode})")
    print(f"written: {out}/matrix.md, matrix.json, data/")


if __name__ == "__main__":
    main()
