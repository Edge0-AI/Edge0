#!/usr/bin/env python3
"""
Strata community benchmark harness — standard library only.

Measures prompt processing (prefill) and decode through the local Strata server,
mirroring the method used in bench/results/2026-09-30-community-rtx-5090:

  * synthetic Python-like filler of a requested size,
  * a unique nonce near the start of every request (defeats conversation-prefix
    reuse so every run processes its whole prompt),
  * greedy decoding, fixed output cap,
  * actual token counts taken from the engine's own log line,
  * client-side time-to-first-token and total latency,
  * optional needle-in-a-haystack recall checks.

Outputs bench-output.json, bench-output.md and bench-output.txt next to this file.

Usage (run from the folder that has this file):
  .venv\\Scripts\\python.exe benchmark.py --url http://127.0.0.1:8080/v1/chat/completions \
      --log D:\\AI\\Strata\\Strata-main\\Strata-main\\strata-iq2_xs.log \
      --lengths 4096,32768,128000 --reps 3 --needle-lengths 32768,128000 --needle-depths 10,50,90
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEEDLE = "7-BLUE-MAGNET-4417"
PROMPT_LINE = re.compile(
    r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
    r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)"
)


# ----------------------------------------------------------------------------- prompts
def code_filler(chars: int, seed: int) -> str:
    """Deterministic Python-like filler of roughly `chars` characters."""
    out = []
    n = 0
    size = 0
    while size < chars:
        n += 1
        k = (seed + n) % 97 + 3
        block = (
            f"def transform_{seed:04d}_{n:05d}(value: int, scale: int = {k}) -> int:\n"
            f'    """Scale a value by {k} and fold it into the working range."""\n'
            f"    folded = (value * scale + {n} * 17) % {1000003 + n}\n"
            f"    if folded % {k} == 0:\n"
            f"        return folded // {k} + {n}\n"
            f"    return folded - {n} * scale\n"
            f"\n"
        )
        out.append(block)
        size += len(block)
    return "".join(out)[:chars]


def fit_target(target_tokens: int, budget: int, overshoot: float = 1.03) -> int:
    """Shrink a nominal target so the finished prompt still fits the context.

    The calibration above under-estimates characters per token at large sizes (measured
    2.519 vs 2.454 at 128K), so a 128,000-token label came back as 131,428 prompt tokens
    and the server rejected prompt + max_tokens > 131,072 with HTTP 400. `budget` is the
    largest prompt token count to aim for; `overshoot` is the observed worst-case factor.
    """
    estimated = target_tokens * overshoot + 60
    if estimated <= budget:
        return target_tokens
    fitted = int((budget - 60) / overshoot)
    print(f"[fit] target {target_tokens} would be about {int(estimated)} tokens with this filler; "
          f"using {fitted} to stay within {budget}", flush=True)
    return fitted


def build_prompt(target_tokens: int, seed: int, chars_per_token: float, needle_depth: int | None) -> str:
    """Assemble one benchmark request: instruction + nonce + filler (+ needle)."""
    filler_chars = max(2000, int(target_tokens * chars_per_token))
    body = code_filler(filler_chars, seed)
    nonce = f"{seed:04d}-{int(time.time() * 1000) % 100000:05d}"
    head = (
        f"You are reviewing a Python file. The file starts with the marker NONCE {nonce}.\n"
        f"Explain in three short paragraphs what the functions in this file do.\n\n"
        f"# NONCE {nonce}\n"
    )
    if needle_depth is not None:
        cut = int(len(body) * needle_depth / 100.0)
        # move the cut to the next newline so the file stays syntactically plausible
        cut = body.find("\n", cut)
        cut = len(body) if cut < 0 else cut
        body = body[:cut] + f'\nVAULT_CODE = "{NEEDLE}"  # keep this exact value\n' + body[cut:]
        head = (
            f"You are reviewing a Python file. The file starts with the marker NONCE {nonce}.\n"
            f"Answer the question about the file at the end, using only what the file says.\n\n"
            f"# NONCE {nonce}\n"
        )
        tail = "\n\nQuestion: what is the exact value assigned to VAULT_CODE in this file? " \
               "Answer with the value only, nothing else.\n"
    else:
        tail = "\n\nAnswer in English, three short paragraphs.\n"
    return head + body + tail


# ----------------------------------------------------------------------------- engine log
class EngineLog:
    def __init__(self, path: Path | None):
        self.path = path
        self.offset = 0
        if path and path.exists():
            self.offset = path.stat().st_size

    def new_lines(self) -> list[str]:
        if not self.path or not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self.offset)
                data = fh.read()
                self.offset = fh.tell()
        except OSError:
            return []
        return [ln.strip() for ln in data.replace("\r", "\n").split("\n") if ln.strip()]

    def last_request_stats(self) -> dict | None:
        for line in reversed(self.new_lines()):
            m = PROMPT_LINE.search(line)
            if m:
                return {
                    "engine_prompt_tokens": int(m.group(1)),
                    "reused_tokens": int(m.group(2)),
                    "new_prompt_tokens": int(m.group(3)),
                    "read_ms": int(m.group(4)),
                    "read_tok_s": float(m.group(5)),
                    "generated_tokens": int(m.group(6)),
                    "gen_ms": int(m.group(7)),
                    "decode_tok_s": float(m.group(8)),
                }
        return None


# ----------------------------------------------------------------------------- HTTP
def post_chat(url: str, payload: dict, timeout: int):
    """POST /v1/chat/completions. Returns (ttft_s, total_s, text, reasoning, usage, finish_reason)."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    started = time.time()
    ttft = None
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage = None
    finish = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        ctype = resp.headers.get("Content-Type", "")
        if "text/event-stream" in ctype:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk in ("", "[DONE]"):
                    continue
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices", []):
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or ""
                    think = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if think:
                        reasoning_parts.append(think)
                    if piece:
                        if ttft is None:
                            ttft = time.time() - started
                        text_parts.append(piece)
        else:
            obj = json.loads(resp.read().decode("utf-8", "replace"))
            usage = obj.get("usage")
            for choice in obj.get("choices", []):
                finish = choice.get("finish_reason") or finish
                msg = choice.get("message") or {}
                content = msg.get("content") or ""
                if msg.get("reasoning_content"):
                    reasoning_parts.append(msg["reasoning_content"])
                if content and ttft is None:
                    ttft = time.time() - started
                text_parts.append(content)
    total = time.time() - started
    return ttft, total, "".join(text_parts), "".join(reasoning_parts), usage, finish


def detect_model(url: str) -> str:
    base = url.split("/v1/")[0]
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=15) as resp:
            data = json.loads(resp.read().decode())
        items = data.get("data") or data.get("models") or []
        if items:
            return items[0].get("id") or items[0].get("name") or "strata"
    except Exception:
        pass
    return "strata"


# ----------------------------------------------------------------------------- measurement
def one_request(url: str, model: str, prompt: str, cap: int, log: EngineLog, timeout: int) -> dict:
    base = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": cap,
    }
    # "reasoning_effort": "none" -> the engine renders the prompt without thinking
    # (serve/frontend.py EFFORT map).  Without it the model spends the whole output cap
    # on reasoning_content and message.content stays empty.
    no_think = dict(base, reasoning_effort="none")
    variants = [
        dict(no_think, stream=True, stream_options={"include_usage": True}),
        dict(no_think, stream=True),
        dict(no_think, stream=False),
        dict(base, stream=True, stream_options={"include_usage": True}),
        dict(base, stream=False),
    ]
    row: dict = {"ok": False}
    last_error = None
    for idx, payload in enumerate(variants):
        try:
            ttft, total, text, reasoning, usage, finish = post_chat(url, payload, timeout)
            row.update(
                ok=True,
                payload_variant=idx,
                thinking_off="reasoning_effort" in payload,
                ttft_s=round(ttft, 3) if ttft is not None else None,
                total_s=round(total, 3),
                text=text,
                answer_len=len(text),
                reasoning_len=len(reasoning),
                reasoning_text=reasoning[:2000],
                finish_reason=finish,
            )
            if usage:
                row["usage"] = usage
            last_error = None
            break
        except urllib.error.HTTPError as exc:
            body = exc.read()[:400].decode("utf-8", "replace")
            last_error = f"HTTP {exc.code}: {body}"
            if exc.code not in (400, 404, 415, 422):
                break
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            break
    if last_error:
        row["error"] = last_error
    stats = log.last_request_stats()
    if stats:
        row.update(stats)
        # Bind the log line to this request: the server reports the same prompt token
        # count in the response usage. A mismatch means the line belongs to another
        # request (or none) and the row must not be trusted.
        server_tokens = (row.get("usage") or {}).get("prompt_tokens")
        if server_tokens is None:
            row["pairing_ok"] = None
        else:
            row["pairing_ok"] = abs(int(server_tokens) - int(stats["engine_prompt_tokens"])) <= 1
            if not row["pairing_ok"]:
                row["pairing_note"] = (f"server reported {server_tokens} prompt tokens, the engine log line "
                                       f"has {stats['engine_prompt_tokens']} - the log line may belong to another request")
    return row


def summarise(rows: list[dict]) -> str:
    cols = ["target", "actual prompt tok", "reused", "new read", "prefill tok/s", "read s",
            "gen tok", "decode tok/s", "thinking off", "answer len", "finish", "TTFT s", "total s"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        if not r.get("ok"):
            lines.append("| " + " | ".join([str(r.get("target"))] +
                                           [f"FAILED: {str(r.get('error'))[:60]}"] +
                                           [""] * (len(cols) - 2)) + " |")
            continue
        vals = [
            r.get("target"), r.get("engine_prompt_tokens", "?"), r.get("reused_tokens", "?"),
            r.get("new_prompt_tokens", "?"), r.get("read_tok_s", "?"),
            f"{(r.get('read_ms') or 0) / 1000:.1f}",
            r.get("generated_tokens", "?"), r.get("decode_tok_s", "?"),
            r.get("thinking_off"), r.get("answer_len"), r.get("finish_reason"),
            r.get("ttft_s"), r.get("total_s"),
        ]
        lines.append("| " + " | ".join(str(v) for v in vals) + " |")
    groups: dict[int, list[float]] = {}
    for r in rows:
        target = r.get("target")
        if r.get("ok") and r.get("read_tok_s") and target is not None:
            groups.setdefault(target, []).append(r["read_tok_s"])
    lines.append("")
    for target, vals in sorted(groups.items()):
        dec = [r["decode_tok_s"] for r in rows if r.get("ok") and r.get("target") == target and r.get("decode_tok_s")]
        lines.append(
            f"* {target} prompt tokens: prefill median {statistics.median(vals):.1f} tok/s "
            f"(range {min(vals):.1f}-{max(vals):.1f}, n={len(vals)})"
            + (f"; decode median {statistics.median(dec):.1f} tok/s "
               f"(range {min(dec):.1f}-{max(dec):.1f})" if dec else "")
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--log", default=r"D:\AI\Strata\Strata-main\Strata-main\strata-iq2_xs.log")
    ap.add_argument("--lengths", default="4096,32768,128000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--needle-lengths", default="32768,128000")
    ap.add_argument("--needle-depths", default="10,50,90")
    ap.add_argument("--cap", type=int, default=256)
    ap.add_argument("--max-prompt-tokens", type=int, default=130500,
                    help="nominal targets that would overshoot this many prompt tokens are shrunk")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--wait-ready", type=int, default=0,
                    help="poll /v1/models until the engine answers, up to this many seconds")
    ap.add_argument("--tag", default="", help="suffix for the output files (bench-output-<tag>.json)")
    ap.add_argument("--skip-needles", action="store_true")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    log = EngineLog(Path(args.log) if args.log else None)
    model = args.model or detect_model(args.url)
    print(f"server   : {args.url}")
    print(f"model id : {model}")
    print(f"engine   : {args.log}")
    if args.tag:
        print(f"tag      : {args.tag}")

    if args.wait_ready:
        base = args.url.split("/v1/")[0]
        ready = False
        deadline = time.time() + args.wait_ready
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(base + "/v1/models", timeout=10) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                pass
            print("  waiting for the engine to come up ...", flush=True)
            time.sleep(5)
        if not ready:
            print(f"the engine did not answer within {args.wait_ready} s - stopping.")
            return 2
        print("engine is up\n", flush=True)

    results: dict = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "url": args.url,
        "model": model,
        "engine_log": args.log,
        "cap_tokens": args.cap,
        "reps": args.reps,
        "speed": [],
        "needles": [],
        "notes": [],
    }

    # --- calibration: one small request to learn characters-per-token -----------
    print("\n[calibration] 8,000-char request ...", flush=True)
    cal_prompt = ("You are reviewing a Python file.\n\n# NONCE 0001\n"
                  + code_filler(8000, 1) + "\n\nAnswer in one short paragraph.\n")
    cal = one_request(args.url, model, cal_prompt, 64, log, args.timeout)
    chars = 8000
    if cal.get("ok") and cal.get("engine_prompt_tokens"):
        # prompt tokens includes the instruction wrapper (~40 tok)
        ratio = (cal["engine_prompt_tokens"] - 40) / chars   # tokens per character
        ratio = min(1.5, max(0.15, ratio))                   # sane bounds
        chars_per_token = 1.0 / ratio
        print(f"[calibration] {cal['engine_prompt_tokens']} prompt tokens for {chars} chars "
              f"-> {chars_per_token:.3f} chars/token", flush=True)
    else:
        chars_per_token = 4.0
        results["notes"].append("calibration failed; assumed 4.0 chars/token")
        print(f"[calibration] FAILED ({cal.get('error')}); assuming 4.0 chars/token", flush=True)
    results["calibration"] = {k: v for k, v in cal.items() if k != "text"}
    results["chars_per_token"] = chars_per_token
    print("warming up (one 2,000-token request) ...", flush=True)
    warm = one_request(args.url, model, build_prompt(2000, 7, chars_per_token, None), 32, log, args.timeout)
    results["warmup"] = {k: v for k, v in warm.items() if k != "text"}

    # --- speed runs ------------------------------------------------------------
    seed = 100
    for target in [int(x) for x in args.lengths.split(",") if x.strip()]:
        target = fit_target(target, args.max_prompt_tokens)
        for rep in range(1, args.reps + 1):
            seed += 1
            prompt = build_prompt(target, seed, chars_per_token, None)
            print(f"[speed] target {target} rep {rep}/{args.reps} "
                  f"({len(prompt):,} chars) ...", flush=True)
            row = one_request(args.url, model, prompt, args.cap, log, args.timeout)
            row.update(target=target, rep=rep, chars=len(prompt))
            row.pop("text", None)
            results["speed"].append(row)
            if row.get("ok"):
                print(f"        -> prompt {row.get('engine_prompt_tokens')} tok, "
                      f"reused {row.get('reused_tokens')}, prefill {row.get('read_tok_s')} tok/s, "
                      f"decode {row.get('decode_tok_s')} tok/s, TTFT {row.get('ttft_s')} s", flush=True)
            else:
                print(f"        -> FAILED: {row.get('error')}", flush=True)

    # --- needle checks ---------------------------------------------------------
    if not args.skip_needles:
        for target in [int(x) for x in args.needle_lengths.split(",") if x.strip()]:
            target = fit_target(target, args.max_prompt_tokens)
            for depth in [int(x) for x in args.needle_depths.split(",") if x.strip()]:
                seed += 1
                prompt = build_prompt(target, seed, chars_per_token, depth)
                print(f"[needle] target {target} depth {depth}% ...", flush=True)
                row = one_request(args.url, model, prompt, 32, log, args.timeout)
                found = NEEDLE in ((row.get("text") or "") + (row.get("reasoning_text") or ""))
                row.update(target=target, depth=depth, needle_found=found, chars=len(prompt))
                results["needles"].append(row)
                print(f"        -> needle {'FOUND' if found else 'MISSED'}; "
                      f"prompt {row.get('engine_prompt_tokens')} tok, "
                      f"prefill {row.get('read_tok_s')} tok/s", flush=True)

    results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["tag"] = args.tag
    suffix = f"-{args.tag}" if args.tag else ""
    out_json = HERE / f"bench-output{suffix}.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    table = summarise(results["speed"])
    needle_lines = ["", "Needle checks (exact string recall):"]
    for r in results["needles"]:
        needle_lines.append(
            f"* {r.get('target')} tokens, depth {r.get('depth')}%: "
            f"{'FOUND' if r.get('needle_found') else 'MISSED'} "
            f"(prompt {r.get('engine_prompt_tokens')} tok, prefill {r.get('read_tok_s')} tok/s)"
        )
    report = (
        f"# Strata benchmark output\n\n"
        f"started {results['started']}, finished {results['finished']}\n"
        f"server {args.url}, model id `{model}`, output cap {args.cap} tokens, "
        f"{args.reps} reps per length\n\n## Speed\n\n{table}\n"
        + ("\n".join(needle_lines) if results["needles"] else "") + "\n"
    )
    (HERE / f"bench-output{suffix}.md").write_text(report, encoding="utf-8")
    (HERE / f"bench-output{suffix}.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"saved: {out_json}")
    print(f"saved: {HERE / f'bench-output{suffix}.md'}")
    print("\nDONE - tell Hermes the run is finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
