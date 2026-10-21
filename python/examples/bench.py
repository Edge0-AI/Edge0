#!/usr/bin/env python3
"""edge0 speed + memory benchmark (any tier).

Runs the same measurement shape as the deployment's bench: prefill ->
greedy warmup steps -> timed sampled decode -> report tok/s and MLX
allocator peak memory (`mlx.core.get_peak_memory`).  Works for both
tiers; the tier is auto-detected from the checkpoint (or forced via the
model name).

Usage:
    python examples/bench.py /path/to/model [--ntok 200] [--warmup 10]
    python examples/bench.py edge0-35b        # via $EDGE0_35B_MODEL
    BENCH_LONG=1 python examples/bench.py edge0-35b   # ~3k-token prefill
    python examples/bench.py edge0-35b --submission --reporter @you
    python examples/bench.py edge0-35b --submission --json > submission.json

Env (sampling knobs, defaults follow the tier's GenerationConfig):
    BENCH_TEMP / BENCH_PROMPT / BENCH_SEED
    BENCH_LONG=1  use a ~3k-token synthetic prompt so the prefill timing
                  is meaningful (short prompts only measure fixed
                  overhead; set BENCH_PROMPT for your own long text)
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext, redirect_stdout
from importlib.metadata import PackageNotFoundError, version
import json
import os
import platform
import subprocess
import sys
import time

PROMPTS = {
    "edge0-35b": "什么是混合专家模型（MoE）？它和普通 Transformer 有什么区别？",
    "edge0-8b": "9.11 和 9.8 哪个大？请仔细比较。",
}
DEFAULT_PROMPT = "什么是混合专家模型（MoE）？简单介绍一下。"

# Synthetic long-context prompt (~3k tokens after tokenization): two
# non-repetitive paragraphs alternating 40 times, prefixed with a task
# instruction.  Prefill timing on a few dozen tokens only measures fixed
# overhead; a long prompt exercises the real chunked prefill path
# (per-layer expert loading, KV growth).
_PARA_ZH = (
    "流式推理框架的设计要点在于把磁盘带宽、缓存层级与计算核心三者重叠"
    "起来。混合专家模型每一层只在少数专家上激活，路由器给出的稀疏选择"
    "恰好为按需加载提供了天然的批粒度。若能在前一步的隐状态基础上预判"
    "下一步的专家集合，固态硬盘的读取延迟就能被计算完全掩盖，这对边缘"
    "设备上的大模型部署具有直接意义。")
_PARA_EN = (
    "The design of a streaming inference framework hinges on overlapping "
    "disk bandwidth, cache hierarchy, and compute. A mixture-of-experts "
    "layer activates only a handful of experts per token, and the router "
    "sparsity provides a natural granularity for on-demand loading. If "
    "the expert set of the next step can be predicted from the previous "
    "hidden state, SSD read latency hides behind compute entirely, which "
    "matters for large-model deployment on edge hardware.")


def _long_prompt() -> str:
    paras = [_PARA_ZH if i % 2 == 0 else _PARA_EN for i in range(40)]
    return "请仔细阅读以下材料并用一句话总结其核心思想：" + " ".join(paras)


def _prompt_for(engine) -> str:
    if os.environ.get("BENCH_PROMPT"):
        return os.environ["BENCH_PROMPT"]
    if os.environ.get("BENCH_LONG") == "1":
        return _long_prompt()
    return PROMPTS.get(engine.name, DEFAULT_PROMPT)


def run_bench(engine, ntok: int, warmup: int, *, submission: bool = False) -> dict:
    from edge0.backends import core
    from edge0.sampling import sample

    tok = engine._tok
    prompt = _long_prompt() if submission else _prompt_for(engine)
    if hasattr(engine, "encode_chat"):
        ids = engine.encode_chat(
            [{"role": "user", "content": prompt}], think=False)
    else:
        ids = list(tok(tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False))["input_ids"])

    gen = getattr(engine.cfg, "gen", None)
    temperature = float(os.environ.get(
        "BENCH_TEMP", getattr(gen, "temperature", 0.7)))
    top_k = getattr(gen, "top_k", 64)
    top_p = getattr(gen, "top_p", 0.95)
    rp = getattr(gen, "repetition_penalty", 1.0)
    seed = os.environ.get("BENCH_SEED")

    results = []
    for run in range(2):
        engine.reset()
        core.reset_peak_memory()
        t0 = time.perf_counter()
        engine.prefill(ids)
        t_prefill = time.perf_counter() - t0
        logits = engine.next_logits()
        if seed is not None:
            core.random.seed(int(seed))

        history = list(ids)
        # warmup: greedy, untimed (first tokens are dominated by expert
        # cold-starts; the timed window is the steady state)
        for _ in range(warmup):
            tid = int(core.argmax(logits, axis=-1).item())
            history.append(tid)
            logits = engine.step(tid)
        out: list[int] = []
        t0 = time.perf_counter()
        for _ in range(ntok):
            tid = sample(logits, temperature=temperature, top_k=top_k,
                         top_p=top_p, repetition_penalty=rp,
                         history=history, seed=None)
            history.append(tid)
            out.append(tid)
            logits = engine.step(tid)
        t_decode = time.perf_counter() - t0
        peak_gib = core.get_peak_memory() / (1024 ** 3)
        results.append(dict(prefill_s=t_prefill, decode_s=t_decode,
                            ntok=len(out), peak_gib=peak_gib,
                            tok_s=len(out) / t_decode if t_decode else 0.0))
        pf_tps = (len(ids) / t_prefill) if t_prefill else 0.0
        if submission:
            results[-1]["prefill_tok_s"] = pf_tps
        print(f"prompt={len(ids)} tok  prefill={t_prefill:.2f}s "
              f"({pf_tps:.0f} tok/s)  "
              f"decode={len(out)}/{t_decode:.2f}s  "
              f"tok/s={results[-1]['tok_s']:.1f}  "
              f"mlx_allocator_peak={peak_gib:.2f} GiB", flush=True)
    mean_ts = sum(r["tok_s"] for r in results) / len(results)
    max_peak = max(r["peak_gib"] for r in results)
    print(f"[bench] mean tok/s={mean_ts:.1f}  mlx_allocator_peak<={max_peak:.2f} GiB",
          flush=True)
    report = {"runs": results, "mean_tok_s": mean_ts, "peak_gib": max_peak}
    if submission:
        report["mean_prefill_tok_s"] = sum(r["prefill_tok_s"] for r in results) / len(results)
    return report


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True,
                              text=True, check=True, timeout=3).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _hardware_metadata() -> dict:
    try:
        mlx_version = version("mlx")
    except PackageNotFoundError:
        mlx_version = ""
    ram_gb = None
    device = chip = ""
    if platform.system() == "Darwin":
        device = _sysctl("hw.model")
        chip = _sysctl("machdep.cpu.brand_string")
        try:
            memory = int(_sysctl("hw.memsize"))
            if memory > 0:
                ram_gb = memory / (1024 ** 3)
        except ValueError:
            pass
    if platform.system() == "Darwin":
        os_name = "macOS " + platform.mac_ver()[0]
    else:
        os_name = platform.system() + " " + platform.release()
    return {"device": device, "chip": chip, "ram_gb": ram_gb,
            "os": os_name.strip(), "mlx_version": mlx_version}


def _single_line(value) -> str:
    return " ".join(str(value).split())


def _submission_block(tier: str, report: dict, args) -> dict:
    metadata = _hardware_metadata()
    notes = [args.notes, "BENCH_LONG; 2 timed runs; "
             f"{args.warmup} greedy warmup steps/run; {args.ntok} sampled decode steps/run; "
             "memory=MLX allocator peak (not process RSS)"]
    for key in ("BENCH_TEMP", "BENCH_SEED"):
        if key in os.environ:
            notes.append(f"{key}={os.environ[key]}")
    return {
        "tier": tier,
        "runtime": "python-mlx",
        "device": _single_line(metadata.get("device", "")),
        "chip": _single_line(metadata.get("chip", "")),
        "ram_gb": metadata.get("ram_gb"),
        "os": _single_line(metadata.get("os", "")),
        "storage": _single_line(args.storage),
        "mlx_version": _single_line(metadata.get("mlx_version", "")),
        "decode_tok_s": report["mean_tok_s"],
        "prefill_tok_s": report["mean_prefill_tok_s"],
        "peak_active_gib": report["peak_gib"],
        "on_battery": args.on_battery,
        "reporter": _single_line(args.reporter),
        "notes": _single_line("; ".join(note for note in notes if note)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="checkpoint dir or tier name")
    ap.add_argument("--ntok", type=int,
                    default=int(os.environ.get("BENCH_NTOK", "200")))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--submission", action="store_true",
                    help="use BENCH_LONG and print the community submission block")
    ap.add_argument("--json", action="store_true",
                    help="with --submission, emit only JSON on stdout; progress goes to stderr")
    ap.add_argument("--reporter", default="", help="GitHub handle for the submission")
    ap.add_argument("--storage", default="", help="storage containing the checkpoint (not auto-detected)")
    ap.add_argument("--on-battery", choices=("yes", "no"), default="",
                    help="power state during measurement (not auto-detected)")
    ap.add_argument("--notes", default="", help="background load, throttling or other context")
    args = ap.parse_args()
    if args.json and not args.submission:
        ap.error("--json requires --submission")
    if args.ntok <= 0 or args.warmup < 0:
        ap.error("--ntok must be positive and --warmup must be nonnegative")
    if args.submission and os.environ.get("BENCH_PROMPT"):
        ap.error("--submission requires the standard long prompt; unset BENCH_PROMPT")
    # tier name -> checkpoint dir via $EDGE0_<TIER>_MODEL (cli parity)
    model = args.model
    name = None
    with redirect_stdout(sys.stderr) if args.json else nullcontext():
        from edge0 import AutoEngine

        if model.startswith("edge0-"):
            from edge0.cli import TIER_ENV
            env = os.environ.get(TIER_ENV.get(model, ""))
            if not env:
                raise SystemExit(
                    f"set {TIER_ENV.get(model, 'EDGE0_<TIER>_MODEL')} to the "
                    f"checkpoint directory for {model}, or pass the dir itself")
            model = env
        engine = AutoEngine.from_pretrained(model, name=name)
        try:
            print(f"[bench] tier={engine.name} ntok={args.ntok} warmup={args.warmup}",
                  flush=True)
            if args.submission:
                report = run_bench(engine, args.ntok, args.warmup, submission=True)
            else:
                report = run_bench(engine, args.ntok, args.warmup)
            tier = engine.name
        finally:
            engine.close()
    if args.submission:
        block = _submission_block(tier, report, args)
        payload = json.dumps(block, ensure_ascii=False, allow_nan=False)
        if args.json:
            print(payload)
        else:
            print()
            for key, value in block.items():
                if value is None:
                    text = ""
                elif isinstance(value, float):
                    text = f"{value:.2f}"
                else:
                    text = str(value)
                print(f"{key + ':':<18} {text}".rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
