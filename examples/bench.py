#!/usr/bin/env python3
"""edge0 speed + memory benchmark (any tier, bench_v7-equivalent protocol).

Runs the same measurement shape as the deployment's bench: prefill ->
sampled warmup steps -> timed sampled decode -> report tok/s and MLX
peak active memory.  Works for both tiers; the tier is auto-detected
from the checkpoint (or forced via the model name).

Usage:
    python examples/bench.py /path/to/model [--ntok 200] [--warmup 10]
    python examples/bench.py edge0-35b        # via $EDGE0_35B_MODEL

Env (sampling knobs, defaults follow the tier's GenerationConfig):
    BENCH_TEMP / BENCH_PROMPT / BENCH_SEED
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from edge0 import AutoEngine
from edge0.backends import core
from edge0.sampling import sample

PROMPTS = {
    "edge0-35b": "什么是混合专家模型（MoE）？它和普通 Transformer 有什么区别？",
    "edge0-10b": "9.11 和 9.8 哪个大？请仔细比较。",
}
DEFAULT_PROMPT = "什么是混合专家模型（MoE）？简单介绍一下。"


def _prompt_for(engine) -> str:
    return os.environ.get("BENCH_PROMPT") or PROMPTS.get(engine.name,
                                                         DEFAULT_PROMPT)


def run_bench(engine, ntok: int, warmup: int) -> dict:
    tok = engine._tok
    prompt = _prompt_for(engine)
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
        # warmup: sampled, untimed (first tokens are dominated by expert
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
        print(f"prompt={len(ids)} tok  prefill={t_prefill:.2f}s  "
              f"decode={len(out)}/{t_decode:.2f}s  "
              f"tok/s={results[-1]['tok_s']:.1f}  "
              f"peak_active={peak_gib:.2f} GiB", flush=True)
    mean_ts = sum(r["tok_s"] for r in results) / len(results)
    max_peak = max(r["peak_gib"] for r in results)
    print(f"[bench] mean tok/s={mean_ts:.1f}  peak_active<={max_peak:.2f} GiB",
          flush=True)
    return {"runs": results, "mean_tok_s": mean_ts, "peak_gib": max_peak}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="checkpoint dir or tier name")
    ap.add_argument("--ntok", type=int,
                    default=int(os.environ.get("BENCH_NTOK", "200")))
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    # tier name -> checkpoint dir via $EDGE0_<TIER>_MODEL (cli parity)
    model = args.model
    name = None
    if model.startswith("edge0-"):
        from edge0.cli import TIER_ENV
        env = os.environ.get(TIER_ENV.get(model, ""))
        if not env:
            raise SystemExit(
                f"set {TIER_ENV.get(model, 'EDGE0_<TIER>_MODEL')} to the "
                f"checkpoint directory for {model}, or pass the dir itself")
        model = env
    engine = AutoEngine.from_pretrained(model, name=name)
    print(f"[bench] tier={engine.name} ntok={args.ntok} warmup={args.warmup}",
          flush=True)
    run_bench(engine, args.ntok, args.warmup)
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
