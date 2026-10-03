"""End-to-end smoke: real checkpoints, real adapters.

Runs both shipped tiers through the full prerouter + lora + staged-decode
path and cross-checks staged vs exact math by comparing GREEDY token
sequences (deterministic, so any staged/exact divergence shows up as a
different sequence).

Usage:
    .venv/bin/python scripts/e2e_smoke.py [--qwen-dir DIR] [--ling-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from edge0 import AutoEngine
from edge0.backends import io
from edge0.config import GenerationConfig
from edge0.streaming.options import LayerOptions

QWEN_DEFAULT = os.environ.get("EDGE0_35B_MODEL", "/path/to/qwen35/model")
LING_DEFAULT = os.environ.get("EDGE0_8B_MODEL", "/path/to/ling/model")


def _available(label: str, model_dir: str) -> bool:
    if os.path.isdir(model_dir):
        return True
    print(f"[skip] {label} checkpoint not found: {model_dir}")
    return False


def _prompt_ids(model_dir: str, text: str) -> list[int]:
    tok = io.load_tokenizer(model_dir)
    return list(tok(text)["input_ids"])


def run(name: str, model_dir: str, prompt_ids: list[int], max_new: int,
        options: LayerOptions | None = None, label: str = ""):
    t0 = time.time()
    kwargs = {"name": name, "model_dir": model_dir}
    if options is not None:
        kwargs["options"] = options  # None would override the tier default
    eng = AutoEngine.from_pretrained(**kwargs)
    t_load = time.time() - t0
    t0 = time.time()
    out = eng.generate(
        prompt_ids,
        GenerationConfig(temperature=0.0, top_k=None, top_p=None,
                         max_new_tokens=max_new),
        max_new_tokens=max_new)
    t_gen = time.time() - t0
    eng.close()
    print(f"[{name}{label}] load={t_load:.1f}s gen={t_gen:.1f}s "
          f"({max_new / t_gen:.1f} tok/s) tokens={out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-dir", default=QWEN_DEFAULT)
    ap.add_argument("--ling-dir", default=LING_DEFAULT)
    ap.add_argument("--max-new", type=int, default=16)
    args = ap.parse_args()

    ran = False
    if _available("edge0-35b", args.qwen_dir):
        qwen_prompt = _prompt_ids(args.qwen_dir, "Hello!")
        staged = run("edge0-35b", args.qwen_dir, qwen_prompt, args.max_new,
                     label="[staged]")
        exact = run("edge0-35b", args.qwen_dir, qwen_prompt, args.max_new,
                    options=LayerOptions(staged=False, staged_replace=False,
                                         full_layer_prefill=False,
                                         hot_per_layer=0, use_compile=False,
                                         history_prefetch=False),
                    label="[exact]")
        if staged != exact:
            print("MISMATCH: staged vs exact greedy sequences differ "
                  f"(staged={staged} exact={exact})")
            return 1
        print("qwen staged == exact greedy: OK")
        ran = True

    if _available("edge0-8b", args.ling_dir):
        ling_prompt = _prompt_ids(args.ling_dir, "你好")
        run("edge0-8b", args.ling_dir, ling_prompt, args.max_new,
            label="[staged]")
        ran = True

    if not ran:
        print("edge0 e2e smoke: no checkpoints available", file=sys.stderr)
        return 1
    print("edge0 e2e smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
