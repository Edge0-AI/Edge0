"""Minimal edge0 API walkthrough — the same path ``edge0 demo`` runs.

Install:
    python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'

Run (checkpoint path or registered tier name):
    .venv/bin/python examples/demo.py --model-dir /path/to/qwen35/model
    .venv/bin/python examples/demo.py --model edge0-10b

Three lines do everything: build the engine (prerouter + LoRA + SSD
offload installed automatically from the tier config), template the
prompt, generate.
"""

from __future__ import annotations

import argparse
import sys

from edge0 import AutoEngine
from edge0.server.chat import ChatMessage, ChatRequest, ChatSession


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None,
                    help="registered tier name, e.g. edge0-35b")
    ap.add_argument("--model-dir", default=None,
                    help="checkpoint directory (tier auto-detected)")
    ap.add_argument("--prompt",
                    default="Hello! Write one short sentence about Zhuhai.")
    ap.add_argument("--max-new", type=int, default=24)
    args = ap.parse_args()

    engine = AutoEngine.from_pretrained(args.model_dir, name=args.model)
    req = ChatRequest(
        model=engine.name,
        messages=[ChatMessage(role="user", content=args.prompt)],
        max_tokens=args.max_new,
    )
    tokens, meta = ChatSession(engine, req).run()
    print(f"user : {args.prompt}")
    print(f"edge0: {engine._tok.decode(tokens)}")
    print(f"# {len(tokens)} tokens in {meta['wall_s']}s", file=sys.stderr)
    engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
