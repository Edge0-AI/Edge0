"""Complete end-to-end example: generate an English introduction to
Zhuhai with edge0-35b (prerouter + LoRA + SSD offload, default tier
profile — staged decode on, K=4).

Usage:
    .venv/bin/python scripts/zhuhai_example.py [--max-new N] [--temp T]

The example walks through the full pipeline a user of the framework
would hit:

1. load the tokenizer and the engine (real checkpoint + adapters),
2. tokenize a prompt,
3. prefill (whole-layer load-drop, hot pins),
4. greedy preview of the first token,
5. sampled generation with the tier's default GenerationConfig,
6. decode back to text.
"""

from __future__ import annotations

import argparse
import sys
import time

from edge0 import AutoEngine
from edge0.backends import io
from edge0.config import GenerationConfig


PROMPT = (
    "Write a short introduction to Zhuhai, a city in southern China, "
    "in English. Keep it to 3-4 sentences."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--thinking", action="store_true",
                    help="enable the <think> block (enable_thinking=True)")
    args = ap.parse_args()

    model_dir = "/Users/linyu/Documents/qwen35-v7-deploy/model"
    print(f"[1] loading tokenizer + edge0-35b from {model_dir}", flush=True)
    tok = io.load_tokenizer(model_dir)
    t0 = time.time()
    eng = AutoEngine.from_pretrained(name="edge0-35b", model_dir=model_dir)
    print(f"[2] engine built in {time.time() - t0:.1f}s", flush=True)

    ids = list(tok(PROMPT)["input_ids"])
    print(f"[3] prompt: {len(ids)} tokens: {ids[:12]} ...", flush=True)

    # Deployment parity (bench_v7.py / serve.py): build the prompt with
    # the chat template. enable_thinking=False closes an empty <think>
    # block so the model answers directly; --thinking leaves the block
    # open so the model emits reasoning first, then </think> + the answer.
    chat = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True, tokenize=True,
        enable_thinking=args.thinking)
    chat = list(chat.input_ids)
    print(f"[3b] chat template: {len(chat)} tokens", flush=True)

    # Greedy preview of the first token (deterministic).
    t0 = time.time()
    eng.prefill(chat)
    l0 = eng.next_logits()
    first = int(l0.argmax().item())
    print(f"[4] prefill in {time.time() - t0:.1f}s; first greedy token "
          f"{first} = {tok.decode([first])!r}", flush=True)
    eng.reset()

    # Full generation with the tier default sampler (temp 0.7, top_p 0.95).
    gc = GenerationConfig(temperature=args.temp, max_new_tokens=args.max_new)
    t0 = time.time()
    out = eng.generate(chat, gc, max_new_tokens=args.max_new)
    dt = time.time() - t0
    text = tok.decode(out)
    print(f"[5] generated {len(out)} tokens in {dt:.1f}s "
          f"({len(out) / dt:.1f} tok/s)", flush=True)
    print("=" * 60, flush=True)
    print(text, flush=True)
    print("=" * 60, flush=True)
    eng.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
