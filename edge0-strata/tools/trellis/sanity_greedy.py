"""sanity_greedy.py - a few greedy answers from an engine config (the same config files tools/quant_quality.py uses), through the
normal batched prompt path (not the yardstick's decode windows).

  python3 tools/trellis/sanity_greedy.py --config CFG.json --out answers.txt [--wrap "flock ... env HIP_VISIBLE_DEVICES=3"]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import make_corpus  # noqa: E402

PROMPTS = [
    "What is the capital of Australia, and what is it known for? Answer in two sentences.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Translate into German: 'The library opens at nine and closes at five, except on Sundays.'",
    "A train leaves at 14:20 and arrives at 17:05. How long is the trip? Think step by step, then give the answer.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wrap", default="")
    ap.add_argument("--new", type=int, default=80)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    tok = make_corpus.load_tokenizer(cfg["tokenizer"])
    env = dict(os.environ)
    env.update(cfg.get("env", {}))
    env["LD_LIBRARY_PATH"] = ":".join(cfg.get("lib_dirs", []) + [env.get("LD_LIBRARY_PATH", "")])
    args = [cfg["exe"], "--serve"] + [x for x in cfg["args"] if x != "--serve"]
    if a.wrap:
        args = a.wrap.split() + args
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(a.out + ".stderr", "w"), text=True,
                         cwd=cfg.get("cwd"), env=env, bufsize=1)
    for line in p.stdout:
        if line.startswith("READY"):
            break
    else:
        raise SystemExit("the engine ended before READY")
    out = []
    for q in PROMPTS:
        text = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" % q
        ids = tok.encode(text, parse_special=True)
        p.stdin.write("GEN %d %s\n" % (a.new, ",".join(map(str, ids))))
        p.stdin.flush()
        gen = []
        for line in p.stdout:
            if line.startswith("T "):
                gen.append(int(line.split()[1]))
            elif line.startswith(("DONE", "ERR")):
                done = line.strip()
                break
        out.append("### %s\n%s\n[%s]\n" % (q, tok.decode(gen), done))
        print(out[-1], flush=True)
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    try:
        p.wait(timeout=300)
    except subprocess.TimeoutExpired:
        p.kill()
    pathlib.Path(a.out).write_text("\n".join(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
