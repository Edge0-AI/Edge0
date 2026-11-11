"""tools/collect_profile_traces.py - routing traces of a varied set of prompts, for tools/make_profile_mix.py.

For every prompt of a corpus (default bench/profile_corpus.json: 45 chat, code, maths, reasoning, writing, science, tool
use and six-language prompts) it runs the engine once (greedy, --max-new tokens) and keeps the routed experts of the
generated text:

    python tools/collect_profile_traces.py CONFIG.json OUTDIR [--corpus FILE] [--max-new 250] [--mode usage|routing]

CONFIG.json is a run config setup wrote (strata-*.json: "exe", "args", "env", "tokenizer", "cwd").  The engine is run as
`exe args... --tokens-file F --max-new N --greedy` (--serve and --batch are dropped).  `--mode usage` (default) is the
SYCL engine's STRATA_USAGE_DUMP=PATH, which counts on the device and works with STRATA_VERIFY_NO_HOST; `--mode routing`
adds `--dump-routing PATH` (the CUDA/HIP engines and the SYCL host path).  Then, for example:

    python tools/make_profile_mix.py OUTDIR/c*.trace:1 --out data/expert-profile-new.bin

A prompt takes a start-up of the engine (20-60 s) and ~10 s of decoding; 45 prompts are ~30 minutes.  The traces are the
generated text's routing only (decode windows); the prompt's own routing is not counted.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_tokenizer as ST  # noqa: E402


def load_tokenizer(tdir):
    tdir = Path(tdir)
    vocab = json.loads((tdir / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    return ST.Tokenizer(toks, (tdir / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((tdir / "token_type.json").read_text()))


def drop(args, flag, with_value=False):
    args = list(args)
    while flag in args:
        i = args.index(flag)
        del args[i:i + (2 if with_value else 1)]
    return args


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("outdir")
    ap.add_argument("--corpus", default=str(ROOT / "bench" / "profile_corpus.json"))
    ap.add_argument("--max-new", type=int, default=250)
    ap.add_argument("--mode", choices=("usage", "routing"), default="usage")
    ap.add_argument("--only", help="comma-separated prompt numbers (default all)")
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    tok = load_tokenizer(cfg["tokenizer"])
    corpus = json.load(open(a.corpus, encoding="utf-8"))
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    args = drop(drop(cfg["args"], "--serve"), "--batch", True)
    only = {int(x) for x in a.only.split(",")} if a.only else None
    env0 = dict(os.environ)
    env0.update({k: str(v) for k, v in cfg.get("env", {}).items()})
    for n, c in enumerate(corpus):
        if only is not None and n not in only:
            continue
        trace = out / f"c{n:02d}.trace"
        if trace.exists() and trace.stat().st_size > 0:
            continue
        ids = tok.encode(f"<|im_start|>user\n{c['p']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                         parse_special=True)
        tf = out / f"c{n:02d}.tok"
        tf.write_text(",".join(str(i) for i in ids))
        cmd = [cfg["exe"]] + args + ["--tokens-file", str(tf), "--max-new", str(a.max_new), "--greedy"]
        env = dict(env0)
        if a.mode == "usage":
            env["STRATA_USAGE_DUMP"] = str(trace)
        else:
            cmd += ["--dump-routing", str(trace)]
        print(f"[{n + 1}/{len(corpus)}] {c['d']}: {len(ids)} prompt tokens", flush=True)
        r = subprocess.run(cmd, cwd=cfg.get("cwd") or None, env=env, capture_output=True, text=True)
        if r.returncode != 0 or not trace.exists():
            print(f"  failed (rc {r.returncode}): {r.stderr[-300:]}", file=sys.stderr)
            sys.exit(1)
    print(f"traces in {out}")


if __name__ == "__main__":
    main()
