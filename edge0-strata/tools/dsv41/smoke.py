"""16 tokens of a fixed prompt through strata-dsv41 and a check of the TEXT (setup runs this once after the pack is built).

    python tools/dsv41/smoke.py --exe strata-dsv41 --pack <pack> [--gpus 0] [--tokens 16]

A pack that loads can still give garbage (a wrong pack recipe does exactly that), so the answer is read, not just the exit code:
the model is asked for the capital of France and the reply must be words, with "Paris" in it.  Exit 0 and the text on success; exit 1
and one plain line (and the engine's last lines) when the reply is not sensible; exit 2 when the engine did not run at all.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

PROMPT = "<｜begin▁of▁sentence｜><｜User｜>What is the capital of France? Answer in one short sentence.<｜Assistant｜></think>"


def tokenizer(pack: Path):
    import strata_tokenizer as ST
    tpath = pack / "tokenizer"
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tpath / "token_type.json").read_text())
    tcfg = {}
    try:
        tcfg = json.loads((tpath / "tokenizer.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return ST.Tokenizer(tokens, merges, types, tcfg.get("pre") or "qwen35", tcfg.get("special_ids") or None)


def sensible(text: str) -> tuple[bool, str]:
    """Words, not noise: letters in a normal share, more than two different words, no replacement characters, and Paris in it."""
    t = text.strip()
    words = re.findall(r"[A-Za-z]{2,}", t)
    if not t:
        return False, "the answer is empty"
    if "�" in t:
        return False, "the answer has broken characters"
    if len(set(w.lower() for w in words)) < 3:
        return False, "the answer is not made of words"
    letters = sum(c.isalpha() or c.isspace() or c in ".,'-" for c in t) / len(t)
    if letters < 0.8:
        return False, "the answer is mostly symbols"
    if "paris" not in t.lower():
        return False, "the answer to a question with a known answer (the capital of France) does not contain it"
    return True, "ok"


def main() -> int:
    for st in (sys.stdout, sys.stderr):       # the reply has DeepSeek's special tokens (fullwidth bars): a Windows console code page cannot print them
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--gpus")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=5400)
    a = ap.parse_args()
    pack = Path(a.pack)
    tok = tokenizer(pack)
    ids = tok.encode(PROMPT, parse_special=True)
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "ids.txt"
        f.write_text(",".join(str(i) for i in ids), encoding="ascii")
        mac = sys.platform == "darwin"
        cmd = [a.exe, "--pack", str(pack), "--ids", "@" + str(f), "--n", str(a.tokens), "--max-context", "1024", "--dspark", "none" if mac else "auto"]
        # (a Mac runs without the drafter: its Metal engine has no kernels for it, and it keeps the engine's own prompt window: with a window of
        # 16 or 17 tokens the Metal engine gave a wrong first word in about half the runs; the engine's default window never did)
        if not mac:
            cmd += ["--window", "16"]
        if a.gpus:
            cmd += ["--gpus", a.gpus]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=a.timeout, env=dict(os.environ))
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f"the engine did not finish the test run: {e}")
            return 2
    err = (r.stderr or "").strip().splitlines()
    if r.returncode != 0:
        print(f"the engine stopped with an error (exit {r.returncode}): " + " | ".join(err[-3:]))
        return 2
    out = [int(x) for x in re.findall(r"-?\d+", (r.stdout or "").strip().splitlines()[-1] if (r.stdout or "").strip() else "")]
    text = tok.decode(out)
    good, why = sensible(text)
    print(text.strip())
    if not good:
        print(f"test question failed: {why}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
