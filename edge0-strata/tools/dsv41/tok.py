# Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
# Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
"""tools/ds4/tok.py - token ids for maya-ds4's --ids, and text back from its output (DeepSeek-V4.1's tokenizer).

    python tools/ds4/tok.py --model ~/models/DSV4 --chat "Why is the sky blue?" [--think]   -> 0,128803,...
    python tools/ds4/tok.py --model ~/models/DSV4 --decode 18033,1                          -> text
"""
from __future__ import annotations

import argparse
import pathlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(pathlib.Path("~/models/DSV4").expanduser()))
    ap.add_argument("--chat")
    ap.add_argument("--think", action="store_true", help="thinking mode (the prompt ends with <think>)")
    ap.add_argument("--text")
    ap.add_argument("--decode")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(pathlib.Path(a.model).expanduser() / "tokenizer.json"))
    if a.decode:
        ids = [int(x) for x in a.decode.replace(" ", "").strip(",").split(",") if x]
        print(tok.decode(ids, skip_special_tokens=False))
        return 0
    if a.chat is not None:
        s = "<｜begin▁of▁sentence｜><｜User｜>" + a.chat + "<｜Assistant｜>" + ("<think>" if a.think else "</think>")
    else:
        s = a.text or ""
    print(",".join(str(i) for i in tok.encode(s, add_special_tokens=False).ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
