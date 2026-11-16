#!/usr/bin/env python3
"""chat_tokens.py REPO TOKENIZER_DIR "question" : the token ids of a one-turn chat prompt (thinking off), comma
separated, for `strata generate --tokens` (the same template as the gates' rbench.py).
With --decode IDS instead of the question: the text of comma-separated ids."""
import json, sys
from pathlib import Path

repo, tdir = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo / "tools"))
import strata_tokenizer as ST

vocab = json.loads((tdir / "vocab.json").read_text(encoding="utf-8"))
toks = [None] * len(vocab)
for t, i in vocab.items():
    toks[i] = t
tok = ST.Tokenizer(toks, (tdir / "merges.txt").read_text(encoding="utf-8").split("\n"),
                   json.loads((tdir / "token_type.json").read_text()))
if sys.argv[3] == "--decode":
    print(tok.decode([int(x) for x in sys.argv[4].replace(" ", "").split(",") if x]))
else:
    ids = tok.encode("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" % sys.argv[3],
                     parse_special=True)
    print(",".join(str(i) for i in ids))
