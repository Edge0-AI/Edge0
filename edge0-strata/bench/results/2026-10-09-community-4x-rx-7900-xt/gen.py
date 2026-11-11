"""Build the bench prompts at exact token counts with Strata's own tokenizer (run in the runner's venv).

usage: gen.py SRC_ROOT TOKENIZER_DIR > prompts.jsonl
Each prompt is "Run id: <uuid>" + a slice of the repository text (tools/needle_bench.py's haystack) +
one question. The slice length is binary-searched so that the server-side prompt (content + the chat
template's 12 tokens) lands on the target.
"""

import json
import sys
import uuid
from pathlib import Path

src, tok_dir = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(src / "tools"))
import needle_bench  # noqa: E402
import strata_tokenizer as ST  # noqa: E402

TEMPLATE_TOKENS = 12
ASK = "\n\nSummarize the material above: what it is, its main parts, and how they fit together."
PLAN = [("cold-first-request", 32768, False), ("warmup-short", 4096, False), ("warmup-long", 32768, False)]
PLAN += [(f"{t}-run{i}", t, True) for t in (4096, 32768, 128000) for i in (1, 2, 3)]

vocab = json.loads((tok_dir / "vocab.json").read_text(encoding="utf-8"))
tokens = [None] * len(vocab)
for t, i in vocab.items():
    tokens[i] = t
tk = ST.Tokenizer(tokens, (tok_dir / "merges.txt").read_text(encoding="utf-8").split("\n"),
                  json.loads((tok_dir / "token_type.json").read_text()))
text = needle_bench.haystack(1_000_000)

for n, (label, target, counted) in enumerate(PLAN):
    head = f"Run id: {uuid.uuid4()}\n\n"
    offset = n * 7919
    goal = target - TEMPLATE_TOKENS

    def count(c: int) -> int:
        return len(tk.encode(head + text[offset:offset + c] + ASK))

    lo, hi = 0, len(text) - offset
    while lo < hi:                      # largest slice whose prompt is <= goal tokens
        mid = (lo + hi + 1) // 2
        if count(mid) <= goal:
            lo = mid
        else:
            hi = mid - 1
    content = head + text[offset:offset + lo] + ASK
    print(json.dumps({"label": label, "target": target, "counted": counted, "offset": offset,
                      "content_tokens": len(tk.encode(content)), "content": content}), flush=True)
