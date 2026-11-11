#!/usr/bin/env python3
"""make_prompts.py - builds this report's prompts from Strata's own repository text, so anyone can rebuild them.

One ~1,000-token warm-up and three prompts each of 4,096, 32,768 and 128,000 tokens. The text is the tag's
docs/*.md, src/**/*.cpp, src/**/*.cu, include/**/*.hpp, serve/**/*.py and tools/**/*.py in that order (the order
tools/needle_bench.py uses, without third_party/), each file under a "=== path ===" line. Every prompt is a different,
non-overlapping slice of that text and starts with its own run id, so no request can reuse another one's prefix beyond
the chat template's first tokens. Token counts are of the rendered prompt, with the model pack's tokenizer and chat
template; the server reports the real count of each request.

    git -C Strata checkout v0.1.41
    python make_prompts.py --tree Strata --tokenizer-dir <pack>/tokenizer --out prompts.json

With the v0.1.41 tree and the IQ3_S pack's tokenizer this writes a file with sha256
d1b10bd59376564bb826b290b50f684991988ec8765af99139e3bfbe044c8895."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True, help="a Strata checkout at v0.1.41")
ap.add_argument("--tokenizer-dir", required=True, help="the model pack's tokenizer/ folder")
ap.add_argument("--out", default="prompts.json")
a = ap.parse_args()
TREE, TDIR = Path(a.tree).resolve(), Path(a.tokenizer_dir)
sys.path.insert(0, str(TREE / "tools"))
sys.path.insert(0, str(TREE))
import strata_tokenizer as ST   # noqa: E402
from serve.frontend import ChatTemplate   # noqa: E402

vocab = json.loads((TDIR / "vocab.json").read_text(encoding="utf-8"))
toks = [None] * len(vocab)
for t, i in vocab.items():
    toks[i] = t
tok = ST.Tokenizer(toks, (TDIR / "merges.txt").read_text(encoding="utf-8").split("\n"),
                   json.loads((TDIR / "token_type.json").read_text()))
tpl = ChatTemplate(TDIR / "chat_template.jinja")

SOURCES = [("docs", "*.md"), ("src", "*.cpp"), ("src", "*.cu"), ("include", "*.hpp"), ("serve", "*.py"), ("tools", "*.py")]
parts = []
for d, pat in SOURCES:
    for f in sorted((TREE / d).rglob(pat)):
        parts.append(f"\n\n=== {f.relative_to(TREE).as_posix()} ===\n{f.read_text(encoding='utf-8', errors='replace')}")
text = "".join(parts)
PLAN = [("warmup", 1000, 1)] + [(str(n), n, 3) for n in (4096, 32768, 128000)]
need = sum(n * r for _, n, r in PLAN)
ids = tok.encode(text[:int(need * 3.6)])
TASK = "\n\nTask: in one short paragraph, say what the material above is about."
n_prompt = lambda m: len(tok.encode(tpl.render(m), parse_special=True))
out, cursor = [], 0
for size, target, reps in PLAN:
    for r in range(reps):
        nonce = hashlib.sha1(f"cb-{size}-{r}".encode()).hexdigest()[:8]
        head = f"[Run cb-{size}-r{r}-{nonce}] Below is source code and documentation from the Strata repository (tag v0.1.41).\n\n"
        n = target - n_prompt([{"role": "user", "content": head + TASK}])
        for _ in range(8):            # decode/re-encode can move the count by a token or two
            body = tok.decode(ids[cursor:cursor + n])
            msgs = [{"role": "user", "content": head + body + TASK}]
            got = n_prompt(msgs)
            if abs(got - target) <= 2:
                break
            n += target - got
        if cursor + n > len(ids):
            raise SystemExit("ran out of text")
        cursor += n
        out.append({"size": size, "rep": r, "target": target, "prompt_tokens": got, "messages": msgs,
                    "first_file": body.split("\n=== ", 1)[-1].split(" ===", 1)[0] if "\n=== " in body else "(mid-file)"})
        print(size, r, got, flush=True)
Path(a.out).write_text(json.dumps(out, ensure_ascii=False))
print(f"wrote {a.out}: {len(out)} prompts, sha256 {hashlib.sha256(Path(a.out).read_bytes()).hexdigest()}")
