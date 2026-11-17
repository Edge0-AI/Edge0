#!/usr/bin/env python3
"""#1604: with --batch N --batch-groups N on a layer split, two long conversations that alternate must reuse their slot
(or the conversation cache) on every later turn instead of reading their prompt from token 0.

  python3 tools/batch_groups_cache_test.py CFG OUT.json
  STRATA_EXTRA_ARGS="--batch 2 --batch-groups 2 --prompt-cache 4 --conversation-cache-mib 4096 --conversation-cache-slots 3"

Three rounds; each round runs conversation A and conversation B at the same time (so both go through the batch slots),
the next round continues each one (prompt + its answer + a new user turn).  The engine log says per request how many
prompt tokens were reused.  PASS when every request of rounds 2 and 3 reused at least 90 % of the previous round's
prompt.  The answers (token ids) are written to OUT.json to compare two runs.
"""
import json, os, re, sys, threading
from pathlib import Path

REPO = Path(os.environ.get("STRATA_REPO", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO / "tools")); sys.path.insert(0, str(REPO))
import strata_tokenizer as ST
from serve.server import StrataEngine, child_env, engine_args

cfg = json.load(open(sys.argv[1]))
tp = Path(cfg["tokenizer"])
vocab = json.loads((tp / "vocab.json").read_text())
toks = [None] * len(vocab)
for t, i in vocab.items():
    toks[i] = t
tok = ST.Tokenizer(toks, (tp / "merges.txt").read_text().split("\n"), json.loads((tp / "token_type.json").read_text()))
if os.environ.get("GPUS"):   # e.g. GPUS=0,1: a layer split over those cards (SPLIT=auto)
    cfg["gpu"] = [int(x) for x in os.environ["GPUS"].split(",")]
    cfg["layer_split"] = os.environ.get("SPLIT", "auto")
args = engine_args(cfg) + os.environ.get("STRATA_EXTRA_ARGS", "").split()
log = os.environ.get("SMOKE_LOG", cfg["log"])
eng = StrataEngine(os.environ.get("STRATA_EXE", cfg["exe"]), args, cwd=cfg.get("cwd"), log=log, env=child_env(cfg))
GEN = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def enc(text):
    return tok.encode(text, parse_special=True)


def first(tag, n):
    return enc(f"<|im_start|>user\n[{tag}] " + " ".join(f"Entry {i} of {tag}: value {i * 13 % 97}." for i in range(n))
               + "\nCount from 1 to 80 separated by commas." + GEN)


conv = {"A": first("alpha", 250), "B": first("bravo", 250)}
answers = {"A": [], "B": []}
prev_len = {}
rounds = []


def one(name, ids, out):
    res = [t for t in eng.generate(ids, 80, {"temperature": 0}, threading.Event()) if t is not None]
    out[name] = res


for rnd in range(3):
    lens = {k: len(v) for k, v in conv.items()}
    out = {}
    ts = [threading.Thread(target=one, args=(k, conv[k], out)) for k in ("A", "B")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    rounds.append({"prompt_lens": lens, "answers": out})
    print(f"round {rnd + 1}: prompts {lens}, answers {[len(v) for v in out.values()]}", flush=True)
    for k in conv:
        answers[k].append(out[k])
        conv[k] = conv[k] + out[k] + enc(f"<|im_end|>\n<|im_start|>user\nNow count from {rnd + 2} to {rnd + 60}." + GEN)
eng.proc.stdin.close()
try:
    eng.proc.wait(60)
except Exception:
    eng.proc.kill()
json.dump(rounds, open(sys.argv[2], "w"))

text = open(log, errors="replace").read()
reqs = [(int(a), int(b)) for a, b in re.findall(r"strata serve: prompt (\d+) tokens = (\d+) reused", text)]
print("requests (prompt tokens, reused):", reqs)
print("slot gave back lines:", len(re.findall(r"gave back", text)), " conversation cache restores:",
      len(re.findall(r"conversation cache: restored", text)))
# rounds 2 and 3 are the last 4 requests of the log
ok = len(reqs) >= 6
for p, r in reqs[2:]:
    if r < 0.9 * (p - 140):   # the previous prompt + answer is ~all of it but the new turn
        ok = False
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
