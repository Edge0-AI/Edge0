#!/usr/bin/env python3
"""#1620: a request cancelled right after it restored a parked conversation must not lose that conversation.

  A (long answer) -> B (an unrelated chat parks A) -> A2 = A + a short next turn, STOPped as soon as the engine
  says RESUME (the parked A is restored, nothing or a few windows read) -> A3 = A + another next turn.

A3 must reuse the whole of A and its answer from the live session; without the fix the cancelled A2 had taken
the parked entry and A3 fell back to the turn checkpoint (the prompt, not the answer).

  python3 tools/cancel_park_test.py CFG   # CFG: a run config json (as tools used by smoke); needs a conversation cache:
  STRATA_EXTRA_ARGS="--prompt-cache 4 --conversation-cache-mib 2048 --conversation-cache-slots 3"
Prints PASS / FAIL and the reused token counts.  Exit status 0 on PASS.
"""
import json, os, queue, sys, threading, time
from pathlib import Path

REPO = Path(os.environ.get("STRATA_REPO", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO / "tools")); sys.path.insert(0, str(REPO))
import strata_tokenizer as ST
from serve.server import StrataEngine, child_env

cfg = json.load(open(sys.argv[1]))
tp = Path(cfg["tokenizer"])
vocab = json.loads((tp / "vocab.json").read_text())
toks = [None] * len(vocab)
for t, i in vocab.items():
    toks[i] = t
tok = ST.Tokenizer(toks, (tp / "merges.txt").read_text().split("\n"), json.loads((tp / "token_type.json").read_text()))
args = cfg["args"] + os.environ.get("STRATA_EXTRA_ARGS", "").split()
eng = StrataEngine(os.environ.get("STRATA_EXE", cfg["exe"]), args, cwd=cfg.get("cwd"),
                   log=os.environ.get("SMOKE_LOG", cfg["log"]), env=child_env(cfg))

HEAD = "<|im_start|>user\n"
GEN = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def enc(text):
    return tok.encode(text, parse_special=True)


def run(ids, max_new=200):
    out = [t for t in eng.generate(ids, max_new, {"temperature": 0}, threading.Event()) if t is not None]
    return out, int(eng.last.get("reused") or 0)


def run_cancelled(ids):
    """GEN, then STOP once the engine has said RESUME (the restore is done, the prompt is not read yet)."""
    eng.proc.stdin.write("GEN 200 " + ",".join(str(int(t)) for t in ids) + "\n")
    eng.proc.stdin.flush()
    stopped, reused, fin = False, None, None
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            line = eng.lines.get(timeout=1)
        except queue.Empty:
            continue
        if line is None:
            raise SystemExit("FAIL: the engine died")
        if line.startswith("RESUME "):
            reused = int(line.split()[1])
            eng.proc.stdin.write("STOP\n"); eng.proc.stdin.flush(); stopped = True
        elif line.startswith("DONE "):
            fin = line.split()[5] if len(line.split()) > 5 else "?"
            break
    return reused, fin, stopped


story = " ".join(f"Item {i}: the quick brown fox number {i} jumps over lazy dog {i * 7}." for i in range(70))
A1 = enc(HEAD + story + "\nNow count from 1 to 120 separated by commas." + GEN)
outA, r0 = run(A1, 300)
print(f"A: prompt {len(A1)} tokens, {len(outA)} generated, reused {r0}", flush=True)
B1 = enc(HEAD + "Say hello in one short sentence." + GEN)
outB, rb = run(B1, 20)
print(f"B: prompt {len(B1)} tokens, reused {rb} (A is parked now)", flush=True)
cont = enc("<|im_end|>\n<|im_start|>user\nNow say only the word done." + GEN)
A2 = A1 + outA + cont
reused2, fin2, stopped = run_cancelled(A2)
print(f"A2 (cancelled): restored/reused at RESUME {reused2}, finish {fin2}, STOP sent {stopped}", flush=True)
cont3 = enc("<|im_end|>\n<|im_start|>user\nNow say only the word finished." + GEN)
A3 = A1 + outA + cont3
outA3, reused3 = run(A3, 20)
want = len(A1) + len(outA) - 3
print(f"A3: prompt {len(A3)} tokens, reused {reused3} (A + its answer = {len(A1) + len(outA)}; need >= {want})", flush=True)
eng.proc.stdin.close()
try:
    eng.proc.wait(60)
except Exception:
    eng.proc.kill()
ok = fin2 == "cancel" and reused3 >= want
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
