#!/usr/bin/env python3
"""SAVE / RESTORE round trip with parked conversations holding the host RAM (#1572 / #1573).

  python3 tools/session_pressure_test.py CFG OUTFILE
  STRATA_EXTRA_ARGS="--prompt-cache 4 --conversation-cache-mib 3072 --conversation-cache-slots 4
                     --conversation-cache-min-free-mib <floor> [--session-save-reclaim]"

Parks three conversations, runs a fourth (D), SAVEs it, runs an unrelated one, RESTOREs, and checks that D's next turn
gives the same token ids after the restore as it does when continued live.  Prints SAVE: ok|refused (...) and PASS/FAIL.
"""
import json, os, sys, threading, time
from pathlib import Path

REPO = Path(os.environ.get("STRATA_REPO", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO / "tools")); sys.path.insert(0, str(REPO))
import strata_tokenizer as ST
from serve import server as S
from serve.server import StrataEngine, child_env

cfg = json.load(open(sys.argv[1]))
out_file = sys.argv[2]
tp = Path(cfg["tokenizer"])
vocab = json.loads((tp / "vocab.json").read_text())
toks = [None] * len(vocab)
for t, i in vocab.items():
    toks[i] = t
tok = ST.Tokenizer(toks, (tp / "merges.txt").read_text().split("\n"), json.loads((tp / "token_type.json").read_text()))
args = cfg["args"] + os.environ.get("STRATA_EXTRA_ARGS", "").split()
eng = StrataEngine(os.environ.get("STRATA_EXE", cfg["exe"]), args, cwd=cfg.get("cwd"),
                   log=os.environ.get("SMOKE_LOG", cfg["log"]), env=child_env(cfg))
GEN = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def enc(text):
    return tok.encode(text, parse_special=True)


def run(ids, max_new=60):
    out = [t for t in eng.generate(ids, max_new, {"temperature": 0}, threading.Event()) if t is not None]
    return out, int(eng.last.get("reused") or 0)


def chat(tag, n):
    return enc(f"<|im_start|>user\n[{tag}] " + " ".join(f"Entry {i} of {tag}: value {i * 13 % 97}." for i in range(n))
               + "\nSummarize in one line." + GEN)


for tag in ("alpha", "bravo", "charlie"):
    p = chat(tag, 70)
    run(p, 20)
D = chat("delta", 80)
outD, _ = run(D, 60)
print(f"D: prompt {len(D)} tokens, {len(outD)} generated", flush=True)
nxt = enc("<|im_end|>\n<|im_start|>user\nNow say only the word done." + GEN)
saved = False
hog = None
if os.environ.get("HOG_TARGET_MIB"):   # take the RAM so that only HOG_TARGET_MIB stay available: a SAVE needs more
    import psutil
    extra = psutil.virtual_memory().available - int(os.environ["HOG_TARGET_MIB"]) * 2**20
    if extra > 0:
        hog = bytearray(extra)
        for i in range(0, len(hog), 4096):
            hog[i] = 1
    print("available now: %d MiB" % (psutil.virtual_memory().available >> 20), flush=True)
try:
    r = eng.session_file("save", out_file)
    saved = True
    print("SAVE: ok", r, flush=True)
except S.SessionRefused as e:
    print("SAVE: refused", e.kind, str(e)[:300], flush=True)
if not saved and "--session-save-reclaim" in args:
    # the server's bounded retry on the same engine
    class Svc:
        engine = eng
    from serve.session_save_retry import session_file as retry
    try:
        r = retry(Svc, "save", out_file, S.SessionRefused)
        saved = True
        print("SAVE after the bounded wait: ok", r, flush=True)
    except S.SessionRefused as e:
        print("SAVE after the bounded wait: refused", e.kind, str(e)[:300], flush=True)
hog = None   # the RAM goes back
if not saved:
    print("NOSAVE")
    eng.proc.stdin.close()
    sys.exit(2)
live_next, reused_live = run(D + outD + nxt, 20)
print(f"D live next turn: reused {reused_live}", flush=True)
run(chat("echo", 70), 20)   # something else takes the session
r = eng.session_file("restore", out_file)
print("RESTORE:", r, flush=True)
again, reused_r = run(D + outD + nxt, 20)
print(f"D after RESTORE: reused {reused_r}", flush=True)
eng.proc.stdin.close()
try:
    eng.proc.wait(60)
except Exception:
    eng.proc.kill()
ok = again == live_next and reused_r > 0
print("PASS" if ok else "FAIL (ids %s)" % ("equal" if again == live_next else "differ"))
sys.exit(0 if ok else 1)
