#!/usr/bin/env python3
"""Metal serve regression probe (owner round 9, G6): one request of the first N token ids of a prompt through serve/server.py on the
config in STRATA_CFG, with its default flags (expert cache auto, prefill auto).  95, 569, 1500 and 8213 ids used to end in
"prefill: relayout needs borrowed buffers and a chunk of at most 0" or in a copy into device allocations that were given back.

    STRATA_REPO=<checkout> STRATA_CFG=<run config json> STRATA_PROMPTS=<dir with agent.prompt.ids> python3 serve_probe.py EXE 95:agent 8

Needs the Gyro-S (or any Flash-Next) model on the Mac; it is not part of the unit tests."""
import os
import sys
import threading
import time
from pathlib import Path

REPO = Path(os.environ["STRATA_REPO"])
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))
import strata_tokenizer as ST  # noqa: E402
from serve.server import StrataEngine, child_env  # noqa: E402

EXE = sys.argv[1]
NAME = sys.argv[2] if len(sys.argv) > 2 else "agent"
MAX_NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 8
S = os.environ.get("STRATA_OUT", os.path.expanduser("~/strata-mac/g023/out"))
cfg = json.load(open(os.environ["STRATA_CFG"], encoding="utf-8-sig"))
tp = Path(cfg["tokenizer"])
vocab = json.loads((tp / "vocab.json").read_text(encoding="utf-8"))
toks = [None] * len(vocab)
for t, i in vocab.items():
    toks[i] = t
tok = ST.Tokenizer(toks, (tp / "merges.txt").read_text(encoding="utf-8").split("\n"),
                   json.loads((tp / "token_type.json").read_text()))
if NAME.split(":")[0].isdigit():          # "4096" / "4096:agent": the first N ids of the agent prompt
    parts = NAME.split(":")
    n = int(parts[0])
    src = parts[1] if len(parts) > 1 else "agent"
    raw = (Path(os.environ.get("STRATA_PROMPTS", os.path.expanduser("~/owner-t9/bundle/prompts"))) / f"{src}.prompt.ids").read_text()
    try:
        ids = json.loads(raw)[:n]
    except Exception:
        ids = [int(x) for x in raw.replace(",", " ").split()][:n]
else:
    raw = (Path(os.environ.get("STRATA_PROMPTS", os.path.expanduser("~/owner-t9/bundle/prompts"))) / f"{NAME}.prompt.ids").read_text()
    try:
        ids = json.loads(raw)
    except Exception:
        ids = [int(x) for x in raw.replace(",", " ").split()]
args = list(cfg["args"]) + os.environ.get("STRATA_EXTRA_ARGS", "").split()
log = os.path.join(S, f"probe-{NAME}.log")
t0 = time.time()
eng = StrataEngine(EXE, args, cwd=cfg.get("cwd"), log=log, env=child_env(cfg))
print(f"READY after {time.time() - t0:.1f} s, prompt {len(ids)} tokens", flush=True)
t1 = time.time()
out = [t for t in eng.generate(ids, MAX_NEW, {"temperature": 0}, threading.Event()) if t is not None]
last = dict(eng.last)
print(f"{NAME}: {len(out)} out in {time.time() - t1:.1f} s; prompt {last.get('prompt_tokens')} tok in "
      f"{last.get('prompt_ms')} ms = {last.get('prompt_tokens', 0) / (last.get('prompt_ms', 1) / 1000):.1f} tok/s; "
      f"decode {last.get('decode_ms')} ms = {len(out) / (last['decode_ms'] / 1000) if last.get('decode_ms') else 0:.2f} tok/s; "
      f"drafts {last.get('drafts_accepted')}/{last.get('drafts_offered')}", flush=True)
print("text:", tok.decode(out)[:120].replace("\n", " "), flush=True)
eng.proc.stdin.close()
try:
    eng.proc.wait(60)
except Exception:
    eng.proc.kill()
print("PROBE_DONE", flush=True)
