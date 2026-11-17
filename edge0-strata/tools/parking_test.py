#!/usr/bin/env python3
"""Conversation parking test: a follow-up to conversation A must decode the same tokens whether A's state was still
live (reference engine: A, then the follow-up) or came back from the parking cache (second engine: A, then B, which
parks A, then the follow-up).  Greedy; run with --extra "--pcie-frac 0 --adapt-every 1000000" for exactness.

  python3 tools/parking_test.py --exe engine/strata --config strata-<model>.json \
      --extra "--layer-split 12,24,36 --conversation-cache-mib 8192 --conversation-cache-slots 4 --pcie-frac 0"
"""
import argparse, json, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_test import Engine, tokenizer  # noqa: E402

DOC = ("The history of the printing press begins with movable type in East Asia and continues with Gutenberg's "
       "press in Mainz around 1450, which spread across Europe within decades, lowered the cost of books, changed "
       "how scholars, merchants and churches shared knowledge, and fed the Reformation and the scientific "
       "revolution. ")


def chat(user):
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def gen(eng, out, ids, n):
    eng.send(f"GEN {n} " + ",".join(map(str, ids)))
    got = []
    for line in out:
        if line.startswith("T "):
            got.append(int(line.split()[1]))
        elif line.startswith("DONE") or line.startswith("ERR"):
            return got, line
    raise SystemExit("the engine ended")


def run(a, cfg, tok, with_b):
    # STRATA_SNAPSHOT_VERIFY: the engine reads the restored draft ring back (on the GPU that holds it) after a restore
    eng = Engine(a.exe, cfg, 0, {"STRATA_IQ_MT_MIN": "1", "STRATA_SNAPSHOT_VERIFY": "1"}, a.extra.split())
    out = eng.lines()
    pa = tok.encode(chat(DOC * a.repeat + "\nSummarize this text in five sentences."), parse_special=True)
    ans, _ = gen(eng, out, pa, a.max_new)
    if with_b:
        pb = tok.encode(chat("Write a short poem about the sea, then explain its metaphors." * 20), parse_special=True)
        gen(eng, out, pb, a.max_new)
    end = tok.encode("<|im_end|>\n", parse_special=True)
    follow = pa + ans + ([] if ans and ans[-1] in end else end) + \
        tok.encode(chat("Now give three keywords for that text, with one sentence each.")[0:], parse_special=True)
    t0 = time.time()
    got, done = gen(eng, out, follow, a.max_new)
    dt = time.time() - t0
    if with_b:   # B again: A is parked a second time, from retained K/V (every stage's) - only its new tail copied
        gen(eng, out, pb, 8)
    eng.send("QUIT")
    eng.p.wait(timeout=180)
    log = open("/tmp/batch_test_engine.log").read()
    restored = re.findall(r"restored \d+ tokens[^\n]*", log)
    if with_b and "SNAPSHOT_VERIFY draft=" not in log:
        print("no SNAPSHOT_VERIFY line: the draft ring was not read back", flush=True)
    reparked = [int(x) for x in re.findall(r"parked \d+ tokens .*reused_kv_bytes=(\d+)", log)]
    return len(pa), len(follow), got, done, dt, restored, reparked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--repeat", type=int, default=60, help="copies of the paragraph in conversation A")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])
    ref = run(a, cfg, tok, with_b=False)
    park = run(a, cfg, tok, with_b=True)
    for name, r in (("live (reference)", ref), ("parked (A, B, A)", park)):
        f = r[3].split()
        print(f"{name}: prompt A {r[0]} tokens, follow-up {r[1]} tokens, reused {f[8] if len(f) > 8 else '?'}, "
              f"follow-up in {r[4]:.2f} s; restore lines: {r[5][-1:] or 'none'}")
    same = ref[2] == park[2]
    first = next((k for k in range(min(len(ref[2]), len(park[2]))) if ref[2][k] != park[2][k]), None)
    print("follow-up tokens:", "IDENTICAL" if same else f"DIFFER at {first}", f"({len(ref[2])} / {len(park[2])})")
    print("   ", repr(tok.decode(park[2])[:200]))
    reused = park[6][-1] if park[6] else 0
    print(f"second park of A: {reused} bytes of K/V reused (retained from its restore)")
    return 0 if same and park[5] and reused > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
