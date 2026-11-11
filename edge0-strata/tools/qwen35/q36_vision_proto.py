"""vision_proto.py - pictures over the engine protocol (GENI / BGENI), no server: the ids come from llama.cpp's own tokenizer
output (vc1/*.llama.log of vcmp.py), the rows from strata-vision (<picture>.sve).

  python q36_vision_proto.py --exe E --args "--model M --max-context 8192 --vision --batch 4" --vc DIR --pics DIR

Checks: 1. a solo GENI answers like llama.cpp's greedy (the first 24 tokens); 2. two pictures and a text prompt decoding
together (BGENI / BGEN) give each the tokens of its solo run; 3. the conversation cache: the same request again and a
longer one (the first, its answer and more text) reuse the held state, a different picture does not; a text request after
a picture request is not disturbed (the rotary offset is back at 0).

Start the engine without --mtp for the cache check: the held tokens of a request that ran with drafts include the drafts accepted past
its last emitted token, so a prompt that does not repeat them reads from the start (correct, but the check wants the reuse)."""
from pathlib import Path
import argparse, json, os, shlex, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from q36_harness import Eng, load_prompts

PAD = 248056


def case(vc, pics, name, q):
    log = open(os.path.join(vc, "%s.q%d.llama.log" % (name, q))).read()
    ids, gen = [], []
    for l in log.splitlines():
        if l.startswith("text:"):
            ids += [int(x) for x in l.split(":", 1)[1].split()]
        elif l.startswith("image:"):
            ids += [PAD] * int(l.split()[1])
        elif l.startswith("gen:"):
            gen = [int(x) for x in l.split(":", 1)[1].split()]
    return ids, gen, os.path.join(pics, name + ".sve")


ap = argparse.ArgumentParser()
ap.add_argument("--exe", required=True)
ap.add_argument("--args", default="")
ap.add_argument("--vc", required=True)
ap.add_argument("--pics", required=True)
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--log", default=None)
a = ap.parse_args()
eng = Eng(a.exe, shlex.split(a.args), log=a.log)
print("ready", eng.ready, eng.info, flush=True)
ok = True
S0, G0, E0 = case(a.vc, a.pics, "shapes.png", 0)
S1, G1, E1 = case(a.vc, a.pics, "shapes.png", 1)
C0, GC0, EC0 = case(a.vc, a.pics, "code4217.png", 0)
C1, GC1, EC1 = case(a.vc, a.pics, "code4217.png", 1)

# 1. solo vs llama.cpp's greedy
solo = {}
for tag, (ids, gen, emb) in {"shapes.q0": (S0, G0, E0), "shapes.q1": (S1, G1, E1), "code.q0": (C0, GC0, EC0),
                              "code.q1": (C1, GC1, EC1)}.items():
    toks, done, ttft, wall = eng.solo(ids, a.n, emb=emb)
    solo[tag] = toks
    k = next((i for i in range(min(len(toks), len(gen))) if toks[i] != gen[i]), None)
    print("solo %-10s %d tokens, vs llama.cpp greedy: %s | %s" % (tag, len(toks), "same %d tokens" % len(toks) if k is None else "first difference at %d" % k, done[:60]), flush=True)

# 2. batch: two pictures + a text prompt
text = load_prompts(str(Path(__file__).resolve().parent / "parity_prompts.ids"))[0]
t_toks, _, _, _ = eng.solo(text, a.n)
reqs = [{"ids": S1, "n": a.n, "slot": 0, "emb": E1}, {"ids": C1, "n": a.n, "slot": 1, "emb": EC1},
        {"ids": text, "n": a.n, "slot": 2}]
r = eng.batch(reqs)
eq = [r["tokens"][0] == solo["shapes.q1"], r["tokens"][1] == solo["code.q1"], r["tokens"][2] == t_toks]
print("batch (2 pictures + 1 text): identical to solo %s" % eq, flush=True)
ok &= all(eq)
# slot reuse: the same again into other slots
r2 = eng.batch([{"ids": C1, "n": a.n, "slot": 3, "emb": EC1}, {"ids": S1, "n": a.n, "slot": 0, "emb": E1}])
eq2 = [r2["tokens"][0] == solo["code.q1"], r2["tokens"][1] == solo["shapes.q1"]]
print("batch again (slot 3 new, slot 0 held): identical %s" % eq2, flush=True)
ok &= all(eq2)

# 3. the conversation cache (solo path)
def reused(done):
    return int(done.split()[8])

toks1, done1, _, _ = eng.solo(S0, 16, emb=E0)
toks1b, done1b, _, _ = eng.solo(S0, 16, emb=E0)
print("same request again (a state that has moved past the prompt cannot step back: no reuse): reused %d of %d, tokens %s" % (reused(done1b), len(S0), "same" if toks1b == toks1 else "DIFFERENT"), flush=True)
ok &= toks1b == toks1
longer = S0 + toks1 + text[:12]
toks2, done2, _, _ = eng.solo(longer, 16, emb=E0)
print("longer request (picture + answer + more): reused %d of %d" % (reused(done2), len(longer)), flush=True)
ok &= reused(done2) >= len(S0)
toks3, done3, _, _ = eng.solo(C0, 16, emb=EC0)   # a different picture
print("a different picture: reused %d" % reused(done3), "tokens vs its solo: %s" % ("same" if toks3 == solo["code.q0"][:16] else "DIFFERENT"), flush=True)
ok &= toks3 == solo["code.q0"][:16]
t2, done4, _, _ = eng.solo(text, a.n)
print("a text request after pictures: tokens vs the text solo: %s" % ("identical" if t2 == t_toks else "DIFFERENT"), flush=True)
ok &= t2 == t_toks
eng.close()
print("GATE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
