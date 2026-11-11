"""batch_gate.py - the parallel-requests gate for strata-q36 (the tools/batch_test.py idea over the serve protocol).

  python q36_batch_gate.py --exe E --args "--model M --max-context 8192 --batch 4" --slots 4 --n 128 [--golden G.json]
                       [--long 3000] [--keys "temperature=0.7 seed=3"] [--out F.json]

1. every prompt alone (GEN) on the same engine -> solo tokens (and, with --golden, equal to the base engine's);
2. the first --slots prompts through BGEN, admitted one after the other, then decoding together -> every slot's tokens
   must equal its solo tokens; the decode rate of the window phase;
3. --waves: the 10 prompts in waves of --slots (slot reuse: the cached state of a finished slot is taken over);
4. --long L: a prompt of about L tokens admitted into the last slot while the others decode (its prompt chunks run
   between their windows); the others' tokens must still equal their solo tokens, and so must the long one's.
"""
from pathlib import Path
import argparse, json, shlex, sys, time
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from q36_harness import Eng, load_prompts


def same(a, b):
    return a == b


ap = argparse.ArgumentParser()
ap.add_argument("--exe", required=True)
ap.add_argument("--args", default="")
ap.add_argument("--slots", type=int, default=4)
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--prompts", default=str(Path(__file__).resolve().parent / "parity_prompts.ids"))
ap.add_argument("--golden", default=None)
ap.add_argument("--keys", default="")
ap.add_argument("--long", type=int, default=0)
ap.add_argument("--waves", action="store_true")
ap.add_argument("--skip-solo-bench", action="store_true")
ap.add_argument("--out", default=None)
ap.add_argument("--log", default=None)
ap.add_argument("--env", default="")
a = ap.parse_args()
env = dict(kv.split("=", 1) for kv in a.env.split() if "=" in kv)
eng = Eng(a.exe, shlex.split(a.args), env=env, log=a.log)
print("ready:", eng.ready, "info:", eng.info, "load %.1fs" % eng.load_s, flush=True)
prompts = load_prompts(a.prompts)
ok = True
res = {"args": a.args, "slots": a.slots}

# 1. solo
solo, solo_rate = [], []
for i, ids in enumerate(prompts):
    toks, done, ttft, wall = eng.solo(ids, a.n, a.keys)
    solo.append(toks)
    solo_rate.append(len(toks) / max(1e-9, wall - (ttft or 0)))
print("solo: %d prompts, decode %.1f tok/s median" % (len(solo), sorted(solo_rate)[len(solo_rate) // 2]), flush=True)
if a.golden:
    g = json.load(open(a.golden))["tokens"]
    eq = [same(x, y) for x, y in zip(solo, g)]
    print("solo vs golden (base engine):", "identical %d of %d" % (sum(eq), len(eq)), flush=True)
    ok &= all(eq)
res["solo_tok_s_median"] = sorted(solo_rate)[len(solo_rate) // 2]

# 2. batch
S = min(a.slots, len(prompts))
reqs = [{"ids": prompts[i], "n": a.n, "slot": i, "keys": a.keys} for i in range(S)]
r = eng.batch(reqs)
eq = [same(r["tokens"][i], solo[i]) for i in range(S)]
print("batch %d: %d of %d slots identical to solo | wall %.2fs, window phase %d tokens in %.2fs = %.1f tok/s (aggregate)"
      % (S, sum(eq), S, r["wall"], r["tail_tokens"], r["tail_s"], r["tail_tok_s"]), flush=True)
for i, e in enumerate(eq):
    if not e:
        x, y = r["tokens"][i], solo[i]
        k = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
        print("  slot %d differs at token %d (lens %d / %d)" % (i, k, len(x), len(y)))
ok &= all(eq)
res["batch"] = {"slots": S, "identical": sum(eq), "wall": r["wall"], "tail_tok_s": r["tail_tok_s"], "tail_tokens": r["tail_tokens"]}

# 3. waves (slot reuse)
if a.waves:
    wave_ok = True
    for w0 in range(0, len(prompts), S):
        idx = list(range(w0, min(w0 + S, len(prompts))))
        rq = [{"ids": prompts[i], "n": a.n, "slot": j, "keys": a.keys} for j, i in enumerate(idx)]
        rr = eng.batch(rq)
        for j, i in enumerate(idx):
            if not same(rr["tokens"][j], solo[i]):
                wave_ok = False
                print("  wave %d prompt %d differs" % (w0 // S, i))
    print("waves of %d: %s" % (S, "all identical" if wave_ok else "DIFFERENT"), flush=True)
    ok &= wave_ok

# 4. a long prompt admitted while the others decode
if a.long and S >= 2:
    base = prompts[0]
    filler = []
    while len(filler) < a.long:
        for p in prompts:
            filler += p[3:-8] if len(p) > 12 else p
    longids = filler[:a.long] + prompts[1][-8:]
    toks, done, ttft, wall = eng.solo(longids, 48, a.keys)
    reqs = [{"ids": prompts[i], "n": a.n * 2, "slot": i, "keys": a.keys} for i in range(S - 1)]
    # the first S-1 are admitted first; then the long one is admitted while they decode
    longreq = {"ids": longids, "n": 48, "slot": S - 1, "keys": a.keys}
    solo2 = []
    for i in range(S - 1):
        t2, _, _, _ = eng.solo(prompts[i], a.n * 2, a.keys)
        solo2.append(t2)
    rr = eng.batch(reqs + [longreq])
    eq = [same(rr["tokens"][i], solo2[i]) for i in range(S - 1)] + [same(rr["tokens"][S - 1], toks)]
    print("long prompt (%d tokens) admitted beside %d decoding slots: %s | admission stamps %s"
          % (len(longids), S - 1, "%d of %d identical" % (sum(eq), len(eq)), ["%.2f" % x for x in rr["admit_stamps"]]), flush=True)
    ok &= all(eq)

res["ok"] = ok
if a.out:
    json.dump(res, open(a.out, "w"))
eng.close()
print("GATE", "PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 1)
