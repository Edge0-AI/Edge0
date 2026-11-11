"""proto_edge.py - the protocol's edges for parallel requests (engine level): max_new 1, BSTOP, a busy slot, STOP during an
admission's prompt read, slot reuse (the held state), sampled requests in a batch against the same request alone, a
request that reaches the context's end."""
from pathlib import Path
import argparse, shlex, sys, time
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from q36_harness import Eng, load_prompts

ap = argparse.ArgumentParser()
ap.add_argument("--exe", required=True)
ap.add_argument("--args", default="")
ap.add_argument("--log", default=None)
a = ap.parse_args()
eng = Eng(a.exe, shlex.split(a.args), log=a.log)
P = load_prompts(str(Path(__file__).resolve().parent / "parity_prompts.ids"))
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (" - " + extra if extra else ""), flush=True)


def lines_until(pred, timeout=120):
    out = []
    t0 = time.time()
    while time.time() - t0 < timeout:
        l = eng.line(timeout)
        out.append(l)
        if pred(l):
            return out
    return out


# 1. max_new 1: T, DONE, BADM slot 0
eng.send("BGEN 0 1 " + ",".join(map(str, P[0])))
r = lines_until(lambda l: l.startswith("BADM"))
check("BGEN max_new 1 ends at the admission", r[-1] == "BADM 0 0" and any(l.startswith("T ") for l in r), str(r[-3:]))

# 2. a long answer, BSTOP after a few tokens, then a longer request on the held state
eng.send("BGEN 0 200 " + ",".join(map(str, P[1])))
r = lines_until(lambda l: l.startswith("BADM"))
check("BGEN max_new 200 continues", r[-1] == "BADM 0 1", r[-1])
bts = []
while len(bts) < 5:
    l = eng.line()
    if l.startswith("BT 0"):
        bts.append(int(l.split()[2]))
# the slot is busy: another BGEN for it is refused (BT lines of the running slot may come first)
eng.send("BGEN 0 5 " + ",".join(map(str, P[2])))
seen = lines_until(lambda l: l.startswith("ERR") or l.startswith("BADM"))
for l in seen:
    if l.startswith("BT 0"):
        bts.append(int(l.split()[2]))
check("BGEN into a busy slot is refused", seen[-1].startswith("ERR BGEN"), seen[-1])
eng.send("BSTOP 0")
done = lines_until(lambda l: l.startswith("BDONE 0"))
for l in done:
    if l.startswith("BT 0"):
        bts.append(int(l.split()[2]))
check("BSTOP ends the slot with cancel", done[-1].split()[3] == "cancel", done[-1])
first = [int(l.split()[1]) for l in r if l.startswith("T ")]
# the slot holds the prompt and every token a window fed (all but the last one): a continuation extends exactly that
eng.send("BGEN 0 3 " + ",".join(map(str, P[1] + first + bts)))
r2 = lines_until(lambda l: l.startswith("BADM"))
dn = [l for l in r2 if l.startswith("DONE")][0].split()
check("a continuation reuses the slot's held state", int(dn[8]) == len(P[1]) + len(first) + len(bts) - 1,
      "reused %s of %d" % (dn[8], len(P[1] + first + bts)))
lines_until(lambda l: l.startswith("BDONE 0"))

# 3. STOP during an admission's prompt read
filler = []
while len(filler) < 3000:
    for p in P:
        filler += p[3:-8]
filler = filler[:3000]
eng.send("BGEN 1 8 " + ",".join(map(str, filler)))
time.sleep(0.15)
eng.send("STOP")
r3 = lines_until(lambda l: l.startswith("BADM"))
dn = [l for l in r3 if l.startswith("DONE")][0].split()
pp = [l for l in r3 if l.startswith("PP ")]
check("STOP during the prompt read ends it with cancel", dn[5] == "cancel" and r3[-1] == "BADM 1 0",
      "%d PP line(s), DONE %s" % (len(pp), " ".join(dn[:6])))
# the engine is in step: a normal request on the same slot works
eng.send("BGEN 1 6 " + ",".join(map(str, P[3])))
r4 = lines_until(lambda l: l.startswith("BADM"))
check("a request after the cancelled read works", r4[-1] == "BADM 1 1", r4[-1])
lines_until(lambda l: l.startswith("BDONE 1"))

# 4. sampled: batch against alone, same seed
keys = "temperature=0.8 top_k=40 top_p=0.95 seed=11"
toks, _, _, _ = eng.solo(P[4], 40, keys)
res = eng.batch([{"ids": P[4], "n": 40, "slot": 2, "keys": keys}, {"ids": P[5], "n": 40, "slot": 3, "keys": "temperature=0.5 seed=3"}])
check("a sampled request in a batch equals the same request alone", res["tokens"][0] == toks,
      "%d / %d tokens" % (len(res["tokens"][0]), len(toks)))

# 5. the context's end: a prompt of ctx - 40 tokens and max_new 400 finishes with length
ctx = int(eng.ready.split()[1])
long_prompt = (filler * 4)[:ctx - 60]
eng.send("BGEN 3 400 " + ",".join(map(str, long_prompt)))
r5 = lines_until(lambda l: l.startswith("ERR") or l.startswith("BADM"))
check("a request that cannot fit its answer is refused with the engine's message", r5[-1].startswith("ERR prompt"), r5[-1][:80])
eng.send("BGEN 3 50 " + ",".join(map(str, long_prompt[:ctx - 80])))
r6 = lines_until(lambda l: l.startswith("BADM"))
bd = lines_until(lambda l: l.startswith("BDONE 3"))
check("a request to the context's end finishes", bd[-1].split()[3] in ("length", "stop"), bd[-1])

eng.close()
print("GATE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
