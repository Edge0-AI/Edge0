"""q36_interleave_probe.py --exe E --args "..." [--long 3000] [--slots 3] : what the slots that are decoding see while a new
request's long prompt is read: the tokens they produce during the read, and their longest pause between two tokens, against the
same without a read.  The read goes chunk by chunk with one decode window of the other slots between two chunks.

  python tools/qwen35/q36_interleave_probe.py --exe build/strata-q36 --args "--model M --max-context 8192 --batch 4" --long 3000
"""
import argparse, shlex, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from q36_harness import Eng, load_prompts

ap = argparse.ArgumentParser()
ap.add_argument("--exe", required=True)
ap.add_argument("--args", default="")
ap.add_argument("--long", type=int, default=3000)
ap.add_argument("--slots", type=int, default=3, help="slots decoding while the long prompt is read")
ap.add_argument("--n", type=int, default=600)
ap.add_argument("--log", default=None)
a = ap.parse_args()
eng = Eng(a.exe, shlex.split(a.args), log=a.log)
P = load_prompts(str(Path(__file__).resolve().parent / "parity_prompts.ids"))
filler = []
while len(filler) < a.long:
    for p in P:
        filler += p[3:-8]
longids = filler[:a.long] + P[1][-8:]
stamps = {s: [] for s in range(a.slots)}
for s in range(a.slots):   # admit the decoding slots
    ids = P[[0, 1, 2, 3, 5, 8][s % 6]]
    eng.send("BGEN %d %d %s" % (s, a.n, ",".join(map(str, ids))))
    while True:
        l = eng.line()
        if l.startswith("BADM"):
            break
        if l.startswith("BT "):
            stamps[int(l.split()[1])].append(time.time())
# 1. a stretch of decoding with no read
t_end = time.time() + 1.0
while time.time() < t_end:
    l = eng.line()
    if l.startswith("BT "):
        stamps[int(l.split()[1])].append(time.time())
base = {s: v[:] for s, v in stamps.items()}
# 2. the long request
t0 = time.time()
eng.send("BGEN %d 8 %s" % (a.slots, ",".join(map(str, longids))))
pp = []
t_adm = None
while True:
    l = eng.line()
    now = time.time()
    if l.startswith("BT "):
        stamps[int(l.split()[1])].append(now)
    elif l.startswith("PP "):
        pp.append(now)
    elif l.startswith("BADM"):
        t_adm = now
        break
read_s = t_adm - t0
during = {s: [t for t in v if t0 <= t <= t_adm] for s, v in stamps.items()}


def gaps(v):
    return [b - a_ for a_, b in zip(v, v[1:])]


rate0 = sum(len(v) for s, v in base.items()) / max(1e-9, 1.0) if False else None
n_during = sum(len(v) for v in during.values())
mx = max((max(gaps(v)) for v in during.values() if len(v) > 1), default=0.0)
mx0 = max((max(gaps(v[-60:])) for v in base.values() if len(v) > 2), default=0.0)
print("a prompt of %d tokens read in %.2f s (%d chunks, %.0f tok/s) beside %d decoding slots" % (len(longids), read_s, len(pp),
                                                                                          len(longids) / read_s, a.slots))
print("the %d decoding slots produced %d tokens during the read (%.1f tok/s together); longest pause between two tokens of a slot %.0f ms"
      " (while only decoding: %.0f ms)" % (a.slots, n_during, n_during / read_s, 1000 * mx, 1000 * mx0))
for s in range(a.slots):
    eng.send("BSTOP %d" % s)
eng.send("BSTOP %d" % a.slots)
time.sleep(1.0)
eng.close()
