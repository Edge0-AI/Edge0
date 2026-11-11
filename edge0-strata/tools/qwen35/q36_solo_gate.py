"""q36_solo_gate.py --exe E --args "..." --out F.json [--n 128] [--prompts F] [--keys K] : greedy tokens of the 10 parity
prompts, one request each through the serve protocol; the JSON is what two builds are compared on (cmp_gate.py)."""
from pathlib import Path
import argparse, json, shlex, sys, time
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from q36_harness import Eng, load_prompts

ap = argparse.ArgumentParser()
ap.add_argument("--exe", required=True)
ap.add_argument("--args", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--prompts", default=str(Path(__file__).resolve().parent / "parity_prompts.ids"))
ap.add_argument("--keys", default="")
ap.add_argument("--log", default=None)
a = ap.parse_args()
eng = Eng(a.exe, shlex.split(a.args), log=a.log)
print("ready", eng.ready, "info", eng.info, "load %.1fs" % eng.load_s, flush=True)
res = []
for i, ids in enumerate(load_prompts(a.prompts)):
    toks, done, ttft, wall = eng.solo(ids, a.n, a.keys)
    dec = done.split()
    print("p%02d %d tok  %.1f tok/s  %s" % (i, len(toks), len(toks) / max(1e-9, wall - (ttft or 0)), done[:70]), flush=True)
    res.append(toks)
json.dump({"args": a.args, "tokens": res}, open(a.out, "w"))
eng.close()
