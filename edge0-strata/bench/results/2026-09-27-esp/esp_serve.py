"""Serve-mode checks of the experimental-speed-projection vector (exact mode: fixed experts, no PCIe share).

    python esp_serve.py run ARM      # ARM = stock | vec; writes esp-serve/<arm>.json
    python esp_serve.py compare

Per arm: the first greedy token after many prefixes of three texts (short ones go through the verify windows, long
ones through the batched prompt path; the conversation cache reuses the earlier prefix), and three 256-token greedy
generations.  The vec arm runs everything with cvec=1, then cvec=0, then cvec=1 again.
"""
import json
import os
import sys
import threading

S = os.path.dirname(os.path.abspath(__file__))
E = r"C:\Users\AI-Server\Desktop\Strata\Public\Engine"
sys.path.insert(0, E)
from serve.server import StrataEngine  # noqa: E402

EXE = E + r"\build-dev\strata.exe"
VEC = r"C:\Users\AI-Server\Desktop\Strata\experimental-speed-projection\Qwen3.8-Flash-Next-experimental-speed-projection.gguf"
TEXTS = {"code": S + r"\p512.ids", "doc": S + r"\4k-02-long-doc-1k.ids", "chat": S + r"\4k-04-chat-1k.ids"}
OUT = S + r"\esp-serve"
VEC_ARGS = ["--control-vector-scaled", VEC + ":1.0", "--control-vector-layer-range", "4", "44",
            "--cvec-mode", "project", "--cvec-dir", "per-layer"]


def ids_of(name):
    return [int(x) for x in open(TEXTS[name]).read().replace(",", " ").split()]


def engine(extra):
    c = json.load(open(S + r"\cfg-rel.json"))
    a = list(c["args"])
    a[a.index("--expert-cache") + 1] = "4000"
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(c["lib_dirs"]) + os.pathsep + env["PATH"]
    return StrataEngine(EXE, a + ["--pcie-frac", "0", "--adapt-swaps", "0"] + extra, cwd=c["cwd"],
                        log=OUT + r"\engine.log", env=env)


def gen(eng, ids, n, sampling):
    toks = [t for t in eng.generate(ids, n, sampling, threading.Event()) if t is not None]
    return toks, dict(eng.last)


def suite(eng, sampling):
    out = {"first": {}, "long": {}}
    for name in TEXTS:
        ids = ids_of(name)
        # short prefixes growing by 4 (each continues the live session through the verify windows), then long ones
        # SHRINKING (no cached prefix, so each is read from 0 by the batched prompt path)
        lens = list(range(32, 72, 4)) + list(range(min(len(ids) - 1, 700), 95, -48))
        out["first"][name] = {L: gen(eng, ids[:L], 1, sampling)[0][0] for L in lens}
    for name in TEXTS:
        toks, last = gen(eng, ids_of(name)[:300], 256, sampling)
        out["long"][name] = {"tokens": toks, "last": last}
    return out


def run(arm):
    os.makedirs(OUT, exist_ok=True)
    eng = engine(VEC_ARGS if arm == "vec" else [])
    res = {"info": eng.info}
    if arm == "stock":
        res["stock"] = suite(eng, {})
    else:
        res["on"] = suite(eng, {"experimental_speed_projection": True})
        res["off"] = suite(eng, {"experimental_speed_projection": False})
        res["on2"] = suite(eng, {"experimental_speed_projection": True})
    eng.close()
    json.dump(res, open(f"{OUT}\\{arm}.json", "w"))
    print(arm, "done", eng.info, flush=True)


def compare():
    import numpy as np
    sys.path.insert(0, S)
    from esp_kl import load, OUT as KL
    st = json.load(open(OUT + r"\stock.json"))
    ve = json.load(open(OUT + r"\vec.json"))
    print("engine INFO (vec):", ve["info"])

    def eq_first(a, b):
        tot = same = 0
        for name in a:
            for L, t in a[name].items():
                tot += 1
                same += int(b[name][L] == t)
        return same, tot

    def eq_long(a, b):
        return all(a[n]["tokens"] == b[n]["tokens"] for n in a), \
            [(n, next((i for i, (x, y) in enumerate(zip(a[n]["tokens"], b[n]["tokens"])) if x != y), None)) for n in a]

    print("1. vector loaded, switched off == no vector")
    print("   first tokens: %d / %d identical" % eq_first(st["stock"]["first"], ve["off"]["first"]))
    print("   256-token generations identical:", *eq_long(st["stock"]["long"], ve["off"]["long"]))
    print("2. repeatable: on, off, on again")
    print("   first tokens: %d / %d identical" % eq_first(ve["on"]["first"], ve["on2"]["first"]))
    print("   256-token generations identical:", *eq_long(ve["on"]["long"], ve["on2"]["long"]))
    print("3. the prompt paths against the per-token path's teacher-forced argmax (row L-1 predicts token L)")
    for arm, first in (("stock", st["stock"]["first"]), ("on", ve["on"]["first"])):
        for ref in ("stock", "proj"):
            agree = tot = 0
            for name in first:
                d = load(f"{KL}\\{name}-{ref}.bin")
                for L, t in first[name].items():
                    L = int(L)
                    if L - 1 < d.shape[0]:
                        tot += 1
                        agree += int(int(np.argmax(d[L - 1])) == t)
            print(f"   served {arm:<5} vs per-token {ref:<5}: {agree} / {tot}")
    print("4. how different the text is: 256-token generations, on vs stock, first difference")
    print("  ", eq_long(st["stock"]["long"], ve["on"]["long"])[1])
    print("5. decode speed (exact mode: fixed experts, no PCIe share - not the normal setup)")
    for label, r in (("stock", st["stock"]), ("vec off", ve["off"]), ("vec on", ve["on"]), ("vec on 2", ve["on2"])):
        g = sum(v["last"]["generated"] for v in r["long"].values())
        ms = sum(v["last"]["decode_ms"] for v in r["long"].values())
        acc = sum(v["last"].get("drafts_accepted", 0) for v in r["long"].values())
        off = sum(v["last"].get("drafts_offered", 0) for v in r["long"].values())
        print(f"   {label:<9} {g / ms * 1000:6.1f} tok/s  drafts accepted {acc}/{off}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        compare()
