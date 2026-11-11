"""Interleaved A/B of per-request strata_tune settings (no restart), with rotated arm order.
usage: python3 ab.py MODE ROUNDS 'A_json' 'B_json' ['C_json' ...]
  MODE: solo  = one 300-token decode per round and arm (tok/s from the stream)
        c2    = 2 concurrent 300-token decodes (aggregate tok/s)
        read  = 24k-token fresh read, 1 token out (prefill tok/s)
        mix   = a fresh ~48k read while one stream decodes (read tok/s + the stream's tok/s overall)
Order: round r visits the arms starting at arm (r mod n) (A,B / B,A / A,B ...), so no arm always follows the same
neighbour; every arm gets a fresh unique prompt. Give the same JSON twice as arms A and B to run an A/A control
(it shows the bias of the method). Prints per-arm median/mean/min/max, the paired ratio against arm A (median of per-round
ratios), and the mean ratio by position in the round."""
import json, statistics as st, sys, threading, time, urllib.request

MODE, ROUNDS = sys.argv[1], int(sys.argv[2])
ARMS = [json.loads(a) for a in sys.argv[3:]]
N = len(ARMS)
BASE = "http://127.0.0.1:8000"
MODEL = "qwen3.8-flash-next-ud-q4_k_xl"
KEY = open("/opt/hyperqwen-swift/api_key.txt").read().strip()
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}
SALT = int(time.time()) % 100000
TOP = ["the history of navigation", "how a jet engine works", "the life cycle of a star", "the rules of chess",
       "the economics of container shipping", "how vaccines train the immune system", "the design of suspension bridges",
       "how compilers optimize loops"]


def hay(n, seed):
    w = ["amber", "birch", "cedar", "delta", "ember", "fjord", "grove", "harbor", "iris", "juniper"]
    return "\n".join(f"Entry {i} (run {seed}): the {w[(i * 3 + seed) % 10]} station {(i * 7 + seed * 13) % 997} logged activity." for i in range(n))


def run(prompt, mt, tune, out, key):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": mt, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    if tune:
        body["strata_tune"] = tune
    r = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), H)
    t0 = time.time(); first = last = None; ct = pt = 0
    with urllib.request.urlopen(r, timeout=1800) as f:
        for raw in f:
            l = raw.decode().strip()
            if not l.startswith("data:") or l.endswith("[DONE]"):
                continue
            d = json.loads(l[5:])
            if d.get("usage"):
                ct, pt = d["usage"]["completion_tokens"], d["usage"]["prompt_tokens"]
            for c in d.get("choices", []):
                if c.get("delta", {}).get("content"):
                    now = time.time(); first = first or now; last = now
    out[key] = dict(ct=ct, pt=pt, t0=t0, first=first, last=last, end=time.time(),
                    dec=((ct - 1) / (last - first)) if first and last and last > first and ct > 1 else float("nan"))


def one(arm, rnd, ai):
    tag = f"{SALT}-{rnd}-{ai}"
    out = {}
    if MODE == "solo":
        run(f"[{tag}] Write a long detailed explanation of {TOP[rnd % 8]}.", 300, arm, out, 0)   # same topic for every arm of a round
        return out[0]["dec"]
    if MODE == "long":   # decode on a ~60k-token context: a shared prefix (read once, then cached), a unique question per arm and round
        run(hay(2500, SALT) + f"\n\n[{tag}] Question: write a long detailed explanation of {TOP[rnd % 8]}, citing the entries above where useful.", 300, arm, out, 0)
        return out[0]["dec"]
    if MODE == "c2":
        ts = [threading.Thread(target=run, args=(f"[{tag}-{i}] Write a long detailed essay about {TOP[(rnd + i) % 8]}.", 300, arm, out, i)) for i in range(2)]
        for t in ts: t.start()
        for t in ts: t.join()
        return sum(o["dec"] for o in out.values())
    if MODE == "read":
        run(hay(1000, SALT * 7 + rnd * 3 + ai) + "\n\nQ: say OK.", 1, arm, out, 0)
        return out[0]["pt"] / (out[0]["end"] - out[0]["t0"])
    if MODE == "mix":
        d = threading.Thread(target=run, args=(f"[{tag}] Write a very long essay about {TOP[rnd % 8]}.", 900, arm, out, "d")); d.start()
        time.sleep(6)
        run(hay(2000, SALT * 11 + rnd * 5 + ai) + "\n\nQ: say OK.", 1, arm, out, "big")
        d.join()
        read_s = out["big"]["end"] - out["big"]["t0"]
        return (out["big"]["pt"] / read_s, out["d"]["ct"] / (out["d"]["end"] - out["d"]["t0"]))
    raise SystemExit("bad mode")


for _ in range(3):
    run(f"[{SALT}w{_}] Say hi.", 8, {}, {}, 0)
res = [[] for _ in ARMS]
pos = [[] for _ in ARMS]
for r in range(ROUNDS):
    order = [(r + k) % N for k in range(N)]
    for p, ai in enumerate(order):
        v = one(ARMS[ai], r, ai)
        res[ai].append(v); pos[ai].append(p)
        print(f"round {r} pos {p} arm {ai} {json.dumps(ARMS[ai])}: {round(v, 1) if isinstance(v, float) else tuple(round(x, 1) for x in v)}", flush=True)


def scal(x, k=0):
    return x[k] if isinstance(x, tuple) else x


for k, label in ((0, "main metric"),) + (((1, "stream tok/s during the read"),) if MODE == "mix" else ()):
    print(f"\n== {MODE}: {label} ==")
    base = [scal(x, k) for x in res[0]]
    for ai, arm in enumerate(ARMS):
        v = [scal(x, k) for x in res[ai]]
        ratios = [b / a for a, b in zip(base, v) if a == a and b == b and a > 0]
        print(f"arm {ai} {json.dumps(arm):<44} median {st.median(v):7.1f} mean {st.mean(v):7.1f} min {min(v):7.1f} max {max(v):7.1f}"
              + (f" | vs A: median ratio {st.median(ratios):.3f} mean {st.mean(ratios):.3f} (n={len(ratios)})" if ai else ""))
    # position bias: mean of (value / round mean) by position in the round
    byp = {}
    for r in range(ROUNDS):
        vals = [scal(res[ai][r], k) for ai in range(N)]
        m = st.mean(vals)
        for ai in range(N):
            byp.setdefault(pos[ai][r], []).append(scal(res[ai][r], k) / m)
    print("position bias (value / round mean): " + ", ".join(f"pos{p}: {st.mean(v):.3f}" for p, v in sorted(byp.items())))
