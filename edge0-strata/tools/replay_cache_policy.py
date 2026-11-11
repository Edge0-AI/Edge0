"""Offline replay of a --dump-routing trace through the expert cache policies.

    python replay_cache_policy.py TRACE.bin PROFILE.bin SLOTS [--every 4 --decay 0.7 --swaps 96 --margin 1.5 --mincand 2.0]

Trace record = int32 layer, int32 k, k int32 ids, k float32 weights; the verify window writes one record per token and
layer, so a layer group is the run of records with one layer and a window starts when the layer does not rise.
The simulated adaptive tier is generate.cpp's adapt(): usage += 1 per routed entry, a round every `every` windows, per
layer the best non-resident (usage >= mincand) against the worst resident (needs margin), the best `swaps` over all
layers, usage *= decay.  Reports the share of routed entries served by the cache (the engine's "hits / all routed").
"""
import sys
import numpy as np

NL, NE = 48, 512


def load_trace(path):
    raw = np.fromfile(path, dtype=np.int32)
    k = int(raw[1])
    rec = 2 + 2 * k
    raw = raw[: (len(raw) // rec) * rec].reshape(-1, rec)
    layer = raw[:, 0]
    ids = raw[:, 2:2 + k]
    # layer groups
    chg = np.flatnonzero(np.diff(layer) != 0) + 1
    starts = np.concatenate([[0], chg])
    ends = np.concatenate([chg, [len(layer)]])
    windows, cur, last = [], None, -1
    for s, e in zip(starts, ends):
        l = int(layer[s])
        if cur is None or l <= last:
            if cur is not None:
                windows.append(cur)
            cur = []
        cur.append((l, ids[s:e]))
        last = l
    if cur:
        windows.append(cur)
    # keep complete windows only
    return [w for w in windows if len(w) == NL], k


def load_profile(path, slots):
    with open(path, "rb") as f:
        assert f.read(4) == b"STRP"
        hdr = np.frombuffer(f.read(20), dtype=np.uint32)
        n_ranked = int(hdr[4])
        pairs = np.frombuffer(f.read(4 * n_ranked), dtype=np.uint16).reshape(-1, 2)
    return pairs[:slots]


class Policy:
    def __init__(self, every=4, decay=0.7, swaps=96, margin=1.5, mincand=2.0, mode='entries', mingain=0.0, posw=1.0):
        self.every, self.decay, self.swaps, self.margin, self.mincand = every, decay, swaps, margin, mincand
        self.mode, self.mingain, self.posw = mode, mingain, posw


def adapt_round(res, usage, pol):
    sw = []
    for l in range(NL):
        u = usage[l]
        r = res[l]
        cand = np.flatnonzero(~r & (u >= pol.mincand))
        vict = np.flatnonzero(r)
        if len(cand) == 0 or len(vict) == 0:
            continue
        cand = cand[np.argsort(-u[cand], kind="stable")]
        vict = vict[np.argsort(u[vict], kind="stable")]
        n = min(len(cand), len(vict))
        gain = u[cand[:n]] - u[vict[:n]]
        ok = u[cand[:n]] >= u[vict[:n]] + pol.margin
        # the engine stops at the first pair that does not clear the margin
        bad = np.flatnonzero(~ok)
        n2 = int(bad[0]) if len(bad) else n
        for i in range(n2):
            sw.append((gain[i], l, int(cand[i]), int(vict[i])))
    sw.sort(key=lambda t: -t[0])
    sw = [t for t in sw if t[0] >= pol.mingain][: pol.swaps]
    for g, l, i, o in sw:
        res[l, i] = True
        res[l, o] = False
    return len(sw)


def simulate(windows, init, pol, reset_every=0):
    res = np.zeros((NL, NE), bool)
    res[init[:, 0], init[:, 1]] = True
    usage = np.zeros((NL, NE))
    hits = ent = swaps = 0
    miss_ent = miss_dist = 0
    for wi, w in enumerate(windows):
        for l, ids in w:
            r = res[l, ids]
            hits += int(r.sum())
            ent += r.size
            ms = ids[~r]
            miss_ent += len(ms)
            miss_dist += len(np.unique(ms))
            if pol.mode == 'distinct':
                usage[l, np.unique(ids)] += 1.0
            elif pol.posw != 1.0:
                wts = np.repeat(pol.posw ** np.arange(ids.shape[0]), ids.shape[1])
                np.add.at(usage[l], ids.ravel(), wts)
            else:
                np.add.at(usage[l], ids.ravel(), 1.0)
        if (wi + 1) % pol.every == 0:
            swaps += adapt_round(res, usage, pol)
            usage *= pol.decay
    return dict(share=hits / max(1, ent), swaps=swaps, windows=len(windows), entries=ent, miss_ent=miss_ent,
                miss_dist=miss_dist)


def static_oracle(windows, slots):
    cnt = np.zeros((NL, NE))
    for w in windows:
        for l, ids in w:
            np.add.at(cnt[l], ids.ravel(), 1.0)
    flat = np.argsort(-cnt.ravel(), kind="stable")[:slots]
    hit = cnt.ravel()[flat].sum() / cnt.sum()
    return hit


if __name__ == "__main__":
    tr, prof, slots = sys.argv[1], sys.argv[2], int(sys.argv[3])

    def opt(name, default, cast=float):
        return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default

    pol = Policy(every=opt("--every", 4, int), decay=opt("--decay", 0.7), swaps=opt("--swaps", 96, int),
                 margin=opt("--margin", 1.5), mincand=opt("--mincand", 2.0))
    windows, k = load_trace(tr)
    print("windows", len(windows), "k", k, "tokens/window", np.mean([len(w[0][1]) for w in windows]))
    init = load_profile(prof, slots)
    print("static oracle (top-%d by this trace's own counts): %.4f" % (slots, static_oracle(windows, slots)))
    base = simulate(windows, init, Policy(every=10**9))
    print("profile only, no adaptation: %.4f" % base["share"])
    cur = simulate(windows, init, pol)
    print("policy every %d, swaps %d, decay %.2f: share %.4f, swaps %d (%.2f/window); miss entries %d, distinct misses %d "
          "(%.1f%% fewer reads)" % (pol.every, pol.swaps, pol.decay, cur["share"], cur["swaps"], cur["swaps"] / cur["windows"],
                                    cur["miss_ent"], cur["miss_dist"],
                                    100 * (1 - cur["miss_dist"] / max(1, cur["miss_ent"]))))
