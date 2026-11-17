"""cachesim.py - offline expert-cache policies on routing traces (DS4_TRACE files of strata-dsv41).

A trace is a run of records: u8 layer, u8 T (bit 7: guesses follow), T x k int16 expert ids [, u8 na1, u8 na2, T x (na1 + na2)
int16 guesses for the next layer and the one after].  One decode step (or verify window) is the records of layers 0..nl-1 in order.

Sizes are in bytes (an expert's real size in its layer: sizes.json), capacities in GB.  Policies:
  lru        one global LRU over (layer, expert)
  lfu        frequency with exponential decay (LRFU / CRF), evicts the lowest score
  static     the N hottest by training-trace frequency (a shipped per-set profile), no adaptation
  static+lru half the capacity pinned from the profile, half LRU
  belady     the offline optimum (farthest next use): the ceiling for any predictor of reuse
  lru+ahead  LRU plus a prefetch of the guesses for the next layer (top m): counts the wasted bytes
Reports: hit rate (by accesses and by bytes), SSD bytes per token (misses + wasted prefetch), the share of misses a guess covered.
"""
import argparse
import collections
import heapq
import json
import math
import os
import struct
import sys


WARM = 0   # accesses before this index warm the cache but are not counted


def read_trace(path, k=6):
    """-> list of steps; a step = list of (layer, [rows of expert ids], guesses or None)."""
    data = open(path, 'rb').read()
    pos, n = 0, len(data)
    steps, cur, last_layer = [], [], -1
    while pos + 2 <= n:
        layer, t = data[pos], data[pos + 1]
        pos += 2
        has_ahead = bool(t & 0x80)
        t &= 0x7f
        rows = []
        for _ in range(t):
            rows.append(list(struct.unpack_from('<%dh' % k, data, pos)))
            pos += 2 * k
        guesses = None
        if has_ahead:
            na1, na2 = data[pos], data[pos + 1]
            pos += 2
            guesses = []
            for _ in range(t):
                g1 = list(struct.unpack_from('<%dh' % na1, data, pos)) if na1 else []
                pos += 2 * na1
                g2 = list(struct.unpack_from('<%dh' % na2, data, pos)) if na2 else []
                pos += 2 * na2
                guesses.append((g1, g2))
        if layer <= last_layer and cur:
            steps.append(cur)
            cur = []
        last_layer = layer
        cur.append((layer, rows, guesses))
    if cur:
        steps.append(cur)
    return steps


def accesses(steps):
    """the flat access stream: (step, layer, expert) with the union over a window's rows, in layer order"""
    out = []
    for si, st in enumerate(steps):
        for layer, rows, _ in st:
            seen = []
            for r in rows:
                for e in r:
                    if e not in seen:
                        seen.append(e)
            for e in seen:
                out.append((si, layer, e))
    return out


class Sizes:
    def __init__(self, path):
        d = json.load(open(path))
        el = d['exp_layer']
        self.s = {int(k): float(v) for k, v in el.items()}

    def __call__(self, layer):
        return self.s.get(layer, 8.6e6)


def sim_lru(acc, size, cap):
    od = collections.OrderedDict()
    used = hits = hitb = tot = totb = 0
    for i, (_, l, e) in enumerate(acc):
        k = (l, e)
        sz = size(l)
        cnt = i >= WARM
        tot += cnt
        totb += sz * cnt
        if k in od:
            od.move_to_end(k)
            hits += cnt
            hitb += sz * cnt
            continue
        od[k] = sz
        used += sz
        while used > cap:
            _, s0 = od.popitem(last=False)
            used -= s0
    return hits, hitb, tot, totb


def sim_lfu(acc, size, cap, half_life_acc=20000):
    lam = math.log(2) / half_life_acc
    score, last, res = {}, {}, {}
    heap = []
    used = hits = hitb = tot = totb = 0
    for t, (_, l, e) in enumerate(acc):
        k = (l, e)
        sz = size(l)
        cnt = t >= WARM
        tot += cnt
        totb += sz * cnt
        # CRF: decayed count, kept in log space: v = log(count) + lam * t
        if k in res:
            hits += cnt
            hitb += sz * cnt
        old = score.get(k)
        v = lam * t
        score[k] = v if old is None else v + math.log1p(math.exp(old - v)) if old - v < 50 else old
        if k not in res:
            res[k] = sz
            used += sz
        heapq.heappush(heap, (score[k], k))
        while used > cap and heap:
            sc, kk = heapq.heappop(heap)
            if kk not in res or score[kk] != sc:
                continue
            if kk == k and len(res) == 1:
                break
            used -= res.pop(kk)
    return hits, hitb, tot, totb


def freq_table(acc):
    c = collections.Counter((l, e) for _, l, e in acc)
    return c


def pick_static(freq, size, cap):
    keys = sorted(freq, key=lambda k: -freq[k] / size(k[0]))
    chosen, used = set(), 0
    for k in keys:
        s = size(k[0])
        if used + s > cap:
            continue
        chosen.add(k)
        used += s
    return chosen


def sim_static(acc, size, chosen):
    hits = hitb = tot = totb = 0
    for i, (_, l, e) in enumerate(acc):
        sz = size(l)
        cnt = i >= WARM
        tot += cnt
        totb += sz * cnt
        if (l, e) in chosen:
            hits += cnt
            hitb += sz * cnt
    return hits, hitb, tot, totb


def sim_static_lru(acc, size, cap, chosen, pinned_bytes):
    od = collections.OrderedDict()
    used = hits = hitb = tot = totb = 0
    room = cap - pinned_bytes
    for i, (_, l, e) in enumerate(acc):
        k = (l, e)
        sz = size(l)
        cnt = i >= WARM
        tot += cnt
        totb += sz * cnt
        if k in chosen:
            hits += cnt
            hitb += sz * cnt
            continue
        if k in od:
            od.move_to_end(k)
            hits += cnt
            hitb += sz * cnt
            continue
        od[k] = sz
        used += sz
        while used > room:
            _, s0 = od.popitem(last=False)
            used -= s0
    return hits, hitb, tot, totb


def sim_belady(acc, size, cap):
    n = len(acc)
    nxt = [0] * n
    last = {}
    for i in range(n - 1, -1, -1):
        k = (acc[i][1], acc[i][2])
        nxt[i] = last.get(k, n + 1)
        last[k] = i
    res = {}   # k -> next use
    heap = []  # (-next use, k)
    used = hits = hitb = tot = totb = 0
    for i, (_, l, e) in enumerate(acc):
        k = (l, e)
        sz = size(l)
        cnt = i >= WARM
        tot += cnt
        totb += sz * cnt
        if k in res:
            hits += cnt
            hitb += sz * cnt
            res[k] = nxt[i]
            heapq.heappush(heap, (-nxt[i], k))
            continue
        if nxt[i] > n:      # never used again: do not keep
            continue
        res[k] = nxt[i]
        used += sz
        heapq.heappush(heap, (-nxt[i], k))
        while used > cap and heap:
            nu, kk = heapq.heappop(heap)
            if kk not in res or res[kk] != -nu:
                continue
            used -= size(kk[0])
            del res[kk]
    return hits, hitb, tot, totb


def sim_lru_ahead(steps, size, cap, m=6, depth=1):
    """LRU + prefetch of the top-m guesses (from the record of layer L-1 for layer L): a miss that a guess covered is a 'hidden'
    miss (its read started a layer early), a guess nobody used is wasted bytes."""
    od = collections.OrderedDict()
    used = 0
    hits = tot = miss = hidden = 0
    ssd = 0.0
    wasted = 0.0
    pend = {}   # (layer, expert) -> bytes prefetched, not yet used this step
    nsteps = len(steps)

    def put(k, sz):
        nonlocal used
        if k in od:
            od.move_to_end(k)
            return
        od[k] = sz
        used += sz
        while used > cap:
            _, s0 = od.popitem(last=False)
            used -= s0

    for st in steps:
        pend.clear()
        for layer, rows, guesses in st:
            sz = size(layer)
            seen = []
            for r in rows:
                for e in r:
                    if e not in seen:
                        seen.append(e)
            for e in seen:
                k = (layer, e)
                tot += 1
                if k in pend:
                    hidden += 1
                    del pend[k]
                    miss += 1
                    put(k, sz)
                elif k in od:
                    hits += 1
                    od.move_to_end(k)
                else:
                    miss += 1
                    ssd += sz
                    put(k, sz)
            for k, b in list(pend.items()):
                if k[0] <= layer:
                    wasted += b
                    del pend[k]
            if guesses:
                nl = layer + 1
                gs = []
                for g1, g2 in guesses:
                    for e in g1[:m]:
                        if e not in gs:
                            gs.append(e)
                for e in gs:
                    k = (nl, e)
                    if k in od or k in pend:
                        continue
                    pend[k] = size(nl)
                    ssd += size(nl)
    for b in pend.values():
        wasted += b
    return dict(hit=hits / max(1, tot), hidden=hidden / max(1, tot), ssd_per_step_MB=ssd / max(1, nsteps) / 1e6,
                wasted_per_step_MB=wasted / max(1, nsteps) / 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('traces', nargs='+', help='trace files: all are test sets; --train-on names the profile source')
    ap.add_argument('--sizes', default='C:/ds-ista-work/sizes.json')
    ap.add_argument('--caps', default='20,30,40,44', help='cache sizes in GB (VRAM hot set + pinned RAM)')
    ap.add_argument('--train', nargs='*', default=[], help='traces the static profile is built from (others than the test)')
    ap.add_argument('--warm-steps', type=int, default=0, help='steps at the start that fill the cache but are not counted')
    ap.add_argument('--skip', default='', help='policies to skip (comma): lru,lfu,static,static+lru,belady,ahead')
    a = ap.parse_args()
    size = Sizes(a.sizes)
    global WARM
    skip = set(a.skip.split(','))
    train_acc = []
    for t in a.train:
        train_acc += accesses(read_trace(t))
    for path in a.traces:
        steps = read_trace(path)
        acc = accesses(steps)
        nsteps = len(steps)
        WARM = 0
        if a.warm_steps:
            WARM = next((i for i, x in enumerate(acc) if x[0] >= a.warm_steps), len(acc))
            nsteps -= a.warm_steps
        uniq = len({(l, e) for _, l, e in acc})
        print('   unique experts touched: %d of 15360 (%.1f%%), %.1f GB' % (uniq, 100 * uniq / 15360, sum(size(l) for l, e in {(l, e) for _, l, e in acc}) / 1e9))
        print('== %s: %d steps, %d expert accesses (%.1f a step)' % (os.path.basename(path), nsteps, len(acc), len(acc) / max(1, nsteps)))
        tb = sum(size(l) for _, l, _ in acc) / nsteps
        print('   bytes a step (union per layer): %.2f GB' % (tb / 1e9))
        print('   %-14s %6s %9s %9s %12s' % ('policy', 'capGB', 'hit(acc)', 'hit(bytes)', 'SSD MB/step'))
        for cg in [float(x) for x in a.caps.split(',')]:
            cap = cg * 1e9
            rows = []
            if 'lru' not in skip:
                rows.append(('lru', sim_lru(acc, size, cap)))
            if 'lfu' not in skip:
                rows.append(('lfu(decay)', sim_lfu(acc, size, cap)))
            if train_acc and 'static' not in skip:
                ch = pick_static(freq_table(train_acc), size, cap)
                rows.append(('static', sim_static(acc, size, ch)))
                ch2 = pick_static(freq_table(train_acc), size, cap / 2)
                rows.append(('static+lru', sim_static_lru(acc, size, cap, ch2, cap / 2)))
            if 'belady' not in skip:
                rows.append(('belady', sim_belady(acc, size, cap)))
            for name, (h, hb, t, tb_) in rows:
                print('   %-14s %6.0f %8.1f%% %8.1f%% %12.1f' % (name, cg, 100 * h / t, 100 * hb / tb_, (tb_ - hb) / nsteps / 1e6))
            if 'ahead' not in skip and any(g for st in steps for _, _, g in st):
                for m in (6, 8, 12):
                    r = sim_lru_ahead(steps, size, cap, m=m)
                    print('   lru+ahead(top%-2d) %4.0f   hit %.1f%% (+%.1f%% hidden)  SSD %.0f MB/step (wasted %.0f)' % (
                        m, cg, 100 * r['hit'], 100 * r['hidden'], r['ssd_per_step_MB'], r['wasted_per_step_MB']))


if __name__ == '__main__':
    main()
