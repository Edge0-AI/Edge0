"""predsim.py - token-level prediction of a decode step's experts from the token history (n-gram match), as a prefetch into an LRU
cache, on DS4_TRACE decode traces.

  predsim.py <trace> <prompt ids file> <output ids file> [--cap GB] [--warm N] [--budget MB] [--k 2]

Before step i (its token known: the one the last step sampled) the experts of the step after the previous occurrence of the same
k-gram are the guess for every layer.  They are read ahead (up to --budget MB a step, hottest layers first), so a miss at use time
is only the ones the guess missed.  Reports demand hit rate, SSD MB a step (demand + wasted guesses) and the guess's recall /
precision.  The engine's lead time for these reads is a whole step (the SSD is idle while the CPU lane computes)."""
import argparse
import collections
import sys
sys.path.insert(0, r'C:\Users\AI-Server\Desktop\Strata\GitHub\wt-dsv41-win\tools\dsv41\win')
import cachesim as cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace')
    ap.add_argument('prompt')
    ap.add_argument('out')
    ap.add_argument('--cap', type=float, default=36.0)
    ap.add_argument('--warm', type=int, default=150)
    ap.add_argument('--budget', type=float, default=400.0, help='MB of reads ahead per step')
    ap.add_argument('--k', type=int, default=2)
    ap.add_argument('--sizes', default=r'C:\ds-ista-work\sizes.json')
    a = ap.parse_args()
    size = cs.Sizes(a.sizes)
    st = cs.read_trace(a.trace)
    p = [int(x) for x in open(a.prompt).read().strip().split(',')]
    o = [int(x) for x in open(a.out).read().strip().strip(',').split(',')]
    S = p + o
    cap = a.cap * 1e9
    routes = {}
    for i, s in enumerate(st):
        if i == 0:
            continue
        routes[i] = {layer: list(dict.fromkeys(rows[0])) for layer, rows, _ in s}
    pos = lambda i: len(p) + i - 1  # noqa: E731
    for mode in ('lru', 'lru+hist'):
        od = collections.OrderedDict()
        used = 0
        last = {}
        n = hit = 0
        demand_b = pref_b = wasted_b = 0.0
        rec = tot = prec = ptot = 0
        steps = 0
        for i in range(1, len(st)):
            key = tuple(S[pos(i) - a.k + 1:pos(i) + 1])
            pred = None
            if mode == 'lru+hist' and key in last:
                pred = routes[last[key]]
            last[key] = i
            counted = i >= a.warm
            pend = {}
            if pred is not None:
                budget = a.budget * 1e6
                for l in range(40):
                    for e in pred[l]:
                        kk = (l, e)
                        if kk in od or kk in pend:
                            continue
                        if budget < size(l):
                            continue
                        budget -= size(l)
                        pend[kk] = size(l)
                if counted:
                    pref_b += sum(pend.values())
                    for l in range(40):
                        a_ = set(routes[i][l])
                        b_ = set(pred[l])
                        rec += len(a_ & b_)
                        tot += len(a_)
                        prec += len(a_ & b_)
                        ptot += len(b_)
            steps += counted
            for l in range(40):
                sz = size(l)
                for e in routes[i][l]:
                    kk = (l, e)
                    if counted:
                        n += 1
                    if kk in od:
                        od.move_to_end(kk)
                        if counted:
                            hit += 1
                        continue
                    if kk in pend:   # read ahead: arrived in time
                        del pend[kk]
                        if counted:
                            hit += 1
                    elif counted:
                        demand_b += sz
                    od[kk] = sz
                    used += sz
                    while used > cap:
                        _, s0 = od.popitem(last=False)
                        used -= s0
            if counted:
                wasted_b += sum(pend.values())   # read ahead, not used this step (they stay in the cache only if inserted: not here)
        print('%-9s cap %.0f GB k=%d: demand hit %.1f%%; SSD per step: demand %.0f MB + guesses %.0f MB (wasted %.0f); guess recall %.1f%% precision %.1f%%' % (
            mode, a.cap, a.k, 100 * hit / max(1, n), demand_b / steps / 1e6, pref_b / steps / 1e6, wasted_b / steps / 1e6,
            100 * rec / max(1, tot), 100 * prec / max(1, ptot)))


if __name__ == '__main__':
    main()
