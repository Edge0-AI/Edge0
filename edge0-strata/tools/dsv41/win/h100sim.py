"""h100sim.py - replay the H100 routing corpus (1.78M tokens, code/agent/chat/multi/math/long, DS4_TRACE format, see
phaseA-tests/h100/trace_tools.py) through an LRU expert cache at this PC's sizes, with and without a profile warm start.

  h100sim.py <trace.bin[.zst]> <profile.bin> [--tokens N] [--skip S] [--caps 30,40,44] [--sizes sizes.json]

Prints the demand hit rate by bytes after a warm-up of `--warm` tokens, for: empty start LRU; LRU started with the profile's
hottest experts in (the engine's preload) and the same cold-start window's hit rate (first 2000 tokens), and a static top-N
(profile only, no adaptation)."""
import argparse
import collections
import io
import json
import struct
import sys

import numpy as np


def decode(path, max_tokens, skip):
    if path.endswith('.zst'):
        import zstandard
        raw = zstandard.ZstdDecompressor().stream_reader(open(path, 'rb')).read()
    else:
        raw = open(path, 'rb').read()
    buf = np.frombuffer(raw, dtype=np.uint8)
    pos, n = 0, len(buf)
    per = {l: [] for l in range(40)}
    got = [0] * 40
    while pos < n:
        b0 = int(buf[pos])
        if b0 & 0x80:
            l = b0 & 0x7F
            T = int(buf[pos + 1]) | (int(buf[pos + 2]) << 8)
            pos += 3
        else:
            l = b0
            T = int(buf[pos + 1]) & 0x7F
            pos += 2
        sz = T * 12
        if pos + sz > n:
            break
        per[l].append(buf[pos:pos + sz].view(np.int16).reshape(T, 6))
        pos += sz
        if min(got) >= skip + max_tokens:
            break
        got[l] += T
    arr = []
    for l in range(40):
        a = np.concatenate(per[l])[skip:skip + max_tokens]
        arr.append(a)
    m = min(len(a) for a in arr)
    return np.stack([a[:m] for a in arr], axis=1)   # [tokens][layer][6]


def lru(acc_iter, size, cap, init=None, warm=0):
    od = collections.OrderedDict()
    used = 0
    if init:
        for k in init:
            od[k] = size[k[0]]
            used += size[k[0]]
    hb = tb = 0.0
    first_hb = first_tb = 0.0
    for i, (t, k) in enumerate(acc_iter):
        sz = size[k[0]]
        cnt = t >= warm
        early = t < 2000
        if k in od:
            od.move_to_end(k)
            if cnt:
                hb += sz
            if early:
                first_hb += sz
        else:
            od[k] = sz
            used += sz
            while used > cap:
                _, s0 = od.popitem(last=False)
                used -= s0
        if cnt:
            tb += sz
        if early:
            first_tb += sz
    return 100 * hb / max(tb, 1), 100 * first_hb / max(first_tb, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace')
    ap.add_argument('profile')
    ap.add_argument('--tokens', type=int, default=40000)
    ap.add_argument('--skip', type=int, default=0)
    ap.add_argument('--warm', type=int, default=4000)
    ap.add_argument('--caps', default='30,40')
    ap.add_argument('--sizes', default=r'C:\ds-ista-work\sizes.json')
    a = ap.parse_args()
    d = json.load(open(a.sizes))
    size = {int(k): float(v) for k, v in d['exp_layer'].items()}
    tr = decode(a.trace, a.tokens, a.skip)
    prof = np.fromfile(a.profile, dtype=np.uint32).reshape(-1, 384)[:40]
    print('trace', tr.shape, 'tokens', len(tr))

    def acc():
        for t in range(len(tr)):
            for l in range(40):
                for e in tr[t, l]:
                    yield t, (l, int(e))
    dens = sorted(((prof[l, e] / size[l], l, e) for l in range(40) for e in range(384)), reverse=True)
    for cg in [float(x) for x in a.caps.split(',')]:
        cap = cg * 1e9
        r0, f0 = lru(acc(), size, cap, None, a.warm)
        order, used = [], 0.0
        for _, l, e in dens:
            if used + size[l] > cap:
                break
            order.append((l, e))
            used += size[l]
        r1, f1 = lru(acc(), size, cap, list(reversed(order)), a.warm)   # hottest last = most recent
        # static: top-N of the profile, no adaptation
        sel = set(order)
        hb = tb = 0.0
        for t, k in acc():
            if t >= a.warm:
                tb += size[k[0]]
                if k in sel:
                    hb += size[k[0]]
        print('cap %.0f GB: LRU from empty: steady hit %.1f%% (first 2000 tokens %.1f%%); LRU from the profile\'s hottest: steady %.1f%% (first 2000 tokens %.1f%%); static top-N alone %.1f%%' % (
            cg, r0, f0, r1, f1, 100 * hb / tb))


if __name__ == '__main__':
    main()
