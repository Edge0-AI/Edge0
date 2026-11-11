"""tools/bench_eviction.py - replay a `--dump-routing` trace against candidate ExpertCache eviction policies.

`ExpertCache::admit` never evicts: eviction policy is the measured question (R4.1's LFU-decay vs LRU sweep,
`include/strata/core/expert_cache.hpp`), and a placeholder policy would set the hit rate that everything
downstream is sized against.  This tool is the harness for that measurement, without a GPU: a policy's hit
rate depends only on the ORDER the run routed (layer, expert) pairs, and `--dump-routing` records exactly
that order.  Replay the trace through a simulated slot pool per policy and the rates are comparable with
the engine's own `h`, at any slot count, on any machine.

    python tools/bench_eviction.py TRACE... [--slots 4105,8000] [--per-layer] [--windows 4]
                                 [--policies fill-only,lru,lfu-decay,oracle] [--decay-window 65536]
                                 [--prefill data/expert-profile.bin] [--n-expert 512] [--csv out.csv]

Policies (`--policies`, default all four):

  fill-only   the engine's current admission: compulsory miss, never evict.  The baseline a policy has
              to beat, and the answer for any run whose trace never exceeds the slot count.
  lru         evict the least-recently-used resident.
  lfu-decay   frequency counts halved every --decay-window lookups, so a hot-once expert fades and the
              cache can follow a topic shift; evict the lowest count, the oldest on a tie.  (Halving the
              table is O(slots) per window; this instead DOUBLES new increments at each window boundary,
              which scales every past access by the same 1/2 and leaves the table untouched.  Comparisons
              stay exact: counts are sums of powers of two, below 2^53 for any realistic trace.)
  oracle      Belady's MIN: evict the resident whose next use is furthest in the future.  Not
              implementable online - it reads the trace ahead - but it is the upper bound any policy can
              reach on this trace, so it says how much headroom a real policy is leaving.

Scope mirrors `ExpertCache`: global (one shared pool, the engine's default) or `--per-layer` (R4.2g:
layer `l` owns slots `[l*q, (l+1)*q)`, `q = slots // n_layers`, the last layer taking the remainder, and
eviction stays inside the layer's own range - `layer_slot_range` in `src/core/expert_cache.cpp`).

`--prefill` starts the run with the profile's top pairs resident (rank 0 counts as most-recently-used),
which is how the engine actually starts with `--expert-profile`; without it the run starts cold.

AGENTS.md: a number needs what it was measured on.  The tool prints the trace files and the lookup and
distinct-pair counts it ran over; quote those with any rate you take from it.  The trace itself comes
from a one-shot engine run (`--dump-routing FILE` on a prompt typical of the workload) - the model,
quant and prompt are the environment, and they belong in any report of these rates.
"""
from __future__ import annotations

import argparse
import heapq
import struct
import sys
from array import array
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_profile as MP  # noqa: E402  (the profile reader; the trace format doc lives there too)

N_LAYER = MP.N_LAYER      # 48, fixed like make_profile's
N_EXPERT = MP.N_EXPERT    # 512; --n-expert for a pruned model


def encode(layer, expert):
    """One (layer, expert) pair as a uint32, so a trace is 4 bytes per lookup instead of two objects."""
    return (layer << 16) | expert


def decode(code):
    return code >> 16, code & 0xFFFF


def read_trace(path, n_expert=N_EXPERT, n_layers=N_LAYER):
    """The routed (layer, expert) pairs of one `--dump-routing` file, IN ORDER, as encode()d codes.

    Same records as make_profile.read_trace - (layer, k, k expert ids, k weights) - but eviction policy
    is a question about order, so the frequencies it returns are not enough.  Out-of-range pairs are
    skipped the same way it skips them."""
    blob = Path(path).read_bytes()
    codes, off = array("I"), 0
    while off + 8 <= len(blob):
        layer, k = struct.unpack_from("<ii", blob, off)
        off += 8
        for e in struct.unpack_from("<%di" % k, blob, off):
            if 0 <= layer < n_layers and 0 <= e < n_expert:
                codes.append(encode(layer, e))
        off += 8 * k  # the ids and the weights
    return codes


class Sim:
    """One policy's slot pool.  `lookup(code, i)` answers hit/miss for the i-th lookup of the trace and
    admits/evicts by the policy; `hits` and `evictions` accumulate.  `prefill(pairs)` seats the profile's
    ranked pairs first (rank 0 = most recently used), respecting each scope's capacity."""

    name = "?"

    def __init__(self, n_layers, n_expert, slots, per_layer, decay_window=65536):
        self.n_layers, self.n_expert, self.slots, self.per_layer = n_layers, n_expert, slots, per_layer
        self.hits = 0
        self.evictions = 0

    def scope(self, layer):
        return layer if self.per_layer else 0

    def capacity(self, layer):
        """The slots a lookup on `layer` may use - ExpertCache::layer_slot_range's range, in size."""
        if not self.per_layer:
            return self.slots
        q = self.slots // self.n_layers
        hi = self.slots if layer == self.n_layers - 1 else (layer + 1) * q
        return hi - layer * q

    def prefill(self, pairs):
        raise NotImplementedError

    def lookup(self, code, i):
        raise NotImplementedError


class FillOnly(Sim):
    """The engine's current admission: the first S distinct pairs stay resident forever."""

    name = "fill-only"

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.resident = set()
        self.used = defaultdict(int)  # scope -> slots taken

    def prefill(self, pairs):
        for layer, e in pairs:
            code, s = encode(layer, e), self.scope(layer)
            if code not in self.resident and self.used[s] < self.capacity(layer):
                self.resident.add(code)
                self.used[s] += 1

    def lookup(self, code, i):
        layer, _ = decode(code)
        if code in self.resident:
            self.hits += 1
            return True
        s = self.scope(layer)
        if self.used[s] < self.capacity(layer):
            self.resident.add(code)
            self.used[s] += 1
        return False


class Lru(Sim):
    """Evict the least-recently-used resident of the scope.  OrderedDict: front = oldest."""

    name = "lru"

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        from collections import OrderedDict
        self.od = defaultdict(OrderedDict)  # scope -> {code: None}

    def prefill(self, pairs):
        # Take each scope's top pairs in RANKED order, then insert them reversed so rank 0 lands
        # at the MRU end.  (Iterating the profile reversed would seat its COLD end instead.)
        chosen, used = [], defaultdict(int)
        for layer, e in pairs:
            s = self.scope(layer)
            if used[s] < self.capacity(layer):
                chosen.append((encode(layer, e), s))
                used[s] += 1
        for code, s in reversed(chosen):
            self.od[s][code] = None

    def lookup(self, code, i):
        layer, _ = decode(code)
        od = self.od[self.scope(layer)]
        if code in od:
            od.move_to_end(code)
            self.hits += 1
            return True
        if len(od) >= self.capacity(layer):
            od.popitem(last=False)
            self.evictions += 1
        od[code] = None
        return False


class LfuDecay(Sim):
    """Evict the lowest decayed-frequency resident of the scope, oldest on a tie.

    Every lookup adds the current increment to the pair's count; the increment doubles every
    `decay_window` lookups, which is the table-wide halving of classic LFU-decay without touching the
    table (see the module docstring).  Victim selection is a lazy heap: entries go stale as counts
    grow, and are popped when they surface."""

    name = "lfu-decay"

    def __init__(self, *args, decay_window=65536, **kw):
        super().__init__(*args, **kw)
        self.decay_window = decay_window
        self.incr = 1.0
        self.tick = 0
        self.counts = {}                    # code -> scaled count
        self.heaps = defaultdict(list)      # scope -> [(count, tick, code)]
        self.used = defaultdict(int)        # scope -> slots taken

    def prefill(self, pairs):
        # Same two-step as Lru.prefill: top pairs per scope in ranked order, newest tick to rank 0.
        chosen, used = [], defaultdict(int)
        for layer, e in pairs:
            s = self.scope(layer)
            if used[s] < self.capacity(layer):
                chosen.append((encode(layer, e), s))
                used[s] += 1
        for code, s in reversed(chosen):
            if code not in self.counts:
                self.counts[code] = self.incr
                heapq.heappush(self.heaps[s], (self.incr, self.tick, code))
                self.used[s] += 1
                self.tick += 1

    def lookup(self, code, i):
        if self.tick > 0 and self.tick % self.decay_window == 0:
            self.incr *= 2.0
        tick, self.tick = self.tick, self.tick + 1
        layer, _ = decode(code)
        s = self.scope(layer)
        if code in self.counts:             # resident (an evicted pair's count is deleted with it)
            self.counts[code] += self.incr
            heapq.heappush(self.heaps[s], (self.counts[code], tick, code))
            self.hits += 1
            return True
        if self.used[s] >= self.capacity(layer):
            h = self.heaps[s]
            while h:                        # drop stale heap entries until the true minimum surfaces
                c, _, cand = h[0]
                if self.counts.get(cand) == c:
                    break
                heapq.heappop(h)
            victim = heapq.heappop(h)[2]
            del self.counts[victim]
            self.used[s] -= 1
            self.evictions += 1
        self.counts[code] = self.incr
        heapq.heappush(self.heaps[s], (self.incr, tick, code))
        self.used[s] += 1
        return False


class Oracle(Sim):
    """Belady's MIN: evict the resident whose next use is furthest away.  Needs the whole trace up
    front (`prepare`), which is exactly why it is a bound and not a candidate."""

    name = "oracle"

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.cur = {}                       # code -> its next use at the current position (T: never)
        self.heaps = defaultdict(list)      # scope -> [(-next_use, code)], lazy like LfuDecay's
        self.used = defaultdict(int)

    def prepare(self, codes):
        T = len(codes)
        nxt = array("I", [T]) * T
        last = {}
        for i in range(T - 1, -1, -1):
            c = codes[i]
            nxt[i] = last.get(c, T)
            last[c] = i
        self.nxt, self.first, self.T = nxt, last, T

    def prefill(self, pairs):
        for layer, e in pairs:
            code, s = encode(layer, e), self.scope(layer)
            if code not in self.cur and self.used[s] < self.capacity(layer):
                self.cur[code] = self.first.get(code, self.T)
                heapq.heappush(self.heaps[s], (-self.cur[code], code))
                self.used[s] += 1

    def lookup(self, code, i):
        layer, _ = decode(code)
        s = self.scope(layer)
        if code in self.cur:
            self.cur[code] = self.nxt[i]
            heapq.heappush(self.heaps[s], (-self.nxt[i], code))
            self.hits += 1
            return True
        if self.used[s] >= self.capacity(layer):
            h = self.heaps[s]
            while h:
                neg, cand = h[0]
                if cand in self.cur and self.cur[cand] == -neg:
                    break
                heapq.heappop(h)
            victim = heapq.heappop(h)[1]
            del self.cur[victim]
            self.used[s] -= 1
            self.evictions += 1
        self.cur[code] = self.nxt[i]
        heapq.heappush(self.heaps[s], (-self.nxt[i], code))
        self.used[s] += 1
        return False


POLICIES = {p.name: p for p in (FillOnly, Lru, LfuDecay, Oracle)}


def run(codes, n_layers, n_expert, slots_list, per_layer, policy_names, windows=1,
        decay_window=65536, prefill_pairs=None):
    """One replay per (slots, policy).  Returns rows of dicts with the overall and per-window counts."""
    T = len(codes)
    rows = []
    for slots in slots_list:
        for name in policy_names:
            sim = POLICIES[name](n_layers, n_expert, slots, per_layer, decay_window=decay_window)
            if name == Oracle.name:
                sim.prepare(codes)
            if prefill_pairs:
                sim.prefill(prefill_pairs)
            w_hits, w_tot = [0] * windows, [0] * windows
            for i, code in enumerate(codes):
                w = i * windows // T if T else 0
                w_tot[w] += 1
                if sim.lookup(code, i):
                    w_hits[w] += 1
            rows.append({
                "slots": slots, "policy": name, "hits": sim.hits, "lookups": T,
                "evictions": sim.evictions, "w_hits": w_hits, "w_tot": w_tot,
            })
    return rows


def fmt_rate(hits, lookups):
    return f"{hits / lookups:.4f}" if lookups else "n/a"


def report(rows, windows):
    for slots in dict.fromkeys(r["slots"] for r in rows):
        block = [r for r in rows if r["slots"] == slots]
        print(f"\nslots = {slots}")
        print(f"  {'policy':<10} {'hits':>10} {'lookups':>10} {'hit-rate':>9} {'evictions':>10}")
        for r in block:
            print(f"  {r['policy']:<10} {r['hits']:>10} {r['lookups']:>10} "
                  f"{fmt_rate(r['hits'], r['lookups']):>9} {r['evictions']:>10}")
        if windows > 1:
            print(f"  per-window hit-rate ({windows} windows, in trace order):")
            print(f"  {'policy':<10} " + " ".join(f"{'w' + str(i + 1):>8}" for i in range(windows)))
            for r in block:
                rates = " ".join(f"{fmt_rate(h, t):>8}" for h, t in zip(r["w_hits"], r["w_tot"]))
                print(f"  {r['policy']:<10} {rates}")


def write_csv(path, rows, scope):
    with open(path, "w") as f:
        f.write("slots,scope,policy,window,lookups,hits,hit_rate,evictions\n")
        for r in rows:
            f.write(f"{r['slots']},{scope},{r['policy']},all,{r['lookups']},{r['hits']},"
                    f"{fmt_rate(r['hits'], r['lookups'])},{r['evictions']}\n")
            for w, (h, t) in enumerate(zip(r["w_hits"], r["w_tot"])):
                if len(r["w_tot"]) > 1:
                    f.write(f"{r['slots']},{scope},{r['policy']},w{w + 1},{t},{h},{fmt_rate(h, t)},\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("traces", nargs="+", help="--dump-routing files, replayed in the order given")
    p.add_argument("--slots", default="4105",
                       help="comma-separated slot counts (default: 4105, the R4 reference sizing)")
    p.add_argument("--per-layer", action="store_true",
                   help="R4.2g admission: each layer its own slot range, eviction stays in-layer")
    p.add_argument("--windows", type=int, default=1,
                   help="split the trace into this many equal windows and report hit-rate per window")
    p.add_argument("--policies", default=",".join(POLICIES),
                   help="comma-separated subset of: " + ",".join(POLICIES))
    p.add_argument("--decay-window", type=int, default=65536,
                   help="lfu-decay: lookups between table-wide halvings (default: 65536)")
    p.add_argument("--prefill", help="profile.bin to start from (the engine's --expert-profile start)")
    p.add_argument("--n-expert", type=int, default=N_EXPERT, help="experts per layer (default: 512)")
    p.add_argument("--csv", help="also write the results to this CSV file")
    a = p.parse_args(argv)

    codes = array("I")
    for t in a.traces:
        codes.extend(read_trace(t, a.n_expert))
    if not codes:
        raise SystemExit("no usable lookups in the trace(s)")
    slots_list = [int(s) for s in a.slots.split(",")]
    policy_names = a.policies.split(",")
    unknown = set(policy_names) - set(POLICIES)
    if unknown:
        raise SystemExit(f"unknown policies: {', '.join(sorted(unknown))} (have: {', '.join(POLICIES)})")
    prefill = MP.read_profile(a.prefill, a.n_expert) if a.prefill else None

    scope = "per-layer" if a.per_layer else "global"
    distinct = len(set(codes))
    print(f"trace: {', '.join(a.traces)}")
    print(f"{len(codes):,} lookups, {distinct:,} distinct (layer, expert) pairs, "
          f"scope: {scope}, prefill: {a.prefill or 'none (cold start)'}"
          + (f", decay-window: {a.decay_window}" if "lfu-decay" in policy_names else ""))
    if distinct <= min(slots_list):
        print("note: the distinct pairs fit in every slot count given - fill-only cannot be beaten "
              "on this trace; use a longer trace or fewer slots to see policy differences")

    rows = run(codes, N_LAYER, a.n_expert, slots_list, a.per_layer, policy_names,
               a.windows, a.decay_window, prefill)
    report(rows, a.windows)
    if a.csv:
        write_csv(a.csv, rows, scope)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
