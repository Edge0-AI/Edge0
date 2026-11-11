"""Tests for tools/bench_eviction.py, without a model: synthetic traces and tiny slot pools.

    python -m unittest tools.test_bench_eviction
"""
from __future__ import annotations

import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import bench_eviction as BE  # noqa: E402

NL, NE = 3, 8   # tiny layer/expert counts; the sims take them as parameters


def codes_of(pairs):
    return BE.array("I", (BE.encode(layer, e) for layer, e in pairs))


def run_trace(sim, pairs):
    codes = codes_of(pairs)
    for i, c in enumerate(codes):
        sim.lookup(c, i)
    return sim


def write_trace(path, records):
    """One record per (layer, [expert ids]); the reader skips k weights after the k ids."""
    with open(path, "wb") as f:
        for layer, experts in records:
            f.write(struct.pack("<ii", layer, len(experts)))
            f.write(struct.pack("<%di" % len(experts), *experts))
            f.write(struct.pack("<%df" % len(experts), *([1.0] * len(experts))))


class ReadTrace(unittest.TestCase):
    def test_reads_pairs_in_order_and_skips_weights_and_out_of_range(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.bin"
            write_trace(p, [(0, [3, 1]), (2, [0]), (99, [1]), (1, [77])])
            got = BE.read_trace(p, n_expert=NE, n_layers=NL)
            self.assertEqual(list(got), [BE.encode(0, 3), BE.encode(0, 1), BE.encode(2, 0)])
            # (99, ..) is a bad layer, (1, 77) a bad expert: skipped like make_profile.read_trace

    def test_n_expert_gates_the_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.bin"
            write_trace(p, [(0, [3, 5])])
            got = BE.read_trace(p, n_expert=4, n_layers=NL)
            self.assertEqual(list(got), [BE.encode(0, 3)])


class FillOnly(unittest.TestCase):
    def test_never_evicts_and_refuses_when_full(self):
        sim = run_trace(BE.FillOnly(NL, NE, 2, False), [(0, 0), (0, 1), (0, 2), (0, 0)])
        self.assertEqual(sim.hits, 1)            # only the repeat of (0,0)
        self.assertEqual(sim.evictions, 0)
        self.assertEqual(sim.resident, {BE.encode(0, 0), BE.encode(0, 1)})   # (0,2) was refused


class Lru(unittest.TestCase):
    def test_evicts_the_least_recently_used(self):
        pairs = [(0, 0), (0, 1), (0, 0), (0, 2), (0, 1), (0, 0)]
        sim = run_trace(BE.Lru(NL, NE, 2, False), pairs)
        self.assertEqual(sim.hits, 1)            # the repeat of (0,0) at index 2
        self.assertEqual(sim.evictions, 3)
        resident = set().union(*[set(od) for od in sim.od.values()])
        self.assertEqual(resident, {BE.encode(0, 1), BE.encode(0, 0)})


class LfuDecay(unittest.TestCase):
    PAIRS = [(0, 0)] * 4 + [(0, 1)] * 2 + [(0, 2)]   # A hot early, B lukewarm late, then C arrives

    def test_decay_lets_a_stale_hot_expert_be_evicted(self):
        sim = run_trace(BE.LfuDecay(NL, NE, 2, False, decay_window=2), self.PAIRS)
        # A's count was built on the small early increments and decays behind B's: C evicts A
        self.assertEqual(set(sim.counts), {BE.encode(0, 1), BE.encode(0, 2)})
        self.assertEqual(sim.evictions, 1)

    def test_without_decay_frequency_wins_forever(self):
        sim = run_trace(BE.LfuDecay(NL, NE, 2, False, decay_window=10**9), self.PAIRS)
        # plain LFU: A(4) > B(2) forever, so C evicts B - the behaviour decay exists to change
        self.assertEqual(set(sim.counts), {BE.encode(0, 0), BE.encode(0, 2)})


class PerLayer(unittest.TestCase):
    def test_a_layers_quota_is_its_own_range(self):
        # layer_slot_range's arithmetic, in capacity() form: q = 7 // 3 = 2, the last layer takes the rest
        sim = BE.FillOnly(NL, NE, 7, True)
        self.assertEqual([sim.capacity(l) for l in range(NL)], [2, 2, 3])

    def test_a_full_layer_cannot_borrow_another_layers_slots(self):
        sim = run_trace(BE.FillOnly(NL, NE, 4, True),
                        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 0)])
        self.assertNotIn(BE.encode(0, 2), sim.resident)   # layer 0's quota is 2: refused
        self.assertIn(BE.encode(1, 0), sim.resident)      # layer 1's quota is its own
        self.assertEqual(sim.hits, 1)


class Oracle(unittest.TestCase):
    def test_sees_past_lrus_blind_spot(self):
        pairs = [(0, 0), (0, 1), (0, 0), (0, 2), (0, 1), (0, 2)]
        codes = codes_of(pairs)
        oracle = BE.Oracle(NL, NE, 2, False)
        oracle.prepare(codes)
        for i, c in enumerate(codes):
            oracle.lookup(c, i)
        lru = run_trace(BE.Lru(NL, NE, 2, False), pairs)
        # at index 3, oracle evicts (0,0) (never used again) where lru evicted (0,1) (used at index 4)
        self.assertEqual(oracle.hits, 3)
        self.assertEqual(lru.hits, 2)

    def test_is_an_upper_bound_on_a_random_trace(self):
        rng = random.Random(20261005)
        pairs = [(rng.randrange(NL), rng.randrange(NE)) for _ in range(4000)]
        codes = codes_of(pairs)
        rows = {r["policy"]: r for r in BE.run(codes, NL, NE, [4], False,
                                               ["lru", "lfu-decay", "oracle"], decay_window=256)}
        self.assertGreaterEqual(rows["oracle"]["hits"], rows["lru"]["hits"])
        self.assertGreaterEqual(rows["oracle"]["hits"], rows["lfu-decay"]["hits"])

    def test_policies_beat_fill_only_when_the_workload_shifts(self):
        rng = random.Random(7)

        def phase(experts, n):  # zipf-ish routing inside one expert set, like a conversation topic
            weights = [1.0 / (i + 1) for i in range(len(experts))]
            return [(rng.randrange(NL), rng.choices(experts, weights)[0]) for _ in range(n)]

        pairs = phase([0, 1, 2, 3, 4, 5], 2000) + phase([6, 7], 2000)
        rows = {r["policy"]: r for r in BE.run(codes_of(pairs), NL, NE, [4], False,
                                               ["fill-only", "lru", "lfu-decay"], decay_window=256)}
        self.assertGreater(rows["lru"]["hits"], rows["fill-only"]["hits"])
        self.assertGreater(rows["lfu-decay"]["hits"], rows["fill-only"]["hits"])


class WindowsAndPrefill(unittest.TestCase):
    def test_windows_split_the_trace_in_order(self):
        codes = codes_of([(0, 0), (0, 1)] * 4)
        (row,) = BE.run(codes, NL, NE, [2], False, ["lru"], windows=2)
        self.assertEqual(row["w_tot"], [4, 4])
        self.assertEqual(row["w_hits"], [2, 4])   # window 1 pays the two cold misses, window 2 all hits
        self.assertEqual(row["hits"], 6)

    def test_prefill_seats_the_profile_with_rank0_as_mru(self):
        sim = BE.Lru(NL, NE, 2, False)
        sim.prefill([(0, 0), (0, 1), (0, 2)])      # the third pair does not fit
        run_trace(sim, [(0, 2)])                   # evicts the prefill's colder end: (0,1)
        self.assertEqual(sim.evictions, 1)
        after = run_trace(sim, [(0, 0), (0, 1)])
        self.assertEqual(after.hits, 1)            # (0,0) still resident, (0,1) gone

    def test_prefill_respects_each_layers_quota(self):
        sim = BE.FillOnly(NL, NE, 6, True)         # q = 6 // 3 = 2 slots per layer
        sim.prefill([(0, 0), (0, 1), (0, 2), (1, 0)])
        self.assertNotIn(BE.encode(0, 2), sim.resident)   # layer 0's quota was already spent
        self.assertEqual(len(sim.resident), 3)


if __name__ == "__main__":
    unittest.main()
