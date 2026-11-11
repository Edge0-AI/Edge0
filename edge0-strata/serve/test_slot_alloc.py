"""serve/test_slot_alloc.py - which batch slot a request gets (StrataEngine.pick_slot), no engine needed.

    python -m unittest serve.test_slot_alloc -v
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import StrataEngine  # noqa: E402


def engine(batch, groups, pipelined=True):
    e = object.__new__(StrataEngine)          # pick_slot reads only the slot tables
    gs = batch // groups
    e.batch = batch
    e.slot_order = [g * gs + t for t in range(gs) for g in range(groups)]
    e.slot_busy = [False] * batch
    e.slot_held = [[] for _ in range(batch)]
    e.slot_used = [0.0] * batch
    e.slot_gs = gs if pipelined and groups > 1 else 0
    return e


class Pipelined(unittest.TestCase):
    def test_spreads_over_the_groups_first(self):
        e = engine(6, 3)
        got = []
        for _ in range(6):
            b = e.pick_slot([1, 2, 3])
            e.slot_busy[b] = True
            got.append(b)
        self.assertEqual(got, [0, 2, 4, 1, 3, 5])

    def test_fills_a_leading_hole_first(self):
        # #793's report: a group left as [_ B] costs a whole row; the next request goes into that hole
        e = engine(6, 3)
        e.slot_busy[3] = True                       # group 1 = [_ B]
        e.slot_used[0] = 0.0                        # least recently used: what the LRU fallback picked
        e.slot_used[2] = 99.0
        self.assertEqual(e.pick_slot([1]), 2)

    def test_never_opens_a_hole_below_the_pick(self):
        for batch, groups in ((6, 3), (8, 4), (8, 2), (4, 2)):
            gs = batch // groups
            e, rng = engine(batch, groups), random.Random(batch * 10 + groups)
            for _ in range(2000):
                busy = [b for b in range(batch) if e.slot_busy[b]]
                if busy and (rng.random() < 0.45 or len(busy) == batch):
                    e.slot_busy[rng.choice(busy)] = False          # a request ends, anywhere
                    continue
                b = e.pick_slot([7])
                g0 = b - b % gs
                with self.subTest(batch=batch, groups=groups):
                    self.assertTrue(all(e.slot_busy[x] for x in range(g0, b)),
                                    f"slot {b} picked with a free slot below it in its group: {e.slot_busy}")
                e.slot_busy[b] = True

    def test_held_tokens_are_ignored(self):
        # the engine keeps no pipelined slot as a cache: a held prefix must not pull a request into a hole-making slot
        e = engine(6, 3)
        e.slot_held[5] = [1, 2]
        self.assertEqual(e.pick_slot([1, 2, 3]), 0)


class OneGroup(unittest.TestCase):
    def test_reuse_then_least_recently_used(self):
        e = engine(4, 1, pipelined=False)
        e.slot_held[2] = [1, 2]
        self.assertEqual(e.pick_slot([1, 2, 3]), 2)        # its conversation's next turn
        e.slot_held[2] = [9]
        e.slot_held = [[5], [], [9], [6]]
        self.assertEqual(e.pick_slot([1, 2, 3]), 1)        # an empty slot before evicting a held one
        e.slot_held[1] = [8]
        e.slot_used = [5.0, 3.0, 9.0, 1.0]
        self.assertEqual(e.pick_slot([1, 2, 3]), 3)        # else the least recently used


class OneSlotGroups(unittest.TestCase):
    """--batch 2 on a two-GPU layer split: two pipelined groups of ONE slot.  No slot sits below another in its group,
    so no pad row writes an idle slot and each one keeps its chat (#2518): the next turn goes back to it."""

    def test_the_slot_of_the_chat(self):
        T = StrataEngine.TURN_ID
        e = engine(2, 2)
        self.assertEqual(e.slot_gs, 1)
        hist_a, hist_b = list(range(1000, 1700)), list(range(2000, 2700))
        e.slot_held = [hist_a + [T, 7, 8] + [50, 51], hist_b + [T, 7, 8] + [60]]
        e.slot_used = [10.0, 20.0]
        again = lambda h: h + [T, 5, 6] + list(range(3000, 3050))
        self.assertEqual(e.pick_slot(again(hist_b)), 1)
        self.assertEqual(e.pick_slot(again(hist_a)), 0)
        self.assertEqual(e.pick_slot(list(range(5000, 5100))), 0)      # a new chat: the least recently used one

    def test_groups_of_several_slots_still_ignore_held_tokens(self):
        T = StrataEngine.TURN_ID
        e = engine(4, 2)
        self.assertEqual(e.slot_gs, 2)
        hist = list(range(1000, 1700))
        e.slot_held[3] = hist + [T, 7, 8] + [50]
        self.assertEqual(e.pick_slot(hist + [T, 5, 6] + list(range(3000, 3050))), 0)


if __name__ == "__main__":
    unittest.main()
