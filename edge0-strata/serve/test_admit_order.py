"""serve/test_admit_order.py - STRATA_ADMIT_SHORT_FIRST: new requests take the control lines short prompts first
(StrataEngine._take_control / _admit_next; no process, no GPU).

    python -m unittest serve.test_admit_order -v
"""
from __future__ import annotations

import sys
import collections
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as server  # noqa: E402
from serve.server import StrataEngine  # noqa: E402


def engine_without_process(n=2):
    """A StrataEngine with the admission state of a started one and no process: alive() is `e.up`."""
    e = StrataEngine("missing-executable", [], lazy=True)
    e.batch = n
    e.slot_busy = [False] * n
    e.slot_claims, e.slot_releasing, e._orphan_seen = set(), set(), {}
    e.slot_cv = threading.Condition()
    e.waiting, e.admitting, e.wait_lens, e.ctl_epoch = 0, 0, [], 0
    e.arrivals = collections.deque(maxlen=64)
    e.ctl = threading.Lock()
    e.gen = 1
    e.up = True
    e.alive = lambda: e.up
    return e


class AdmitOrderTest(unittest.TestCase):
    def setUp(self):
        self.e = engine_without_process(16)
        self.order, self.cancels = [], []
        self.order_lock = threading.Lock()

    def tearDown(self):
        for c in self.cancels:
            c.set()

    def take(self, name, plen, **kw):
        """A request waiting for the control lines; once it has them it notes its name and gives them back."""
        c = threading.Event()
        self.cancels.append(c)

        def go():
            gen = self.e._take_control(c, plen, **kw)
            try:
                while True:
                    next(gen)
            except StopIteration as s:
                if s.value:
                    with self.order_lock:
                        self.order.append(name)
                    time.sleep(0.05)                # its read
                    self.e.ctl.release()
        threading.Thread(target=go, daemon=True).start()

    def until(self, cond, s=5.0):
        end = time.time() + s
        while time.time() < end and not cond():
            time.sleep(0.01)
        self.assertTrue(cond())

    def queue(self, *reqs):
        """While a long read holds the control lines, the requests arrive in this order; then the read ends."""
        self.assertTrue(self.e.ctl.acquire(timeout=1))
        for name, plen in reqs:
            self.take(name, plen, new=True)
            self.until(lambda n=len(self.order) + sum(1 for _ in self.e.wait_lens): self.e.waiting >= 1)
            time.sleep(0.02)                        # (distinct arrival times)
        self.until(lambda: self.e.waiting == len(reqs))
        self.e.ctl.release()
        self.until(lambda: len(self.order) == len(reqs), s=15.0)
        return self.order

    def test_short_first_puts_the_chat_before_the_agent_reads(self):
        with mock.patch.object(server, "ADMIT_SHORT_FIRST", 8192), mock.patch.object(server, "ADMIT_MAX_WAIT_S", 60.0):
            order = self.queue(("agent1", 120000), ("agent2", 190000), ("chat", 500), ("agent3", 150000))
        self.assertEqual(order[0], "chat")
        self.assertEqual(order[1:], ["agent1", "agent2", "agent3"])      # arrival order among the long ones

    def test_a_long_request_that_waited_too_long_goes_first(self):
        with mock.patch.object(server, "ADMIT_SHORT_FIRST", 8192), mock.patch.object(server, "ADMIT_MAX_WAIT_S", 0.2):
            self.assertTrue(self.e.ctl.acquire(timeout=1))
            self.take("agent", 120000, new=True)
            self.until(lambda: self.e.waiting == 1)
            time.sleep(0.4)                         # past STRATA_ADMIT_MAX_WAIT_S
            self.take("chat", 500, new=True)
            self.until(lambda: self.e.waiting == 2)
            self.e.ctl.release()
            self.until(lambda: len(self.order) == 2, s=15.0)
        self.assertEqual(self.order, ["agent", "chat"])

    def test_off_by_default(self):
        self.assertEqual(server.ADMIT_SHORT_FIRST if "STRATA_ADMIT_SHORT_FIRST" not in __import__("os").environ
                         else 0, 0)
        with mock.patch.object(server, "ADMIT_SHORT_FIRST", 0):
            order = self.queue(("agent1", 120000), ("chat", 500))
        self.assertEqual(sorted(order), ["agent1", "chat"])            # both get their turn, in any order

    def test_a_continuation_is_not_held_back(self):
        """A request already admitted once (new=False: promoted from the solo path) is not part of the order."""
        with mock.patch.object(server, "ADMIT_SHORT_FIRST", 8192), mock.patch.object(server, "ADMIT_MAX_WAIT_S", 60.0):
            self.assertTrue(self.e.ctl.acquire(timeout=1))
            self.take("chat", 500, new=True)
            self.take("continuation", 190000)
            self.until(lambda: self.e.waiting == 2)
            self.e.ctl.release()
            self.until(lambda: len(self.order) == 2, s=15.0)
        self.assertEqual(sorted(self.order), ["chat", "continuation"])


if __name__ == "__main__":
    unittest.main()
