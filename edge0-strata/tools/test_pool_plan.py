"""The default count of CPU expert workers on mock CPUs, and what setup writes about thread counts.

The engine's choice is include/strata/kernels/cpu/pool_plan.hpp (tested on its own by the pool_plan_test binary); setup mirrors
it so it can warn about a configured count.  This test keeps the two in step and checks the mirror on the same mock
topologies (8c/16t, 16c/32t, 64c/128t, a 2-socket 128c/256t box) plus the reports behind it: a 256-thread machine started
~256 workers and slowed down.

    python -m unittest tools.test_pool_plan
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

HEADER = ROOT / "include" / "strata" / "kernels" / "cpu" / "pool_plan.hpp"


class InStepWithTheEngine(unittest.TestCase):
    def header_constant(self, name):
        m = re.search(rf"constexpr int {name}\s*=\s*(\d+);", HEADER.read_text(encoding="utf-8"))
        self.assertIsNotNone(m, name)
        return int(m.group(1))

    def test_constants(self):
        self.assertEqual(setup.POOL_WORKERS_CAP, self.header_constant("kPoolWorkersCap"))
        self.assertEqual(setup.POOL_RESERVE_FROM, self.header_constant("kPoolReserveFrom"))
        self.assertEqual(setup.POOL_RESERVE, self.header_constant("kPoolReserve"))


class DefaultWorkers(unittest.TestCase):
    def test_mock_topologies(self):
        cap = setup.POOL_WORKERS_CAP
        # physical cores, not threads, minus the host thread's core
        self.assertEqual(setup.plan_pool_workers(16, [8]), 7)            # 8c/16t
        self.assertEqual(setup.plan_pool_workers(12, [6]), 5)            # 6c/12t
        self.assertEqual(setup.plan_pool_workers(32, [16]), 15)          # 16c/32t
        self.assertEqual(setup.plan_pool_workers(128, [64]), cap)        # 64c/128t: the cap
        self.assertEqual(setup.plan_pool_workers(256, [64, 64]), cap)    # 2-socket 128c/256t: the cap, never 255
        self.assertLess(setup.plan_pool_workers(256, [64, 64]), 128)
        # two sockets of 20 cores: the host's socket only (bench #674 #707: 17-18 workers beat 35)
        self.assertEqual(setup.plan_pool_workers(80, [20, 20]), 18)
        self.assertEqual(setup.plan_pool_workers(48, [6, 18]), 5)
        self.assertEqual(setup.plan_pool_workers(1, [1]), 1)

    def test_hybrid(self):
        self.assertEqual(setup.plan_pool_workers(20, [20], hybrid=(8, 12)), 13)     # Core Ultra 7 265 (measured)
        self.assertEqual(setup.plan_pool_workers(32, [24], hybrid=(8, 16)), 15)     # i9-14900KF (#642)
        self.assertEqual(setup.plan_pool_workers(24, [16], hybrid=(8, 8)), 15)      # not more E-cores than P-cores
        self.assertEqual(setup.hybrid_pool_workers((8, 12)), 13)                    # setup's own rule agrees

    def test_unknown_layout_and_quota(self):
        self.assertLessEqual(setup.plan_pool_workers(256, []), setup.POOL_WORKERS_CAP)
        self.assertLess(setup.plan_pool_workers(256, None), 255)
        self.assertEqual(setup.plan_pool_workers(8, None), 7)
        self.assertEqual(setup.plan_pool_workers(128, [64], quota=4.0), 3)   # a container limit of 4 CPUs

    def test_never_writes_a_big_count(self):
        # the hybrid rule (P-cores + half the E-cores) is capped like every other count
        with mock.patch.object(setup, "cpu_cores", return_value=(8, 120)):
            args = setup.recommend_pool_workers(["--spec", "4"])
        n = int(args[args.index("--pool-workers") + 1])
        self.assertLessEqual(n, setup.POOL_WORKERS_CAP)
        with mock.patch.object(setup, "cpu_cores", return_value=(8, 16)):   # an ordinary hybrid: the usual 15
            args = setup.recommend_pool_workers([])
        self.assertEqual(args, ["--pool-workers", "15"])
        with mock.patch.object(setup, "cpu_cores", return_value=None):      # not hybrid: nothing written
            self.assertEqual(setup.recommend_pool_workers(["--spec", "4"]), ["--spec", "4"])
        # a count that is already there is never changed
        with mock.patch.object(setup, "cpu_cores", return_value=(8, 120)):
            self.assertEqual(setup.recommend_pool_workers(["--pool-workers", "200"]), ["--pool-workers", "200"])

    def test_vision_threads(self):
        self.assertEqual(setup.vision_threads(16), 8)
        self.assertEqual(setup.vision_threads(256), setup.VISION_THREADS_CAP)
        self.assertEqual(setup.vision_threads(1), 1)


class ConfiguredCountWarns(unittest.TestCase):
    big = (256, [64, 64])

    def test_warns_above_the_plan_and_keeps_the_count(self):
        args = ["--pool-workers", "255"]
        note = setup.pool_workers_note(args, self.big)
        self.assertEqual(len(note), 1)
        self.assertIn("--pool-workers %d" % setup.plan_pool_workers(*self.big), note[0])
        self.assertIn("stays as you set it", note[0])
        self.assertEqual(args, ["--pool-workers", "255"])     # nothing rewritten

    def test_quiet_when_fine(self):
        self.assertEqual(setup.pool_workers_note([], self.big), [])
        self.assertEqual(setup.pool_workers_note(["--pool-workers", "8"], self.big), [])
        self.assertEqual(setup.pool_workers_note(["--pool-workers", str(setup.plan_pool_workers(*self.big))], self.big), [])
        self.assertEqual(setup.pool_workers_note(["--pool-workers"], self.big), [])
        self.assertEqual(setup.pool_workers_note(["--pool-workers", "x"], self.big), [])

    def test_reasons(self):
        self.assertIn("CPU threads", setup.pool_workers_note(["--pool-workers", "300"], self.big)[0])
        self.assertIn("physical cores", setup.pool_workers_note(["--pool-workers", "200"], self.big)[0])
        self.assertIn("one CPU socket", setup.pool_workers_note(["--pool-workers", "30"], (80, [20, 20]))[0])


if __name__ == "__main__":
    unittest.main()
