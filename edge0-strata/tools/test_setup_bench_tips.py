"""Setup's recommendations from the community bench data (0.1.40): text only, never a changed default.

    python -m unittest tools.test_setup_bench_tips
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_hybrid import fake_sys  # noqa: E402

BASE = ["--expert-cache", "auto", "--prefill", "auto", "--max-context", "65536"]


def tips(args=BASE, env=None, ram=32.0, model=20.0, vram=12.0, vision="none", win=False):
    return setup.bench_tips(list(args), env, ram, model, vram, vision, win)


class BenchTips(unittest.TestCase):
    def test_nothing_to_say_on_a_plain_pc(self):
        self.assertEqual(tips(), [])

    def test_explicit_cache_warns_on_windows_only(self):
        a = ["--expert-cache", "2400", "--prefill", "auto"]
        self.assertTrue(any("--expert-cache 2400" in t and "7x" in t for t in tips(a, win=True)))
        self.assertEqual(tips(a, win=False), [])
        self.assertEqual(tips(BASE, win=True), [])                      # auto: nothing to warn about

    def test_prefill_auto_32768_follows_the_ram(self):
        self.assertTrue(any("auto:32768" in t and t.startswith("tip") for t in tips(ram=128)))
        self.assertEqual([t for t in tips(ram=64) if "auto:32768" in t], [])
        big = ["--prefill", "auto:32768"]
        self.assertTrue(any(t.startswith("warning") and "auto:32768" in t for t in tips(big, ram=32)))
        self.assertEqual([t for t in tips(big, ram=96) if "auto:32768" in t], [])

    def test_resident_headroom_on_small_ram(self):
        res = [*BASE, "--resident-experts"]
        self.assertTrue(any("STRATA_RESIDENT_HEADROOM_GIB" in t for t in tips(res, ram=32)))
        self.assertTrue(any("STRATA_RESIDENT_HEADROOM_GIB" in t for t in tips([*BASE, "--resident-budget-gib", "40"], ram=48)))
        self.assertEqual([t for t in tips(res, ram=64) if "HEADROOM" in t], [])
        self.assertEqual([t for t in tips(res, {"STRATA_RESIDENT_HEADROOM_GIB": "8"}, ram=32) if "HEADROOM" in t], [])
        self.assertEqual([t for t in tips(BASE, ram=32) if "HEADROOM" in t], [])   # not a resident config

    def test_no_conversation_cache_tip_the_engine_sizes_it_itself(self):
        # 0.2.3 (#3493): the engine sizes the conversation cache from the free RAM when no value is given, so setup no longer tips about it
        for ram, model in ((64, 40), (64, 41), (128, 40)):
            self.assertEqual([t for t in tips(ram=ram, model=model) if "conversation" in t], [])
        self.assertEqual([t for t in tips([*BASE, "--conversation-cache-mib", "4096"], ram=128) if "conversation" in t], [])

    def test_vision_cpu_on_small_cards(self):
        self.assertTrue(any("--vision cpu" in t for t in tips(vision="gpu", vram=12)))
        self.assertEqual([t for t in tips(vision="gpu", vram=24) if "vision" in t], [])
        self.assertEqual([t for t in tips(vision="cpu", vram=12) if "vision" in t], [])

    def test_tips_never_touch_the_arguments(self):
        a = ["--expert-cache", "2400", "--prefill", "auto", "--resident-experts"]
        before = list(a)
        env = {}
        tips(a, env, ram=128, vision="gpu", win=True)
        self.assertEqual(a, before)
        self.assertEqual(env, {})


class Sockets(unittest.TestCase):
    def fake(self, d, sockets, cores):
        n = 0
        for s in range(sockets):
            for c in range(cores):
                t = Path(d) / "devices" / "system" / "cpu" / f"cpu{n}" / "topology"
                t.mkdir(parents=True)
                (t / "physical_package_id").write_text(str(s))
                (t / "core_id").write_text(str(c))
                n += 1

    def test_linux_sockets(self):
        with tempfile.TemporaryDirectory() as d:
            self.fake(d, 2, 18)
            self.assertEqual(setup.linux_sockets(d), (2, 18))
        with tempfile.TemporaryDirectory() as d:
            self.fake(d, 1, 8)
            self.assertEqual(setup.linux_sockets(d), (1, 8))
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(setup.linux_sockets(d))

    def test_two_socket_tip(self):
        t = setup.two_socket_note((2, 18))
        self.assertEqual(len(t), 1)
        self.assertIn("--pool-workers 17", t[0])
        self.assertEqual(setup.two_socket_note((1, 18)), [])
        self.assertEqual(setup.two_socket_note(None), [])


if __name__ == "__main__":
    unittest.main()
