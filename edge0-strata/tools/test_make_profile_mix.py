"""Tests for tools/make_profile_mix.py, without a model: synthetic routing traces.

    python -m unittest tools.test_make_profile_mix
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_profile as MP  # noqa: E402
from test_make_profile import write_trace  # noqa: E402

NE = 4


def run(args):
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "make_profile_mix.py")] + args, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


class MixProfile(unittest.TestCase):
    def test_a_long_trace_does_not_outvote_a_short_one(self):
        with tempfile.TemporaryDirectory() as d:
            long_t, short_t, out = Path(d, "long"), Path(d, "short"), Path(d, "p.bin")
            # the long trace uses pair (0, 1) 1000 times and (0, 2) 100 times; the short one uses (0, 3) 5 times
            write_trace(long_t, [(0, [1] * 10)] * 100 + [(0, [2] * 10)] * 10)
            write_trace(short_t, [(0, [3] * 5)])
            run([str(long_t), str(short_t), "--out", str(out), "--n-expert", str(NE)])
            ranked = MP.read_profile(out, NE)
            # as distributions: (0,1) 0.909, (0,2) 0.091 in the long one; (0,3) 1.0 in the short one -> (0,3) first
            self.assertEqual(ranked[:3], [(0, 3), (0, 1), (0, 2)])
            self.assertEqual(len(ranked), MP.N_LAYER * NE)           # every pair, once
            self.assertEqual(len(set(ranked)), len(ranked))

    def test_weights(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, out = Path(d, "a"), Path(d, "b"), Path(d, "p.bin")
            write_trace(a, [(1, [0] * 4)])
            write_trace(b, [(1, [2] * 4)])
            run([f"{a}:1", f"{b}:3", "--out", str(out), "--n-expert", str(NE)])
            self.assertEqual(MP.read_profile(out, NE)[:2], [(1, 2), (1, 0)])
            run([f"{a}:3", f"{b}:1", "--out", str(out), "--n-expert", str(NE)])
            self.assertEqual(MP.read_profile(out, NE)[:2], [(1, 0), (1, 2)])


if __name__ == "__main__":
    unittest.main()
