"""Tests for setup.py's experimental older-CPU build (#394 #595 #623): which floor a CPU without AVX2 gets, the CMake
definition, a build folder configured afresh when the floor changes, and the MCP server's note instead of a refusal.
No compiler, no GPU.

    python -m unittest tools.test_setup_oldcpu
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


def cpuinfo(flags: str):
    return mock.patch("builtins.open", lambda *a, **k: io.StringIO(f"model name\t: Old Xeon\nflags\t\t: {flags}\n"))


class FloorTest(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"STRATA_ISA_FLOOR": ""})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_avx2_cpu_gets_the_normal_engine(self):
        self.assertEqual(setup.cpu_floor(True), "")

    def test_linux_flags(self):
        with mock.patch.object(setup, "WIN", False):
            with cpuinfo("fpu sse4_1 sse4_2 popcnt avx xsave"):
                self.assertEqual(setup.cpu_floor(False), "avx")
            with cpuinfo("fpu sse4_1 sse4_2 popcnt"):
                self.assertEqual(setup.cpu_floor(False), "none")
            with cpuinfo("fpu sse3 ssse3"):
                self.assertEqual(setup.cpu_floor(False), "unsupported")

    def test_windows_probe(self):
        with mock.patch.object(setup, "WIN", True):
            for probe, want in (("avx", "avx"), ("sse4.2", "none"), ("", "unsupported")):
                with mock.patch.object(setup, "_cpuid_floor", lambda p=probe: p):
                    self.assertEqual(setup.cpu_floor(False), want)

    def test_env_asks_for_a_floor_build_on_any_cpu(self):
        for v in ("avx", "none", "AVX"):
            with mock.patch.dict(os.environ, {"STRATA_ISA_FLOOR": v}):
                self.assertEqual(setup.cpu_floor(True), v.lower())
        with mock.patch.dict(os.environ, {"STRATA_ISA_FLOOR": "bogus"}):
            self.assertEqual(setup.cpu_floor(True), "")

    @unittest.skipUnless(os.name == "nt", "the CPUID stub runs on Windows")
    def test_this_pc_probe(self):
        self.assertIn(setup._cpuid_floor(), ("avx", "sse4.2"))   # any x86-64 PC that runs the tests


class DefsTest(unittest.TestCase):
    def test_defs_and_fresh_configure(self):
        with tempfile.TemporaryDirectory() as d:
            b = Path(d)
            cache = b / "CMakeCache.txt"
            self.assertEqual(setup.isa_floor_defs("", b, {}), [])            # the normal build: nothing added
            cache.write_text("x")
            self.assertEqual(setup.isa_floor_defs("avx", b, {"isa_floor": "avx"}), ["-DSTRATA_ISA_FLOOR=avx"])
            self.assertTrue(cache.exists())                                  # same floor: the folder is kept
            self.assertEqual(setup.isa_floor_defs("none", b, {"isa_floor": "avx"}), ["-DSTRATA_ISA_FLOOR=none"])
            self.assertFalse(cache.exists())                                 # another floor: configured afresh
            cache.write_text("x")
            self.assertEqual(setup.isa_floor_defs("", b, {"isa_floor": "none"}), [])
            self.assertFalse(cache.exists())                                 # back to the normal build
            cache.write_text("x")
            setup.isa_floor_defs("", b, {})
            self.assertTrue(cache.exists())                                  # the normal build's folder untouched


class McpTest(unittest.TestCase):
    def test_no_avx2_is_a_note_not_a_refusal(self):
        import strata_mcp as M
        s = M.Strata(ROOT)
        gpu = [{"index": 0, "name": "x", "vram_gb": 24.0, "usable": True, "vendor": "nvidia"}]
        r = s.recommend({"ram_gb": 64, "gpus": gpu, "cpu": {"avx2": False}})
        self.assertIsNotNone(r["model"])
        self.assertTrue(any("no AVX2" in n and "EXPERIMENTAL" in n for n in r["notes"]))


if __name__ == "__main__":
    unittest.main()
