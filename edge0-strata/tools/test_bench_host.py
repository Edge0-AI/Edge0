"""Tests for the server address tools/strata_bench.py polls, without starting one.

    python -m unittest tools.test_bench_host
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_bench as SB  # noqa: E402


class ServerBase(unittest.TestCase):
    def test_uses_the_configured_host(self):
        # The readiness poll was hardcoded to 127.0.0.1 while the server was started on the config's
        # host, so a LAN bind left BENCH saying "still loading" until it was killed (#2330).
        self.assertEqual(SB.server_base("192.168.1.10", 8080), "http://192.168.1.10:8080")

    def test_defaults_to_loopback_without_a_host(self):
        self.assertEqual(SB.server_base("", 9000), "http://127.0.0.1:9000")
        self.assertEqual(SB.server_base(None, 9000), "http://127.0.0.1:9000")

    def test_wildcard_binds_are_connected_through_loopback(self):
        self.assertEqual(SB.server_base("0.0.0.0", 8080), "http://127.0.0.1:8080")
        self.assertEqual(SB.server_base("::", 8080), "http://[::1]:8080")

    def test_ipv6_literals_are_bracketed(self):
        self.assertEqual(SB.server_base("fd00::5", 8080), "http://[fd00::5]:8080")
        self.assertEqual(SB.server_base("[fd00::5]", 8080), "http://[fd00::5]:8080")

    def test_a_hostname_is_left_alone(self):
        self.assertEqual(SB.server_base("bench.local", 8080), "http://bench.local:8080")


if __name__ == "__main__":
    unittest.main()
