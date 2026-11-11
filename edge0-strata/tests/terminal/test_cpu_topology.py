"""serve/telemetry.py `_cpu_topology`: how many CPU sockets and NUMA nodes the computer has (for "2 CPUs · 64 cores" in the monitor)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve import telemetry


def fake_sysfs(root: Path, packages: list[int], nodes: int) -> None:
    for i, pkg in enumerate(packages):
        d = root / "cpu" / f"cpu{i}" / "topology"
        d.mkdir(parents=True)
        (d / "physical_package_id").write_text(f"{pkg}\n")
    (root / "cpu" / "cpufreq").mkdir()                       # not a cpuN folder: never counted
    for n in range(nodes):
        (root / "node" / f"node{n}").mkdir(parents=True)


class Topology(unittest.TestCase):
    def read(self, root):
        with mock.patch.object(telemetry.os, "name", "posix"), mock.patch.object(telemetry.sys, "platform", "linux"):
            return telemetry._cpu_topology(str(root))

    def test_two_sockets_two_numa_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_sysfs(Path(tmp), [0, 0, 0, 0, 1, 1, 1, 1], 2)           # 8 logical CPUs on 2 packages
            self.assertEqual(self.read(tmp), (2, 2))

    def test_one_socket(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_sysfs(Path(tmp), [0, 0, 0, 0], 1)
            self.assertEqual(self.read(tmp), (1, 1))

    def test_one_socket_with_several_memory_nodes(self):             # e.g. an EPYC with NPS4: one package, four nodes
        with tempfile.TemporaryDirectory() as tmp:
            fake_sysfs(Path(tmp), [0] * 8, 4)
            self.assertEqual(self.read(tmp), (1, 4))

    def test_nothing_readable_is_none_not_a_guess(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.read(tmp), (None, None))

    def test_this_computer_answers_in_numbers(self):
        sockets, numa = telemetry._cpu_topology()
        self.assertTrue(sockets is None or sockets >= 1)
        self.assertTrue(numa is None or numa >= 1)

    def test_the_hardware_info_carries_them(self):
        src = (Path(telemetry.__file__)).read_text(encoding="utf-8")
        self.assertIn('"sockets"', src)
        self.assertIn('"numa_nodes"', src)


if __name__ == "__main__":
    unittest.main()
