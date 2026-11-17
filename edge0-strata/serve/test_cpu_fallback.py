"""Linux fallback CPU accounting from kernel-format /proc/stat snapshots."""
import io
import os
import unittest
from unittest.mock import patch

from serve.telemetry import _CpuRamFallback


@unittest.skipIf(os.name == "nt", "Linux /proc/stat accounting")
class CpuFallback(unittest.TestCase):
    def usage(self, counters):
        before = "cpu  " + " ".join("1000" for _ in counters) + "\n"
        after = "cpu  " + " ".join(str(1000 + value) for value in counters) + "\n"
        with patch("builtins.open", side_effect=[io.StringIO(before), io.StringIO(after)]):
            return _CpuRamFallback().cpu()

    def test_guest_and_guest_nice_are_already_in_user_and_nice(self):
        # Linux account_guest_time increments USER and GUEST together, or NICE and GUEST_NICE together.
        self.assertAlmostEqual(self.usage([50, 10, 0, 40, 0, 0, 0, 0, 30, 10]), 60.0)

    def test_guest_only_workload(self):
        self.assertAlmostEqual(self.usage([40, 0, 0, 60, 0, 0, 0, 0, 40, 0]), 40.0)

    def test_niced_guest_workload(self):
        self.assertAlmostEqual(self.usage([0, 30, 0, 70, 0, 0, 0, 0, 0, 30]), 30.0)

    def test_host_only_workload_and_iowait_keep_their_meaning(self):
        self.assertAlmostEqual(self.usage([30, 0, 20, 40, 10, 0, 0, 0, 0, 0]), 50.0)

    def test_eight_field_stat_keeps_its_total(self):
        self.assertAlmostEqual(self.usage([20, 10, 10, 40, 10, 5, 3, 2]), 50.0)


if __name__ == "__main__":
    unittest.main()
