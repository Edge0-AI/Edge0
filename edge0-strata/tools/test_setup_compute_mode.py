"""#1445: setup warns when nvidia-smi reports a compute mode other than Default (never refuses).

    python -m unittest tools.test_setup_compute_mode
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class Mode(unittest.TestCase):
    def test_default_is_silent(self):
        for m in ("Default", "default", "", "[N/A]", "N/A"):
            self.assertIsNone(setup.compute_mode_warning(0, m), m)

    def test_other_modes_warn(self):
        for m in ("Exclusive_Process", "Prohibited", "Exclusive_Thread"):
            w = setup.compute_mode_warning(2, m)
            self.assertIn(m, w)
            self.assertIn("GPU 2", w)
            self.assertIn("nvidia-smi -i 2 -c DEFAULT", w)

    def test_query(self):
        with mock.patch.object(setup, "out", return_value="Exclusive_Process\n") as o:
            self.assertEqual(setup.gpu_compute_mode(1), "Exclusive_Process")
        self.assertIn("compute_mode", " ".join(o.call_args[0][0]))


if __name__ == "__main__":
    unittest.main()
