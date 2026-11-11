"""#4675: a child process that writes non-UTF-8 bytes (a GBK display adapter name) must not kill Step 1.

    python -m unittest tools.test_setup_gbk_adapters
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

GBK_LINE = b"Microsoft \xbb\xf9\xb1\xbe\xcf\xd4\xca\xbe\xca\xca\xc5\xe4\xc6\xf7|PCI/VEN_10DE&DEV_0A60&SUBSYS|0\r\n"


class GbkOutput(unittest.TestCase):
    def test_out_survives_bytes_that_are_not_utf8(self):
        code = "import sys; sys.stdout.buffer.write(%r)" % (GBK_LINE,)
        text = setup.out([sys.executable, "-X", "utf8", "-c", code])
        self.assertIsInstance(text, str)
        self.assertIn("Microsoft", text)
        self.assertIn("VEN_10DE", text)

    def test_adapter_list_never_gets_none(self):
        from unittest import mock
        with mock.patch.object(setup, "out", return_value=None):
            self.assertEqual(setup._win_display_adapters(), [])
        with mock.patch.object(setup, "out", return_value=GBK_LINE.decode("utf-8", "replace")):
            found = setup._win_display_adapters()
        self.assertEqual(len(found), 1)
        self.assertIn("VEN_10DE", found[0]["pnp"])

    def test_a_mangled_name_still_matches_its_registry_entry_by_pci_id(self):
        adapters = [{"name": "AMD Radeon(TM) Graphics �", "pnp": "PCI/VEN_1002&DEV_744C&SUBSYS", "ram": 0}]
        registry = [{"DriverDesc": "AMD Radeon RX 7900 XTX", "MatchingDeviceId": "PCI/VEN_1002&DEV_744C",
                     "qwMemorySize": 24 << 30}]
        g = setup.amd_gpus_windows(adapters, registry)
        self.assertEqual(len(g), 1)


if __name__ == "__main__":
    unittest.main()
