"""#1587: a Windows PC with only an AMD card - the server's hardware_static names the card from the registry's display
class, and strata_bench no longer labels the run "CPU only".  No GPU, no registry needed (a fake winreg).

    python -m unittest tools.test_windows_amd_gpu
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve import telemetry as T  # noqa: E402

GB = 1 << 30


def fake_winreg(adapters):
    """A winreg with the display class holding `adapters` (dicts of value name -> value) under 0000, 0001, ..."""
    w = types.ModuleType("winreg")
    w.HKEY_LOCAL_MACHINE = "HKLM"
    keys = {f"{i:04d}": a for i, a in enumerate(adapters)}
    subs = list(keys) + ["Properties"]

    def OpenKey(root, sub):
        if root == "HKLM":
            return "CLASS"
        if root == "CLASS" and sub in keys:
            return keys[sub]
        raise OSError("denied")

    def EnumKey(k, i):
        if i >= len(subs):
            raise OSError("no more")
        return subs[i]

    def QueryValueEx(k, name):
        if name not in k:
            raise OSError("no value")
        return k[name], 1
    w.OpenKey, w.EnumKey, w.QueryValueEx = OpenKey, EnumKey, QueryValueEx
    return w


IGPU = {"DriverDesc": "AMD Radeon(TM) Graphics", "ProviderName": "Advanced Micro Devices, Inc.",
        "DriverVersion": "32.0.13066.12", "HardwareInformation.qwMemorySize": 512 << 20}
CARD = {"DriverDesc": "AMD Radeon RX 6900 XT", "ProviderName": "Advanced Micro Devices, Inc.",
        "DriverVersion": "32.0.21001.9024", "HardwareInformation.MemorySize": (16 * GB).to_bytes(8, "little")}
GEFORCE = {"DriverDesc": "NVIDIA GeForce RTX 5070", "ProviderName": "NVIDIA", "DriverVersion": "32.0.16.1047",
           "HardwareInformation.qwMemorySize": 12 * GB}


class Adapters(unittest.TestCase):
    def read(self, adapters, vendor=None):
        with mock.patch.dict(sys.modules, {"winreg": fake_winreg(adapters)}), mock.patch.object(T.os, "name", "nt"):
            return T.windows_display_adapters(vendor)

    def test_reads_names_vram_and_drivers(self):
        a = self.read([IGPU, CARD, GEFORCE])
        self.assertEqual([x["name"] for x in a], ["AMD Radeon(TM) Graphics", "AMD Radeon RX 6900 XT", "NVIDIA GeForce RTX 5070"])
        self.assertEqual(a[1]["vram_bytes"], 16 * GB)       # the binary form of the memory size
        self.assertEqual(a[2]["vram_bytes"], 12 * GB)
        self.assertEqual([x["name"] for x in self.read([IGPU, CARD, GEFORCE], "amd")],
                         ["AMD Radeon(TM) Graphics", "AMD Radeon RX 6900 XT"])

    def test_the_integrated_radeon_is_left_out_beside_a_card(self):
        self.assertEqual([c["name"] for c in T.windows_amd_cards(self.read([IGPU, CARD], "amd"))], ["AMD Radeon RX 6900 XT"])
        self.assertEqual([c["name"] for c in T.windows_amd_cards(self.read([IGPU], "amd"))], ["AMD Radeon(TM) Graphics"])

    def test_off_windows_nothing(self):
        with mock.patch.object(T.os, "name", "posix"):
            self.assertEqual(T.windows_display_adapters(), [])

    def test_hardware_static_names_the_card(self):
        cards = [{"name": "AMD Radeon RX 6900 XT", "provider": "AMD", "driver": "x", "vram_bytes": 16 * GB}]
        with mock.patch.object(T.os, "name", "nt"), mock.patch.object(T, "windows_amd_cards", return_value=cards), \
                mock.patch.object(T, "gpu_reader", return_value=mock.Mock(ok=lambda: False)), \
                mock.patch.object(T, "_CpuRamFallback"), mock.patch.object(T, "_cpu_name", return_value="cpu"), \
                mock.patch.object(T.threading, "Thread"):
            t = T.Telemetry(amd=True)
        self.assertEqual(t.static["gpu_name"], "AMD Radeon RX 6900 XT")
        self.assertEqual(t.static["gpu_vram_total_mib"], 16384)

    def test_bench_labels_the_card(self):
        import strata_bench as B
        cards = [{"name": "AMD Radeon RX 6900 XT", "provider": "AMD", "driver": "32.0.21001.9024", "vram_bytes": 16 * GB}]
        with mock.patch.object(B, "run_cmd", return_value=""), mock.patch.object(B.os, "name", "nt"), \
                mock.patch.object(T, "windows_amd_cards", return_value=cards):
            info = B.gpu_info_local()
        self.assertEqual(info["gpus"], [{"name": "AMD Radeon RX 6900 XT", "vram_mib": 16384, "driver": "32.0.21001.9024"}])


if __name__ == "__main__":
    unittest.main()
