"""`bench.sh` (tools/strata_bench.py) has its own GPU probe, so the ROCm 7+ fix in strata_cli/bench_facts.py did not
reach it: with no `rocm-smi` on the box it wrote the server's vendor stub as the card ("AMD Radeon",
`...-community-unknown-...`) and `GPU driver: not read`. The strings below are the real answers of `amd-smi static
--asic --json`, `static --driver --json` and `list --json` on one AI PRO R9700 (gfx1201) plus a Ryzen iGPU, with ROCm
10.1, trimmed to the fields the probe reads."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import strata_bench as sb

ASIC = {"gpu_data": [{"gpu": 0, "asic": {"market_name": "AMD Radeon AI PRO R9700", "target_graphics_version": "gfx1201"}},
                     {"gpu": 1, "asic": {"market_name": "AMD Radeon Graphics", "target_graphics_version": "gfx1036"}}]}
DRIVER = {"gpu_data": [{"gpu": 0, "driver": {"name": "amdgpu", "version": "N/A",
                                             "os_kernel_version": "7.3.0-rc6-2-cachyos-rc"}},
                       {"gpu": 1, "driver": {"name": "amdgpu", "version": "6.16.4"}}]}
LIST = [{"gpu": 0, "bdf": "0000:03:00.0", "node_id": 1}, {"gpu": 1, "bdf": "0000:7c:00.0", "node_id": 2}]
BDF_TO_TOTAL = {"0000:03:00.0": "34208743424", "0000:7c:00.0": "805306368"}


def run_cmd(cmd: list[str], timeout: float = 15) -> str:
    """ROCm 10.1 answers, with `rocm-smi` missing entirely."""
    if cmd[:1] == ["rocm-smi"] or cmd[:1] == ["nvidia-smi"]:
        return ""
    if cmd[:3] == ["amd-smi", "list", "--json"]:
        return json.dumps(LIST)
    if cmd[:2] == ["amd-smi", "static"]:
        return json.dumps(DRIVER if "--driver" in cmd else ASIC)
    return ""


@unittest.skipIf(os.name == "nt", "the probe is the non-Windows branch")
class AmdSmiProbe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # /sys/class/drm/cardN/device is a symlink whose target ends in the PCI address; the fake tree repeats the
        # shape, and the probe matches that address against `amd-smi list`
        root = Path(self.tmp.name)
        drm = root / "drm"
        for card, bdf in (("card1", "0000:03:00.0"), ("card2", "0000:7c:00.0")):
            real = root / "devices" / bdf
            real.mkdir(parents=True)
            (real / "mem_info_vram_total").write_text(BDF_TO_TOTAL[bdf], encoding="utf-8")
            drm.mkdir(parents=True, exist_ok=True)
            (drm / card).mkdir(exist_ok=True)
            os.symlink(real, drm / card / "device")
        self.drm = str(drm)
        real_path = Path

        class DrmPath(real_path):
            def __new__(cls, *a, **kw):
                return real_path(self.drm) if a == ("/sys/class/drm",) else real_path(*a, **kw)

        self.drm_patch = mock.patch.object(sb, "Path", DrmPath)
        self.drm_patch.start()
        self.addCleanup(self.drm_patch.stop)
        self.cmd_patch = mock.patch.object(sb, "run_cmd", run_cmd)
        self.cmd_patch.start()
        self.addCleanup(self.cmd_patch.stop)

    def test_the_probe_names_both_cards(self):
        info = sb.gpu_info_local()
        self.assertEqual([g["name"] for g in info["gpus"]], ["AMD Radeon AI PRO R9700", "AMD Radeon Graphics"])
        self.assertEqual([g["vram_mib"] for g in info["gpus"]], [34208743424 >> 20, 805306368 >> 20])
        self.assertEqual([g["driver"] for g in info["gpus"]], [None, "6.16.4"])
        self.assertEqual(info.get("rocm_driver"), "6.16.4")

    def test_the_folder_names_the_card_it_used(self):
        # the suite reports the card the engine runs on: the words gpu_slug drops are the vendor ones only
        self.assertEqual(sb.gpu_slug({"gpu": ["AMD Radeon AI PRO R9700"]}), "ai-pro-r9700")

    def test_the_vendor_stub_does_not_replace_the_card(self):
        metrics = {"hardware_static": {"gpu_name": "AMD Radeon", "cpu_name": "AMD Ryzen 9 9950X3D",
                                       "cores": 16, "threads": 32}, "engine": {}}
        with mock.patch.object(sb, "try_json", lambda base, path, key="", timeout=20: metrics if path == "/metrics" else {}):
            m = sb.collect_machine("http://127.0.0.1:1", "", None)
        self.assertEqual(m["gpu"], ["AMD Radeon AI PRO R9700"])
        self.assertEqual(m["gpu_vram_mib"], 34208743424 >> 20)
        self.assertEqual(m["gpu_driver"], None)


if __name__ == "__main__":
    unittest.main()
