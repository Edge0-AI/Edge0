"""ROCm 7 and newer ship `amd-smi` and no `rocm-smi`, and the amdgpu driver's `product_name` is empty on gfx1201, so a
report from a Radeon wrote the vendor as the card ("AMD Radeon", `...-standard-unknown-...`) and `runtime: not
measured`. The strings below are the real output of `amd-smi static --asic --json`, `static --driver --json`,
`list --json` and `-h` on one AI PRO R9700 (gfx1201) with ROCm 10.1, trimmed to the fields the probe reads."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from strata_cli import bench_facts as bf

ASIC = {"gpu_data": [{"gpu": 0, "asic": {"market_name": "AMD Radeon AI PRO R9700", "target_graphics_version": "gfx1201"}},
                     {"gpu": 1, "asic": {"market_name": "AMD Radeon Graphics", "target_graphics_version": "gfx1036"}}]}
DRIVER = {"gpu_data": [{"gpu": 0, "driver": {"name": "amdgpu", "version": "N/A",
                                             "os_kernel_version": "7.3.0-rc6-2-cachyos-rc"}}]}
LIST = [{"gpu": 0, "bdf": "0000:03:00.0", "node_id": 1}, {"gpu": 1, "bdf": "0000:7c:00.0", "node_id": 2}]
HELP = "AMD System Management Interface | Version: 27.1.0+7da43026 | ROCm version: 10.1.0 |"
BDF_TO_TOTAL = {"0000:03:00.0": "34208743424", "0000:7c:00.0": "805306368"}


def run_cmd(cmd: list[str], timeout: float = 15) -> str:
    """ROCm 10.1 answers, with `rocm-smi` missing entirely."""
    if cmd[:1] == ["rocm-smi"]:
        return ""
    if cmd[:3] == ["amd-smi", "list", "--json"]:
        return json.dumps(LIST)
    if cmd[:2] == ["amd-smi", "static"]:
        return json.dumps(DRIVER if "--driver" in cmd else ASIC)
    if cmd == ["amd-smi", "-h"]:
        return HELP
    return ""


@unittest.skipIf(os.name == "nt", "the probe is the non-Windows branch")
class AmdSmiProbe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # /sys/class/drm/cardN/device is a symlink whose target's last component is the PCI address; the probe matches
        # that address against `amd-smi list`, so the fake tree repeats the shape instead of the path
        devices = []
        for card, bdf in (("card1", "0000:03:00.0"), ("card2", "0000:7c:00.0")):
            pci = os.path.join(self.tmp.name, "pci", bdf)
            os.makedirs(pci)
            with open(os.path.join(pci, "mem_info_vram_total"), "w", encoding="utf-8") as f:
                f.write(BDF_TO_TOTAL[bdf] + "\n")
            link = os.path.join(self.tmp.name, "drm", card, "device")
            os.makedirs(os.path.dirname(link))
            os.symlink(pci, link)
            devices.append(link)
        self.globs = mock.patch.object(bf.glob, "glob", lambda pattern: devices)
        self.globs.start()
        self.addCleanup(self.globs.stop)
        self.cmds = mock.patch.object(bf, "run_cmd", run_cmd)
        self.cmds.start()
        self.addCleanup(self.cmds.stop)

    def test_names_both_cards_with_their_vram(self):
        cards, driver = bf.other_gpus()
        self.assertEqual([c["name"] for c in cards], ["AMD Radeon AI PRO R9700", "AMD Radeon Graphics"])
        self.assertEqual([c["memory_bytes"] for c in cards], [34208743424, 805306368])
        self.assertEqual([c["device_index"] for c in cards], [0, 1])

    def test_a_stub_driver_version_stays_unmeasured(self):
        # `amd-smi static --driver` says N/A when amdgpu is built into the kernel: that is not a version, so the report
        # must keep saying it did not measure one instead of writing "N/A"
        _, driver = bf.other_gpus()
        self.assertIsNone(driver)

    def test_an_asic_without_a_market_name_is_skipped(self):
        blank = {"gpu_data": [{"gpu": 0, "asic": {"market_name": ""}},
                              {"gpu": 1, "asic": {"market_name": "AMD Radeon AI PRO R9700"}}]}
        with mock.patch.object(bf, "run_cmd", lambda cmd, timeout=15: json.dumps(blank) if "static" in cmd else ""):
            cards, _ = bf.other_gpus()
        self.assertEqual([c["name"] for c in cards], ["AMD Radeon AI PRO R9700"])

    def test_rocm_version_comes_from_the_install(self):
        base = os.path.join(self.tmp.name, "rocm")
        os.makedirs(os.path.join(base, "core", ".info"))
        with open(os.path.join(base, "core", ".info", "version"), "w", encoding="utf-8") as f:
            f.write("10.1.0\n")
        with mock.patch.object(bf.os, "environ", {"ROCM_PATH": base}):
            self.assertEqual(bf.rocm_version(), "10.1.0")

    def test_rocm_version_falls_back_to_the_help_text(self):
        real_open = open

        def refuse(path, *args, **kwargs):
            if "/opt/rocm" in str(path):
                raise OSError("no ROCm install here")
            return real_open(path, *args, **kwargs)

        env = {k: v for k, v in os.environ.items() if k != "ROCM_PATH"}
        with mock.patch.object(bf.os, "environ", env), mock.patch("builtins.open", side_effect=refuse):
            self.assertEqual(bf.rocm_version(), "10.1.0")


@unittest.skipIf(os.name == "nt", "the probe is the non-Windows branch")
class ServedName(unittest.TestCase):
    """The server's `gpu` string is the driver's product_name, empty on gfx1201, so it degrades to the vendor. That
    must not overwrite the card the local probe named, while a card only the server can see still wins."""

    def cards(self):
        return [{"device_index": 0, "name": "AMD Radeon AI PRO R9700", "memory_bytes": 34208743424,
                 "memory_kind": "dedicated", "power_limit_w": None}]

    def hardware(self, served_gpu):
        with mock.patch.object(bf, "nvidia_gpus", lambda: ([], None, None)), \
             mock.patch.object(bf, "other_gpus", lambda: (self.cards(), None)):
            block, machine = bf.hardware(bf.Why(), selected=[], served_gpu=served_gpu, model_dir=None,
                                         declared_background=[])
        return block, machine

    def test_a_vendor_only_name_does_not_replace_the_card(self):
        block, machine = self.hardware("AMD Radeon")
        self.assertEqual([c["name"] for c in block["gpus"]], ["AMD Radeon AI PRO R9700"])
        self.assertEqual(block["gpus"][0]["memory_bytes"], 34208743424)

    def test_a_card_only_the_server_lists_still_wins(self):
        block, _ = self.hardware("AMD Radeon RX 7900 XTX")
        self.assertEqual([c["name"] for c in block["gpus"]], ["AMD Radeon RX 7900 XTX"])
        self.assertIsNone(block["gpus"][0]["memory_bytes"])

    def test_the_rocm_runtime_reaches_the_report(self):
        why = bf.Why()
        with mock.patch.object(bf, "nvidia_gpus", lambda: ([], None, None)), \
             mock.patch.object(bf, "other_gpus", lambda: (self.cards(), None)), \
             mock.patch.object(bf, "rocm_version", lambda: "10.1.0"):
            _, machine = bf.hardware(why, selected=[], served_gpu="AMD Radeon", model_dir=None,
                                     declared_background=[])
            software = bf.software(why, dict(machine), engine_version="0.1.46.2", build_options=None)
        self.assertEqual(machine["rocm"], "10.1.0")
        self.assertIn({"name": "rocm", "version": "10.1.0"}, software["runtime"])


class MacStillReadsItsChip(unittest.TestCase):
    """The amd-smi branch is for Linux; a Mac (os.name is posix there too) still takes its card from the chip name."""

    def test_a_mac_is_not_sent_down_the_amd_smi_branch(self):
        calls = []

        def run(cmd, timeout=15):
            calls.append(cmd[0])
            return ""

        with mock.patch.object(bf.os, "name", "posix"), mock.patch.object(bf.sys, "platform", "darwin"),                 mock.patch.object(bf, "run_cmd", run), mock.patch.object(bf, "cpu_name", lambda: "Apple M5 Max"),                 mock.patch.object(bf, "ram_bytes", lambda: 137438953472):
            cards, driver = bf.other_gpus()
        self.assertEqual([c["name"] for c in cards], ["Apple M5 Max"])
        self.assertEqual(cards[0]["memory_kind"], "shared")
        self.assertNotIn("amd-smi", calls)


if __name__ == "__main__":
    unittest.main()
