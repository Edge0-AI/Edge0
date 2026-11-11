"""Linux on ARM64 (the NVIDIA DGX Spark, 0.2.7.1): the platform name, the CUDA engine package for it, and the release tools that make
the Terminal and the engine packages for it. Nothing here needs an ARM machine; tests/terminal/test_release_matrix.py holds the feed
selection rows."""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packaging"))

from strata_control import engine_component as ec  # noqa: E402
from strata_control import feed as feedmod  # noqa: E402
from strata_control import targets  # noqa: E402
import feed_tool  # noqa: E402
import lock_wheels  # noqa: E402
import package_terminal  # noqa: E402


def host(system: str, machine: str) -> str:
    with mock.patch.object(targets.platform, "system", return_value=system), mock.patch.object(targets.platform, "machine", return_value=machine):
        return targets.host_platform()


class PlatformName(unittest.TestCase):
    def test_linux_arm64_is_linux_aarch64_and_nothing_else_changed(self):
        self.assertEqual(host("Linux", "aarch64"), "linux-aarch64")
        self.assertEqual(host("Linux", "arm64"), "linux-aarch64")
        self.assertEqual(host("Linux", "x86_64"), "linux-x86_64")
        self.assertEqual(host("Windows", "AMD64"), "windows-x86_64")
        self.assertEqual(host("Darwin", "arm64"), "macos-arm64")        # a Mac keeps its name
        self.assertEqual(host("Darwin", "x86_64"), "macos-x86_64")

    def test_cuda_is_committed_to_on_linux_aarch64_and_no_other_backend_is(self):
        self.assertTrue(targets.backend_supported("cuda", "linux-aarch64")[0])
        for backend in ("hip", "sycl", "metal", "cpu"):
            self.assertFalse(targets.backend_supported(backend, "linux-aarch64")[0], backend)
        # the rows that were there are still there
        self.assertTrue(targets.backend_supported("cuda", "linux-x86_64")[0])
        self.assertTrue(targets.backend_supported("hip", "linux-x86_64")[0])
        self.assertTrue(targets.backend_supported("metal", "macos-arm64")[0])

    def test_the_feed_knows_the_platform(self):
        self.assertIn("linux-aarch64", feedmod.PLATFORMS)
        self.assertEqual(feedmod.TAURI_TARGETS["linux-aarch64"], "linux-aarch64")
        self.assertEqual(feedmod.PLATFORMS[:2], ("windows-x86_64", "linux-x86_64"))


class GB10Cards(unittest.TestCase):
    def test_sm_121_code_covers_a_gb10_and_only_a_12_x_card(self):
        archs = ("sm_121", "compute_121")
        self.assertTrue(ec.arch_covers(archs, ["sm_121"]))
        self.assertFalse(ec.arch_covers(archs, ["sm_120"]))      # machine code is for the card's own minor or a newer one; PTX needs >= 121
        self.assertFalse(ec.arch_covers(archs, ["sm_86"]))
        self.assertTrue(ec.arch_native(archs, ["sm_121"]))


class ReleaseTools(unittest.TestCase):
    def test_the_terminal_has_a_pinned_interpreter_for_linux_aarch64(self):
        pin = package_terminal.PYTHON_PINS["linux-aarch64"]
        self.assertIn("aarch64-unknown-linux-gnu-install_only.tar.gz", pin["file"])
        self.assertRegex(pin["sha256"], r"^[0-9a-f]{64}$")
        self.assertNotEqual(pin["sha256"], package_terminal.PYTHON_PINS["linux-x86_64"]["sha256"])
        self.assertTrue(pin["url"].startswith(package_terminal.PBS_BASE) and "%2B" in pin["url"])
        self.assertEqual(pin["python"], "python/bin/python3")

    def test_host_target_names_linux_arm64(self):
        with mock.patch.object(package_terminal.sys, "platform", "linux"), mock.patch.object(package_terminal.platform, "machine", return_value="aarch64"):
            self.assertEqual(package_terminal.host_target(), "linux-aarch64")
        with mock.patch.object(package_terminal.sys, "platform", "linux"), mock.patch.object(package_terminal.platform, "machine", return_value="x86_64"):
            self.assertEqual(package_terminal.host_target(), "linux-x86_64")

    def test_the_runtime_lock_for_linux_aarch64_is_made_the_same_way_and_is_complete(self):
        spec = lock_wheels.TARGETS["linux-aarch64"]
        self.assertTrue(all(p.endswith("aarch64") for p in spec["platforms"]))
        self.assertEqual(spec["env"], lock_wheels.TARGETS["linux-x86_64"]["env"])
        lock = ROOT / "packaging" / "locks" / "runtime-linux-aarch64-cp312.txt"
        self.assertTrue(lock.is_file(), "run packaging/lock_wheels.py --target linux-aarch64 on the Spark")
        names = package_terminal.lock_names(lock)
        x86 = package_terminal.lock_names(ROOT / "packaging" / "locks" / "runtime-linux-x86_64-cp312.txt")
        self.assertEqual(set(names), set(x86), "the same packages as the x86_64 Linux lock")
        text = lock.read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"^[A-Za-z0-9_.-]+==", text, re.M)), len(names))
        hashed = re.findall(r"^[A-Za-z0-9_.-]+==\S+ \\\n(?:\s+--hash=sha256:[0-9a-f]{64}(?: \\)?\n?)+", text, re.M)
        self.assertEqual(len(hashed), len(names), "every package of the lock names the SHA-256 of its wheel")

    def test_pack_control_takes_the_platform(self):
        parser_choices = feedmod.PLATFORMS
        self.assertIn("linux-aarch64", parser_choices)
        # the installers' one-line manifest wants tar.gz for every non-Windows platform
        self.assertTrue(feed_tool.feedmod.PLATFORMS is feedmod.PLATFORMS)

    def test_install_sh_installs_on_linux_arm64_and_says_what_the_engine_is_for(self):
        text = (ROOT / "packaging" / "install.sh").read_text(encoding="utf-8")
        self.assertIn("Linux-aarch64|Linux-arm64) platform=linux-aarch64", text)
        self.assertNotIn("has no engine for ARM Linux", text)
        self.assertIn("GB10", text)
        self.assertIn("Linux-x86_64|Linux-amd64) platform=linux-x86_64", text)        # unchanged
        self.assertIn("Darwin-x86_64) die", text)

    def test_the_spark_build_script_is_lf_and_names_the_platform(self):
        raw = (ROOT / "spark" / "build_engine.sh").read_bytes()
        self.assertNotIn(b"\r", raw)
        self.assertIn(b'linux-aarch64', raw)
        self.assertIn(b"-DSTRATA_PORTABLE=ON", raw)


class GB10DeepSeekPlan(unittest.TestCase):
    def test_the_plan_for_a_gb10_speaks_of_one_pool_and_the_system_not_windows_and_a_bios(self):
        from strata_control import dsv41_fit
        card = {"index": 0, "name": "NVIDIA GB10", "vram_gb": 115.6, "shared_gb": 115.6, "dedicated_gb": 0.0, "uma": True}
        plan = dsv41_fit.plan([card], 121.6, 400, 8192, model="IST-DAS-2.00bpw")
        text = " ".join(plan["lines"])
        self.assertNotIn("Windows", text)
        self.assertNotIn("BIOS", text)
        self.assertIn("DGX Spark", plan["measured"])
        self.assertIn("one pool of 122 GB", text)
        self.assertGreater(plan["experts_in_vram"], 11000)          # the engine placed 12,163 of 15,744 on the Spark itself

    def test_the_windows_strix_halo_plan_is_unchanged(self):
        from strata_control import dsv41_fit
        card = {"index": 0, "name": "AMD Radeon(TM) 8060S Graphics", "vram_gb": 96.0, "shared_gb": 32.0, "dedicated_gb": 64.0, "uma": True}
        plan = dsv41_fit.plan([card], 64.0, 400, 8192, model="IST-DAS-2.00bpw")
        text = " ".join(plan["lines"])
        self.assertIn("BIOS carve-out", text)
        self.assertIn("kept for Windows", text)
        self.assertEqual(plan["measured"], dsv41_fit.UMA_MEASURED)


class GB10Probes(unittest.TestCase):
    """nvidia-smi gives a GB10's memory.total as [N/A]: the probes the Terminal reads must neither drop the card nor invent a size."""
    LINE = "0, NVIDIA GB10, GPU-0a1b2c3d-0000-1111-2222-333344445555, [N/A], 580.173.02\n"

    def cards(self, machine):
        from strata_control import doctor
        done = mock.Mock(returncode=0, stdout=self.LINE)
        with mock.patch.object(doctor.shutil, "which", return_value="/usr/bin/nvidia-smi"), \
                mock.patch.object(doctor.subprocess, "run", return_value=done), \
                mock.patch.object(doctor.platform, "machine", return_value=machine):
            return doctor._nvidia_cards()

    def test_doctor_lists_a_gb10_with_no_memory_figure(self):
        cards, why = self.cards("aarch64")
        self.assertIsNone(why)
        self.assertEqual(cards[0]["name"], "NVIDIA GB10")
        self.assertIsNone(cards[0]["memory_mib"])

    def test_doctor_on_x86_still_refuses_a_card_without_a_memory_figure(self):
        cards, why = self.cards("x86_64")
        self.assertIsNone(cards)
        self.assertIn("unrecognized", why)

    def test_the_launch_probe_counts_a_gb10_as_unified_memory(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import strata_mcp
        smi = "0, NVIDIA GB10, [N/A], 12.1, 580.173.02\n"
        with mock.patch.object(strata_mcp, "run_quiet", return_value=smi), \
                mock.patch.object(strata_mcp, "own_ram_gb", return_value=121.6), \
                mock.patch.object(strata_mcp.platform, "machine", return_value="aarch64"):
            found = strata_mcp.own_nvidia_gpus()
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0]["uma"])
        self.assertEqual(found[0]["arch"], "121")
        self.assertAlmostEqual(found[0]["vram_gb"], 115.6, places=1)
        self.assertEqual(found[0]["dedicated_gb"], 0.0)
        with mock.patch.object(strata_mcp, "run_quiet", return_value=smi), \
                mock.patch.object(strata_mcp.platform, "machine", return_value="x86_64"):
            self.assertEqual(strata_mcp.own_nvidia_gpus(), [])


if __name__ == "__main__":
    unittest.main()
