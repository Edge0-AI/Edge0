"""Intel Arc engine package (owner test round 8, track L3): the stable feed had no `linux-x86_64-sycl` package, so a clean Arc PC got a
Terminal that could not run a model and `strata update` kept saying the feed had no engine for it. The feed is whatever packages the release
builds (packaging/feed_tool.py takes every `*.entry.json`), so the gap was the missing build recipe, not the choice code. These tests keep the
choice honest once the package exists: an Arc PC (found by its PCI vendor, with no oneAPI tools installed) gets it by itself, nobody else does,
and the recipe that makes it carries the checks that a clean PC needs."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import engine_component as ec, feed as feedmod, updater

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))
import package_engine  # noqa: E402

LINUX, WIN = "linux-x86_64", "windows-x86_64"


def entry(platform, backend, variant, archs, version="0.1.47"):
    return {"version": version, "platform": platform, "backend": backend, "variant": variant, "archs": list(archs), "min_driver": {},
            "min_runtime": None, "build_id": f"rel-{platform}-{backend}-{variant}", "format": "tar.gz",
            "url": f"strata-engine-{platform}-{backend}-{variant}.tar.gz", "size": 1000, "sha256": "a" * 64, "manifest_sha256": "b" * 64,
            "installed_bytes": 2000, "notes": ""}


FEED = [feedmod._engine_package(e) for e in (
    entry(LINUX, "cuda", "cuda13", ("sm_75", "sm_86", "sm_89", "sm_120", "compute_120")),
    entry(LINUX, "hip", "rx9070", ("gfx1200", "gfx1201")),
    entry(WIN, "cuda", "cuda13", ("sm_86",)),
    entry(LINUX, "sycl", "", ("intel-arc",)))]


def fake_drm(root: Path, *vendors: str) -> str:
    """A /sys/class/drm with one card per vendor id (plus a connector entry, which must be skipped)."""
    for i, vendor in enumerate(vendors):
        dev = root / f"card{i}" / "device"
        dev.mkdir(parents=True)
        (dev / "vendor").write_text(vendor + "\n")
        (root / f"card{i}-HDMI-A-1").mkdir()
    return str(root)


class ArcChoice(unittest.TestCase):
    def detect(self, *vendors, platform=LINUX, environ=None):
        """What `updater.detect_hardware` says on a PC with these display vendors and none of the vendor tools installed."""
        with tempfile.TemporaryDirectory() as t:
            drm = fake_drm(Path(t), *vendors)
            real = ec._drm_vendor_backend
            with mock.patch.object(ec.shutil, "which", return_value=None), \
                    mock.patch.object(ec, "host_platform", return_value=platform), \
                    mock.patch.object(updater, "host_platform", return_value=platform), \
                    mock.patch.object(ec, "_drm_vendor_backend", lambda: real(drm)), \
                    mock.patch.object(ec, "windows_display_adapters", return_value=[]),                     mock.patch.object(ec, "_windows_vendor_backend", lambda: real(drm)):      # the test PC may be Windows: either probe answers
                return updater.detect_hardware(environ or {})

    def pick(self, hardware):
        best, reasons = feedmod.select_engine(FEED, hardware)
        return best, reasons

    def test_a_clean_arc_pc_without_oneapi_gets_the_sycl_package(self):
        hw = self.detect("0x8086")
        self.assertEqual((hw["platform"], hw["backend"]), (LINUX, "sycl"))
        best, _ = self.pick(hw)
        self.assertEqual((best.backend, best.platform, best.archs), ("sycl", LINUX, ("intel-arc",)))

    def test_nvidia_and_amd_pcs_never_get_it_even_with_an_intel_igpu_beside_the_card(self):
        for vendors, want in ((("0x10de",), "cuda"), (("0x8086", "0x10de"), "cuda"), (("0x8086", "0x1002"), "hip")):
            hw = self.detect(*vendors)
            best, _ = self.pick(hw)
            self.assertEqual(best.backend, want, vendors)

    def test_windows_has_no_intel_package_and_says_why(self):
        hw = self.detect("0x8086", platform=WIN)
        best, why = self.pick(hw)
        self.assertIsNone(best)
        self.assertTrue(any("sycl" in r and WIN in r for r in why), why)
        self.assertEqual(ec.backend_supported("sycl", WIN)[0], False)
        self.assertEqual(ec.backend_supported("sycl", LINUX)[0], True)

    def test_the_package_is_not_offered_to_a_mac_or_to_a_forced_other_backend(self):
        self.assertIsNone(self.pick({"platform": "macos-arm64", "backend": "metal", "runtime": "0.2.2"})[0])
        best, _ = self.pick(self.detect("0x8086", environ={ec.ENV_BACKEND: "cuda"}))
        self.assertEqual(best.backend, "cuda")

    def test_an_older_sycl_package_does_not_beat_the_new_one_and_a_newer_one_wins(self):
        packages = [feedmod._engine_package(entry(LINUX, "sycl", "", ("intel-arc",), v)) for v in ("0.1.45", "0.1.47", "0.1.46")]
        best, _ = feedmod.select_engine(packages, self.detect("0x8086"))
        self.assertEqual(best.version, "0.1.47")

    def test_no_package_at_all_gives_the_plain_no_engine_reason(self):
        best, why = feedmod.select_engine([p for p in FEED if p.backend != "sycl"], self.detect("0x8086"))
        self.assertIsNone(best)
        self.assertTrue(any("needs sycl" in r for r in why), why)


class Recipe(unittest.TestCase):
    SCRIPT = ROOT / "packaging" / "build_sycl_linux.sh"

    def test_the_recipe_exists_and_is_valid_shell(self):
        self.assertTrue(self.SCRIPT.is_file())
        sh = shutil.which("sh")
        if sh:
            self.assertEqual(subprocess.run([sh, "-n", str(self.SCRIPT)], capture_output=True).returncode, 0)

    def test_the_recipe_bundles_the_runtime_and_checks_a_clean_start(self):
        text = self.SCRIPT.read_text(encoding="utf-8")
        for needed in ("collect_libs.py", "libur_adapter_level_zero", "libur_adapter_opencl", "patchelf --set-rpath",
                       "package_engine.py --verify", "binary_deps.py", "env -i", "--version", "--backend sycl", "intel-arc"):
            self.assertIn(needed, text)
        self.assertNotIn("\r", text, "a shell script with CRLF endings does not run on Linux")

    def test_a_sycl_package_needs_an_arch_tag_and_only_linux(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(SystemExit):
                package_engine.build(Path(t), "sycl", "0.1.47", LINUX, Path(t) / "out")
        self.assertIn(LINUX, ec.BACKEND_MATRIX["sycl"])
        self.assertNotIn(WIN, ec.BACKEND_MATRIX["sycl"])


if __name__ == "__main__":
    unittest.main()
