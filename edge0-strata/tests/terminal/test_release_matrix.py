"""The 0.2.0-beta release set as a signed feed offers it: one engine package per platform/backend/variant (values copied from the
packages' engine-component.json), and what `feed.select_engine` hands each kind of PC. This is the evidence that engine updates
reach every installer variant by card, so the app update can stay engine-less (tauri.update.conf.json)."""
from __future__ import annotations

import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from strata_control import feed as feedmod

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packaging"))
import feed_tool  # noqa: E402

V = "0.1.45.3"
SM_NEW = ("sm_75", "sm_86", "sm_89", "sm_120", "compute_120")                       # the CUDA 13 packages
SM_OLD = ("sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "compute_89")   # the CUDA 12 packages (Pascal and Volta)


def pkg(platform, backend, variant, archs, driver=None):
    return {"version": V, "platform": platform, "backend": backend, "variant": variant, "archs": list(archs),
            "min_driver": {"nvidia": driver} if driver else {}, "min_runtime": None,
            "build_id": f"rel-{platform}-{backend}-{variant}", "format": "tar.gz", "url": f"strata-engine-{platform}-{variant}.tar.gz",
            "size": 1000, "sha256": "a" * 64, "manifest_sha256": "b" * 64, "installed_bytes": 2000, "notes": ""}


RELEASE = [
    pkg("windows-x86_64", "cuda", "cuda13", SM_NEW, "580.65"),
    pkg("windows-x86_64", "cuda", "cuda12", SM_OLD, "576.02"),
    pkg("windows-x86_64", "hip", "rx9070", ("gfx1200", "gfx1201")),
    pkg("windows-x86_64", "hip", "rdna3", ("gfx1100", "gfx1101", "gfx1102", "gfx1151")),
    pkg("windows-x86_64", "hip", "rdna2", ("gfx1030", "gfx1031")),
    pkg("linux-x86_64", "cuda", "cuda13", SM_NEW, "580.65"),
    pkg("linux-x86_64", "cuda", "cuda12", SM_OLD, "575.51"),
    pkg("linux-x86_64", "hip", "rx9070", ("gfx1200", "gfx1201")),
    pkg("linux-x86_64", "hip", "rx7900", ("gfx1100", "gfx1101", "gfx1102")),
    pkg("linux-x86_64", "hip", "halo", ("gfx1151",)),
    pkg("linux-x86_64", "hip", "rx6000", ("gfx1030", "gfx1031")),
    pkg("linux-aarch64", "cuda", "cuda13", ("sm_121", "compute_121"), "580.65"),     # the NVIDIA DGX Spark (GB10), 0.2.7.1
    pkg("linux-x86_64", "sycl", "", ("intel-arc",)),                 # the Intel Arc (A and B series) engine, GPU code JIT-compiled
    pkg("macos-arm64", "metal", "", ("apple-gpu",)),
]


def pc(platform, backend, driver=None, sm=(), gfx=()):
    return {"platform": platform, "backend": backend, "runtime": "0.2.0", "nvidia_driver": driver, "cuda_archs": list(sm), "gfx": list(gfx)}


class ReleaseSelection(unittest.TestCase):
    def pick(self, hardware):
        best, reasons = feedmod.select_engine([feedmod._engine_package(p) for p in RELEASE], hardware)
        return (best.platform, best.backend, best.variant) if best else None, reasons

    def test_every_release_variant_is_chosen_by_some_pc(self):
        chosen = {
            self.pick(pc("windows-x86_64", "cuda", "590.10", ["sm_120"]))[0],     # RTX 5070
            self.pick(pc("windows-x86_64", "cuda", "576.02", ["sm_61"]))[0],      # GTX 1080 on an older driver
            self.pick(pc("windows-x86_64", "hip", gfx=["gfx1201"]))[0],           # RX 9070
            self.pick(pc("windows-x86_64", "hip", gfx=["gfx1100"]))[0],           # RX 7900 XTX
            self.pick(pc("windows-x86_64", "hip", gfx=["gfx1030"]))[0],           # RX 6800 XT
            self.pick(pc("linux-x86_64", "cuda", "615.71", ["sm_86"]))[0],        # RTX 3060
            self.pick(pc("linux-x86_64", "cuda", "575.51", ["sm_60"]))[0],        # Tesla P100
            self.pick(pc("linux-x86_64", "hip", gfx=["gfx1201"]))[0],             # Radeon AI PRO R9700
            self.pick(pc("linux-x86_64", "hip", gfx=["gfx1100"]))[0],             # RX 7900 XTX
            self.pick(pc("linux-x86_64", "hip", gfx=["gfx1151"]))[0],             # Ryzen AI Max (Strix Halo)
            self.pick(pc("linux-x86_64", "hip", gfx=["gfx1031"]))[0],             # RX 6700 XT
            self.pick(pc("linux-aarch64", "cuda", "580.173", ["sm_121"]))[0],     # DGX Spark (GB10)
            self.pick(pc("linux-x86_64", "sycl"))[0],                             # Arc A750 / B580 / Pro B70
            self.pick(pc("macos-arm64", "metal"))[0],
        }
        self.assertEqual(chosen, {(p["platform"], p["backend"], p["variant"]) for p in RELEASE})

    def test_cuda_13_or_12_follows_the_driver_and_the_card(self):
        win = "windows-x86_64"
        self.assertEqual(self.pick(pc(win, "cuda", "580.65", ["sm_86"]))[0][2], "cuda13")
        self.assertEqual(self.pick(pc(win, "cuda", "590.10", ["sm_120"]))[0][2], "cuda13")
        got, why = self.pick(pc(win, "cuda", "576.02", ["sm_120"]))                 # a new card on an older driver: CUDA 12 reaches it by PTX
        self.assertEqual(got[2], "cuda12")
        self.assertTrue(any("580.65" in r for r in why))
        self.assertEqual(self.pick(pc(win, "cuda", "615.71", ["sm_70"]))[0][2], "cuda12", "Volta is only in the CUDA 12 build")
        self.assertEqual(self.pick(pc("linux-x86_64", "cuda", "575.51", ["sm_75"]))[0][2], "cuda12")
        self.assertEqual(self.pick(pc("linux-x86_64", "cuda", "580.65", ["sm_75"]))[0][2], "cuda13")

    def test_the_dgx_spark_gets_the_aarch64_build_and_no_other_pc_does(self):
        self.assertEqual(self.pick(pc("linux-aarch64", "cuda", "580.173", ["sm_121"]))[0], ("linux-aarch64", "cuda", "cuda13"))
        for hw in (pc("linux-x86_64", "cuda", "590.10", ["sm_120"]), pc("linux-x86_64", "cuda", "590.10", ["sm_121"]),
                   pc("windows-x86_64", "cuda", "590.10", ["sm_121"])):
            self.assertNotEqual(self.pick(hw)[0][0], "linux-aarch64", hw)
        got, why = self.pick(pc("linux-aarch64", "cuda", "580.173", ["sm_86"]))      # an arm64 server with an A-series card: no machine code for it
        self.assertIsNone(got)
        self.assertTrue(any("sm_121" in r for r in why))
        got, why = self.pick(pc("linux-aarch64", "cuda", "570.10", ["sm_121"]))      # the driver is too old
        self.assertIsNone(got)
        self.assertTrue(any("580.65" in r for r in why))
        self.assertIsNone(self.pick(pc("linux-aarch64", "hip", gfx=["gfx1201"]))[0])    # no AMD engine for arm64 Linux

    def test_a_driver_that_is_too_old_gets_no_engine_and_a_reason(self):
        got, why = self.pick(pc("windows-x86_64", "cuda", "551.00", ["sm_86"]))
        self.assertIsNone(got)
        self.assertTrue(any("576.02" in r for r in why) and any("580.65" in r for r in why))

    def test_amd_gets_its_own_build_and_never_a_cuda_one(self):
        self.assertEqual(self.pick(pc("linux-x86_64", "hip", gfx=["gfx1200"]))[0][2], "rx9070")
        self.assertEqual(self.pick(pc("linux-x86_64", "hip", gfx=["gfx1102"]))[0][2], "rx7900")
        for hw in (pc("windows-x86_64", "hip", gfx=["gfx1201"]), pc("linux-x86_64", "hip", gfx=["gfx1201"])):
            self.assertEqual(self.pick(hw)[0][1], "hip")

    def test_rx7000_and_strix_halo_on_windows_get_the_rdna3_build(self):
        for gfx in ("gfx1100", "gfx1101", "gfx1102", "gfx1151"):
            self.assertEqual(self.pick(pc("windows-x86_64", "hip", gfx=[gfx]))[0], ("windows-x86_64", "hip", "rdna3"), gfx)
        self.assertEqual(self.pick(pc("windows-x86_64", "hip", gfx=["gfx1201"]))[0][2], "rx9070")
        self.assertEqual(self.pick(pc("windows-x86_64", "hip", gfx=["gfx1200"]))[0][2], "rx9070")

    def test_strix_halo_on_linux_gets_the_halo_build_and_rx7000_keeps_rx7900(self):
        self.assertEqual(self.pick(pc("linux-x86_64", "hip", gfx=["gfx1151"]))[0], ("linux-x86_64", "hip", "halo"))
        for gfx in ("gfx1100", "gfx1101", "gfx1102"):
            self.assertEqual(self.pick(pc("linux-x86_64", "hip", gfx=[gfx]))[0][2], "rx7900", gfx)

    def test_no_amd_card_matches_two_hip_builds_of_one_platform(self):
        """A tie between two builds of the same version would be settled by list order: no arch may be in two hip builds of a platform."""
        for platform in ("windows-x86_64", "linux-x86_64"):
            seen = {}
            for p in RELEASE:
                if p["platform"] == platform and p["backend"] == "hip":
                    for arch in p["archs"]:
                        self.assertNotIn(arch, seen, f"{arch} is in {seen.get(arch)} and {p['variant']} on {platform}")
                        seen[arch] = p["variant"]

    def test_other_unified_memory_radeons_get_no_engine_and_the_plain_family_line(self):
        from strata_control import engine_component as ec
        for platform in ("windows-x86_64", "linux-x86_64"):
            for gfx in ("gfx1103", "gfx1150", "gfx1152", "gfx1032"):
                got, why = self.pick(pc(platform, "hip", gfx=[gfx]))
                self.assertIsNone(got, f"{gfx} on {platform}")
                self.assertTrue(any(gfx in r for r in why))
        self.assertEqual(ec.arch_notice(("gfx1100", "gfx1151"), ["gfx1201"]),
                         "This build of Strata is for AMD RX 7000 and Ryzen AI Max cards. This computer has an AMD RX 9000 card (gfx1201). Ask for the RX 9000 build.")
        self.assertEqual(ec.arch_notice(("gfx1151",), ["gfx1201"]),
                         "This build of Strata is for AMD Ryzen AI Max cards. This computer has an AMD RX 9000 card (gfx1201). Ask for the RX 9000 build.")

    def test_rx6000_has_its_own_package_and_only_gfx103x_cards_get_it(self):
        """0.2.2: gfx1030 / gfx1031 are in the RX 6000 package (Linux rx6000, Windows rdna2) and in no other; no RX 7000 / 9000 card gets it."""
        from strata_control import engine_component as ec
        want = {"windows-x86_64": "rdna2", "linux-x86_64": "rx6000"}
        for platform, variant in want.items():
            for gfx in ("gfx1030", "gfx1031"):
                self.assertEqual(self.pick(pc(platform, "hip", gfx=[gfx]))[0], (platform, "hip", variant), (platform, gfx))
            for gfx in ("gfx1100", "gfx1101", "gfx1102", "gfx1151", "gfx1200", "gfx1201"):
                got = self.pick(pc(platform, "hip", gfx=[gfx]))[0]
                self.assertNotEqual(got[2] if got else None, variant, (platform, gfx))
        self.assertEqual(ec.NO_PACKAGE_HIP, {}, "every card family setup supports has a package; the table is for the next family without one")
        covered = {a for p in RELEASE if p["backend"] == "hip" for a in p["archs"]}
        self.assertTrue({"gfx1030", "gfx1031"} <= covered)
        # the plain family line now names a build that exists
        self.assertIn("Ask for the RX 6000 build.", ec.arch_notice(("gfx1200", "gfx1201"), ["gfx1030"]))
        self.assertIsNone(ec.no_package_note(["gfx1030", "gfx1031"]))

    def test_a_pc_with_an_rx7000_beside_an_rx6000_keeps_the_package_it_always_got(self):
        """Before the RX 6000 package a PC with both cards got the RX 7000 one: the new package must not win that tie, in any feed order."""
        for platform, old, new in (("windows-x86_64", "rdna3", "rdna2"), ("linux-x86_64", "rx7900", "rx6000")):
            hardware = pc(platform, "hip", gfx=["gfx1030", "gfx1100"])
            for order in (RELEASE, list(reversed(RELEASE))):
                best, _ = feedmod.select_engine([feedmod._engine_package(p) for p in order], hardware)
                self.assertEqual(best.variant, old, (platform, new))
        # a Radeon AI PRO (gfx1201) beside an RX 6800 keeps rx9070 too
        for platform in ("windows-x86_64", "linux-x86_64"):
            for order in (RELEASE, list(reversed(RELEASE))):
                best, _ = feedmod.select_engine([feedmod._engine_package(p) for p in order], pc(platform, "hip", gfx=["gfx1201", "gfx1030"]))
                self.assertEqual(best.variant, "rx9070")

    def test_installed_packages_are_chosen_the_same_way(self):
        from strata_control import engine_component as ec
        self.assertEqual(ec.hip_preference(("gfx1030", "gfx1031")), 0)
        self.assertEqual(ec.hip_preference(("gfx1100", "gfx1101", "gfx1102")), 1)
        self.assertEqual(ec.hip_preference(("gfx1030", "gfx1100")), 1)
        self.assertEqual(ec.hip_preference(()), 1)

    def test_the_platform_decides_first(self):
        self.assertIsNone(self.pick(pc("macos-arm64", "cuda", "590.10", ["sm_86"]))[0])
        self.assertEqual(self.pick(pc("macos-arm64", "metal"))[0], ("macos-arm64", "metal", ""))


class PackedOnWindows(unittest.TestCase):
    def test_programs_are_recognised_by_their_header_so_a_linux_package_packed_on_windows_keeps_them_executable(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "strata-vision").write_bytes(b"\x7fELF\x02\x01" + bytes(20))
            (root / "libx.dylib").write_bytes(b"\xcf\xfa\xed\xfe" + bytes(20))
            (root / "run.sh").write_bytes(b"#!/bin/sh\n")
            (root / "BUILD.json").write_text("{}")
            (root / "kernel.hsaco").write_bytes(b"\x00\x01\x02\x03")
            names = ("strata-vision", "libx.dylib", "run.sh", "BUILD.json", "kernel.hsaco")
            self.assertEqual([feed_tool._is_native_executable(root / n) for n in names], [True, True, True, False, False])
            if os.name == "nt":                     # the owner's release PC: the archive is what matters
                archive = root / "a.tar.gz"
                feed_tool._write_archive([(n, root / n) for n in ("strata-vision", "BUILD.json")], archive, "tar.gz", None)
                with tarfile.open(archive) as tar:
                    modes = {m.name: m.mode for m in tar.getmembers()}
                self.assertEqual(modes, {"strata-vision": 0o755, "BUILD.json": 0o644})


if __name__ == "__main__":
    unittest.main()
