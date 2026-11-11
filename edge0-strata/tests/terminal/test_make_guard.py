"""#3688: the update path never crosses GPU makes. An AMD R9700 PC (hip rx9070 installed) was given the cuda13 engine because a leftover
nvidia-smi made detection say cuda. Three guards, all tested here with faked hardware:
  1. detection: on Windows the display adapters decide first, and a tool for a make with no adapter is ignored;
  2. the update decision: an installed engine of another make is never replaced by one of the detected make (unless the person asks);
  3. the stage step refuses a package of another make even if a decision slipped it through."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from strata_control import engine_component as ec, feed as feedmod, updater

WIN = "windows-x86_64"

R9700_ONLY = [{"vendor": "1002", "device": 0x7551, "name": "AMD Radeon AI PRO R9700", "vram_gb": 32.0}]
R9700_AND_NVIDIA = R9700_ONLY + [{"vendor": "10de", "device": 0x2504, "name": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0}]
AMD_IGPU_ONLY = [{"vendor": "1002", "device": 0x164E, "name": "AMD Radeon(TM) Graphics", "vram_gb": 0.5}]
NVIDIA_ONLY = [{"vendor": "10de", "device": 0x2504, "name": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0}]


def package(backend, variant, archs, version="0.1.47"):
    return feedmod._engine_package({
        "version": version, "platform": WIN, "backend": backend, "variant": variant, "archs": list(archs),
        "min_driver": {"nvidia": "580.65"} if backend == "cuda" else {}, "min_runtime": None, "build_id": f"b-{variant}",
        "format": "zip", "url": f"e-{variant}.zip", "size": 10, "sha256": "a" * 64, "manifest_sha256": "b" * 64,
        "installed_bytes": 20, "notes": ""})


FEED = SimpleNamespace(engine=[package("cuda", "cuda13", ("sm_75", "sm_86", "sm_120")),
                               package("hip", "rdna3", ("gfx1100", "gfx1151")),
                               package("hip", "rx9070", ("gfx1200", "gfx1201"))])


class Detection(unittest.TestCase):
    def detect(self, adapters, tools=("nvidia-smi",)):
        def fake_run(command, **kw):
            return subprocess.CompletedProcess(command, 0, stdout="NVIDIA GeForce RTX 3060\n", stderr="")
        with mock.patch.object(ec.os, "name", "nt"), mock.patch.object(ec, "host_platform", return_value=WIN), \
                mock.patch.object(ec.shutil, "which", lambda name: name if name in tools else None), \
                mock.patch.object(ec.subprocess, "run", fake_run), mock.patch.object(ec, "windows_display_adapters", return_value=adapters):
            return ec.detect_backend({})

    def test_a_leftover_nvidia_smi_does_not_make_an_amd_pc_cuda(self):
        self.assertEqual(self.detect(R9700_ONLY), ("hip", "display adapters"))

    def test_a_leftover_nvidia_smi_beside_only_an_integrated_radeon_is_not_believed_either(self):
        self.assertEqual(self.detect(AMD_IGPU_ONLY), (None, "no GPU tool answered"))

    def test_an_nvidia_pc_is_still_cuda(self):
        self.assertEqual(self.detect(NVIDIA_ONLY), ("cuda", "display adapters"))

    def test_both_makes_keep_the_usual_order(self):
        self.assertEqual(self.detect(R9700_AND_NVIDIA)[0], "cuda")
        self.assertEqual(ec.adapter_makes(R9700_AND_NVIDIA), {"cuda", "hip"})

    def test_adapters_that_cannot_be_read_leave_the_tools_in_charge(self):
        self.assertEqual(self.detect([]), ("cuda", "nvidia-smi"))

    def test_the_environment_override_still_wins(self):
        with mock.patch.object(ec, "windows_display_adapters", return_value=R9700_ONLY):
            self.assertEqual(ec.detect_backend({"STRATA_ENGINE_BACKEND": "cuda"}), ("cuda", "environment"))


class Decision(unittest.TestCase):
    """Faked hardware and a faked installed engine (a HIP package cannot be built on a machine without the HIP tuning files)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-makeguard-")
        self.addCleanup(self.temp.cleanup)
        self.components = Path(self.temp.name) / "components"
        self.state = updater.State(self.components)

    def installed(self, backend, variant, version="0.1.46"):
        """Pretend this engine is installed: what the folder scan and the start path would report."""
        engine = SimpleNamespace(version=version, platform=WIN, backend=backend, variant=variant, build_id="b", root=Path("x") / f"{version}-{backend}", signature="s")
        for patch in (mock.patch.object(ec, "scan", return_value=([engine], [])),
                      mock.patch.object(updater, "installed_engine", lambda root, hardware: (engine if hardware.get("backend") == backend else None, None))):
            patch.start()
            self.addCleanup(patch.stop)

    def hardware(self, backend, makes, source="nvidia-smi", **extra):
        return {"platform": WIN, "backend": backend, "backend_source": source, "runtime": None, "nvidia_driver": "610.47",
                "cuda_archs": ["sm_86"] if backend == "cuda" else [], "gfx": ["gfx1201"] if backend == "hip" else [], "gpu_names": [],
                "makes": list(makes), **extra}

    def ctx(self, hardware):
        return updater.Context(components=self.components, source="", trusted={}, hardware=hardware, environ={})

    def test_the_r9700_case_detection_says_cuda_but_the_adapters_list_only_the_amd_card_so_the_update_stays_hip(self):
        self.installed("hip", "rx9070")
        ctx = self.ctx(self.hardware("cuda", ["hip"]))           # what a leftover nvidia-smi said
        with mock.patch.object(updater, "detect_hardware", return_value=self.hardware("hip", ["hip"])):
            report, pkg = updater._choose_engine(FEED, ctx, self.state)
        self.assertEqual(ctx.hardware["backend"], "hip")
        self.assertEqual(ctx.hardware["backend_source"], "installed engine")
        self.assertEqual((pkg.backend, pkg.variant), ("hip", "rx9070"))
        self.assertEqual(report["status"], "available")

    def test_with_no_card_of_the_installed_make_the_update_is_refused_in_plain_words(self):
        self.installed("hip", "rx9070")
        report, pkg = updater._choose_engine(FEED, self.ctx(self.hardware("cuda", ["cuda"])), self.state)
        self.assertIsNone(pkg)
        self.assertEqual(report["status"], "wrong-make")
        for word in ("AMD", "NVIDIA", "Nothing was changed"):
            self.assertIn(word, report["note"])
        self.assertIsNone(report["offered"])
        self.assertEqual(report["installed"]["backend"], "hip")

    def test_cards_unreadable_and_a_different_make_installed_also_refuses(self):
        self.installed("hip", "rx9070")
        report, pkg = updater._choose_engine(FEED, self.ctx(self.hardware("cuda", [])), self.state)
        self.assertEqual((report["status"], pkg), ("wrong-make", None))

    def test_nothing_installed_yet_is_not_blocked(self):
        with mock.patch.object(ec, "scan", return_value=([], [])):
            report, pkg = updater._choose_engine(FEED, self.ctx(self.hardware("cuda", ["cuda"])), self.state)
        self.assertEqual((report["status"], pkg.backend), ("install", "cuda"))

    def test_the_same_make_updates_as_before(self):
        self.installed("cuda", "cuda13", version="0.1.46")
        report, pkg = updater._choose_engine(FEED, self.ctx(self.hardware("cuda", ["cuda"])), self.state)
        self.assertEqual((report["status"], pkg.backend, pkg.variant), ("available", "cuda", "cuda13"))

    def test_the_cuda_pc_with_an_nvidia_engine_does_not_flip_to_amd(self):
        self.installed("cuda", "cuda13")
        ctx = self.ctx(self.hardware("cuda", ["cuda", "hip"]))
        report, pkg = updater._choose_engine(FEED, ctx, self.state)
        self.assertEqual(pkg.backend, "cuda")

    def test_an_explicit_backend_is_the_persons_decision_and_passes(self):
        self.installed("hip", "rx9070")
        for source in ("requested", "environment"):
            self.assertIsNone(updater.make_guard(self.ctx(self.hardware("cuda", ["cuda"], source=source))), source)

    def test_the_stage_step_refuses_a_package_of_another_make_even_if_a_decision_slipped_it_through(self):
        self.installed("hip", "rx9070")
        ctx = self.ctx(self.hardware("cuda", ["cuda"]))
        cuda = package("cuda", "cuda13", ("sm_86",))
        with mock.patch.object(updater, "_choose_engine", return_value=({"status": "available", "installed": None}, cuda)),                 mock.patch.object(updater, "_download", side_effect=AssertionError("downloaded")):
            with self.assertRaises(updater.UpdateError) as caught:
                updater.update_engine(ctx, FEED, self.state)
        self.assertIn("another make", str(caught.exception))


class Pascal(unittest.TestCase):
    """#3655: a GTX 1080 (sm_61) is a working machine on the CUDA 12 engine; the CUDA 13 build has no code for it and is never chosen."""

    FEED = [package("cuda", "cuda13", ("sm_75", "sm_86", "sm_89", "sm_120")),
            feedmod._engine_package({"version": "0.1.47", "platform": WIN, "backend": "cuda", "variant": "cuda12", "min_driver": {"nvidia": "525.60"},
                                     "archs": ["sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_89"], "min_runtime": None, "build_id": "b-12",
                                     "format": "zip", "url": "e-12.zip", "size": 10, "sha256": "a" * 64, "manifest_sha256": "b" * 64,
                                     "installed_bytes": 20, "notes": ""})]

    def hardware(self, archs):
        return {"platform": WIN, "backend": "cuda", "runtime": None, "nvidia_driver": "581.15", "cuda_archs": archs, "gfx": [], "gpu_names": []}

    def test_two_gtx_1080_get_the_cuda_12_engine(self):
        best, reasons = feedmod.select_engine(self.FEED, self.hardware(["sm_61", "sm_61"]))
        self.assertEqual(best.variant, "cuda12")
        self.assertTrue(any("cuda13" in r or "sm_75" in r for r in reasons), "the CUDA 13 build is refused for the card")

    def test_a_newer_card_still_gets_the_newest_build(self):
        best, _ = feedmod.select_engine(self.FEED, self.hardware(["sm_86"]))
        self.assertEqual(best.variant, "cuda13")


class Host(unittest.TestCase):
    def test_the_app_shows_the_sentence_not_the_no_engine_for_this_card_text(self):
        text = (Path(__file__).resolve().parents[2] / "apps" / "host" / "server" / "updates.mjs").read_text(encoding="utf-8")
        self.assertIn("engine.status === 'wrong-make'", text)


if __name__ == "__main__":
    unittest.main()
