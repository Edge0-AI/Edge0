"""I1 / I2 (tools and images review, 2026-10-22): the picture reader in an engine package must run on every card and Mac the package does.

#3669: the Windows cuda12 package reused a strata-vision.exe built before Pascal support; the engine ran on a P100 and the reader stopped
with "device kernel image is invalid". #3697: a macOS package with no reader at all. These tests need no GPU and no toolkit."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))
import package_engine  # noqa: E402

CUDA12 = ("sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "compute_89")
CUDA13 = ("sm_75", "sm_86", "sm_89", "sm_120", "compute_120")


class ReaderCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = Path(self.tmp.name)
        (self.src / "strata").write_text("x")

    def reader(self, vision_archs=None):
        (self.src / "strata-vision").write_text("x")
        data = {"version": "0.2.3"}
        if vision_archs is not None:
            data["vision_archs"] = vision_archs
        (self.src / "BUILD.json").write_text(json.dumps(data))

    def check(self, backend, target, archs, **kw):
        return package_engine.check_picture_reader(self.src, backend, target, archs, **kw)

    def test_a_reader_with_no_recorded_archs_is_refused_for_cuda12(self):
        self.reader()
        with self.assertRaises(SystemExit) as cm:
            self.check("cuda", "windows-x86_64", CUDA12, variant="cuda12")
        self.assertIn("vision_archs", str(cm.exception))
        self.assertIn("3669", str(cm.exception))

    def test_a_cuda13_reader_does_not_pass_for_the_pascal_package(self):
        self.reader("75;86;89;120")
        with self.assertRaises(SystemExit) as cm:
            self.check("cuda", "windows-x86_64", CUDA12, variant="cuda12")
        for n in ("sm_60", "sm_61", "sm_70"):
            self.assertIn(n, str(cm.exception))

    def test_a_reader_built_for_the_packages_archs_passes(self):
        self.reader("60;61;70;75;80;86;89")
        self.assertEqual(self.check("cuda", "linux-x86_64", CUDA12, variant="cuda12"), [])
        self.reader(["75", "sm_86", "89-real", 120])
        self.assertEqual(self.check("cuda", "windows-x86_64", CUDA13, variant="cuda13"), [])

    def test_allow_stale_vision_packages_it_and_says_what_is_missing(self):
        self.reader("75;86;89;120")
        notes = self.check("cuda", "windows-x86_64", CUDA12, variant="cuda12", allow_stale=True)
        self.assertIn("sm_60", notes[0])

    def test_a_cuda_package_without_a_reader_is_text_only_not_an_error(self):
        notes = self.check("cuda", "linux-x86_64", CUDA13, variant="cuda13")
        self.assertIn("text only", notes[0])

    def test_a_macos_package_without_a_reader_is_refused(self):
        with self.assertRaises(SystemExit) as cm:
            self.check("metal", "macos-arm64", ())
        self.assertIn("build_vision_mac.sh", str(cm.exception))
        self.assertIn("pictures are off", self.check("metal", "macos-arm64", (), allow_none=True)[0])

    def test_a_macos_package_with_a_reader_passes(self):
        self.reader()
        self.assertEqual(self.check("metal", "macos-arm64", ()), [])

    def test_build_runs_the_check_only_when_asked(self):
        self.reader("75;86;89;120")
        out = Path(self.tmp.name) / "out"
        out.mkdir()
        with self.assertRaises(SystemExit) as cm:
            package_engine.build(self.src, "cuda", "0.2.3", "linux-x86_64", out, archs=CUDA12, variant="cuda12", build_id="t",
                                 check_reader=True)
        self.assertIn("picture reader", str(cm.exception))

    def test_stamp_vision_writes_the_record(self):
        (self.src / "BUILD.json").write_text(json.dumps({"version": "0.2.3", "archs": "60;61"}))
        self.assertEqual(package_engine.main(["--stamp-vision", str(self.src), "--vision-archs", "60;61;70"]), 0)
        data = json.loads((self.src / "BUILD.json").read_text())
        self.assertEqual(data["vision_archs"], "60;61;70")
        self.assertEqual(data["archs"], "60;61")


if __name__ == "__main__":
    unittest.main()
