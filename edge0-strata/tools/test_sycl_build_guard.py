"""#3219: sycl/tools/build.sh stops before the long compile when this image's ocloc is the one that crashes on the Battlemage int8 kernels,
and I8_ESIMD=OFF reaches cmake. A fake ocloc and a fake cmake stand in for the oneAPI image; skipped without bash."""
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_SH = ROOT / "sycl" / "tools" / "build.sh"
BASH = shutil.which("bash")


@unittest.skipUnless(BASH, "needs bash")
class BuildGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bin = Path(self.tmp.name) / "bin"
        self.bin.mkdir()
        self.log = Path(self.tmp.name) / "cmake.log"
        (self.bin / "cmake").write_text('#!/bin/sh\necho "cmake $*" >> "%s"\nexit 0\n' % self.log.as_posix(), encoding="utf-8", newline="\n")
        self.build = Path(self.tmp.name) / "b"

    def tearDown(self):
        self.tmp.cleanup()

    def ocloc(self, version):
        path = self.bin / "ocloc"
        path.write_text('#!/bin/sh\necho "Intel(R) Compute Runtime ocloc %s"\n' % version, encoding="utf-8", newline="\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        (self.bin / "cmake").chmod(0o755)

    def run_build(self, **env):
        (self.bin / "cmake").chmod(0o755)
        merged = dict(os.environ, PATH=self.bin.as_posix() + os.pathsep + os.environ["PATH"], REPO=ROOT.as_posix(),
                      BUILD_DIR=self.build.as_posix())
        for key in ("AOT", "I8_ESIMD", "ALLOW_BAD_OCLOC"):
            merged.pop(key, None)
        merged.update(env)
        return subprocess.run([BASH, BUILD_SH.as_posix(), "strata"], env=merged, capture_output=True, text=True, timeout=120)

    def test_known_bad_ocloc_stops_a_battlemage_aot_build(self):
        self.ocloc("26.22.38646.4")
        r = self.run_build(AOT="bmg-g31")
        self.assertEqual(r.returncode, 2, r.stderr)
        for word in ("26.22.38646.4", "docker build --pull", "I8_ESIMD=OFF", "ALLOW_BAD_OCLOC=1"):
            self.assertIn(word, r.stderr)
        self.assertFalse(self.log.exists(), "cmake never started")

    def test_other_versions_other_devices_and_the_switches_go_on(self):
        self.ocloc("26.31.39395.14")
        r = self.run_build(AOT="bmg-g31")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("-DSTRATA_SYCL_I8_ESIMD=AUTO", self.log.read_text())
        self.log.unlink()
        self.ocloc("26.22.38646.4")
        self.assertEqual(self.run_build(AOT="acm-g10").returncode, 0)              # not a Battlemage build
        self.assertEqual(self.run_build().returncode, 0)                           # the JIT build
        self.assertEqual(self.run_build(AOT="bmg-g31", ALLOW_BAD_OCLOC="1").returncode, 0)
        self.log.unlink()
        r = self.run_build(AOT="bmg-g31", I8_ESIMD="OFF")                          # the workaround: stubs, no check needed
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("-DSTRATA_SYCL_I8_ESIMD=OFF", self.log.read_text())

    def test_the_cmake_switch_is_declared(self):
        text = (ROOT / "sycl" / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn('set(STRATA_SYCL_I8_ESIMD "AUTO" CACHE STRING', text)
        self.assertIn("add_compile_definitions(STRATA_I8_NO_ESIMD=1)", text)


if __name__ == "__main__":
    unittest.main()
