"""sycl/tools/build.sh exits with the build's own status (#2990): 0 after a good build, nonzero after a failed one.

It once ended with a grep pipeline, so a good build exited 1 (nothing to find) and a failed compile exited 0, and a `build.sh && next-step` chain ran
the wrong way round. The test runs the real script with a stub cmake on PATH; no compiler, no oneAPI, no docker.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sycl" / "tools" / "build.sh"
BASH = shutil.which("bash")      # the full path: on Windows a bare "bash" can start the WSL launcher in System32 instead of Git bash
CYGPATH = shutil.which("cygpath")


def sh_path(p) -> str:
    if CYGPATH is None:
        return str(p)
    return subprocess.run([CYGPATH, "-u", str(p)], capture_output=True, text=True, check=True).stdout.strip()


STUB = """#!/bin/sh
# a stand-in for cmake: `cmake --build DIR ...` prints what ninja prints and exits with STUB_RC; any other call does nothing
case "$1" in
  --build)
    echo "[1/2] Building CXX object CMakeFiles/strata.dir/main.cpp.o"
    if [ "${STUB_RC:-0}" != 0 ]; then
      echo "FAILED: [code=1] CMakeFiles/strata.dir/main.cpp.o"
      echo "main.cpp:1:21: error: 'x' was not declared in this scope"
      echo "ninja: build stopped: cannot make progress due to previous errors."
    else
      echo "[2/2] Linking CXX executable strata"
    fi
    exit "${STUB_RC:-0}" ;;
esac
exit 0
"""


@unittest.skipIf(BASH is None, "needs bash")
class BuildScriptStatus(unittest.TestCase):
    def run_script(self, rc: int):
        with tempfile.TemporaryDirectory() as tmp:
            bindir, build = Path(tmp) / "bin", Path(tmp) / "build"
            bindir.mkdir()
            build.mkdir()
            (build / "build.ninja").write_text("", encoding="utf-8")          # an existing build folder: the script does not configure it
            (bindir / "cmake").write_text(STUB, encoding="utf-8", newline="\n")
            (bindir / "cmake").chmod(0o755)
            env = dict(os.environ, PATH=sh_path(bindir) + os.pathsep + os.environ["PATH"], STUB_RC=str(rc), REPO=sh_path(ROOT),
                       BUILD_DIR=sh_path(build), AOT="", I8_ESIMD="OFF")
            done = subprocess.run([BASH, sh_path(SCRIPT), "strata"], capture_output=True, text=True, env=env, timeout=120)
            return done

    def test_a_good_build_exits_zero(self):
        done = self.run_script(0)
        self.assertIn("BUILD EXIT 0", done.stdout)
        self.assertIn("errors: 0", done.stdout)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_a_failed_compile_exits_nonzero_with_the_builds_status(self):
        done = self.run_script(1)
        self.assertIn("BUILD EXIT 1", done.stdout)
        self.assertIn("errors: 1", done.stdout)
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)


if __name__ == "__main__":
    unittest.main()
