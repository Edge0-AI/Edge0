"""Build-script behavior checks; run in Docker with a cached pinned llama.cpp.

Set EDGE0_TEST_LLAMA_SOURCE to the cache, or assemble vendor/llama.cpp first.
The fake CMake records backend selection and injects failures; git/patch replay
uses the real pin. No models, GPU, network access or Python packages are needed.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("EDGE0_TEST_LLAMA_SOURCE", ROOT / "vendor/llama.cpp"))


class VendorBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="edge0-build-")
        self.addCleanup(self.temp.cleanup)
        self.depot = Path(self.temp.name) / "depot with spaces"
        self.copy_inputs()
        self.env = dict(os.environ, EDGE0_LLAMA_URL=str(SOURCE),
                        EDGE0_DEPOT=str(self.depot))
        self.vendor = self.depot / "vendor/llama.cpp"
        self.worktree = self.depot / "wt/linux"
        self.pin = (self.depot / "vendor.llama.pin").read_text().strip()
        self.install_fake_cmake()

    def copy_inputs(self):
        self.script = self.depot / "linux/scripts/vendor-build.sh"
        self.script.parent.mkdir(parents=True)
        shutil.copy2(ROOT / "linux/scripts/vendor-build.sh", self.script)
        shutil.copy2(ROOT / "vendor.llama.pin", self.depot)
        for band in ("common", "linux"):
            shutil.copytree(ROOT / "patches/llama.cpp" / band,
                            self.depot / "patches/llama.cpp" / band)

    def install_fake_cmake(self):
        bindir = self.depot / "fake-bin"
        bindir.mkdir()
        cmake = bindir / "cmake"
        cmake.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CMAKE_LOG"\n'
                         'if [[ "$1" == --build ]]; then exit "${BUILD_RC:-0}"; fi\n'
                         'exit "${CONFIGURE_RC:-0}"\n')
        cmake.chmod(0o755)
        self.cmake_log = self.depot / "cmake.log"
        self.env.update(PATH=f"{bindir}:{self.env['PATH']}",
                        CMAKE_LOG=str(self.cmake_log))

    def run_build(self, *args, ok=True):
        result = subprocess.run(["bash", str(self.script), *args], env=self.env,
                                text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("ERROR", result.stdout)
        return result

    def git(self, repo, *args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    def assert_pristine(self):
        self.assertEqual(self.git(self.vendor, "rev-parse", "HEAD"), self.pin)
        self.assertEqual(self.git(self.vendor, "status", "--porcelain"), "")

    def test_repeat_assembly_preserves_vendor_and_removes_stale_serve_files(self):
        self.run_build("--assemble-only")
        first = self.git(self.worktree, "rev-parse", "HEAD^{tree}")
        stale = self.worktree / "src/edge0/stale.cc"
        stale.parent.mkdir(exist_ok=True)
        stale.write_text("stale")
        self.run_build("--assemble-only")
        self.assertEqual(first, self.git(self.worktree, "rev-parse", "HEAD^{tree}"))
        patches = list((self.depot / "patches/llama.cpp").glob("*/*.patch"))
        count = self.git(self.worktree, "rev-list", "--count", f"{self.pin}..HEAD")
        self.assertEqual(count, str(len(patches)))
        self.assertFalse(stale.exists())
        self.assertFalse(self.cmake_log.exists())
        self.assert_pristine()

    def test_stage_linux_sources_only(self):
        for platform in ("linux", "windows"):
            serve = self.depot / platform / "serve"
            serve.mkdir(parents=True)
            (serve / f"{platform}.cc").write_text("// fixture\n")
        self.run_build("--assemble-only")
        self.assertTrue((self.worktree / "src/edge0/linux.cc").exists())
        self.assertFalse((self.worktree / "src/edge0/windows.cc").exists())

    def test_cpu_release_avx2_and_build_targets(self):
        self.run_build("--jobs", "2", "--build-dir", "build-custom")
        configure, build = self.cmake_log.read_text().splitlines()
        for flag in ("CMAKE_BUILD_TYPE=Release", "GGML_NATIVE=OFF", "GGML_AVX2=ON",
                     "GGML_AVX512=OFF", "GGML_CUDA=OFF", "GGML_VULKAN=OFF",
                     "LLAMA_BUILD_UI=OFF", "LLAMA_USE_PREBUILT_UI=OFF"):
            self.assertIn(f"-D{flag}", configure)
        self.assertIn("build-custom", configure)
        self.assertIn("--parallel 2", build)
        self.assertIn("--target llama-server llama-bench llama-batched-bench", build)
        self.assert_pristine()

    def test_gpu_options_are_explicit_and_use_separate_build_directories(self):
        for backend in ("cuda", "vulkan"):
            with self.subTest(backend=backend):
                self.run_build(f"--{backend}", "--configure-only")
                configure = self.cmake_log.read_text().splitlines()[-1]
                self.assertIn(f"build-{backend}", configure)
                for option in ("cuda", "vulkan"):
                    state = "ON" if option == backend else "OFF"
                    self.assertIn(f"-DGGML_{option.upper()}={state}", configure)
                self.assertNotIn("--build", configure)

    def test_invalid_options_fail_before_materialization(self):
        for args in (("--bogus",), ("--jobs",), ("--jobs", "0"), ("--jobs", "-1"),
                     ("--cuda", "--vulkan"), ("--build-dir", "../outside"),
                     ("--build-dir", "/outside")):
            with self.subTest(args=args):
                self.run_build(*args, ok=False)
                self.assertFalse(self.vendor.exists())

    def test_help_does_not_require_a_depot(self):
        self.env["EDGE0_DEPOT"] = str(self.depot / "absent")
        self.assertIn("Usage:", self.run_build("--help").stdout)
        self.assertFalse(self.vendor.exists())

    def test_invalid_pin_and_missing_band_fail_before_clone(self):
        pinfile = self.depot / "vendor.llama.pin"
        pinfile.write_text("invalid\n")
        self.run_build("--assemble-only", ok=False)
        pinfile.write_text(self.pin + "\n")
        shutil.rmtree(self.depot / "patches/llama.cpp/common")
        self.run_build("--assemble-only", ok=False)
        self.assertFalse(self.vendor.exists())

    def test_dirty_and_moved_vendor_are_refused_without_reset(self):
        self.run_build("--assemble-only")
        sentinel = self.vendor / "local-file"
        sentinel.write_text("keep")
        self.run_build("--assemble-only", ok=False)
        self.assertEqual(sentinel.read_text(), "keep")
        sentinel.unlink()
        parent = self.git(self.vendor, "rev-parse", "HEAD^")
        self.git(self.vendor, "update-ref", "HEAD", parent)
        self.run_build("--assemble-only", ok=False)
        self.assertEqual(self.git(self.vendor, "rev-parse", "HEAD"), parent)

    def test_unrelated_worktree_directory_is_refused(self):
        self.worktree.mkdir(parents=True)
        sentinel = self.worktree / "keep"
        sentinel.write_text("keep")
        self.run_build("--assemble-only", ok=False)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_tracked_vendor_edits_are_preserved(self):
        self.run_build("--assemble-only")
        readme = self.vendor / "README.md"
        changed = readme.read_text() + "\nlocal edit\n"
        readme.write_text(changed)
        self.run_build("--assemble-only", ok=False)
        self.assertEqual(readme.read_text(), changed)

    def test_patch_failure_is_nonzero_and_can_be_retried(self):
        patch = self.depot / "patches/llama.cpp/linux/9999-bad.patch"
        patch.write_text("invalid patch\n")
        self.run_build("--assemble-only", ok=False)
        self.assert_pristine()
        patch.unlink()
        self.run_build("--assemble-only")
        self.assertEqual(self.git(self.worktree, "status", "--porcelain"), "")

    def test_configure_and_build_failures_propagate(self):
        self.env["CONFIGURE_RC"] = "23"
        self.assertEqual(self.run_build(ok=False).returncode, 23)
        self.assertEqual(len(self.cmake_log.read_text().splitlines()), 1)
        self.env.update(CONFIGURE_RC="0", BUILD_RC="24")
        self.assertEqual(self.run_build(ok=False).returncode, 24)
        self.assert_pristine()

    def test_clean_removes_only_selected_build_and_rejects_symlinks(self):
        self.run_build("--assemble-only")
        selected, other = self.worktree / "build-cpu", self.worktree / "build-vulkan"
        selected.mkdir()
        other.mkdir()
        (selected / "stale").touch()
        (other / "keep").touch()
        self.run_build("--clean")
        self.assertFalse((selected / "stale").exists())
        self.assertTrue((other / "keep").exists())
        selected.symlink_to(other, target_is_directory=True)
        self.run_build("--clean", ok=False)
        self.assertTrue((other / "keep").exists())


if __name__ == "__main__":
    unittest.main()
