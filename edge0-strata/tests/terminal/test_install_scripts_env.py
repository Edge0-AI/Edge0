"""install.ps1 / install.sh: an ARM computer is refused with the reason and the way out; the Windows script survives the usual environment trouble."""

import os
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PS1 = ROOT / "packaging" / "install.ps1"
SH = ROOT / "packaging" / "install.sh"


class Text(unittest.TestCase):
    def test_the_readme_one_liner_cannot_succeed_on_a_failed_download(self):
        for name in ("README.md", "README.de.md", "docs/INSTALL.md", "docs/TERMINAL_LAUNCH.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertNotIn("install.sh | sh", text, name)                    # `curl ... | sh` runs nothing and exits 0 when curl fails
            self.assertIn("install.sh -o install.sh && sh install.sh", text, name)

    def setUp(self):
        self.ps1 = PS1.read_text(encoding="utf-8")
        self.sh = SH.read_text(encoding="utf-8")

    def test_arm_is_refused_with_a_reason_and_a_next_step(self):
        self.assertIn("Windows on ARM PC", self.ps1)
        self.assertIn("no engine for ARM", self.ps1)
        self.assertNotIn("has no Strata Terminal yet", self.ps1)
        self.assertIn("NVIDIA DGX Spark", self.sh)      # 0.2.7.1: Linux ARM64 installs the Terminal (the engine is for the Spark only)
        self.assertIn("Apple silicon", self.sh)

    def test_tls12_proxy_sign_in_and_the_real_error_text(self):
        self.assertIn("Tls12", self.ps1)
        self.assertIn("DefaultNetworkCredentials", self.ps1)
        self.assertIn("$($_.Exception.Message)", self.ps1)

    def test_long_paths_junction_and_running_version(self):
        self.assertIn("tar.exe", self.ps1)
        self.assertIn("STRATA_DATA_DIR to a short folder", self.ps1)
        self.assertIn("$LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $current", self.ps1)
        self.assertIn("a program is using it", self.ps1)
        self.assertIn("Windows did not say where your user folder is", self.ps1)
        self.assertNotIn("is the disk full?", self.ps1)


@unittest.skipUnless(shutil.which("powershell"), "no PowerShell")
class Run(unittest.TestCase):
    def test_native_arm_stops_before_any_download(self):
        env = dict(os.environ, PROCESSOR_ARCHITECTURE="ARM64", STRATA_FEED="http://127.0.0.1:9")
        env.pop("PROCESSOR_ARCHITEW6432", None)
        done = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PS1)],
                              capture_output=True, text=True, env=env, timeout=120)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("Windows on ARM PC", done.stderr + done.stdout)
        self.assertNotIn("Downloading", done.stdout)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(shutil.which("sh") and os.uname().sysname in ("Linux", "Darwin") if hasattr(os, "uname") else False, "needs sh on Linux or macOS")
class InstallShFailsVisibly(unittest.TestCase):
    """A broken download of install.sh must not look like a finished install."""

    def run_sh(self, text, **env):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "install.sh"
            f.write_text(text, encoding="utf-8", newline="\n")
            home = Path(d) / "home"
            home.mkdir()
            r = subprocess.run(["sh", str(f)], capture_output=True, text=True, timeout=120,
                               env={**os.environ, "HOME": str(home), "STRATA_DATA_DIR": str(home / "data"), "STRATA_BIN_DIR": str(home / "bin"), **env})
            return r, home

    def test_the_script_is_one_function_called_on_the_last_line(self):
        text = SH.read_text(encoding="utf-8")
        self.assertIn("\nmain() {", text)
        self.assertEqual(text.rstrip().splitlines()[-1], 'main "$@"')

    def test_syntax(self):
        self.assertEqual(subprocess.run(["sh", "-n", str(SH)], capture_output=True).returncode, 0)

    def test_a_half_downloaded_script_runs_nothing_and_fails(self):
        lines = SH.read_text(encoding="utf-8").splitlines(True)
        r, home = self.run_sh("".join(lines[:int(len(lines) * 0.8)]))
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((home / "data").exists() or (home / "bin").exists())

    def test_an_unreachable_release_folder_stops_with_a_reason_and_a_failing_exit(self):
        if not shutil.which("curl") and not shutil.which("wget"):
            self.skipTest("no curl or wget")
        r, home = self.run_sh(SH.read_text(encoding="utf-8"), STRATA_FEED="http://127.0.0.1:9")
        self.assertEqual(r.returncode, 1)
        self.assertIn("could not read", r.stderr)
        self.assertFalse((home / "bin" / "strata").exists())
