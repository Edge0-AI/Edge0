"""fetch-node.ps1 (Windows) and fetch-node.sh pin the same Node archive digest, and the PowerShell
script refuses a download that does not match it."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "apps" / "desktop" / "scripts"
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")


class PinTests(unittest.TestCase):
    def test_both_scripts_pin_the_same_windows_digest_and_version(self):
        sh = (SCRIPTS / "fetch-node.sh").read_text()
        ps = (SCRIPTS / "fetch-node.ps1").read_text()
        sh_digest = re.search(r'win-x64\)\s+echo "([0-9a-f]{64})"', sh).group(1)
        ps_digest = re.search(r"\$Expected = '([0-9a-f]{64})'", ps).group(1)
        self.assertEqual(sh_digest, ps_digest)
        self.assertEqual(re.search(r'STRATA_NODE_VERSION:-([\d.]+)', sh).group(1),
                         re.search(r"else \{ '([\d.]+)' \}", ps).group(1))


@unittest.skipUnless(os.name == "nt" and POWERSHELL, "Windows PowerShell")
class PowerShellTests(unittest.TestCase):
    def run_script(self, *args):
        return subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                               str(SCRIPTS / "fetch-node.ps1"), *args],
                              capture_output=True, text=True, timeout=120)

    def test_a_zip_with_the_wrong_digest_is_refused_and_nothing_is_installed(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "node.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("node-v24.12.0-win-x64/node.exe", b"not node")
            dest = Path(directory) / "out"
            result = self.run_script("-FromFile", str(archive), "-Dest", str(dest))
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("REFUSING", result.stderr)
            self.assertFalse((dest / "node.exe").exists())

    def test_check_reports_a_missing_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_script("-Check", "-Dest", directory)
            self.assertEqual(result.returncode, 1)
            self.assertIn("not installed", result.stderr)


if __name__ == "__main__":
    unittest.main()
