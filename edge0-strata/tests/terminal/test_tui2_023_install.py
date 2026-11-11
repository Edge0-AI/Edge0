"""0.2.3 RC stress test: #11 (A01b/A05) an install from an older feed shows the real reason with both sequence numbers, the folder that holds the
old state and one next step, and the installer prints the whole cause; #12 (A04) the Windows installer unpacks into a folder whose name has
accented letters."""

import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from strata_control import feed as feedmod, updater
from tests.terminal import test_updater

ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = ROOT / "packaging" / "install.ps1"
POWERSHELL = shutil.which("powershell") if os.name == "nt" else None


class ReplayedFeedWords(test_updater.UpdaterCase):
    def test_the_refusal_gives_both_numbers_the_folder_and_one_next_step(self):
        self.publish(("0.1.46", test_updater.fx.GOOD_ENGINE), sequence=108)
        updater.check(self.ctx())                                      # accepts sequence 108 and remembers it
        self.publish(("0.1.46", test_updater.fx.GOOD_ENGINE), sequence=107)
        with self.assertRaises(feedmod.FeedDowngrade) as caught:
            updater.check(self.ctx())
        text = str(caught.exception)
        self.assertIn("older than the one already accepted", text)
        self.assertIn("its sequence is 107", text)
        self.assertIn("the accepted one is 108", text)
        self.assertIn(str(self.components / ".updates"), text)
        self.assertIn("state.json", text)
        self.assertEqual(text.count("Use the newest release folder"), 1)


def stub_release(folder: Path, body: str) -> str:
    folder.mkdir(parents=True)
    archive = folder / "terminal.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("strata-terminal-0.1.48-windows-x86_64/bin/strata.cmd", body)
        z.writestr("strata-terminal-0.1.48-windows-x86_64/strata.py", "# stub\n")
    (folder / "install-manifest.txt").write_text(
        f"windows-x86_64 0.1.48 terminal.zip {hashlib.sha256(archive.read_bytes()).hexdigest()} {archive.stat().st_size}\n")
    return "file:///" + folder.as_posix()


@unittest.skipUnless(POWERSHELL, "the Windows installer runs only on Windows")
class WindowsInstaller(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-inst-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_installer(self, feed: str, data: Path):
        runner = self.root / "run.ps1"
        runner.write_text("$ErrorActionPreference = 'Continue'\n"
                          f"$env:STRATA_FEED = '{feed}'; $env:STRATA_DATA_DIR = '{data}'; $env:STRATA_NO_PATH = '1'\n"
                          f"try {{ Invoke-Expression (Get-Content -Raw -LiteralPath '{INSTALL_PS1}') }} catch {{ Write-Host \"STOPPED\" }}\n",
                          encoding="utf-8-sig")
        done = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(runner)], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=300)
        return done.stdout + done.stderr

    def test_a_folder_with_accented_letters_is_unpacked(self):
        feed = stub_release(self.root / "feed", "@echo off\r\nexit /b 0\r\n")
        data = self.root / "Ünï test"
        out = self.run_installer(feed, data)
        self.assertIn("Installed. Run: strata launch", out, out)
        self.assertTrue((data / "terminal" / "0.1.48" / "bin" / "strata.cmd").is_file())
        self.assertEqual([p.name for p in (data / "terminal").iterdir() if p.name.startswith("u") and p.suffix in ("", ".zip")], [])

    def test_the_whole_cause_is_printed_not_the_first_six_lines(self):
        lines = ["echo strata update: this update source is older than the one already accepted (its sequence is 107, the accepted one is 108)."]
        lines += [f"echo detail {i}" for i in range(2, 9)] + ["echo the last line of the cause"]
        feed = stub_release(self.root / "feed2", "@echo off\r\n" + "\r\n".join(lines) + "\r\nexit /b 5\r\n")
        out = self.run_installer(feed, self.root / "data")
        self.assertIn("the last line of the cause", out)
        self.assertIn("its sequence is 107", out)
        self.assertIn("The reason is printed above", out)
        self.assertIn("STOPPED", out)
        self.assertFalse((self.root / "data" / "terminal" / "0.1.48").exists(), "nothing is installed")


if __name__ == "__main__":
    unittest.main()
