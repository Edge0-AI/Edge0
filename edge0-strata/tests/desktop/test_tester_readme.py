"""The tester READMEs get their checksums from the files themselves (apps/desktop/scripts/make-tester-readme.sh)."""
import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "apps" / "desktop" / "scripts" / "make-tester-readme.sh"
TEMPLATES = ROOT / "apps" / "desktop" / "tester"
NOTICE = "Strata opened in your browser because the app window couldn't start on this computer."


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args, check=True):
    return subprocess.run(["bash", str(SCRIPT), *map(str, args)], capture_output=True, text=True, check=check)


@unittest.skipIf(shutil.which("bash") is None or shutil.which("sha256sum") is None, "needs bash and sha256sum")
class TesterReadmeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="strata-readme-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.files = self.tmp / "files"
        self.files.mkdir()
        self.out = self.tmp / "README.txt"

    def put(self, name, data):
        (self.files / name).write_bytes(data)
        return self.files / name

    def test_every_hash_and_size_in_the_readme_comes_from_the_file(self):
        app = self.put("Strata-tester-linux.AppImage", b"app image bytes " * 1000)
        run(TEMPLATES / "README-linux.txt.in", self.out, "--appimage", app.name, "--files-dir", self.files)
        text = self.out.read_text()
        self.assertIn(sha(app), text)
        self.assertNotIn("@APPIMAGE@", text)
        for needle in ("libfuse2t64", "Ubuntu 26.04", "--browser", "--quit", "--save-logs", "Strata 0.2.3", NOTICE):
            self.assertIn(needle, text)
        run("--verify", self.out, "--files-dir", self.files)

    def test_amd_readmes_hash_both_files(self):
        for variant in ("rx9070", "rx7900"):
            app, zipname = f"Strata-tester-linux-amd-{variant}.AppImage", f"Strata-Terminal-Linux-tester-amd-{variant}.zip"
            a, z = self.put(app, b"a" * 5000), self.put(zipname, b"z" * 3000)
            run(TEMPLATES / f"README-amd-{variant}.txt.in", self.out, "--appimage", app, "--files-dir", self.files)
            text = self.out.read_text()
            self.assertIn(sha(a), text)
            self.assertIn(sha(z), text)
            self.assertIn(f"./{app} --browser", text)
            run("--verify", self.out, "--files-dir", self.files)

    def test_a_missing_file_stops_the_readme_instead_of_leaving_a_stale_hash(self):
        app = self.put("Strata-tester-linux-amd-rx9070.AppImage", b"a" * 5000)  # the zip is not there
        result = run(TEMPLATES / "README-amd-rx9070.txt.in", self.out, "--appimage", app.name, "--files-dir", self.files, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must not carry a hash it did not compute", result.stderr)
        self.assertFalse(self.out.exists())

    def test_verify_catches_a_stale_hash(self):
        app = self.put("Strata-tester-linux.AppImage", b"first build")
        run(TEMPLATES / "README-linux.txt.in", self.out, "--appimage", app.name, "--files-dir", self.files)
        app.write_bytes(b"rebuilt")  # the file changed after the README was written
        result = run("--verify", self.out, "--files-dir", self.files, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("STALE", result.stderr)


if __name__ == "__main__":
    unittest.main()
