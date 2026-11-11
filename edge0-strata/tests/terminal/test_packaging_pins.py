"""The terminal archive's pins: the interpreter digests, the hash-pinned wheel locks, and the launchers that prefer the
bundled runtime. Offline: nothing is downloaded (the build and the smoke test do that)."""

import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))

import package_terminal as pt  # noqa: E402


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


class InterpreterPins(unittest.TestCase):
    def test_every_target_has_a_digest_and_a_matching_url(self):
        self.assertEqual(sorted(pt.PYTHON_PINS), ["darwin-arm64", "linux-aarch64", "linux-x86_64", "windows-x86_64"])
        for target, pin in pt.PYTHON_PINS.items():
            with self.subTest(target):
                self.assertRegex(pin["sha256"], r"^[0-9a-f]{64}$")
                self.assertIn(pt.PYTHON_VERSION, pin["url"])
                self.assertIn(pt.PBS_RELEASE, pin["url"])
                self.assertTrue(pin["url"].startswith("https://github.com/astral-sh/python-build-standalone/"))
                self.assertTrue(pin["url"].endswith("install_only.tar.gz"))
        self.assertEqual(len({p["sha256"] for p in pt.PYTHON_PINS.values()}), 4)

    def test_the_mac_pin_is_the_one_setup_sh_downloads(self):
        text = (ROOT / "setup.sh").read_text(encoding="utf-8")
        self.assertIn(pt.PYTHON_PINS["darwin-arm64"]["sha256"], text)
        self.assertIn(pt.PYTHON_PINS["darwin-arm64"]["url"].split("/")[-1], text)

    def test_fetch_refuses_a_download_that_does_not_match_its_pin(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            pin = pt.PYTHON_PINS["linux-x86_64"]
            (cache / pin["file"].replace("+", "_")).write_bytes(b"not the interpreter")
            with self.assertRaises(SystemExit) as cm:
                pt.fetch_python("linux-x86_64", cache)
            self.assertIn("does not match its pin", str(cm.exception))


class WheelLocks(unittest.TestCase):
    def locks(self):
        found = sorted((ROOT / "packaging" / "locks").glob("runtime-*-cp312.txt"))
        self.assertEqual([p.name for p in found], ["runtime-darwin-arm64-cp312.txt", "runtime-linux-aarch64-cp312.txt",
                                                    "runtime-linux-x86_64-cp312.txt", "runtime-windows-x86_64-cp312.txt"])
        return found

    def test_every_entry_is_pinned_and_hashed_and_nothing_else_is_in_the_file(self):
        for lock in self.locks():
            with self.subTest(lock.name):
                entries = {}
                current = None
                for line in lock.read_text(encoding="utf-8").splitlines():
                    if not line.strip() or line.startswith("#"):
                        continue
                    m = re.match(r"^([A-Za-z0-9_.-]+)==([0-9][^\s\\]*) \\$", line)
                    if m:
                        current = norm(m.group(1))
                        entries[current] = {"version": m.group(2), "hashes": []}
                        continue
                    h = re.match(r"^\s+--hash=sha256:([0-9a-f]{64})( \\)?$", line)
                    self.assertTrue(h and current, f"unexpected line in {lock.name}: {line!r}")
                    entries[current]["hashes"].append(h.group(1))
                self.assertGreater(len(entries), 15)
                for name, e in entries.items():
                    self.assertTrue(e["hashes"], name)

    def test_the_locks_agree_with_requirements_txt(self):
        pins = {}
        for name in ("requirements.txt", "requirements-monitor.txt"):
            for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
                m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;]+)(?:\s*;\s*(.*))?$", line.split(" #")[0].strip())
                if m:
                    pins.setdefault(norm(m.group(1)), []).append((m.group(2), m.group(3) or ""))
        for lock in self.locks():
            names = pt.lock_names(lock)
            with self.subTest(lock.name):
                for pkg, versions in pins.items():
                    if pkg == "colorama":
                        self.assertEqual(pkg in names, "windows" in lock.name)
                        continue
                    if pkg == "numpy":                       # python_version markers: the 3.12 line applies
                        self.assertEqual(names[pkg], "2.5.3")
                        continue
                    self.assertEqual(names.get(pkg), versions[0][0], pkg)
                self.assertNotIn("cmake", names)             # build tools never ship in the bundle
                self.assertNotIn("ninja", names)

    def test_the_build_installs_with_hash_checking_and_no_resolution(self):
        text = (ROOT / "packaging" / "package_terminal.py").read_text(encoding="utf-8")
        for flag in ("--require-hashes", "--no-deps", "--only-binary=:all:"):
            self.assertIn(flag, text)
        self.assertNotIn('"--upgrade"', text)               # pip itself is the interpreter's, not a fresh download


class Launchers(unittest.TestCase):
    def test_wrappers_prefer_the_bundled_runtime(self):
        cmd = (ROOT / "bin" / "strata.cmd").read_text(encoding="utf-8")
        ps1 = (ROOT / "bin" / "strata.ps1").read_text(encoding="utf-8")
        sh = (ROOT / "bin" / "strata").read_text(encoding="utf-8")
        self.assertLess(cmd.index("python\\python.exe"), cmd.index("py -3"))
        self.assertLess(ps1.index("python\\python.exe"), ps1.index("'py', '-3'"))
        self.assertLess(sh.index("python/bin/python3"), sh.index("command -v python3"))
        for text in (cmd, ps1, sh):
            self.assertIn("PYTHONNOUSERSITE", text)

    def test_start_here_setup_and_update_use_it(self):
        bat = (ROOT / "START-HERE.bat").read_text(encoding="utf-8")
        self.assertLess(bat.index('if exist "python\\python.exe" goto bundled'), bat.index("call :findpy"))
        self.assertIn("START-HERE.bat", (ROOT / "SETUP.bat").read_text(encoding="utf-8"))
        self.assertIn("START-HERE.bat", (ROOT / "UPDATE.bat").read_text(encoding="utf-8"))
        sh = (ROOT / "setup.sh").read_text(encoding="utf-8")
        self.assertLess(sh.index("exec python/bin/python3"), sh.index("if [ ! -x .venv/bin/python ]"))
        self.assertIn("setup.sh", (ROOT / "update.sh").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
