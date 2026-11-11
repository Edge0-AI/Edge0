"""The one-line installers and the local test feed: syntax, the placeholder guard, the feed builder, and (POSIX) a real install.sh run against a feed."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))
sys.path.insert(0, str(ROOT))
import make_local_feed as mlf  # noqa: E402

# the fixture Terminal is built for the machine the test runs on: install.sh picks its archive by uname
PLAT = "linux-aarch64" if sys.platform.startswith("linux") and platform.machine().lower() in ("aarch64", "arm64") else "linux-x86_64"


class Scripts(unittest.TestCase):
    def test_install_sh_syntax_and_guard(self):
        sh = shutil.which("sh") or shutil.which("bash")
        if not sh:
            self.skipTest("no sh")
        r = subprocess.run([sh, "-n", str(ROOT / "packaging" / "install.sh")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        text = (ROOT / "packaging" / "install.sh").read_text(encoding="utf-8")
        self.assertNotIn("__FEED__", text)                                # no placeholder: the GitHub Releases address is the built-in default
        self.assertIn(":-" + mlf.RELEASE_FEED + "}", text)                # the default a local feed bakes its own address over
        self.assertIn("Installed. Run: ", text)
        self.assertNotIn("\r", text)

    def test_install_ps1_parses_and_guard(self):
        text = (ROOT / "packaging" / "install.ps1").read_text(encoding="utf-8")
        self.assertNotIn("__FEED__", text)
        self.assertIn("else { '" + mlf.RELEASE_FEED + "' }", text)
        self.assertIn("Installed. Run: strata launch", text)
        ps = shutil.which("powershell") or shutil.which("pwsh")
        if not ps:
            self.skipTest("no PowerShell")
        cmd = ("$e=$null; $t=$null; [System.Management.Automation.Language.Parser]::ParseFile('%s',[ref]$t,[ref]$e) | Out-Null; "
               "if ($e.Count) { $e | ForEach-Object { $_.Message }; exit 1 } else { 'ok' }" % str(ROOT / "packaging" / "install.ps1").replace("'", "''"))
        r = subprocess.run([ps, "-NoProfile", "-Command", cmd], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_the_built_in_address_is_the_default_feed(self):
        """install.sh / install.ps1 and `strata update` use the same GitHub Releases folder; STRATA_FEED stays an override."""
        from strata_cli import update_cmd
        self.assertEqual(update_cmd.DEFAULT_FEED, mlf.RELEASE_FEED + "/strata-feed.json")
        self.assertIn(":-" + mlf.RELEASE_FEED + "}", (ROOT / "packaging" / "install.sh").read_text(encoding="utf-8"))
        self.assertIn("$env:STRATA_FEED", (ROOT / "packaging" / "install.ps1").read_text(encoding="utf-8"))


class Feed(unittest.TestCase):
    def make_bundle(self, work: Path) -> Path:
        """A tiny terminal archive that is a working `strata` (the repo's own source with the system Python)."""
        top = work / "bundle" / ("strata-terminal-9.9.9-" + PLAT)
        top.mkdir(parents=True)
        for name in ("strata.py", "strata_cli", "strata_control", "serve"):
            src = ROOT / name
            (shutil.copytree(src, top / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")) if src.is_dir() else shutil.copy2(src, top / name))
        (top / "bin").mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "bin" / "strata", top / "bin" / "strata")
        (top / "bin" / "strata").chmod(0o755)
        (top / "VERSION").write_text("9.9.9\n", encoding="utf-8")
        archive = work / ("strata-terminal-9.9.9-" + PLAT + ".tar.gz")
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(top, arcname=top.name)
        return archive

    def test_make_local_feed_and_a_real_install(self):
        import make_local_feed
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            archive = self.make_bundle(work)
            feed = work / "feed"
            rc = make_local_feed.main(["--out", str(feed), "--terminal", str(archive), "--keys", str(work / "keys"), "--url", "http://127.0.0.1:1"])
            self.assertEqual(rc, 0)
            for name in ("strata-feed.json", "strata-feed.json.sig", "install-manifest.txt", "install.sh", "install.ps1", "test-trust.json"):
                self.assertTrue((feed / name).is_file(), name)
            line = (feed / "install-manifest.txt").read_text().split()
            self.assertEqual(line[0], PLAT)
            self.assertEqual(line[1], "9.9.9")
            control = feed / line[2]
            self.assertEqual(hashlib.sha256(control.read_bytes()).hexdigest(), line[3])
            self.assertEqual(control.stat().st_size, int(line[4]))
            sh_text = (feed / "install.sh").read_text()
            self.assertIn("http://127.0.0.1:1", sh_text)
            self.assertNotIn(mlf.RELEASE_FEED, sh_text)                  # the GitHub address was replaced everywhere by the local one
            self.assertIn("Installed. Run:", (feed / "install.ps1").read_text())
            doc = json.loads((feed / "strata-feed.json").read_text())
            self.assertEqual(doc["control"]["packages"][0]["sha256"], line[3])
            self.assertNotIn("seed", (feed / "test-trust.json").read_text())      # only the public key is published
            if os.name == "nt" or not (shutil.which("sh") and shutil.which("curl") and shutil.which("tar")):
                self.skipTest("the real install.sh run needs POSIX sh, curl and tar")
            handler = partial(SimpleHTTPRequestHandler, directory=str(feed))
            handler.log_message = lambda *a, **k: None                                   # type: ignore[attr-defined]
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                url = f"http://127.0.0.1:{server.server_address[1]}"
                home = work / "home"
                home.mkdir()
                env = {**os.environ, "HOME": str(home), "STRATA_FEED": url, "STRATA_INSTALL_TEST_TRUST": "1", "STRATA_PYTHON": sys.executable,
                       "PATH": os.environ["PATH"], "STRATA_NO_ACQUIRE": ""}
                for k in ("XDG_DATA_HOME", "STRATA_UPDATE_TRUSTED_KEYS"):
                    env.pop(k, None)
                r = subprocess.run(["sh", str(feed / "install.sh")], capture_output=True, text=True, env=env, timeout=240)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertIn("Checksum ok", r.stdout)
                self.assertIn("Signature ok", r.stdout)
                self.assertIn("Installed. Run:", r.stdout)
                self.assertTrue((home / ".local" / "bin" / "strata").exists())
                v = subprocess.run([str(home / ".local" / "bin" / "strata"), "--version"], capture_output=True, text=True, env=env, timeout=60)
                self.assertEqual(v.returncode, 0, v.stderr)
                # an archive with one top folder (a terminal archive, not the feed's control package) installs too
                data = archive.read_bytes()
                (feed / "top.tar.gz").write_bytes(data)
                (feed / "install-manifest.txt").write_text(f"{PLAT} 9.9.9 top.tar.gz {hashlib.sha256(data).hexdigest()} {len(data)}" + chr(10))
                home3 = work / "home3"
                home3.mkdir()
                r = subprocess.run(["sh", str(feed / "install.sh")], capture_output=True, text=True, env={**env, "HOME": str(home3)}, timeout=240)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertTrue((home3 / ".local" / "share" / "strata" / "terminal" / "9.9.9" / "bin" / "strata").exists())
                (feed / "install-manifest.txt").write_text(" ".join(line) + chr(10))
                # a damaged download is refused and nothing is installed
                data = (feed / line[2]).read_bytes()
                (feed / line[2]).write_bytes(data[:-10] + b"x" * 10)
                home2 = work / "home2"
                home2.mkdir()
                r = subprocess.run(["sh", str(feed / "install.sh")], capture_output=True, text=True, env={**env, "HOME": str(home2)}, timeout=240)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("does not match the release's checksum", r.stderr)
                self.assertFalse((home2 / ".local" / "share" / "strata" / "terminal" / "9.9.9").exists())
            finally:
                server.shutdown()

    def test_the_release_path_writes_install_manifest_and_the_unchanged_install_sh_installs_from_it(self):
        """pack-control (flat names, as for GitHub assets) + build-feed, the files a release uploads; not make_local_feed."""
        import feed_tool
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            archive = self.make_bundle(work)
            unpacked = work / "unpacked"
            unpacked.mkdir()
            folder = mlf.extract_terminal(archive, unpacked)
            release = work / "release"
            feed_tool.keygen(work / "keys")
            key = feed_tool._load_key(work / "keys" / "feed-key.json")
            entry = feed_tool.pack_control(folder, release / "control", version="9.9.9", platform=PLAT, url_prefix="")
            shutil.move(str(release / "control" / entry["url"]), str(release / entry["url"]))        # GitHub assets are flat
            feed_tool.build_feed(release, key=key, sequence=1, control_dir=release / "control")
            line = (release / "install-manifest.txt").read_text().split()
            self.assertEqual(line, [PLAT, "9.9.9", entry["url"], entry["sha256"], str(entry["size"])])
            self.assertEqual(json.loads((release / "strata-feed.json").read_text())["control"]["packages"][0]["sha256"], entry["sha256"])
            bad = work / "bad"                           # a Windows pack that is not a zip cannot be installed by install.ps1: left out of the manifest
            feed_tool.build_feed(bad, key=key, sequence=1, control_dir=self._windows_tar_pack(folder, work))
            self.assertFalse((bad / "install-manifest.txt").exists())
            if PLAT == "linux-aarch64":                  # the Linux ARM64 Terminal reads its own feed file (same bytes here)
                for suffix in ("", ".sig"):
                    shutil.copy2(release / ("strata-feed.json" + suffix), release / ("strata-feed-linux-aarch64.json" + suffix))
            shutil.copy2(ROOT / "packaging" / "install.sh", release / "install.sh")                   # uploaded unchanged
            shutil.copy2(work / "keys" / "feed-key.pub.json", release / "test-trust.json")
            if os.name == "nt" or not (shutil.which("sh") and shutil.which("curl") and shutil.which("tar")):
                self.skipTest("the real install.sh run needs POSIX sh, curl and tar")
            handler = partial(SimpleHTTPRequestHandler, directory=str(release))
            handler.log_message = lambda *a, **k: None                                   # type: ignore[attr-defined]
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                home = work / "home"
                home.mkdir()
                env = {**os.environ, "HOME": str(home), "STRATA_FEED": f"http://127.0.0.1:{server.server_address[1]}",
                       "STRATA_INSTALL_TEST_TRUST": "1", "STRATA_PYTHON": sys.executable, "STRATA_NO_ACQUIRE": ""}
                for k in ("XDG_DATA_HOME", "STRATA_UPDATE_TRUSTED_KEYS"):
                    env.pop(k, None)
                r = subprocess.run(["sh", str(release / "install.sh")], capture_output=True, text=True, env=env, timeout=240)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertIn("Checksum ok", r.stdout)
                self.assertIn("Signature ok", r.stdout)
                self.assertTrue((home / ".local" / "bin" / "strata").exists())
            finally:
                server.shutdown()

    def _windows_tar_pack(self, folder: Path, work: Path) -> Path:
        import feed_tool
        out = work / "badcontrol"
        feed_tool.pack_control(folder, out, version="9.9.9", platform="windows-x86_64", fmt="tar.gz", url_prefix="")
        return out


if __name__ == "__main__":
    unittest.main()
