"""docker/fetch-engines.py against a LOCAL signed release folder: it fetches the engine package for an image flavor and refuses what the
signed feed does not vouch for (a feed signed by an unknown key, an archive that does not match the feed's SHA-256).
Linux x86-64 only: the script picks `linux-x86_64` packages, and the fixture builds a package for the host."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from strata_control import components
from tests.terminal import update_fixture as fx

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docker" / "fetch-engines.py"


@unittest.skipUnless(components.host_platform() == "linux-x86_64", "the Docker images are for linux-x86_64 packages")
class FetchEngines(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-dockfetch-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.key = fx.make_key()
        self.release = self.root / "release"
        self.release.mkdir()
        self.trust = self.root / "trust.json"
        self.trust.write_text(json.dumps({"keys": [{k: self.key[k] for k in ("alg", "key_id", "public")}]}), encoding="utf-8")
        folder = fx.build_package(self.root / "pub", "0.1.45", key=self.key, backend="cuda", variant="cuda13")
        from feed_tool import build_feed, pack_engine
        pack_engine(folder, self.release, url_prefix="")
        build_feed(self.release, key=self.key, sequence=1, engine_dir=self.release)
        self.out = self.root / "out"

    def run_script(self, *args, trust=True):
        env = dict(os.environ)
        env.pop("STRATA_UPDATE_TRUSTED_KEYS", None)
        if trust:
            env["STRATA_UPDATE_TRUSTED_KEYS"] = str(self.trust)
        return subprocess.run([sys.executable, str(SCRIPT), "--source", str(self.release), "--out", str(self.out), *args],
                              capture_output=True, text=True, env=env, cwd=str(ROOT))

    def test_fetches_and_verifies_a_package_for_the_flavor(self):
        r = self.run_script("--flavors", "cuda13")
        self.assertEqual(r.returncode, 0, r.stderr)
        flavor, build_flavor, folder = r.stdout.strip().split("\t")
        self.assertEqual((flavor, build_flavor), ("cuda13", "cuda"))
        self.assertTrue((Path(folder) / "engine-component.json").is_file())
        self.assertIn("signature and every file verified", r.stderr)

    def test_a_flavor_without_a_package_is_an_error_unless_skipped(self):
        r = self.run_script("--flavors", "cuda13,sycl")
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("no linux-x86_64 package", r.stderr)
        r = self.run_script("--flavors", "sycl", "--skip-missing")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip().split("\t")[-1], "SKIP")

    def test_a_feed_signed_by_a_key_we_do_not_trust_is_refused(self):
        r = self.run_script("--flavors", "cuda13", trust=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not trusted", r.stderr + r.stdout)
        self.assertFalse(self.out.exists() and any(self.out.iterdir()))

    def test_an_archive_that_is_not_the_one_in_the_feed_is_refused(self):
        archive = next(self.release.glob("strata-engine-*.tar.gz"))
        data = bytearray(archive.read_bytes())
        data[len(data) // 2] ^= 0xFF
        archive.write_bytes(bytes(data))
        r = self.run_script("--flavors", "cuda13")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("SHA-256", r.stderr + r.stdout)
        self.assertEqual(r.stdout.strip(), "")

    def test_a_tampered_feed_is_refused(self):
        feed = self.release / "strata-feed.json"
        feed.write_bytes(feed.read_bytes().replace(b'"sequence": 1', b'"sequence": 2'))
        r = self.run_script("--flavors", "cuda13")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not trusted", r.stderr + r.stdout)


if __name__ == "__main__":
    unittest.main()
