"""Tests for setup.py's verification of the ready-made engine archive against its published SHA-256.

`get_prebuilt()` used to install the engine on nothing more than the byte count matching the server's
Content-Length, so a substituted file of the same length passed. These cover the check that closes that, the
refusal and what a refusal does to the caller, the cases where there is nothing published to check against,
and the order it happens in - it must run before the archive is opened, or a wrong engine still reaches the
disk. Mocked network - nothing is downloaded.

    python -m unittest tools.test_setup_engine_hash
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import setup  # noqa: E402

ASSET = "strata-windows-x64.zip"
HIP_ASSET = "strata-windows-x64-hip.zip"
TAG_BASE = "https://github.com/Niko1221/Strata/releases/download/v0.1.40/"
LATEST_BASE = "https://github.com/Niko1221/Strata/releases/latest/download/"
FORK_BASE = "https://github.com/someone/Strata/releases/download/v1.0/"
LOCAL_BASE = r"C:\my\mirror"


def zip_bytes() -> bytes:
    """A stand-in engine archive. Small; its hash is what matters, not its size."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("strata.exe", b"not a real engine")
        zf.writestr("BUILD.json", b'{"version": "0.1.40"}')
    return buf.getvalue()


def sha_of(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class WhereTheHashComesFrom(unittest.TestCase):
    """`release_of` and `is_local`: which base names a published release, and which is the user's own disk."""

    def test_a_tag_url_names_its_tag(self):
        self.assertEqual(setup.release_of(TAG_BASE),
                         ("https://api.github.com/repos/Niko1221/Strata/releases", "0.1.40"))

    def test_the_latest_url_names_no_tag(self):
        self.assertEqual(setup.release_of(LATEST_BASE),
                         ("https://api.github.com/repos/Niko1221/Strata/releases", None))

    def test_a_fork_names_the_fork_not_this_repository(self):
        # The repository used to be hardcoded, so a fork's own tag was looked up under Niko1221/Strata,
        # where it does not exist - every fork install would be refused.
        self.assertEqual(setup.release_of(FORK_BASE),
                         ("https://api.github.com/repos/someone/Strata/releases", "1.0"))

    def test_something_that_is_not_a_github_release_url_names_nothing(self):
        # This is the case that mattered: a local folder or a plain mirror used to match no tag, fall
        # through to releases/latest, and be checked against a release the user did not ask for.
        for base in (LOCAL_BASE, "https://mirror.invalid/dl/", "/mnt/mirror", "\\\\share\\engine"):
            self.assertIsNone(setup.release_of(base), base)

    def test_a_local_folder_is_local_and_a_remote_mirror_is_not(self):
        # The distinction the policy turns on: nothing is between a local file and setup; a remote mirror
        # has a network in the middle and still no release to check against.
        for base in (LOCAL_BASE, r"C:\\mirror", "/mnt/mirror", "file:///srv/engine",
                     r"\\\\share\\engine"):
            self.assertTrue(setup.is_local(base), base)
        for base in (setup.PREBUILT_URL, TAG_BASE, FORK_BASE, "https://mirror.invalid/dl/",
                     "http://example.invalid/engine/"):
            self.assertFalse(setup.is_local(base), base)

    def test_a_local_folder_asks_the_api_nothing_at_all(self):
        # Air-gapped and shared-IP installs must not depend on GitHub answering.
        with mock.patch.object(setup.urllib.request, "urlopen",
                               side_effect=AssertionError("the API was called")):
            self.assertIsNone(setup.engine_digest(ASSET, LOCAL_BASE))


class DigestLookup(unittest.TestCase):
    """`engine_digest`: the size and hash for one asset, from the releases API."""

    def release_json(self, tag, assets):
        return json.dumps({"tag_name": tag, "assets": assets}).encode("utf-8")

    def asked(self, payload):
        """Patch urlopen with a fake that serves `payload`; returns the list the URL goes into.

        A list rather than a mock, because patching urlopen with a plain function hands back the function,
        and the URL asked for is part of what is under test.
        """
        seen = []

        class R:
            def __init__(self, data):
                self._b = io.BytesIO(data)

            def read(self, *a):
                return self._b.read(*a)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def open_it(req, timeout=None):
            seen.append(req.full_url)
            return R(payload)

        patcher = mock.patch.object(setup.urllib.request, "urlopen", open_it)
        patcher.start()
        self.addCleanup(patcher.stop)
        return seen

    def test_reads_the_tag_out_of_the_download_url(self):
        # The URL already names the release, so the API call must ask for THAT release and not "latest":
        # the tag URL is tried first (#214) so a checkout normally gets its own version's engine.
        seen = self.asked(self.release_json(
            "v0.1.40", [{"name": ASSET, "size": 42, "digest": "sha256:" + "a" * 64}]))
        self.assertEqual(setup.engine_digest(ASSET, TAG_BASE), (42, "a" * 64))
        self.assertEqual(seen, ["https://api.github.com/repos/Niko1221/Strata/releases/tags/v0.1.40"])

    def test_latest_when_the_url_says_latest(self):
        seen = self.asked(self.release_json(
            "v0.1.40.1", [{"name": ASSET, "size": 7, "digest": "sha256:" + "b" * 64}]))
        setup.engine_digest(ASSET, LATEST_BASE)
        self.assertEqual(seen, ["https://api.github.com/repos/Niko1221/Strata/releases/latest"])

    def test_asks_the_fork_for_a_fork_url(self):
        seen = self.asked(self.release_json(
            "v1.0", [{"name": ASSET, "size": 9, "digest": "sha256:" + "c" * 64}]))
        self.assertEqual(setup.engine_digest(ASSET, FORK_BASE), (9, "c" * 64))
        self.assertIn("/repos/someone/Strata/releases/tags/v1.0", seen[0])

    def test_none_when_no_digest_is_published(self):
        # No digest is not a pass. The caller refuses.
        self.asked(self.release_json("v0.1.40", [{"name": ASSET, "size": 42}]))
        self.assertIsNone(setup.engine_digest(ASSET, TAG_BASE))

    def test_none_when_the_digest_is_not_a_sha256(self):
        for other in ("md5:" + "d" * 32, "sha512:" + "e" * 128, "", None):
            self.asked(self.release_json("v0.1.40",
                                         [{"name": ASSET, "size": 42, "digest": other}]))
            self.assertIsNone(setup.engine_digest(ASSET, TAG_BASE), other)

    def test_none_when_the_asset_is_not_in_that_release(self):
        # A CUDA 12 asset asked for on a release that has none, or a renamed asset. Refuse, do not verify
        # against some other file's hash.
        self.asked(self.release_json("v0.1.40",
                                     [{"name": "other.zip", "size": 1, "digest": "sha256:" + "f" * 64}]))
        self.assertIsNone(setup.engine_digest(ASSET, TAG_BASE))

    def test_none_when_github_cannot_be_reached(self):
        for exc in (OSError("no network"), TimeoutError("slow")):
            with mock.patch.object(setup.urllib.request, "urlopen", side_effect=exc):
                self.assertIsNone(setup.engine_digest(ASSET, TAG_BASE), exc)

    def test_picks_the_asset_the_url_actually_wants(self):
        # v0.1.40.1 ships several engine builds with different hashes; the one asked for must be the one
        # used, not whichever the API lists first.
        self.asked(self.release_json("v0.1.40.1", [
            {"name": "strata-windows-x64.zip", "size": 1, "digest": "sha256:" + "1" * 64},
            {"name": "strata-windows-x64-cuda12.zip", "size": 2, "digest": "sha256:" + "2" * 64},
            {"name": HIP_ASSET, "size": 3, "digest": "sha256:" + "3" * 64},
        ]))
        self.assertEqual(setup.engine_digest("strata-windows-x64-cuda12.zip", LATEST_BASE), (2, "2" * 64))
        self.assertEqual(setup.engine_digest(HIP_ASSET, LATEST_BASE), (3, "3" * 64))


class ArchiveVerification(unittest.TestCase):
    """`verify_engine_archive`: the downloaded file against the published size and hash."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.z = Path(self.tmp.name) / ASSET
        self.data = zip_bytes()
        self.z.write_bytes(self.data)
        patch = mock.patch.dict(os.environ, {}, clear=False)
        env = patch.start()
        self.addCleanup(patch.stop)
        env.pop("STRATA_SKIP_SHA256", None)

    def digest(self, value):
        return mock.patch.object(setup, "engine_digest", return_value=value)

    def test_passes_when_the_hash_matches(self):
        with self.digest((len(self.data), sha_of(self.data))):
            setup.verify_engine_archive(self.z, ASSET, TAG_BASE)      # must not raise
        self.assertTrue(self.z.exists())

    def test_refuses_a_wrong_hash_and_deletes_the_file(self):
        # A wrong engine left on disk is the thing this exists to prevent, so it goes.
        with self.digest((len(self.data), "0" * 64)):
            with self.assertRaises(setup.UnverifiedEngine) as caught:
                setup.verify_engine_archive(self.z, ASSET, TAG_BASE)
        self.assertIn("wrong SHA-256", str(caught.exception))
        self.assertFalse(self.z.exists())
        self.assertFalse(self.z.with_name(self.z.name + ".done").exists())

    def test_refuses_a_wrong_size_before_hashing(self):
        with self.digest((len(self.data) + 1, sha_of(self.data))):
            with self.assertRaises(setup.UnverifiedEngine) as caught:
                setup.verify_engine_archive(self.z, ASSET, TAG_BASE)
        self.assertIn("not the published", str(caught.exception))
        self.assertFalse(self.z.exists())

    def test_a_same_size_impostor_is_caught_by_the_hash_alone(self):
        # The case Content-Length cannot see, and the reason the hash is there at all.
        impostor = b"x" * len(self.data)
        self.z.write_bytes(impostor)
        with self.digest((len(impostor), sha_of(self.data))):
            with self.assertRaises(setup.UnverifiedEngine):
                setup.verify_engine_archive(self.z, ASSET, TAG_BASE)
        self.assertFalse(self.z.exists())

    def test_no_digest_warns_and_installs_anyway(self):
        # "recommend, never force": no digest (offline, rate limit, mirror, folder) is a warning, not a refusal.
        for base in (TAG_BASE, LOCAL_BASE, "https://mirror.invalid/dl/"):
            said = io.StringIO()
            with self.digest(None), mock.patch.object(setup, "warn", said.write):
                setup.verify_engine_archive(self.z, ASSET, base)     # must not raise
            self.assertIn("NOT verified", said.getvalue(), base)
            self.assertTrue(self.z.exists())

    def test_skip_env_skips_the_check_loudly(self):
        os.environ["STRATA_SKIP_SHA256"] = "1"
        said = io.StringIO()
        with mock.patch.object(setup, "engine_digest", return_value=(1, "0" * 64)) as m,                 mock.patch.object(setup, "warn", said.write):
            setup.verify_engine_archive(self.z, ASSET, TAG_BASE)     # a wrong digest, but skipped
        self.assertEqual(m.call_count, 0)
        self.assertIn("STRATA_SKIP_SHA256", said.getvalue())
        self.assertIn("NOT checking", said.getvalue())
        self.assertTrue(self.z.exists())

    def test_skip_env_needs_exactly_1(self):
        os.environ["STRATA_SKIP_SHA256"] = "0"
        with self.digest((1, "0" * 64)):
            with self.assertRaises(setup.UnverifiedEngine):
                setup.verify_engine_archive(self.z, ASSET, TAG_BASE)

    def test_a_truncated_archive_is_refused(self):
        # There is no finish mark any more, so nothing can vouch for a file that changed after it was
        # checked. Measured cost of that decision: 190 MB hashes in 0.79 s at 242 MB/s on this machine,
        # against 3.0 s to download the same file, so re-checking is not worth saving.
        with self.digest((len(self.data), sha_of(self.data))):
            setup.verify_engine_archive(self.z, ASSET, TAG_BASE)
            with open(self.z, "r+b") as f:
                f.truncate(len(self.data) // 2)
            with self.assertRaises(setup.UnverifiedEngine) as caught:
                setup.verify_engine_archive(self.z, ASSET, TAG_BASE)
        self.assertIn("not the published", str(caught.exception))

    def test_a_same_length_change_after_verification_is_refused(self):
        # The hole the mark left: same length, so the size check passes, and the mark would have vouched
        # for it. With no mark, the hash is recomputed and it is caught.
        with self.digest((len(self.data), sha_of(self.data))):
            setup.verify_engine_archive(self.z, ASSET, TAG_BASE)
            with open(self.z, "r+b") as f:
                f.seek(0)
                f.write(b"X")
            with self.assertRaises(setup.UnverifiedEngine) as caught:
                setup.verify_engine_archive(self.z, ASSET, TAG_BASE)
        self.assertIn("wrong SHA-256", str(caught.exception))

    def test_the_api_is_asked_once_not_twice(self):
        # It is a rate-limited API and an install is not the place to spend two calls on one lookup.
        with mock.patch.object(setup, "engine_digest",
                               return_value=(len(self.data), sha_of(self.data))) as m:
            setup.verify_engine_archive(self.z, ASSET, TAG_BASE)
        self.assertEqual(m.call_count, 1, f"engine_digest was called {m.call_count} times")


class WhatARefusalDoesToTheCaller(unittest.TestCase):
    """`engine_refused`, and the paths that call it.

    This is where the adversarial pass found the worst bug: a refusal used to end in `fail()`, i.e.
    `sys.exit(1)`, which is a BaseException, so the engine-UPDATE paths' `except Exception` could not catch
    it and setup died instead of starting with the engine it already had.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        (Path(self.tmp.name) / "engine").mkdir(parents=True, exist_ok=True)
        # get_prebuilt looks for the release with a HEAD request before it downloads (0.2.0), and keeps the engine in
        # an engine folder of its own: no network, and never this checkout's engine folder.  A published release answers the probe.
        for patcher in (mock.patch.object(setup, "engine_dir", lambda toolkit=13: Path(self.tmp.name) / "engine"),
                        mock.patch.object(setup, "_urlopen", lambda req, timeout=None: mock.MagicMock())):
            patcher.start()
            self.addCleanup(patcher.stop)

    def fake_download(self, url, dst, what=None):
        with zipfile.ZipFile(dst, "w") as z:
            z.writestr("BUILD.json", json.dumps({"version": "0.1.40", "archs": [89]}))
            z.writestr(setup.EXE, b"engine")

    def get_prebuilt(self, updating, digest):
        out = io.StringIO()
        with contextlib_redirect(out),                 mock.patch.object(setup, "download", self.fake_download),                 mock.patch.object(setup, "engine_digest", return_value=digest):
            return setup.get_prebuilt(setup.PREBUILT_URL, {"arch": 89}, "gpu", updating=updating), out

    def test_updating_keeps_the_installed_engine_instead_of_stopping(self):
        # The call site's own words: "a failed download must not stop the model from starting".
        for digest in ((999, "f" * 64), (1, "0" * 64)):
            eng, out = self.get_prebuilt(updating=True, digest=digest)
            self.assertIsNone(eng, f"digest={digest!r} did not return None")
        self.assertNotIn("Setup stopped", out.getvalue(),
                         "a refusal while updating must not claim setup stopped")
        self.assertIn("keeping the engine that is installed", out.getvalue())

    def test_a_refusal_is_an_exception_the_update_path_can_catch(self):
        # The regression guard for the actual bug: SystemExit escaped this guard before.
        self.assertTrue(issubclass(setup.UnverifiedEngine, Exception))
        self.assertFalse(issubclass(SystemExit, Exception))
        eng, _ = self.get_prebuilt(updating=True, digest=(999, "f" * 64))
        self.assertIsNone(eng, "a refusal must not escape get_prebuilt as SystemExit")

    def test_a_first_install_stops_because_there_is_nothing_to_fall_back_to(self):
        with mock.patch.object(setup, "download", self.fake_download),                 mock.patch.object(setup, "engine_digest", return_value=(999, "f" * 64)):
            with self.assertRaises(SystemExit):
                setup.get_prebuilt(setup.PREBUILT_URL, {"arch": 89}, "gpu")

    def test_the_cuda_and_amd_paths_both_verify(self):
        # get_prebuilt_hip downloads the AMD engine from a SECOND `download(base + ...)` call site, which
        # the first version of this change missed entirely - so AMD users got nothing from it.
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        for call in ('download(base + asset, z, "Strata engine")',
                     'download(base + WIN_HIP_ASSET, z, "Strata AMD engine")'):
            i = src.index(call)
            self.assertIn("verify_engine_archive", src[i:i + 400], f"{call} is not verified")


class Wiring(unittest.TestCase):
    """The check is reached, and reached before the archive is opened."""

    def test_get_prebuilt_verifies_before_it_unpacks(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        body = src[src.index("def get_prebuilt("):]
        body = body[:body.index("\ndef ")]
        i_verify = body.index("verify_engine_archive(")
        for opener in ("ZipFile(", "_unpack", "install_unpacked"):
            self.assertGreater(body.index(opener), i_verify,
                               f"{opener} happens before the SHA-256 is checked")

    def test_the_digest_is_read_from_the_api_not_the_download_host(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        # release_of() builds the API URL, from the OWNER/REPO in the download URL - so a mirror of this
        # repository is checked against this repository's published hash, not against whatever the mirror
        # happens to be serving.
        body = src[src.index("def release_of("):]
        body = body[:body.index("\ndef ")]
        self.assertIn("api.github.com/repos/", body)
        self.assertIn("m.group(1)", body, "the repository comes out of the URL, not a constant")
        # and engine_digest asks release_of rather than the download URL
        body = src[src.index("def engine_digest("):]
        body = body[:body.index("\ndef ")]
        self.assertIn("release_of(base)", body)


class contextlib_redirect:
    """stdout into a StringIO, without importing contextlib under a name setup.py also uses."""

    def __init__(self, buf):
        self.buf = buf

    def __enter__(self):
        self.old = sys.stdout
        sys.stdout = self.buf
        return self

    def __exit__(self, *a):
        sys.stdout = self.old
        return False


if __name__ == "__main__":
    unittest.main()
