"""Component store tests (checklist P06, P07, P11, P12, P13).

The security tests here use the actual attack shapes rather than a generic "bad path", because the
failure this guards against is arbitrary file writes from a hostile archive, and each shape defeats a
different naive check:

* ``../`` escapes a destination that only joins paths;
* a leading ``/`` defeats a check that only looks for ``..``;
* a Windows drive letter defeats a POSIX-only check on a Windows host;
* a backslash separator defeats a check that only splits on ``/``;
* a symlink pointing outside defeats every name-based check, because the *name* is innocent.

Every refusal is asserted to leave the destination directory **empty**: checking after extraction
would be too late.
"""

from __future__ import annotations

import io
import json
import os
import stat
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strata_control import components as C  # noqa: E402


def manifest(**over) -> dict:
    base = {
        "schema_version": 1,
        "id": "strata-engine",
        "version": "0.1.44.1",
        "kind": "engine",
        "backend": "metal",
        "platforms": ["macos-arm64"],
        "artifacts": [{"name": "engine.tar.gz", "url": "https://example.invalid/e.tar.gz",
                       "sha256": "a" * 64, "bytes": 1024, "format": "tar.gz"}],
        "license": "proprietary",
    }
    base.update(over)
    return base


def tar_bytes(entries: list[tuple[str, bytes]], *, links: list[tuple[str, str]] | None = None,
              devices: bool = False) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, payload in entries:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(payload))
        for name, target in (links or []):
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)
        if devices:
            info = tarfile.TarInfo("dev/null")
            info.type = tarfile.CHRTYPE
            info.devmajor, info.devminor = 1, 3
            tar.addfile(info)
    return buffer.getvalue()


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="strata-components-"))
        self.store = C.ComponentStore(self.tmp / "store")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def payload(self, data: bytes, name: str = "engine.tar.gz") -> tuple[C.Component, dict]:
        import hashlib
        man = manifest(artifacts=[{"name": name, "url": None,
                                   "sha256": hashlib.sha256(data).hexdigest(),
                                   "bytes": len(data), "format": "tar.gz"}])
        return C.parse_manifest(man), {name: data}


class TestManifestParsing(StoreCase):
    def test_a_valid_manifest_parses(self):
        c = C.parse_manifest(manifest())
        self.assertEqual(c.id, "strata-engine")
        self.assertEqual(c.backend, "metal")
        self.assertEqual(len(c.artifacts), 1)

    def test_a_manifest_without_a_digest_is_refused(self):
        bad = manifest(artifacts=[{"name": "e.tar.gz", "sha256": "", "bytes": 10}])
        with self.assertRaises(C.ComponentError) as ctx:
            C.parse_manifest(bad)
        self.assertIn("sha256", str(ctx.exception))

    def test_a_short_or_non_hex_digest_is_refused(self):
        for digest in ("abc", "z" * 64, "A" * 63):
            with self.assertRaises(C.ComponentError):
                C.parse_manifest(manifest(artifacts=[{"name": "e", "sha256": digest, "bytes": 1}]))

    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(C.ComponentError) as ctx:
            C.parse_manifest(manifest(kind="kernel"))
        self.assertIn("unknown component kind", str(ctx.exception))

    def test_a_manifest_that_does_not_state_its_platforms_is_refused(self):
        """An unstated platform would be read as every platform, which is the unsafe default."""
        bad = manifest()
        bad.pop("platforms")
        with self.assertRaises(C.ComponentError) as ctx:
            C.parse_manifest(bad)
        self.assertIn("which platforms", str(ctx.exception))

    def test_an_artifact_name_that_is_a_path_is_refused(self):
        for name in ("../../etc/passwd", "sub/dir", "..", "a\\b"):
            with self.assertRaises(C.ComponentError):
                C.parse_manifest(manifest(artifacts=[
                    {"name": name, "sha256": "a" * 64, "bytes": 1}]))

    def test_an_unknown_schema_version_is_refused(self):
        with self.assertRaises(C.ComponentError) as ctx:
            C.parse_manifest(manifest(schema_version=99))
        self.assertIn("schema_version", str(ctx.exception))

    def test_malformed_json_is_refused_with_the_parse_error(self):
        with self.assertRaises(C.ComponentError) as ctx:
            C.parse_manifest("{not json")
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_the_cache_key_is_the_bytes_not_the_version_label(self):
        """A rebuilt artifact with the same version must be a different key."""
        a = C.parse_manifest(manifest())
        b = C.parse_manifest(manifest(artifacts=[{"name": "engine.tar.gz", "url": None,
                                                  "sha256": "b" * 64, "bytes": 1024}]))
        self.assertEqual(a.version, b.version, "the label is the same")
        self.assertNotEqual(a.cache_key, b.cache_key, "the bytes differ, so the key must differ")


class TestCompatibility(StoreCase):
    def test_a_component_for_this_platform_is_compatible(self):
        self.assertEqual(C.check_compatible(C.parse_manifest(manifest()), target="macos-arm64"), [])

    def test_another_platform_is_refused_with_both_named(self):
        reasons = C.check_compatible(C.parse_manifest(manifest()), target="linux-x86_64")
        self.assertTrue(reasons)
        self.assertIn("macos-arm64", reasons[0])
        self.assertIn("linux-x86_64", reasons[0])

    def test_a_backend_that_is_not_committed_to_on_a_platform_is_refused(self):
        """Windows SYCL and Intel Macs are explicitly not supported, not merely untested."""
        ok, why = C.backend_supported("sycl", "windows-x86_64")
        self.assertFalse(ok)
        self.assertIn("not a supported combination", why)
        ok, why = C.backend_supported("metal", "macos-x86_64")
        self.assertFalse(ok, "Intel Macs are excluded from local Metal support")
        self.assertTrue(C.backend_supported("metal", "macos-arm64")[0])
        self.assertTrue(C.backend_supported("cuda", "linux-x86_64")[0])
        self.assertTrue(C.backend_supported("sycl", "linux-x86_64")[0])

    def test_an_unknown_backend_is_refused(self):
        ok, why = C.backend_supported("vulkan", "linux-x86_64")
        self.assertFalse(ok)
        self.assertIn("unknown backend family", why)

    def test_a_component_for_another_backend_is_refused(self):
        c = C.parse_manifest(manifest(backend="metal", platforms=["macos-arm64"]))
        reasons = C.check_compatible(c, target="macos-arm64", backend="cuda")
        self.assertTrue(any("built for the metal backend" in r for r in reasons), reasons)


class TestSafeExtraction(StoreCase):
    """The attack shapes, each asserted to write nothing."""

    def extract(self, data: bytes, name: str = "a.tar.gz") -> Path:
        archive = self.tmp / name
        archive.write_bytes(data)
        destination = self.tmp / "out"
        C.safe_extract(archive, destination)
        return destination

    def assert_refused_and_empty(self, data: bytes, expect: str):
        destination = self.tmp / "out"
        with self.assertRaises(C.UnsafeArchive) as ctx:
            archive = self.tmp / "bad.tar.gz"
            archive.write_bytes(data)
            C.safe_extract(archive, destination)
        self.assertIn(expect, str(ctx.exception))
        present = sorted(p.name for p in destination.rglob("*")) if destination.exists() else []
        self.assertEqual(present, [], f"a refusal must leave nothing behind, found {present}")

    def test_a_normal_archive_extracts(self):
        out = self.extract(tar_bytes([("bin/strata", b"#!/bin/sh\n"), ("README", b"hi")]))
        self.assertEqual((out / "bin/strata").read_bytes(), b"#!/bin/sh\n")
        self.assertEqual((out / "README").read_bytes(), b"hi")

    def test_a_parent_segment_is_refused(self):
        self.assert_refused_and_empty(tar_bytes([("../escaped", b"x")]), "'..' segment")

    def test_a_nested_parent_segment_is_refused(self):
        self.assert_refused_and_empty(tar_bytes([("a/b/../../../escaped", b"x")]), "'..' segment")

    def test_an_absolute_path_is_refused(self):
        self.assert_refused_and_empty(tar_bytes([("/etc/passwd", b"x")]), "absolute path")

    def test_a_windows_drive_path_is_refused(self):
        self.assert_refused_and_empty(tar_bytes([("C:\\Windows\\evil", b"x")]), "drive-letter")

    def test_a_backslash_parent_segment_is_refused(self):
        self.assert_refused_and_empty(tar_bytes([("..\\escaped", b"x")]), "'..' segment")

    def test_a_unc_path_is_refused(self):
        self.assert_refused_and_empty(tar_bytes([("//server/share/evil", b"x")]), "absolute path")

    def test_a_symlink_is_refused(self):
        """A symlink's *name* is innocent, so only refusing links catches it."""
        self.assert_refused_and_empty(
            tar_bytes([("link", b"")], links=[("escape", "/etc/passwd")]), "symlink")

    def test_a_device_member_is_refused(self):
        self.assert_refused_and_empty(tar_bytes([("ok", b"x")], devices=True), "device")

    def test_a_good_member_before_a_bad_one_does_not_leave_the_good_one(self):
        """Validation happens over the whole archive first, so nothing is written before a refusal."""
        self.assert_refused_and_empty(
            tar_bytes([("fine.txt", b"hello"), ("../escaped", b"x")]), "'..' segment")

    def test_a_zip_with_a_parent_segment_is_refused(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("../escaped", "x")
        archive = self.tmp / "bad.zip"
        archive.write_bytes(buffer.getvalue())
        destination = self.tmp / "out"
        with self.assertRaises(C.UnsafeArchive):
            C.safe_extract(archive, destination, fmt="zip")
        self.assertFalse(destination.exists() and any(destination.rglob("*")))

    def test_the_executable_bit_survives_and_nothing_else_does(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            info = tarfile.TarInfo("bin/run")
            info.size = 2
            info.mode = 0o4755                     # setuid, which must not survive
            tar.addfile(info, io.BytesIO(b"#\n"))
        out = self.extract(buffer.getvalue())
        mode = (out / "bin/run").stat().st_mode
        if os.name != "nt":
            # NTFS has no execute bit: Windows decides by file extension (.exe), so there is nothing to keep.
            self.assertTrue(mode & stat.S_IXUSR, "a bundled runtime needs its executable bit")
        self.assertFalse(mode & stat.S_ISUID, "setuid from an archive is never trusted")


class TestInstall(StoreCase):
    def test_a_verified_component_installs_into_the_cache(self):
        component, payloads = self.payload(tar_bytes([("bin/strata", b"engine")]))
        path = self.store.install(component, payloads=payloads, target="macos-arm64")
        self.assertTrue((path / ".complete").exists())
        self.assertEqual((path / "bin/strata").read_bytes(), b"engine")
        self.assertEqual(path.name, component.cache_key)

    def test_a_digest_mismatch_installs_nothing(self):
        data = tar_bytes([("bin/strata", b"engine")])
        component, _ = self.payload(data)
        wrong = b"different bytes entirely"
        with self.assertRaises(C.ComponentError) as ctx:
            self.store.install(component, payloads={"engine.tar.gz": wrong}, target="macos-arm64")
        self.assertIn("does not match its manifest", str(ctx.exception))
        cache = self.store.root / "cache"
        present = [p.name for p in cache.iterdir()] if cache.exists() else []
        self.assertEqual(present, [], "a mismatch must leave no cache entry and no staging directory")

    def test_a_declared_size_mismatch_is_refused(self):
        data = tar_bytes([("a", b"x")])
        component, _ = self.payload(data)
        object.__setattr__(component.artifacts[0], "bytes", len(data) + 5)
        with self.assertRaises(C.ComponentError) as ctx:
            self.store.install(component, payloads={"engine.tar.gz": data}, target="macos-arm64")
        self.assertIn("bytes", str(ctx.exception))

    def test_an_incompatible_component_is_refused_before_anything_is_written(self):
        component, payloads = self.payload(tar_bytes([("a", b"x")]))
        with self.assertRaises(C.ComponentError) as ctx:
            self.store.install(component, payloads=payloads, target="linux-x86_64")
        self.assertEqual(ctx.exception.code, 2)
        self.assertFalse(any((self.store.root / "cache").iterdir()))

    def test_a_second_install_of_the_same_bytes_reuses_the_cache(self):
        component, payloads = self.payload(tar_bytes([("a", b"x")]))
        first = self.store.install(component, payloads=payloads, target="macos-arm64")
        stamp = (first / ".complete").stat().st_mtime_ns
        second = self.store.install(component, target="macos-arm64")
        self.assertEqual(first, second)
        self.assertEqual((second / ".complete").stat().st_mtime_ns, stamp,
                         "the cached payload was reused rather than rebuilt")

    def test_a_hostile_archive_installs_nothing(self):
        data = tar_bytes([("fine.txt", b"ok"), ("../escape", b"x")])
        component, payloads = self.payload(data)
        with self.assertRaises(C.UnsafeArchive):
            self.store.install(component, payloads=payloads, target="macos-arm64")
        self.assertFalse(any((self.store.root / "cache").iterdir()))
        self.assertFalse((self.tmp / "escape").exists(), "nothing escaped the destination")


class TestActivation(StoreCase):
    def build(self, version: str, marker: bytes) -> tuple[C.Component, dict]:
        import hashlib
        data = tar_bytes([("marker", marker)])
        man = manifest(version=version, artifacts=[
            {"name": "engine.tar.gz", "url": None, "sha256": hashlib.sha256(data).hexdigest(),
             "bytes": len(data), "format": "tar.gz"}])
        return C.parse_manifest(man), {"engine.tar.gz": data}

    def test_activation_puts_the_payload_at_the_active_path(self):
        component, payloads = self.build("1.0", b"one")
        self.store.install(component, payloads=payloads, target="macos-arm64")
        result = self.store.activate(component, target="macos-arm64")
        self.assertEqual((Path(result["active"]) / "marker").read_bytes(), b"one")

    def test_the_cache_survives_activation(self):
        """A compatible application update must not have to download the engine again."""
        component, payloads = self.build("1.0", b"one")
        cached = self.store.install(component, payloads=payloads, target="macos-arm64")
        self.store.activate(component, target="macos-arm64")
        self.assertTrue((cached / "marker").exists(), "the cached payload was consumed")

    def test_activating_a_newer_version_retains_the_previous_one(self):
        first, p1 = self.build("1.0", b"one")
        self.store.install(first, payloads=p1, target="macos-arm64")
        self.store.activate(first, target="macos-arm64")
        second, p2 = self.build("2.0", b"two")
        self.store.install(second, payloads=p2, target="macos-arm64")
        result = self.store.activate(second, target="macos-arm64")
        self.assertTrue(result["moved_aside"], "the displaced version is kept, not deleted")
        self.assertEqual((Path(result["active"]) / "marker").read_bytes(), b"two")

    def test_rollback_restores_the_previous_version_exactly(self):
        first, p1 = self.build("1.0", b"one")
        self.store.install(first, payloads=p1, target="macos-arm64")
        self.store.activate(first, target="macos-arm64")
        second, p2 = self.build("2.0", b"two")
        self.store.install(second, payloads=p2, target="macos-arm64")
        self.store.activate(second, target="macos-arm64")

        result = self.store.rollback("strata-engine")
        self.assertEqual(result["restored"], "1.0")
        active = self.store.active_path("strata-engine")
        self.assertEqual((active / "marker").read_bytes(), b"one")

    def test_rollback_without_a_previous_version_is_refused(self):
        """'Rollback succeeded' and 'the component is gone' must not be the same outcome."""
        component, payloads = self.build("1.0", b"one")
        self.store.install(component, payloads=payloads, target="macos-arm64")
        self.store.activate(component, target="macos-arm64")
        with self.assertRaises(C.ComponentError) as ctx:
            self.store.rollback("strata-engine")
        self.assertEqual(ctx.exception.code, 2)
        self.assertTrue(self.store.active_path("strata-engine").exists(),
                        "the active version must survive a refused rollback")

    def test_activating_something_not_installed_is_refused(self):
        component, _ = self.build("1.0", b"one")
        with self.assertRaises(C.ComponentError) as ctx:
            self.store.activate(component, target="macos-arm64")
        self.assertIn("not installed", str(ctx.exception))

    def test_activating_an_incompatible_component_is_refused(self):
        component, payloads = self.build("1.0", b"one")
        self.store.install(component, payloads=payloads, target="macos-arm64")
        with self.assertRaises(C.ComponentError) as ctx:
            self.store.activate(component, target="linux-x86_64")
        self.assertEqual(ctx.exception.code, 2)


class TestStoreState(StoreCase):
    def build(self, version: str) -> tuple[C.Component, dict]:
        import hashlib
        data = tar_bytes([("marker", version.encode())])
        man = manifest(version=version, artifacts=[
            {"name": "engine.tar.gz", "url": None, "sha256": hashlib.sha256(data).hexdigest(),
             "bytes": len(data), "format": "tar.gz"}])
        return C.parse_manifest(man), {"engine.tar.gz": data}

    def test_installed_versions_are_listed_and_the_active_one_marked(self):
        one, p1 = self.build("1.0")
        self.store.install(one, payloads=p1, target="macos-arm64")
        two, p2 = self.build("2.0")
        self.store.install(two, payloads=p2, target="macos-arm64")
        self.store.activate(one, target="macos-arm64")
        listed = self.store.installed("strata-engine")
        self.assertEqual([c.version for c in listed], ["1.0", "2.0"])
        self.assertEqual([c.version for c in listed if c.active], ["1.0"])

    def test_a_damaged_state_file_does_not_make_the_store_unusable(self):
        (self.store.root / "state.json").write_text("{ broken", encoding="utf-8")
        self.assertEqual(self.store.state()["components"], {})

    def test_pruning_keeps_what_is_active_and_what_rollback_needs(self):
        one, p1 = self.build("1.0")
        self.store.install(one, payloads=p1, target="macos-arm64")
        self.store.activate(one, target="macos-arm64")
        two, p2 = self.build("2.0")
        kept = self.store.install(two, payloads=p2, target="macos-arm64")
        self.store.activate(two, target="macos-arm64")

        removed = self.store.prune_cache()
        self.assertNotIn(one.cache_key, removed, "the rollback target must survive a prune")
        self.assertNotIn(two.cache_key, removed, "the active version must survive a prune")
        self.assertTrue((kept / "marker").exists())

    def test_pruning_removes_a_cache_entry_nothing_refers_to(self):
        orphan = self.store.root / "cache" / ("f" * 64)
        orphan.mkdir(parents=True)
        (orphan / "junk").write_text("x", encoding="utf-8")
        self.assertIn("f" * 64, self.store.prune_cache())
        self.assertFalse(orphan.exists())


class TestInterruptedTransfer(StoreCase):
    """P12: an interrupted download keeps its bytes and resumes."""

    def test_a_partial_file_larger_than_the_declared_size_is_discarded(self):
        """A stale partial from a different artifact must not be appended to."""
        partial = self.store.root / "partial" / "engine.tar.gz"
        partial.write_bytes(b"x" * 5000)
        component, _ = self.payload(tar_bytes([("a", b"x")]))
        artifact = component.artifacts[0]
        # A URL is needed to reach the transfer logic at all; it is never fetched, because the stale
        # partial is discarded and the ranged request then fails on the unreachable host.
        object.__setattr__(artifact, "url", "https://example.invalid/e.tar.gz")
        object.__setattr__(artifact, "bytes", 100)
        with self.assertRaises(C.ComponentError):
            self.store._fetch(artifact)
        self.assertFalse(partial.exists(), "a stale partial is removed, not appended to")

    def test_a_short_download_is_an_error_and_the_partial_is_kept(self):
        """The bytes are kept so a later attempt resumes rather than restarting."""
        import urllib.request
        body = b"only part of it"

        class FakeResponse(io.BytesIO):
            status = 200

            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

        original = urllib.request.urlopen
        urllib.request.urlopen = lambda *a, **k: FakeResponse(body)
        try:
            component, _ = self.payload(tar_bytes([("a", b"x")]))
            artifact = component.artifacts[0]
            object.__setattr__(artifact, "url", "https://example.invalid/e.tar.gz")
            with self.assertRaises(C.ComponentError) as ctx:
                self.store._fetch(artifact)
            self.assertIn("resume", str(ctx.exception))
        finally:
            urllib.request.urlopen = original
        self.assertTrue((self.store.root / "partial" / "engine.tar.gz").exists(),
                        "the partial bytes must be kept for a resume")

    def test_an_unreachable_url_is_reported_as_an_interruption_not_a_crash(self):
        import urllib.request
        original = urllib.request.urlopen

        def boom(*a, **k):
            raise OSError("network unreachable")

        urllib.request.urlopen = boom
        try:
            component, _ = self.payload(tar_bytes([("a", b"x")]))
            artifact = component.artifacts[0]
            object.__setattr__(artifact, "url", "https://example.invalid/e.tar.gz")
            with self.assertRaises(C.ComponentError) as ctx:
                self.store._fetch(artifact)
            self.assertIn("interrupted", str(ctx.exception))
        finally:
            urllib.request.urlopen = original

    def test_a_crash_during_install_leaves_no_complete_entry(self):
        """The `.complete` marker is written last, so a partial directory is never trusted."""
        data = tar_bytes([("a", b"x")])
        component, _ = self.payload(data)
        with self.assertRaises(C.ComponentError):
            self.store.install(component, payloads={"engine.tar.gz": b"wrong"}, target="macos-arm64")
        self.assertIsNone(self.store.cached(component))


if __name__ == "__main__":
    unittest.main()
