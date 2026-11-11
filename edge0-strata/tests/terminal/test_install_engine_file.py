"""The offline path of the first-run card: an engine package the person downloaded and chose is checked (signature, every file, fit for
this PC), test-started and switched to, with no feed and no network. The file they chose stays where it is."""

import tarfile
from pathlib import Path
from unittest import mock

from strata_control import updater
from tests.terminal import update_fixture as fx
from feed_tool import pack_engine      # noqa: E402  (update_fixture puts packaging/ on the path)
from tests.terminal.test_updater import PLATFORM, UpdaterCase


class InstallEngineFile(UpdaterCase):
    def package(self, version="0.1.46", key="default", **options):
        folder = fx.build_package(self.root / f"file-{version}", version, key=self.key if key == "default" else key, **options)
        out = self.root / f"file-out-{version}"
        out.mkdir(exist_ok=True)
        pack_engine(folder, out)
        return next(p for p in sorted(out.iterdir()) if p.name.endswith((".tar.gz", ".zip")))

    def install(self, archive, **over):
        with mock.patch.object(updater, "discover_run_config", return_value=None):
            return updater.install_engine_file(self.ctx(run_config=None, **over), archive)

    def test_a_signed_package_is_installed_without_any_network(self):
        archive = self.package()
        out = self.install(archive)
        self.assertEqual((out["status"], out["changed"]), ("updated", True), out)
        self.assertEqual(self.resolved(), "0.1.46")
        self.assertTrue(archive.is_file(), "the person's file is not deleted")
        self.assertEqual(self.server.requests, [], "nothing was fetched")
        self.assertEqual(sorted(p.name for p in (self.components / ".s").iterdir()), [], "staging is cleaned")

    def test_an_unsigned_package_is_refused(self):
        archive = self.package(key=None)
        with self.assertRaises(updater.UpdateError) as caught:
            self.install(archive)
        self.assertIn("security check", str(caught.exception))
        self.assertEqual(self.engine_folders(), [])

    def test_a_package_signed_by_another_key_is_refused(self):
        archive = self.package(key=fx.make_key())
        with self.assertRaises(updater.UpdateError):
            self.install(archive)
        self.assertEqual(self.engine_folders(), [])

    def test_a_package_that_does_not_fit_this_card_says_so(self):
        archive = self.package(archs=("sm_100",))
        with self.assertRaises(updater.UpdateError) as caught:
            self.install(archive)
        self.assertIn("not a Strata engine for this PC", str(caught.exception))
        self.assertEqual(self.engine_folders(), [])

    def test_something_that_is_not_a_package_is_refused_in_plain_words(self):
        junk = self.root / "notes.txt"
        junk.write_text("hello")
        with self.assertRaises(updater.UpdateError) as caught:
            self.install(junk)
        self.assertIn("not a Strata engine package", str(caught.exception))
        bad = self.root / "bad.tar.gz"
        bad.write_bytes(b"this is not gzip")
        with self.assertRaises(updater.UpdateError):
            self.install(bad)

    def test_an_older_package_does_not_replace_a_newer_engine(self):
        self.install_current("0.1.47")
        out = self.install(self.package("0.1.46"))
        self.assertEqual((out["status"], out["changed"]), ("current", False))
        self.assertEqual(self.resolved(), "0.1.47")

    def test_a_package_with_a_changed_file_is_refused(self):
        archive = self.package()
        tampered = self.root / "tampered.tar.gz"
        with tarfile.open(archive) as src, tarfile.open(tampered, "w:gz") as dst:
            for member in src.getmembers():
                data = src.extractfile(member) if member.isfile() else None
                if member.isfile() and member.name.endswith("engine_main.py"):
                    import io
                    payload = b"print('changed')\n"
                    member.size = len(payload)
                    data = io.BytesIO(payload)
                dst.addfile(member, data)
        with self.assertRaises(updater.UpdateError):
            self.install(tampered)
        self.assertEqual(self.engine_folders(), [])
