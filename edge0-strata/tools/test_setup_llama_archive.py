"""#1797: a damaged llama.cpp source zip must not be kept as an already-downloaded source.

    python tools/test_setup_llama_archive.py

Only tiny local zip files and the standard library are used; no GPU, model or network is needed.
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class LlamaSourceArchive(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "source.zip"
        self.prefix = "llama.cpp-" + setup.LLAMA_CPP_COMMIT
        with zipfile.ZipFile(self.source, "w", compression=zipfile.ZIP_STORED) as z:
            z.writestr(self.prefix + "/ggml/CMakeLists.txt", "ggml fixture")
            z.writestr(self.prefix + "/gguf-py/fixture.txt", "gguf fixture")
            z.writestr(self.prefix + "/tools/ui/not-needed.txt", "unused web UI")
        self.archive = self.root / "third_party" / f"llama.cpp-{setup.LLAMA_CPP_COMMIT[:7]}.zip"
        self.marker = self.archive.with_name(self.archive.name + ".done")
        self.llama = self.archive.parent / "llama.cpp"
        self.archive.parent.mkdir()
        self.unrelated = self.archive.parent / "user-notes.txt"
        self.unrelated.write_text("keep this", encoding="utf-8")
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(setup, "ROOT", self.root))
        # a plain path: download() copies it (file:// plus a Windows drive letter is not one)
        stack.enter_context(mock.patch.object(setup, "LLAMA_CPP_ZIP", str(self.source)))
        stack.enter_context(mock.patch.object(setup.urllib.request, "urlopen",
                                             side_effect=AssertionError("test must not use the network")))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def cached(self, data):
        self.archive.write_bytes(data)
        setup.mark(self.archive)

    def assert_cache_removed(self):
        self.assertFalse(self.archive.exists())
        self.assertFalse(self.marker.exists())
        self.assertEqual(self.unrelated.read_text(encoding="utf-8"), "keep this")

    def assert_installed(self, result):
        self.assertEqual(result, self.llama)
        self.assertEqual((result / "ggml/CMakeLists.txt").read_text(), "ggml fixture")
        self.assertEqual((result / "gguf-py/fixture.txt").read_text(), "gguf fixture")
        self.assertFalse((result / "tools/ui").exists())
        self.assertFalse((self.archive.parent / "_unpack").exists())
        self.assert_cache_removed()

    def test_success_uses_local_source_and_removes_archive_and_mark(self):
        self.assert_installed(setup.get_llama_cpp())

    def test_installed_source_is_reused_without_downloading(self):
        self.assert_installed(setup.get_llama_cpp())
        with mock.patch.object(setup, "download", side_effect=AssertionError("already installed")):
            self.assertEqual(setup.get_llama_cpp(), self.llama)

    def test_bad_cached_zip_is_removed_and_the_next_call_recovers(self):
        self.cached(b"not a zip")
        with self.assertRaises(zipfile.BadZipFile):
            setup.get_llama_cpp()
        self.assert_cache_removed()
        self.assert_installed(setup.get_llama_cpp())

    def test_a_bad_new_download_is_also_removed(self):
        self.source.write_bytes(b"not a zip")
        with self.assertRaises(zipfile.BadZipFile):
            setup.get_llama_cpp()
        self.assert_cache_removed()

    def test_truncated_cached_zip_is_removed(self):
        self.cached(self.source.read_bytes()[:-22])  # missing the end-of-central-directory record
        with self.assertRaises(zipfile.BadZipFile):
            setup.get_llama_cpp()
        self.assert_cache_removed()
        self.assert_installed(setup.get_llama_cpp())

    def test_crc_failure_during_extraction_is_removed_and_the_next_call_recovers(self):
        # Keep the ZIP directory intact; the second member fails its CRC after the first was extracted.
        self.cached(self.source.read_bytes().replace(b"gguf fixture", b"BAD! fixture", 1))
        self.llama.mkdir()
        user_file = self.llama / "user-file.txt"
        user_file.write_text("keep until a source was unpacked", encoding="utf-8")
        with self.assertRaisesRegex(zipfile.BadZipFile, "Bad CRC-32"):
            setup.get_llama_cpp()
        self.assert_cache_removed()
        self.assertTrue((self.archive.parent / "_unpack" / self.prefix / "ggml/CMakeLists.txt").is_file())
        self.assertEqual(user_file.read_text(), "keep until a source was unpacked")
        self.assert_installed(setup.get_llama_cpp())

    def test_original_bad_zip_exception_is_preserved(self):
        for target in ("ZipFile", "ZipFile.extractall"):
            with self.subTest(target=target):
                self.cached(self.source.read_bytes())
                error = zipfile.BadZipFile("original error")
                with mock.patch("setup.zipfile." + target, side_effect=error), \
                        self.assertRaises(zipfile.BadZipFile) as caught:
                    setup.get_llama_cpp()
                self.assertIs(caught.exception, error)
                self.assert_cache_removed()

    def test_missing_archive_or_mark_during_cleanup_is_harmless(self):
        for absent in (self.archive, self.marker):
            with self.subTest(absent=absent.name):
                self.cached(b"not a zip")
                error = zipfile.BadZipFile("original error")

                def fail_after_removal(*args, **kwargs):
                    absent.unlink()
                    raise error

                with mock.patch.object(zipfile, "ZipFile", side_effect=fail_after_removal), \
                        self.assertRaises(zipfile.BadZipFile) as caught:
                    setup.get_llama_cpp()
                self.assertIs(caught.exception, error)
                self.assert_cache_removed()

    def test_cleanup_failure_does_not_replace_the_bad_zip_error(self):
        real_unlink = Path.unlink
        for blocked in (self.archive, self.marker):
            with self.subTest(blocked=blocked.name):
                self.cached(b"not a zip")
                error = zipfile.BadZipFile("original error")

                def deny_unlink(path, *args, **kwargs):
                    if path == blocked:
                        raise PermissionError("held open")
                    return real_unlink(path, *args, **kwargs)

                with mock.patch.object(zipfile, "ZipFile", side_effect=error), \
                        mock.patch.object(Path, "unlink", deny_unlink), mock.patch.object(setup, "warn") as warn, \
                        self.assertRaises(zipfile.BadZipFile) as caught:
                    setup.get_llama_cpp()
                self.assertIs(caught.exception, error)
                self.assertEqual(self.archive.exists(), blocked == self.archive)
                self.assertTrue(self.marker.exists())
                self.assertEqual(self.unrelated.read_text(), "keep this")
                warn.assert_called_once()
                self.assertIn("held open", warn.call_args.args[0])

    def test_permission_and_disk_errors_do_not_discard_a_valid_archive(self):
        for target in ("ZipFile", "ZipFile.extractall"):
            for error in (PermissionError("access denied"), OSError("disk full")):
                with self.subTest(target=target, error=type(error).__name__):
                    data = self.source.read_bytes()
                    self.cached(data)
                    with mock.patch("setup.zipfile." + target, side_effect=error), \
                            self.assertRaises(OSError) as caught:
                        setup.get_llama_cpp()
                    self.assertIs(caught.exception, error)
                    self.assertEqual(self.archive.read_bytes(), data)
                    self.assertTrue(self.marker.exists())
                    self.assertEqual(self.unrelated.read_text(), "keep this")


if __name__ == "__main__":
    unittest.main()
