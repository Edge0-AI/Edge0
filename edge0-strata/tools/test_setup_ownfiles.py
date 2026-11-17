"""setup.py and the person's own model files (--gguf-dir): setup checks them, it never deletes them (0.2.2).

    python -m unittest tools.test_setup_ownfiles
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


def run(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            fn(*args, **kw)
            code = 0
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


class OwnFileChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = Path(self.tmp.name) / "model-00001-of-00002.gguf"
        self.file.write_bytes(b"abcdefgh" * 100)
        self.sha = hashlib.sha256(self.file.read_bytes()).hexdigest()

    def test_a_downloaded_file_of_the_wrong_size_asks_to_delete_it(self):
        code, text = run(setup.verify_sha256, self.file, 801, self.sha)
        self.assertEqual(code, 1)
        self.assertIn("delete it and run setup again", text)

    def test_an_own_file_of_the_wrong_size_is_left_and_the_text_says_so(self):
        code, text = run(setup.verify_sha256, self.file, 801, self.sha, own=True)
        self.assertEqual(code, 1)
        self.assertIn("your file was left as it is", text)
        self.assertTrue(self.file.exists())

    def test_an_own_file_with_another_hash_is_not_deleted(self):
        code, text = run(setup.verify_sha256, self.file, 800, "0" * 64, own=True)
        self.assertEqual(code, 1)
        self.assertIn("does not match the published file", text)
        self.assertTrue(self.file.exists())

    def test_a_downloaded_file_with_another_hash_is_deleted_as_before(self):
        code, _text = run(setup.verify_sha256, self.file, 800, "0" * 64)
        self.assertEqual(code, 1)
        self.assertFalse(self.file.exists())

    def test_a_complete_own_gguf_of_another_size_or_hash_is_used_with_a_warning(self):     # #3777
        import importlib.util                                                          # by path: another `tests` package may already be loaded
        spec = importlib.util.spec_from_file_location("own_files_helper", Path(__file__).resolve().parents[1] / "tests" / "terminal" / "test_own_files.py")
        helper = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        spec.loader.exec_module(helper)
        gguf_bytes = helper.gguf_bytes
        gguf = Path(self.tmp.name) / "fine-tune.gguf"
        gguf.write_bytes(gguf_bytes(["Q8_0"]))
        for size, sha in ((gguf.stat().st_size + 96, "0" * 64), (gguf.stat().st_size, "0" * 64)):
            code, text = run(setup.verify_sha256, gguf, size, sha, own=True)
            self.assertEqual(code, 0, text)
            self.assertIn("Strata uses it as it is", text)
        cut = Path(self.tmp.name) / "cut.gguf"
        cut.write_bytes(gguf.read_bytes()[:-40])                    # cut short: the tensor payload is not all there
        code, text = run(setup.verify_sha256, cut, gguf.stat().st_size, "0" * 64, own=True)
        self.assertEqual(code, 1)
        self.assertIn("your file was left as it is", text)
        code, text = run(setup.verify_sha256, gguf, gguf.stat().st_size + 96, "0" * 64)          # a download is still checked
        self.assertEqual(code, 1)

    def test_a_matching_own_file_passes(self):
        code, _text = run(setup.verify_sha256, self.file, 800, self.sha, own=True)
        self.assertEqual(code, 0)


class CheckingShowsItsProgress(unittest.TestCase):
    def test_a_big_file_is_hashed_with_a_progress_line_of_its_own(self):
        """Hashing a 60 GB shard takes minutes: the app and the Terminal show it as 'checking', with a figure (not a download stuck at 99%)."""
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "big-00001-of-00002.gguf"
            size = (1 << 30) + 4096
            with open(f, "wb") as g:
                g.truncate(size)
            sha = hashlib.sha256()
            with open(f, "rb") as g:
                for block in iter(lambda: g.read(16 << 20), b""):
                    sha.update(block)
            import time
            original = time.time
            ticks = iter(range(0, 10_000, 3))
            setup.time.time = lambda: float(next(ticks))             # every read looks 3 s later: the progress line prints
            try:
                code, text = run(setup.verify_sha256, f, size, sha.hexdigest())
            finally:
                setup.time.time = original
        self.assertEqual(code, 0)
        self.assertIn("checking big-00001-of-00002.gguf (SHA-256", text)
        self.assertRegex(text, r"checking big-00001-of-00002\.gguf: +\d+\.\d\d / 1\.07 GB \(\d+%\)")


if __name__ == "__main__":
    unittest.main()
