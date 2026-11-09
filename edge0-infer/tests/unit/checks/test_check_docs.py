"""check_docs: broken links, root-style links and machine-local paths are rejected, clean docs pass."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[3] / "tools/check_docs.py"
FENCE = "`" * 3


class DocsTests(unittest.TestCase):
    def run_check(self, files, ignored=(), use_git=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, text in files.items():
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text(text, encoding="utf-8")
            if use_git:
                subprocess.run(["git", "init", "-q"], cwd=root, check=True)
                (root / ".gitignore").write_text("\n".join(ignored) + "\n")
            result = subprocess.run(
                [sys.executable, "-I", "-X", "utf8", str(SCRIPT), str(root)],
                text=True, capture_output=True, encoding="utf-8")
            return result.returncode, result.stdout

    def expect_error(self, files, rule):
        code, out = self.run_check(files)
        self.assertEqual(code, 1, out)
        self.assertIn(rule, out)

    def test_clean(self):
        code, out = self.run_check({"a.md": "[b](b.md)\n", "b.md": "text\n"})
        self.assertEqual(code, 0, out)

    def test_missing_file_and_anchor(self):
        self.expect_error({"a.md": "[b](c.md)\n"}, "missing-file a.md:1")
        self.expect_error({"a.md": "[b](b.md#nope)\n", "b.md": "# Title\n"}, "missing-anchor a.md:1")
        code, out = self.run_check({"a.md": "[b](b.md#title)\n", "b.md": "# Title\n"})
        self.assertEqual(code, 0, out)

    def test_root_style_link(self):
        self.expect_error({"a.md": "[b](/b.md)\n", "b.md": "text\n"}, "absolute-link a.md:1")

    def test_local_path_in_prose(self):
        for path in ["/Users/someone/work", "/home/someone/work", "/private/var/x", "/private/tmp/x", "/var/folders/ab/c"]:
            with self.subTest(path=path):
                self.expect_error({"a.md": f"see {path} here\n"}, "local-path a.md:1")

    def test_home_relative_path(self):
        self.expect_error({"a.md": "see `~/workspace/repo` here\n"}, "local-path a.md:1")
        self.expect_error({"a.md": f"{FENCE}sh\ncd ~/aiModelSets\n{FENCE}\n"}, "local-path a.md:2")

    def test_tilde_without_path_is_fine(self):
        code, out = self.run_check({"a.md": "~~struck~~ and about ~5 s and a~/b\n"})
        self.assertEqual(code, 0, out)

    def test_local_path_in_code_block(self):
        self.expect_error({"a.md": f"{FENCE}sh\nls /Users/someone/x\n{FENCE}\n"}, "local-path a.md:2")

    def test_local_path_in_inline_code(self):
        self.expect_error({"a.md": "run `UV_CACHE_DIR=/private/tmp/x/uv uv sync`\n"}, "local-path a.md:1")

    def test_tail_of_longer_path_or_url_is_fine(self):
        text = "https://example.com/home/page and https://host/Users/list and repo/home/x\n"
        code, out = self.run_check({"a.md": text})
        self.assertEqual(code, 0, out)

    def test_variables_are_fine(self):
        code, out = self.run_check({"a.md": "`$MOE_TINY/x` and `$TMPDIR/uv-cache`\n"})
        self.assertEqual(code, 0, out)

    def test_ignored_files_are_not_scanned(self):
        files = {"a.md": "ok\n", "local.md": "/Users/someone/x\n"}
        code, out = self.run_check(files, ignored=["local.md"], use_git=True)
        self.assertEqual(code, 0, out)
        code, out = self.run_check(files, ignored=[], use_git=True)
        self.assertEqual(code, 1, out)


if __name__ == "__main__":
    unittest.main()
