"""check_docs: root-style links, machine-local paths, model names in framework
headings and change references in component status lines are rejected, clean docs pass."""
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

    def test_archive_link_to_retired_plan_is_fine(self):
        link = "[plan](../../../../docs-internal/plan/m1.md#x)\n"
        code, out = self.run_check({"openspec/changes/archive/c/tasks.md": link})
        self.assertEqual(code, 0, out)
        # Outside the archive the same link is still an error, and so is any other missing target.
        self.expect_error({"openspec/changes/c/tasks.md": link}, "missing-file openspec/changes/c/tasks.md:1")
        self.expect_error({"openspec/changes/archive/c/tasks.md": "[x](../../../../docs-internal/plan/m2.md)\n"},
                          "missing-file openspec/changes/archive/c/tasks.md:1")

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

    def test_evidence_size_limit(self):
        small = {"a.md": "ok\n", "openspec/changes/demo/evidence/run.log": "x" * 1000}
        code, out = self.run_check(small)
        self.assertEqual(code, 0, out)
        big = {"a.md": "ok\n", "openspec/changes/archive/2026-01-01-demo/evidence/run.log": "x" * (300 * 1024)}
        self.expect_error(big, "evidence-too-large openspec/changes/archive/2026-01-01-demo/evidence/run.log")

    def test_model_heading_in_framework_doc(self):
        for name in ["docs-internal/adr/0001-x.md", "docs-internal/architecture/c4_model/03-component/a.md", "docs-internal/interfaces/modules/b.md"]:
            for heading in ["## Example: 35B one turn", "### Qwen3 template", "# DeltaNet state"]:
                with self.subTest(name=name, heading=heading):
                    self.expect_error({name: heading + "\n"}, f"model-heading {name}:1")

    def test_model_heading_outside_scope_or_in_body_is_fine(self):
        files = {
            "docs-internal/architecture/c4_model/03-component/details/models/edge0-35b.md": "# 35B model page\n",
            "docs-internal/interfaces/reference/x.md": "## Qwen notes\n",
            "docs-internal/adr/evidence/x.md": "## 35B measurement\n",
            "docs-internal/interfaces/modules/b.md": "## General\n\nThe 35B model and MLA are named in prose only.\n"
                                           + FENCE + "\n# 8B in a code block\n" + FENCE + "\n",
            "docs-internal/requirements/a.md": "## 8B target\n",
        }
        code, out = self.run_check(files)
        self.assertEqual(code, 0, out)

    def test_change_reference_in_component_status_line(self):
        status = "\u72b6\u6001\uff1a"
        name = "docs-internal/architecture/c4_model/03-component/a.md"
        for line in [status + "partly done (\u2465, \u246b)", status + "see m1-multi-turn"]:
            with self.subTest(line=line):
                self.expect_error({name: "# A\n\n" + line + "\n"}, f"status-change-ref {name}:3")

    def test_status_line_without_change_reference_is_fine(self):
        status = "\u72b6\u6001\uff1a"
        files = {
            "docs-internal/architecture/c4_model/03-component/a.md": status + "partly done; see the interface page.\n\nBody may name \u2465.\n",
            "docs-internal/architecture/c4_model/03-component/details/models/m.md": status + "done in \u2465.\n",
            "docs-internal/interfaces/modules/b.md": status + "done in \u2465.\n",
        }
        code, out = self.run_check(files)
        self.assertEqual(code, 0, out)

    def test_ignored_files_are_not_scanned(self):
        files = {"a.md": "ok\n", "local.md": "/Users/someone/x\n"}
        code, out = self.run_check(files, ignored=["local.md"], use_git=True)
        self.assertEqual(code, 0, out)
        code, out = self.run_check(files, ignored=[], use_git=True)
        self.assertEqual(code, 1, out)


if __name__ == "__main__":
    unittest.main()
