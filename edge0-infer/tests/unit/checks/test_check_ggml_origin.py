"""Exercise provenance against real temporary Git trees."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[3] / "tools/checks/check_ggml_origin.py"


class OriginTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        # A machine-wide core.fileMode=false would hide the mode bit this fixture records.
        self.git("config", "core.fileMode", "true")
        (self.root / "third_party/ggml/sub").mkdir(parents=True)
        (self.root / "third_party/ggml/a").write_bytes(b"one")
        (self.root / "third_party/ggml/sub/b").write_bytes(b"two")
        (self.root / "third_party/ggml/tool.sh").write_bytes(b"#!/bin/sh\n")
        (self.root / "third_party/ggml/tool.sh").chmod(0o755)
        (self.root / "third_party/GGML_LICENSE").write_bytes(b"license")
        self.git("add", ".")
        tree = self.git("write-tree").strip()
        ggml = self.git("rev-parse", tree + ":third_party/ggml").strip()
        license_hash = self.git("hash-object", "third_party/GGML_LICENSE").strip()
        (self.root / "third_party/GGML_ORIGIN").write_text(f"tree: {ggml}\nlicense_blob: {license_hash}\n")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, text=True)

    def run_check(self, expected):
        result = subprocess.run([sys.executable, "-I", str(SCRIPT), str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        self.assertIn("ggml-origin: third_party/ggml: calculated=", result.stdout)
        self.assertIn("recorded=", result.stdout)

    def test_clean(self):
        self.run_check(0)

    def test_changed_byte(self):
        (self.root / "third_party/ggml/a").write_bytes(b"One")
        self.run_check(1)

    def test_deleted(self):
        (self.root / "third_party/ggml/a").unlink()
        self.run_check(1)

    def test_added_tracked(self):
        (self.root / "third_party/ggml/c").write_bytes(b"new")
        self.git("add", "third_party/ggml/c")
        self.run_check(1)

    def test_executable_bit_from_index_without_file_mode(self):
        # Windows has no executable bit; Git sets core.fileMode=false there and keeps the
        # mode in the index. A file system without the bit must not change the tree.
        # This OS cannot show the opposite (a chmod that Git records), so only the
        # fileMode=false half is checked here.
        (self.root / "third_party/ggml/tool.sh").chmod(0o644)
        if os.name != "nt":
            self.run_check(1)
        self.git("config", "core.fileMode", "false")
        self.run_check(0)

    def test_tracked_subdirectory_of_a_larger_repository(self):
        # Published inside another repository: the engine is a subdirectory of that work tree.
        # TemporaryDirectory also removes the read-only files Git writes, which Windows refuses to delete.
        outer_temp = tempfile.TemporaryDirectory()
        self.addCleanup(outer_temp.cleanup)
        outer = Path(outer_temp.name)
        subprocess.check_output(["git", "init", "-q"], cwd=outer)
        subprocess.check_output(["git", "config", "core.fileMode", "true"], cwd=outer)
        os.rename(self.root / ".git", self.root / ".git-unused")
        engine = outer / "engine"
        # copy2 (the default) keeps permission bits and timestamps.
        shutil.copytree(self.root, engine, symlinks=True, ignore=shutil.ignore_patterns(".git-unused"))
        subprocess.check_output(["git", "add", "."], cwd=outer)
        self.root = engine
        self.run_check(0)
        (engine / "third_party/ggml/a").write_bytes(b"One")
        self.run_check(1)

    def test_not_a_repository(self):
        os.rename(self.root / ".git", self.root / ".git-unused")
        result = subprocess.run([sys.executable, "-I", str(SCRIPT), str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 77, result.stdout + result.stderr)

    def test_license_changed(self):
        (self.root / "third_party/GGML_LICENSE").write_bytes(b"other")
        self.run_check(1)
