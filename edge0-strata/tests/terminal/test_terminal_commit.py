"""package_terminal.py must work where there is no git (the Docker build stage copies the sources without .git and the image has no git):
the commit comes from STRATA_BUILD_COMMIT, falls back to git when it exists, and is simply left out when neither answers."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))
import package_terminal  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"


class SourceCommit(unittest.TestCase):
    def test_env_wins_and_is_lower_cased(self):
        with mock.patch.dict(os.environ, {"STRATA_BUILD_COMMIT": " " + SHA.upper() + " "}), \
             mock.patch.object(package_terminal.subprocess, "run", side_effect=AssertionError("git must not run")):
            self.assertEqual(package_terminal.source_commit(), SHA)

    def test_no_git_installed_gives_empty_not_a_crash(self):
        env = {k: v for k, v in os.environ.items() if k != "STRATA_BUILD_COMMIT"}
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(package_terminal.subprocess, "run", side_effect=FileNotFoundError("git")):
            self.assertEqual(package_terminal.source_commit(), "")

    def test_bad_env_value_falls_back_to_git(self):
        done = subprocess.CompletedProcess([], 0, stdout=SHA + "\n", stderr="")
        with mock.patch.dict(os.environ, {"STRATA_BUILD_COMMIT": "not-a-sha"}), \
             mock.patch.object(package_terminal.subprocess, "run", return_value=done):
            self.assertEqual(package_terminal.source_commit(), SHA)

    def test_not_a_checkout_gives_no_commit_file_value(self):
        done = subprocess.CompletedProcess([], 128, stdout="", stderr="fatal: not a git repository")
        env = {k: v for k, v in os.environ.items() if k != "STRATA_BUILD_COMMIT"}
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(package_terminal.subprocess, "run", return_value=done):
            self.assertEqual(package_terminal.source_commit(), "")


class DockerPassesTheCommit(unittest.TestCase):
    def test_dockerfile_turns_the_build_arg_into_the_env_the_packager_reads(self):
        text = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
        self.assertRegex(text, r"ARG STRATA_COMMIT")
        self.assertRegex(text, r"ENV STRATA_BUILD_COMMIT=\$\{STRATA_COMMIT\}")
        self.assertLess(text.index("ENV STRATA_BUILD_COMMIT"), text.index("RUN python3 /src/packaging/package_terminal.py"))

    def test_build_sh_passes_the_commit(self):
        text = (ROOT / "docker" / "build.sh").read_text(encoding="utf-8")
        self.assertIn('--build-arg "STRATA_COMMIT=$commit"', text)

    def test_the_packager_reads_the_same_name(self):
        self.assertIn("STRATA_BUILD_COMMIT", (ROOT / "packaging" / "package_terminal.py").read_text(encoding="utf-8"))
        self.assertTrue(re.fullmatch(r"[0-9a-f]{40}", SHA))


if __name__ == "__main__":
    unittest.main()
