"""Tests for serve/server.py's effort_end_args (#458): "effort_position": "end" is used when the engine binary knows
--tail-role-token.  With the Intel launcher (sycl/serve/strata-sycl.sh) as the configured exe, the check reads the
binary the launcher runs (STRATA_SYCL_BIN, default build-sycl-aot/strata), not the shell script, which never names
the option.  Fake files in a temp repo - nothing is built or run.

    python -m unittest tools.test_effort_end_args
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import server as S  # noqa: E402


class Tok:
    def encode(self, text, parse_special=False):
        return [42] if text == "system" else [1, 2]


ENGINE_WITH = b"\x7fELF ... --tail-role-token ... --serve ..."
ENGINE_WITHOUT = b"\x7fELF ... --serve ..."


class EffortEndArgs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "engine-repo"
        (self.repo / "sycl" / "serve").mkdir(parents=True)
        self.launcher = self.repo / "sycl" / "serve" / "strata-sycl.sh"
        self.launcher.write_text("#!/usr/bin/env bash\nexec ${STRATA_SYCL_BIN:-build-sycl-aot/strata}\n")
        (self.repo / "build-sycl-aot").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def call(self, cfg, exe):
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATA_SYCL_BIN", None)
            return S.effort_end_args(cfg, str(exe), Tok())

    def test_sycl_launcher_reads_the_binary(self):
        (self.repo / "build-sycl-aot" / "strata").write_bytes(ENGINE_WITH)
        self.assertEqual(self.call({"effort_position": "end"}, self.launcher), ["--tail-role-token", "42"])

    def test_sycl_binary_without_the_option(self):
        (self.repo / "build-sycl-aot" / "strata").write_bytes(ENGINE_WITHOUT)
        self.assertIsNone(self.call({"effort_position": "end"}, self.launcher))

    def test_sycl_bin_from_the_config_env(self):
        (self.repo / "build-sycl-aot" / "strata").write_bytes(ENGINE_WITHOUT)
        (self.repo / "other").mkdir()
        (self.repo / "other" / "strata").write_bytes(ENGINE_WITH)
        cfg = {"effort_position": "end", "env": {"STRATA_SYCL_BIN": "other/strata"}}
        self.assertEqual(self.call(cfg, self.launcher), ["--tail-role-token", "42"])

    def test_sycl_binary_missing_falls_back_to_the_launcher(self):
        self.assertIsNone(self.call({"effort_position": "end"}, self.launcher))

    def test_a_plain_engine_is_read_as_before(self):
        exe = Path(self.tmp.name) / "strata"
        exe.write_bytes(ENGINE_WITH)
        self.assertEqual(self.call({"effort_position": "end"}, exe), ["--tail-role-token", "42"])
        exe.write_bytes(ENGINE_WITHOUT)
        self.assertIsNone(self.call({"effort_position": "end"}, exe))

    def test_start_is_the_default(self):
        (self.repo / "build-sycl-aot" / "strata").write_bytes(ENGINE_WITH)
        self.assertIsNone(self.call({}, self.launcher))
        self.assertIsNone(self.call({"effort_position": "start"}, self.launcher))


if __name__ == "__main__":
    unittest.main()
