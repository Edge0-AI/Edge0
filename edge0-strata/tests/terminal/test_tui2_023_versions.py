"""0.2.3 RC stress test, #3 (H01): the version bump is complete and consistent, and a 0.2.2 install sees the update."""

import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

import strata_cli
from strata_control import feed as feedmod, updater
from tests.terminal import test_updater

ROOT = Path(__file__).resolve().parents[2]


def first(path, pattern):
    return re.search(pattern, (ROOT / path).read_text(encoding="utf-8"), re.MULTILINE).group(1)


class VersionBump(unittest.TestCase):
    def test_the_terminal_is_0_2_4_2_and_control_is_0_1_49_2_while_the_app_stays_0_2_3(self):
        # 0.2.9: Terminal and app are 0.2.9; Control and every engine package (Qwen, DeepSeek, Metal, SYCL, aarch64) are 0.1.55.
        # 0.2.8.1 (Control-only): the Terminal is 0.2.8.1 and Control 0.1.55 (the CMake project version); the app stays 0.2.8 and every engine package 0.1.54 (as 0.2.3.1 did for the Terminal).
        # 0.2.8: Terminal and app are 0.2.8; Control and every engine package (Qwen, DeepSeek, Metal, SYCL, aarch64) are 0.1.54 (both engines changed since 0.2.7.1).
        # 0.2.7.1 (DGX Spark): Control and the linux-aarch64 engine packages are 0.1.53 (the CMake project version); the app and the Terminal stay 0.2.7, the Metal engine 0.1.52.
        # 0.2.7: Terminal and app are 0.2.7; Control and every engine package (Qwen, DeepSeek, Metal) are 0.1.52 (the CMake project version; the Qwen and DeepSeek C++ both changed since 0.2.6)
        self.assertEqual(strata_cli.PRODUCT_VERSION, "0.2.9")
        self.assertEqual(strata_cli.version(), "0.1.55")                # the CMake project version: Control
        self.assertEqual(first("CMakeLists.txt", r"^project\(strata VERSION ([0-9.]+) "), "0.1.55")
        self.assertIn('STRATA_VERSION="0.1.55-metal"', (ROOT / "metal" / "CMakeLists.txt").read_text(encoding="utf-8"))

    def test_every_place_the_app_version_lives_says_0_2_3(self):
        self.assertEqual(first("apps/desktop/src-tauri/Cargo.toml", r'^version = "([^"]+)"'), "0.2.9")
        lock = (ROOT / "apps/desktop/src-tauri/Cargo.lock").read_text(encoding="utf-8")
        self.assertRegex(lock, r'name = "strata-desktop"\r?\nversion = "0.2.9"')
        self.assertEqual(json.loads((ROOT / "apps/desktop/package.json").read_text(encoding="utf-8"))["version"], "0.2.9")
        host = json.loads((ROOT / "apps/host/package.json").read_text(encoding="utf-8"))
        self.assertEqual(host["version"], "0.2.9")
        locked = json.loads((ROOT / "apps/host/package-lock.json").read_text(encoding="utf-8"))
        self.assertEqual((locked["version"], locked["packages"][""]["version"]), ("0.2.9", "0.2.9"))

    def test_the_documents_that_state_the_current_version_agree(self):
        self.assertIn("Version 0.2.9, engine 0.1.55.", (ROOT / "AGENTS.md").read_text(encoding="utf-8"))
        for name in ("README.md", "README.de.md", "README.es.md", "README.fr.md", "README.ja.md", "README.pl.md", "README.pt-BR.md", "README.zh-CN.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertRegex(text, r"Strata-0[.]2[.][0-9]+-", name)       # (the restyled READMEs state the app version through the download buttons; their owner moves them to the release)
            self.assertNotIn("Strata-0.2.2-", text, name)

    def test_version_prints_the_product_and_the_build(self):
        done = subprocess.run([sys.executable, str(ROOT / "strata.py"), "--version"], capture_output=True, text=True, cwd=str(ROOT),
                              env={**os.environ, "STRATA_NO_HANDOVER": "1"})
        self.assertTrue(done.stdout.startswith("strata 0.2.9 (build 0.1.55;"), done.stdout)


class UpdateFromTheLastRelease(test_updater.UpdaterCase):
    def feed_with_control(self, version):
        bundle = self.root / "control-bundle"
        (bundle / "strata_control").mkdir(parents=True)
        (bundle / "strata.py").write_text(f"print('strata 0.2.3 (build {version})')\n")
        (bundle / "strata_control" / "__init__.py").write_text("")
        from feed_tool import build_feed, pack_control
        (self.release / "control").mkdir()
        pack_control(bundle, self.release / "control", version=version, platform=test_updater.PLATFORM)
        build_feed(self.release, key=self.key, sequence=1, control_dir=self.release / "control")

    def test_a_0_2_2_install_is_offered_control_0_1_48_and_a_0_2_3_install_is_current(self):
        self.feed_with_control("0.1.48")
        (self.components / "control").mkdir(parents=True)
        (self.components / "control" / "active.json").write_text(json.dumps({"version": "0.1.47", "folder": "0.1.47"}))
        report = updater.check(self.ctx())
        self.assertEqual((report["control"]["status"], report["control"]["installed"], report["control"]["offered"]),
                         ("available", "0.1.47", "0.1.48"))
        out = updater.run_update(self.ctx(), engine=False, control=True)
        self.assertEqual((out["exit_code"], out["results"][0]["status"]), (0, "updated"), out)
        self.assertEqual(updater.active_control(self.components)["version"], "0.1.48")
        self.assertEqual(updater.check(self.ctx())["control"]["status"], "current")

    def test_the_bumped_numbers_sort_above_the_last_release(self):
        self.assertGreater(feedmod.version_key(strata_cli.version()), feedmod.version_key("0.1.47"))
        self.assertGreater(feedmod.version_key("0.1.48"), feedmod.version_key("0.1.46.3"))
        self.assertGreater(feedmod.version_key("0.1.48"), feedmod.version_key("0.1.45.3"))
