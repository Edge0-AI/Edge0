"""The Terminal tells people the app's version (0.2.3), and its README no longer contradicts itself."""

from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path

import strata_cli

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))
import package_terminal  # noqa: E402


class ProductVersion(unittest.TestCase):
    def test_it_is_the_apps_own_version_or_a_terminal_only_release_after_it(self):
        cargo = ROOT / "apps" / "desktop" / "src-tauri" / "Cargo.toml"
        if not cargo.is_file():
            self.skipTest("no app source in this tree")
        m = re.search(r'^version\s*=\s*"([^"]+)"', cargo.read_text(encoding="utf-8"), re.MULTILINE)
        # 0.2.3.1 is a Control-only hotfix: the Terminal says 0.2.3.1 while the app (and its updater entry) stay 0.2.3
        # 0.2.4 is a Terminal release: the app stays 0.2.3 (the Terminal is never older than the app)
        as_tuple = lambda v: tuple(int(p) for p in v.split("."))
        self.assertGreaterEqual(as_tuple(strata_cli.PRODUCT_VERSION), as_tuple(m.group(1)))
        self.assertEqual(as_tuple(strata_cli.PRODUCT_VERSION)[:2], as_tuple(m.group(1))[:2])

    def test_dash_dash_version_leads_with_it_and_keeps_the_build_number(self):
        out = subprocess.run([sys.executable, str(ROOT / "strata.py"), "--version"], capture_output=True, text=True, check=True).stdout
        self.assertTrue(out.startswith(f"strata {strata_cli.PRODUCT_VERSION} (build {strata_cli.version()};"), out)
        self.assertIn(sys.executable, out)                                  # the packaging smoke test reads the interpreter back from this line

    def test_the_build_number_stays_numeric_for_the_engine_compatibility_checks(self):
        self.assertRegex(strata_cli.version(), r"^[0-9]+(\.[0-9]+)+$")


class Readme(unittest.TestCase):
    def readme(self, target: str) -> str:
        return package_terminal._readme("0.2.3", target)

    def test_it_does_not_say_nothing_downloads_and_then_that_launch_fetches(self):
        text = " ".join(self.readme("windows-x86_64").split())
        self.assertNotIn("Starting and serving never install or download anything", text)
        self.assertIn("Serving and watching", text)
        self.assertIn("Only two commands fetch files", text)
        self.assertIn("`strata launch`", text)

    def test_the_launch_doc_it_points_at_is_in_the_archive(self):
        self.assertIn("docs/TERMINAL_LAUNCH.md", self.readme("linux-x86_64"))
        self.assertIn("docs/TERMINAL_LAUNCH.md", package_terminal.PAYLOAD)
        self.assertTrue((ROOT / "docs" / "TERMINAL_LAUNCH.md").is_file())

    def test_free_memory_guidance_fits_the_system(self):
        win, lin, mac = (self.readme(t) for t in ("windows-x86_64", "linux-x86_64", "darwin-arm64"))
        self.assertNotIn("free -g", win)
        self.assertIn("FreePhysicalMemory", win)
        self.assertIn("Task Manager", win)
        self.assertIn("free -g", lin)
        self.assertNotIn("free -g", mac)


if __name__ == "__main__":
    unittest.main()
