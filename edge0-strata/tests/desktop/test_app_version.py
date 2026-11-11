"""The desktop app's version is written in one place: apps/desktop/src-tauri/Cargo.toml (Tauri reads it for the installer and file names when
tauri.conf.json has no version; the shell passes it to the host as STRATA_APP_VERSION; the host reads the same file in a checkout)."""
import json
import re
import unittest
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[2] / "apps" / "desktop"


def cargo_version() -> str:
    return re.search(r'^version\s*=\s*"([^"]+)"', (DESKTOP / "src-tauri" / "Cargo.toml").read_text(encoding="utf-8"), re.M).group(1)


class AppVersionTests(unittest.TestCase):
    def test_the_version_is_the_product_version_not_the_placeholder(self):
        self.assertRegex(cargo_version(), r"^\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?$")
        self.assertNotEqual(cargo_version(), "0.1.0")

    def test_tauri_conf_does_not_carry_a_second_copy(self):
        conf = json.loads((DESKTOP / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8"))
        self.assertNotIn("version", conf)

    def test_the_npm_package_follows_it(self):
        self.assertEqual(json.loads((DESKTOP / "package.json").read_text(encoding="utf-8"))["version"], cargo_version())

    def test_no_script_hard_codes_a_file_name_with_a_version(self):
        text = (DESKTOP / "scripts" / "make-dmg.sh").read_text(encoding="utf-8")
        self.assertNotRegex(text, r"Strata_\d+\.\d+\.\d+")
        self.assertIn("Cargo.toml", text)


if __name__ == "__main__":
    unittest.main()
