"""The Linux ARM64 (DGX Spark) Terminal reads strata-feed-linux-aarch64.json; every other machine reads strata-feed.json, as before.

Why: a Control older than 0.1.53 refuses a feed that names a platform it does not know, so the linux-aarch64 entries cannot be in the common
feed. Same format, same signing key; only the file name differs.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))
sys.path.insert(0, str(ROOT))

import feed_tool  # noqa: E402
from strata_cli import update_cmd  # noqa: E402
from strata_control import feed as feedmod  # noqa: E402
from strata_control import updater  # noqa: E402

COMMON = "strata-feed.json"
ARM = "strata-feed-linux-aarch64.json"
RELEASE = "https://github.com/Niko1221/Strata/releases/latest/download"


def host(platform):
    return mock.patch("strata_control.feed.host_platform", return_value=platform)


class FeedFileName(unittest.TestCase):
    def test_names(self):
        self.assertEqual(feedmod.feed_name("linux-aarch64"), ARM)
        for plat in ("windows-x86_64", "linux-x86_64", "macos-arm64", "macos-x86_64"):
            self.assertEqual(feedmod.feed_name(plat), COMMON)

    def test_the_machine_decides(self):
        with host("linux-aarch64"):
            self.assertEqual(feedmod.feed_name(), ARM)
        with host("linux-x86_64"):
            self.assertEqual(feedmod.feed_name(), COMMON)

    def test_an_unknown_os_reads_the_common_feed(self):
        with mock.patch("strata_control.feed.host_platform", side_effect=RuntimeError("no")):
            self.assertEqual(feedmod.feed_name(), COMMON)

    def test_the_common_name_and_default_are_unchanged(self):
        self.assertEqual(feedmod.FEED_NAME, COMMON)
        self.assertEqual(update_cmd.DEFAULT_FEED, RELEASE + "/" + COMMON)


class DefaultSource(unittest.TestCase):
    def source(self, remembered=None, feed=None, env=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            if remembered:
                state = updater.State(root)
                state.data["feed"] = {"source": remembered}
                state.save()
            with mock.patch.dict("os.environ", {}, clear=False):
                import os
                os.environ.pop(update_cmd.ENV_FEED, None)
                if env:
                    os.environ[update_cmd.ENV_FEED] = env
                return update_cmd._feed_source(SimpleNamespace(feed=feed), root)

    def test_arm64_linux_defaults_to_its_own_file(self):
        with host("linux-aarch64"):
            self.assertEqual(update_cmd.default_feed(), RELEASE + "/" + ARM)
            self.assertEqual(self.source(), RELEASE + "/" + ARM)

    def test_the_common_feed_remembered_from_before_does_not_stick_on_arm64(self):
        with host("linux-aarch64"):
            self.assertEqual(self.source(remembered=update_cmd.DEFAULT_FEED), RELEASE + "/" + ARM)

    def test_other_machines_are_unchanged(self):
        for plat in ("windows-x86_64", "linux-x86_64", "macos-arm64"):
            with host(plat):
                self.assertEqual(self.source(), update_cmd.DEFAULT_FEED)
                self.assertEqual(self.source(remembered=update_cmd.DEFAULT_FEED), update_cmd.DEFAULT_FEED)

    def test_an_explicit_feed_wins_on_arm64(self):
        with host("linux-aarch64"):
            self.assertEqual(self.source(feed="https://mirror.example/f.json"), "https://mirror.example/f.json")
            self.assertEqual(self.source(env="/tmp/feed"), "/tmp/feed")
            self.assertEqual(self.source(feed="http://127.0.0.1:9", env="/tmp/feed"), "http://127.0.0.1:9")

    def test_a_remembered_mirror_is_kept_on_arm64(self):
        with host("linux-aarch64"):
            self.assertEqual(self.source(remembered="https://mirror.example/f.json"), "https://mirror.example/f.json")


class ReadsTheRightFile(unittest.TestCase):
    """feed.fetch against a folder and against a release address: which file is asked for."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        feed_tool.keygen(self.dir / "keys")
        self.key = feed_tool._load_key(self.dir / "keys" / "feed-key.json")
        self.trusted = {self.key["key_id"]: __import__("base64").b64decode(self.key["public"])}

    def write(self, folder, name, sequence):
        tmp = folder / "_tmp"
        feed_tool.build_feed(tmp, key=self.key, sequence=sequence)
        for suffix in ("", ".sig"):
            shutil.copy2(tmp / ("strata-feed.json" + suffix), folder / (name + suffix))
        shutil.rmtree(tmp)

    def test_a_folder_with_both_files(self):
        folder = self.dir / "both"
        folder.mkdir()
        self.write(folder, COMMON, 115)
        self.write(folder, ARM, 116)
        with host("linux-aarch64"):
            self.assertEqual(feedmod.fetch(str(folder), self.trusted).sequence, 116)
        for plat in ("windows-x86_64", "linux-x86_64", "macos-arm64"):
            with host(plat):
                self.assertEqual(feedmod.fetch(str(folder), self.trusted).sequence, 115)

    def test_a_folder_made_for_the_common_feed_still_works_on_arm64(self):
        folder = self.dir / "common"
        folder.mkdir()
        self.write(folder, COMMON, 7)
        with host("linux-aarch64"):
            self.assertEqual(feedmod.fetch(str(folder), self.trusted).sequence, 7)

    def test_a_named_file_is_used_as_given(self):
        folder = self.dir / "named"
        folder.mkdir()
        self.write(folder, COMMON, 115)
        self.write(folder, ARM, 116)
        with host("linux-aarch64"):
            self.assertEqual(feedmod.fetch(str(folder / COMMON), self.trusted).sequence, 115)
        with host("linux-x86_64"):
            self.assertEqual(feedmod.fetch(str(folder / ARM), self.trusted).sequence, 116)

    def test_a_release_address_asks_for_the_machines_file(self):
        asked = []

        def read_small(location, **kw):
            asked.append(location)
            raise feedmod.transfer.TransferError("stop here")

        with mock.patch("strata_control.feed.transfer.read_small", side_effect=read_small):
            for plat, want in (("linux-aarch64", ARM), ("linux-x86_64", COMMON), ("windows-x86_64", COMMON), ("macos-arm64", COMMON)):
                asked.clear()
                with host(plat), self.assertRaises(feedmod.FeedError):
                    feedmod.fetch(RELEASE, self.trusted)
                self.assertEqual(asked[0], RELEASE + "/" + want, plat)
            asked.clear()
            with host("linux-aarch64"), self.assertRaises(feedmod.FeedError):
                feedmod.fetch(RELEASE + "/" + COMMON, self.trusted)
            self.assertEqual(asked[0], RELEASE + "/" + COMMON)


class CommonFeedStaysReadableByOldClients(unittest.TestCase):
    def test_the_published_parser_set_is_the_reason(self):
        """The platforms a 0.1.52 Control accepts are the four below; a feed with a fifth is refused whole, which is why linux-aarch64 has its own file."""
        self.assertEqual(sorted(set(feedmod.PLATFORMS) - {"linux-aarch64"}),
                         ["linux-x86_64", "macos-arm64", "macos-x86_64", "windows-x86_64"])


if __name__ == "__main__":
    unittest.main()
