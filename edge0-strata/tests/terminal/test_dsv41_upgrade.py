"""An installed DeepSeek engine package is updated by itself when the signed feed's `dsv41` member is newer (0.1.49 -> 0.1.49.1):
fetched, checked, started once, switched to, the old folder removed; and left alone, with one plain line, when anything goes wrong."""
from __future__ import annotations

import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strata_control import dsv41_engine  # noqa: E402
from tests.terminal import update_fixture as fx  # noqa: E402
from tests.terminal.test_dsv41_engine import Dsv41Engine, build  # noqa: E402
import feed_tool  # noqa: E402


class Upgrade(Dsv41Engine):
    # (the parent's own tests are not repeated here)
    test_downloads_verifies_installs_beside_the_engine = None
    test_a_cut_download_resumes = None
    test_no_package_for_this_pc_is_a_plain_line_not_a_crash = None
    test_a_feed_signed_with_another_key_is_refused = None

    def setUp(self):
        super().setUp()
        self.dest = self.root / "data" / "engine-dsv41"
        self.url = self.server.url + "/strata-feed.json"
        self.publish("0.1.49")
        self.old = self.install()
        self.assertIsNotNone(self.old)

    def upgrade(self, lines=None, start=(True, "strata-dsv41 0.1.49.1 build x")):
        out = [] if lines is None else lines
        with mock.patch.object(dsv41_engine, "_test_start", return_value=start):
            return dsv41_engine.upgrade(self.dest, self.url, installed=self.old, hardware=self.hw, emit=out.append)

    def folders(self):
        return sorted(p.name.split("-")[0] for p in self.dest.iterdir() if p.is_dir())

    def test_a_newer_package_in_the_feed_is_installed_next_to_the_old_and_the_old_goes(self):
        self.publish("0.1.49.1")
        lines = []
        exe = self.upgrade(lines)
        self.assertIsNotNone(exe, lines)
        self.assertEqual(self.folders(), ["0.1.49.1"])
        self.assertTrue(exe.is_file())
        self.assertEqual(dsv41_engine.find(self.dest), exe)
        self.assertIn("updated from 0.1.49 to 0.1.49.1", " ".join(lines))
        self.assertEqual(self.tree(self.eng), self.before, "the Qwen engine's files must not change")

    def test_the_old_one_is_still_there_until_the_new_one_has_started(self):
        self.publish("0.1.49.1")
        seen = {}

        def start(exe):
            seen["folders"] = self.folders()
            return True, "strata-dsv41 0.1.49.1"
        with mock.patch.object(dsv41_engine, "_test_start", side_effect=start):
            self.assertIsNotNone(dsv41_engine.upgrade(self.dest, self.url, installed=self.old, hardware=self.hw, emit=lambda m: None))
        self.assertEqual(seen["folders"], ["0.1.49", "0.1.49.1"])
        self.assertEqual(self.folders(), ["0.1.49.1"])

    def test_the_same_version_in_the_feed_changes_nothing(self):
        lines = []
        self.assertIsNone(self.upgrade(lines))
        self.assertEqual((self.folders(), lines), (["0.1.49"], []))
        self.assertTrue(self.old.is_file())

    def test_an_older_version_in_the_feed_changes_nothing(self):
        self.publish("0.1.48")
        self.assertIsNone(self.upgrade())
        self.assertEqual(self.folders(), ["0.1.49"])

    def test_versions_compare_as_numbers(self):
        key = dsv41_engine.version_key
        self.assertLess(key("0.1.49-x"), key("0.1.49.1-x"))
        self.assertLess(key("0.1.9-x"), key("0.1.10-x"))
        self.assertLess(key("0.1.49.1-x"), key("0.1.49.10-x"))
        self.assertGreater(key("0.1.49.1-x"), key("0.1.49-x"))

    def test_an_unreachable_feed_is_silent_and_the_installed_engine_stays(self):
        for dead in ("http://127.0.0.1:9/strata-feed.json", str(self.root / "no-such-feed")):       # nothing answers / the folder is gone
            lines = []
            with mock.patch.object(dsv41_engine, "_test_start", return_value=(True, "x")):
                self.assertIsNone(dsv41_engine.upgrade(self.dest, dead, installed=self.old, hardware=self.hw, emit=lines.append))
            self.assertEqual(lines, [], dead)
            self.assertEqual(self.folders(), ["0.1.49"])
        self.assertFalse(dsv41_engine.source_reachable("http://127.0.0.1:9/strata-feed.json", timeout=3))

    def test_a_feed_with_a_bad_signature_is_refused_and_the_old_engine_stays(self):
        self.publish("0.1.49.1")
        sig = self.release / "strata-feed.json.sig"
        sig.write_bytes(sig.read_bytes().replace(b"a", b"b", 1) if b"a" in sig.read_bytes() else sig.read_bytes() + b" ")
        lines = []
        self.assertIsNone(self.upgrade(lines))
        self.assertEqual(self.folders(), ["0.1.49"])
        self.assertIn("still using 0.1.49", " ".join(lines))

    def test_a_package_signed_with_another_key_is_refused_and_the_old_engine_stays(self):
        shutil.rmtree(self.release, ignore_errors=True)
        (self.release / "dsv41").mkdir(parents=True)
        other = fx.make_key()
        folder = build(self.root / "other", "0.1.49.1", other)
        feed_tool.pack_engine(folder, self.release / "dsv41", url_prefix="dsv41/")
        feed_tool.build_feed(self.release, key=self.key, sequence=3, dsv41_dir=self.release / "dsv41")      # (the feed is ours, the package is not)
        lines = []
        self.assertIsNone(self.upgrade(lines))
        self.assertEqual(self.folders(), ["0.1.49"])
        self.assertIn("could not be installed", " ".join(lines))
        self.assertIn("still using 0.1.49", " ".join(lines))

    def test_a_new_engine_that_does_not_start_is_removed_and_the_old_one_keeps_running(self):
        self.publish("0.1.49.1")
        lines = []
        self.assertIsNone(self.upgrade(lines, start=(False, "it did not start correctly (libcudart.so.13: not found)")))
        self.assertEqual(self.folders(), ["0.1.49"])
        self.assertTrue(self.old.is_file())
        self.assertIn("A newer DeepSeek engine is available but could not be installed", " ".join(lines))
        self.assertIn("libcudart.so.13", " ".join(lines))
        self.assertIn("still using 0.1.49", " ".join(lines))

    def test_source_reachable(self):
        self.assertTrue(dsv41_engine.source_reachable(str(self.release)))
        self.assertFalse(dsv41_engine.source_reachable(str(self.root / "no-such-feed")))
        self.assertTrue(dsv41_engine.source_reachable(self.url, timeout=5))


class LaunchChoice(unittest.TestCase):
    """The launch does not touch an engine the person chose."""

    def runner(self, package=None):
        from strata_cli.launch import runner as R
        r = R.Runner.__new__(R.Runner)
        r.s = mock.Mock(dsv41_package=package)
        r.feed = None
        r.emit = lambda *a, **k: None
        return r

    def test_a_dsv41_package_or_a_build_of_your_own_is_never_replaced(self):
        found = Path("/x/engine-dsv41/0.1.49-a/strata-dsv41")
        with mock.patch.object(dsv41_engine, "upgrade", side_effect=AssertionError("must not run")):
            self.assertEqual(self.runner(package="/some/file.tar.gz").upgrade_dsv41_engine(found), found)
            with mock.patch.dict(os.environ, {"STRATA_DSV41_EXE": "/my/build/strata-dsv41"}):
                self.assertEqual(self.runner().upgrade_dsv41_engine(found), found)
            self.assertEqual(self.runner().upgrade_dsv41_engine(Path("/x/engine/strata-dsv41")), Path("/x/engine/strata-dsv41"))     # not a package folder

    def test_an_unreachable_feed_does_not_start_an_upgrade(self):
        found = Path("/x/engine-dsv41/0.1.49-a/strata-dsv41")
        with mock.patch.object(dsv41_engine, "source_reachable", return_value=False), \
                mock.patch.object(dsv41_engine, "upgrade", side_effect=AssertionError("must not run")):
            self.assertEqual(self.runner().upgrade_dsv41_engine(found), found)

    def test_a_newer_package_replaces_the_choice(self):
        found = Path("/x/engine-dsv41/0.1.49-a/strata-dsv41")
        new = Path("/x/engine-dsv41/0.1.49.1-a/strata-dsv41")
        with mock.patch.object(dsv41_engine, "source_reachable", return_value=True), mock.patch.object(dsv41_engine, "upgrade", return_value=new):
            self.assertEqual(self.runner().upgrade_dsv41_engine(found), new)


class SetupPathLines(unittest.TestCase):
    """The upgrade lines setup prints are shown by strata launch, the failure one especially (0.2.5 list)."""

    def test_the_failure_line_is_a_warning_and_the_success_line_is_news(self):
        from strata_cli.launch.runner import deepseek_engine_note
        bad = deepseek_engine_note("  A newer DeepSeek engine is available but could not be installed: no space left; still using 0.1.49.")
        self.assertEqual(bad[0], "warn")
        self.assertTrue(bad[1].endswith("still using 0.1.49."))
        self.assertEqual(deepseek_engine_note("[!] A newer DeepSeek engine may be available but could not be checked (OSError); still using 0.1.49.")[0], "warn")
        good = deepseek_engine_note("  DeepSeek engine updated from 0.1.49 to 0.1.49.2.")
        self.assertEqual(good, ("info", "DeepSeek engine updated from 0.1.49 to 0.1.49.2."))
        self.assertIsNone(deepseek_engine_note("  checking model.gguf (SHA-256, 5 GB) ..."))


if __name__ == "__main__":
    unittest.main()
