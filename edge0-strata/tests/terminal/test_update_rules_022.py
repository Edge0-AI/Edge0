"""0.2.2 update rules: a slow first run of Control is waited for, a busy card is not held against an engine, a pre-release Control is
handed over to (or not) by its number, and a feed that looks expired on a PC with a wrong clock says so."""

import datetime as dt
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from strata_cli import handover
from strata_control import feed as feedmod, signing, updater
from tests.terminal import update_fixture as fx
from tests.terminal.test_handover import make_control
from tests.terminal.test_update_feed import document, signed
from tests.terminal import test_updater


class ControlVersionTest(unittest.TestCase):
    def ctx(self):
        said = []
        return SimpleNamespace(environ={}, say=lambda *a, **k: said.append(a)), said

    def test_a_first_run_that_times_out_is_tried_once_more_with_more_time(self):
        ctx, said = self.ctx()
        good = SimpleNamespace(returncode=0, stdout="strata 0.2.2\n")
        calls = []

        def fake(cmd, **kw):
            calls.append(kw["timeout"])
            if len(calls) == 1:
                raise subprocess.TimeoutExpired(cmd, kw["timeout"])
            return good
        with mock.patch.object(updater.subprocess, "run", fake):
            done = updater._control_version("py", Path("strata.py"), Path("."), ctx, "0.2.2")
        self.assertIs(done, good)
        self.assertEqual(calls, list(updater.CONTROL_VERSION_TIMEOUTS))
        self.assertGreater(calls[0], 120)
        self.assertEqual(len(said), 1)

    def test_two_failures_end_as_a_failed_test_start(self):
        ctx, _ = self.ctx()
        with mock.patch.object(updater.subprocess, "run", side_effect=subprocess.TimeoutExpired("x", 1)):
            with self.assertRaises(updater.TestStartFailed):
                updater._control_version("py", Path("strata.py"), Path("."), ctx, "0.2.2")


class BusyCardTest(test_updater.UpdaterCase):
    def refuse(self, reason):
        def runner(stage_root, staged, ctx):
            raise updater.TestStartFailed(staged.version, reason)
        return runner

    def test_a_busy_card_is_not_held_against_the_engine(self):
        self.install_current()
        self.publish(("0.1.47", fx.GOOD_ENGINE))
        out = updater.run_update(self.ctx(test_start=self.refuse("it timed out waiting for the graphics card")), control=False)
        result = out["results"][0]
        self.assertEqual(result["status"], "failed")
        self.assertIn("looks busy", result["error"])
        again = updater.run_update(self.ctx(), control=False)
        self.assertEqual(again["results"][0]["status"], "updated", again)

    def test_a_real_failure_is_remembered_for_a_day_and_then_tried_again(self):
        self.install_current()
        self.publish(("0.1.47", fx.GOOD_ENGINE))
        updater.run_update(self.ctx(test_start=self.refuse("the engine crashed on start")), control=False)
        soon = updater.run_update(self.ctx(), control=False)
        self.assertEqual(soon["results"][0]["status"], "failed-before")
        later = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=25)
        retry = updater.run_update(self.ctx(now=later), control=False)
        self.assertEqual(retry["results"][0]["status"], "updated", retry)


class HandoverPreRelease(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-rc-")
        self.addCleanup(self.temp.cleanup)
        self.comp = Path(self.temp.name) / "components"
        self.comp.mkdir()
        self.env = {"STRATA_COMPONENTS_DIR": str(self.comp)}

    def target(self, running):
        return handover.find_target(running, self.env)

    def test_the_order_of_releases_and_candidates(self):
        key = handover.version_key
        self.assertLess(key("0.2.2-rc1"), key("0.2.2"))
        self.assertLess(key("0.2.2-rc1"), key("0.2.2-rc2"))
        self.assertLess(key("0.2.2-beta"), key("0.2.2-rc1"))
        self.assertLess(key("0.2.1"), key("0.2.2-rc1"))
        self.assertIsNone(key("banana"))

    def test_a_release_candidate_is_handed_over_to_from_an_older_release(self):
        make_control(self.comp, "0.2.2-rc1")
        folder, version, problem = self.target("0.2.1")
        self.assertIsNone(problem)
        self.assertEqual(version, "0.2.2-rc1")
        self.assertEqual(folder.name, "0.2.2-rc1")

    def test_the_final_release_replaces_its_candidate_but_not_the_other_way(self):
        make_control(self.comp, "0.2.2")
        self.assertEqual(self.target("0.2.2-rc1")[1], "0.2.2")
        make_control(self.comp, "0.2.2-rc1", point=False)
        (self.comp / "control" / "active.json").write_text(json.dumps({"version": "0.2.2-rc1", "folder": "0.2.2-rc1"}))
        self.assertEqual(self.target("0.2.2"), (None, None, None))

    def test_a_version_nobody_can_read_is_said_not_ignored(self):
        make_control(self.comp, "banana")
        folder, version, problem = self.target("0.2.1")
        self.assertIsNone(folder)
        self.assertIn("cannot be read", problem)


class ClockTest(unittest.TestCase):
    def setUp(self):
        self.key = signing.generate_key()
        self.trusted = fx.trusted_of(self.key)

    def load(self, **guards):
        payload, sig = signed(document(expires_at="2026-10-01T00:00:00Z"), self.key)
        return feedmod.verify_and_parse(payload, sig, source="/feeds/strata-feed.json", trusted=self.trusted, **guards)

    def test_the_server_clock_proves_the_pc_clock_wrong(self):
        pc = dt.datetime(2027, 3, 1, tzinfo=dt.timezone.utc)
        server = dt.datetime(2026, 9, 20, tzinfo=dt.timezone.utc)
        with self.assertRaises(feedmod.FeedError) as caught:
            self.load(now=pc, server_now=server)
        text = str(caught.exception)
        self.assertIn("your PC clock is wrong", text)
        self.assertIn("2026-09-20", text)
        self.assertNotIn("refusing a stale feed", text)

    def test_without_a_server_date_the_clock_is_named_as_a_cause(self):
        with self.assertRaisesRegex(feedmod.FeedError, "your PC clock is wrong.*new feed"):
            self.load(now=dt.datetime(2027, 3, 1, tzinfo=dt.timezone.utc))

    def test_a_feed_the_server_also_calls_expired_is_still_refused_as_expired(self):
        pc = dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc)
        with self.assertRaisesRegex(feedmod.FeedError, "expired on"):
            self.load(now=pc, server_now=pc)

    def test_a_certificate_error_hints_at_the_clock(self):
        import urllib.error
        from strata_control import transfer
        err = urllib.error.URLError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
        with mock.patch.object(transfer._opener, "open", side_effect=err):
            with self.assertRaisesRegex(transfer.TransferError, "date and time"):
                transfer.read_small("https://example.org/strata-feed.json")


if __name__ == "__main__":
    unittest.main()
