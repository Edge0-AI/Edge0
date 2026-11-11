"""The app reports its version as 0.2.0-beta; the update check must compare it, not choke on it."""
import unittest

from strata_control import feed as feedmod


class AppPrereleaseVersionTests(unittest.TestCase):
    def test_a_pre_release_sorts_below_the_same_number_and_above_the_older_one(self):
        key = feedmod.app_version_key
        self.assertLess(key("0.2.0-beta"), key("0.2.0"))
        self.assertGreater(key("0.2.0-beta"), key("0.1.46"))
        self.assertLess(key("0.2.0-beta"), key("0.2.0.1"))
        self.assertEqual(key("0.2.0"), key("0.2.0"))

    def test_a_feed_whose_app_entry_is_a_pre_release_parses(self):
        """The release feed carries app version 0.2.0-beta (found building the real feed: parse() used the strict key)."""
        import json
        from tests.terminal import update_fixture as fx
        key = fx.make_key()
        doc = {"schema_version": 1, "feed": "strata-updates", "channel": "stable", "sequence": 1, "generated_at": "2026-10-19T00:00:00Z", "expires_at": None,
               "app": {"version": "0.2.0-beta", "notes": "", "pub_date": "2026-10-19T00:00:00Z",
                       "platforms": {"windows-x86_64": {"url": "a.exe", "signature": "s", "sha256": "a" * 64, "size": 1}}}}
        from strata_control import signing
        payload = (json.dumps(doc, sort_keys=True) + "\n").encode()
        loaded = feedmod.verify_and_parse(payload, json.dumps(signing.sign_bytes(key, payload)).encode(), source="x", trusted=fx.trusted_of(key))
        self.assertEqual(loaded.app.version, "0.2.0-beta")

    def test_the_feeds_own_versions_stay_strict(self):
        with self.assertRaises(feedmod.FeedError):
            feedmod.version_key("0.2.0-beta")
        with self.assertRaises(feedmod.FeedError):
            feedmod.app_version_key("0.2.0-be ta")


if __name__ == "__main__":
    unittest.main()
