"""A feed that lists something this Control does not know (a platform, a backend or an archive format added in a later release) is still a feed:
the entry is left out with one plain line, and the rest is read as usual. A Control up to 0.1.52 refused the whole feed in that case, which is why
the first release with a new platform must not put it into the shared feed (see test_feed_aarch64.py). An entry of a KNOWN kind that is malformed
still refuses the feed."""

from __future__ import annotations

import copy
import json
import unittest

from strata_control import feed as feedmod


def engine(**kw):
    e = {"version": "0.1.53", "platform": "linux-x86_64", "backend": "cuda", "variant": "cuda13", "archs": ["sm_86"], "min_driver": {},
         "min_runtime": None, "build_id": "b", "format": "tar.gz", "url": "e.tar.gz", "size": 10, "sha256": "a" * 64,
         "manifest_sha256": "b" * 64, "installed_bytes": 20, "notes": ""}
    e.update(kw)
    return e


def control(**kw):
    c = {"platform": "linux-x86_64", "format": "tar.gz", "url": "c.tar.gz", "size": 10, "sha256": "a" * 64, "installed_bytes": 20, "entry": "strata.py"}
    c.update(kw)
    return c


def doc(**members):
    d = {"schema_version": 1, "feed": "strata-updates", "channel": "stable", "sequence": 1, "generated_at": "2026-10-28T00:00:00Z", "expires_at": None}
    d.update(members)
    return d


def parse(d):
    return feedmod.parse(json.dumps(d).encode(), source="x/strata-feed.json", key_id="k")


class UnknownEntriesAreSkipped(unittest.TestCase):
    def test_unknown_platform_in_every_member(self):
        f = parse(doc(engine={"packages": [engine(), engine(platform="freebsd-riscv", url="u1")]},
                      dsv41={"packages": [engine(url="d"), engine(platform="plan9-mips", url="u2")]},
                      control={"version": "0.1.53", "packages": [control(), control(platform="linux-m68k", url="c2")]}))
        self.assertEqual([p.platform for p in f.engine], ["linux-x86_64"])
        self.assertEqual([p.platform for p in f.dsv41], ["linux-x86_64"])
        self.assertEqual([p.platform for p in f.control], ["linux-x86_64"])
        self.assertEqual(len(f.skipped), 3)
        for line in f.skipped:
            self.assertNotIn("\n", line)
            self.assertIn("skipped", line)

    def test_unknown_backend_and_format(self):
        f = parse(doc(engine={"packages": [engine(), engine(backend="npu", url="u1"), engine(format="7z", url="u2")]},
                      control={"version": "0.1.53", "packages": [control(), control(format="rar", url="c2")]}))
        self.assertEqual(len(f.engine), 1)
        self.assertEqual(len(f.control), 1)
        self.assertEqual(len(f.skipped), 3)

    def test_an_unknown_variant_is_just_a_name(self):
        f = parse(doc(engine={"packages": [engine(variant="cuda14-new")]}))
        self.assertEqual(f.engine[0].variant, "cuda14-new")
        self.assertEqual(f.skipped, ())

    def test_unknown_app_platform(self):
        app = {"version": "0.2.8", "notes": "", "pub_date": "2026-10-28T00:00:00Z", "platforms": {
            "windows-x86_64": {"url": "a", "signature": "s", "sha256": "a" * 64, "size": 5},
            "plan9-mips": {"url": "b", "signature": "s", "sha256": "a" * 64, "size": 5}}}
        f = parse(doc(app=app))
        self.assertEqual(sorted(f.app.platforms), ["windows-x86_64"])
        self.assertEqual(len(f.skipped), 1)

    def test_the_known_entries_are_unchanged_by_the_skipped_ones(self):
        base = doc(engine={"packages": [engine()]}, control={"version": "0.1.53", "packages": [control()]})
        more = copy.deepcopy(base)
        more["engine"]["packages"].append(engine(platform="linux-aarch64x", url="u"))
        a, b = parse(base), parse(more)
        self.assertEqual(a.engine, b.engine)
        self.assertEqual(a.control, b.control)
        self.assertEqual(a.skipped, ())

    def test_linux_aarch64_is_known_here(self):
        f = parse(doc(engine={"packages": [engine(platform="linux-aarch64", archs=["sm_121"], url="a")]}))
        self.assertEqual(len(f.engine), 1)
        self.assertEqual(f.skipped, ())

    def test_a_feed_of_only_unknown_entries_is_empty_not_an_error(self):
        f = parse(doc(engine={"packages": [engine(platform="freebsd-riscv")]}))
        self.assertEqual(f.engine, ())
        self.assertEqual(len(f.skipped), 1)


class MalformedKnownEntriesStillRefuse(unittest.TestCase):
    def test_missing_fields(self):
        for bad in (engine(sha256="zz"), engine(size=0), engine(archs=[]), engine(version="x.y")):
            with self.assertRaises(feedmod.FeedError, msg=str(bad)):
                parse(doc(engine={"packages": [bad]}))
        with self.assertRaises(feedmod.FeedError):
            parse(doc(control={"version": "0.1.53", "packages": [control(size=0)]}))

    def test_a_duplicate_known_entry(self):
        with self.assertRaises(feedmod.FeedError):
            parse(doc(engine={"packages": [engine(), engine(url="again")]}))

    def test_the_strict_reading_is_still_there_for_a_direct_caller(self):
        with self.assertRaises(feedmod.FeedError):
            feedmod._engine_package(engine(platform="freebsd-riscv"))


if __name__ == "__main__":
    unittest.main()
