"""Disk-space checks stop only when the bytes to write do not fit; the working-room margin warns; a resumed download's partial file is not counted twice."""

import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

from strata_control import updater

Usage = namedtuple("Usage", "total used free")
MiB = 1 << 20


class Ctx:
    def __init__(self, root):
        self.components = root
        self.said = []

    def say(self, phase, message, **extra):
        self.said.append((phase, message))


class FreeSpace(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def check(self, free_mib, need_mib, ctx=None, sha=""):
        with mock.patch.object(updater.shutil, "disk_usage", return_value=Usage(10 ** 12, 0, free_mib * MiB)):
            updater._free_space_check(self.root, need_mib * MiB, ctx, sha, "engine")

    def test_plenty_of_room_is_silent(self):
        ctx = Ctx(self.root)
        self.check(5000, 1000, ctx)
        self.assertEqual(ctx.said, [])

    def test_between_the_real_need_and_the_margin_warns_and_goes_on(self):
        ctx = Ctx(self.root)
        self.check(1050, 1000, ctx)                      # 1.15x would be 1150
        self.assertEqual(len(ctx.said), 1)
        self.assertEqual(ctx.said[0][0], "warn")

    def test_below_the_real_need_stops_and_names_the_drive_and_the_step(self):
        with self.assertRaises(updater.UpdateError) as caught:
            self.check(800, 1000, Ctx(self.root))
        self.assertEqual(caught.exception.exit_code, 6)
        self.assertIn("800 MiB free", str(caught.exception))
        self.assertIn("Free up some space", str(caught.exception))

    def test_the_partial_file_of_a_resumed_download_is_not_counted_twice(self):
        ctx = Ctx(self.root)
        sha = "ab" * 32
        part = updater.updates_dir(self.root) / "partial"
        part.mkdir(parents=True)
        (part / f"{sha[:16]}.part").write_bytes(b"x" * 600)
        with mock.patch.object(updater.shutil, "disk_usage", return_value=Usage(10 ** 12, 0, 500)):
            updater._free_space_check(self.root, 1000, ctx, sha, "engine")        # 400 still to write, 500 free: fine and silent
            self.assertEqual(ctx.said, [])
            with self.assertRaises(updater.UpdateError):
                updater._free_space_check(self.root, 1000, ctx, "")                # without the partial file's credit 1000 > 500


if __name__ == "__main__":
    unittest.main()
