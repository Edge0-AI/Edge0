"""POSIX owner-only checks: our own folders with drifted modes are repaired; another account's folders get a message with the exact fix."""

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import posix_owner

posix_only = unittest.skipUnless(os.name == "posix", "POSIX modes")


@posix_only
class Repair(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_our_folder_with_open_modes_is_put_back_to_0700(self):
        d = self.root / "recipes"
        d.mkdir()
        os.chmod(d, 0o755)
        self.assertFalse(posix_owner.owner_only_wrong(d, d.lstat()))
        self.assertEqual(stat.S_IMODE(d.lstat().st_mode), 0o700)

    def test_our_file_with_open_modes_is_put_back_to_0600(self):
        f = self.root / "x.json"
        f.write_text("{}")
        os.chmod(f, 0o666)
        self.assertFalse(posix_owner.owner_only_wrong(f, f.lstat()))
        self.assertEqual(stat.S_IMODE(f.lstat().st_mode), 0o600)

    def test_a_private_one_is_left_alone(self):
        d = self.root / "ok"
        d.mkdir(mode=0o700)
        self.assertFalse(posix_owner.owner_only_wrong(d, d.lstat()))

    def test_a_link_is_never_followed_or_repaired(self):
        target = self.root / "t"
        target.mkdir()
        os.chmod(target, 0o755)
        link = self.root / "l"
        os.symlink(target, link)
        self.assertTrue(posix_owner.owner_only_wrong(link, link.lstat()))
        self.assertEqual(stat.S_IMODE(target.lstat().st_mode), 0o755)

    def test_a_folder_owned_by_someone_else_is_refused_and_not_touched(self):
        d = self.root / "root-made"
        d.mkdir()
        os.chmod(d, 0o755)
        with mock.patch.object(posix_owner.os, "getuid", return_value=os.getuid() + 1):
            self.assertTrue(posix_owner.owner_only_wrong(d, d.lstat()))
            text = posix_owner.owner_only_text("recipe directory must be owner-only (mode 0700)", d)
        self.assertEqual(stat.S_IMODE(d.lstat().st_mode), 0o755)
        self.assertIn("recipe directory belongs to", text)
        self.assertIn("sudo", text)
        self.assertIn("sudo chown -R \"$USER\"", text)
        self.assertNotIn("0700", text)

    def test_a_filesystem_that_keeps_no_modes_warns_once_and_goes_on(self):
        d = self.root / "drvfs"
        d.mkdir()
        os.chmod(d, 0o777)
        posix_owner._NOTED.discard(str(d))
        with mock.patch.object(posix_owner.os, "chmod"), mock.patch("sys.stderr") as err:
            self.assertFalse(posix_owner.owner_only_wrong(d, d.lstat()))
            self.assertFalse(posix_owner.owner_only_wrong(d, d.lstat()))
        printed = "".join(str(c.args[0]) for c in err.write.call_args_list)
        self.assertEqual(printed.count("cannot keep files private"), 1)

    def test_the_text_of_our_own_folder_is_unchanged(self):
        d = self.root / "mine"
        d.mkdir()
        self.assertEqual(posix_owner.owner_only_text("x must be owner-only", d), "x must be owner-only")


class Windows(unittest.TestCase):
    def test_nothing_changes_off_posix(self):
        with mock.patch.object(posix_owner, "POSIX", False):
            self.assertFalse(posix_owner.owner_only_wrong(".", os.stat(".")))
            self.assertEqual(posix_owner.owner_only_text("x", "."), "x")


if __name__ == "__main__":
    unittest.main()
