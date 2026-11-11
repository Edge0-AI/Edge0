"""Step 1's page-file check: only the size the page files have for sure counts - their size now (the commit limit
minus RAM), or a file's configured initial size when larger - never what Windows may grow them to, and the warning
(below 4 GB, as before) advises a fixed size.  When the Virtual memory setting (the registry's PagingFiles) has Windows
grow a file on demand ("System managed", or an initial size below the maximum), the warning says why that is not
enough (issue #60).  The registry, the drives and GlobalMemoryStatusEx are mocked, so it runs on Linux too.

    python -m unittest tools.test_setup_page_file
"""
from __future__ import annotations

import contextlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_golden import PROFILES, install  # noqa: E402

PAGE_FILE_GB, PAGE_FILE_SETTING = setup.page_file_gb, setup.page_file_setting   # the golden harness mocks the first
GROWS = "It is set to grow on demand"
FIXED = f"Custom size with initial and maximum size {setup.PAGE_FILE_FIXED_MB} MB"


def windows(paging, ram=31.9, now_mb=0, drives=None, existing=("\\??\\C:\\pagefile.sys",)):
    """What Windows answers page_file_gb() and page_file_setting(), mocked: PagingFiles (None: it cannot be read),
    ExistingPageFiles, each drive's (free MB, page file MB now), and GlobalMemoryStatusEx (commit limit = RAM + the
    files' size now)."""
    drives = {"C": (400000, now_mb)} if drives is None else drives
    ms = types.SimpleNamespace(ullTotalPhys=int(ram * 2**30), ullTotalPageFile=int(ram * 2**30) + now_mb * 2**20)
    return [mock.patch.object(setup, "_mm_multi_sz", lambda name: None if paging is None else
                              list(paging) if name == "PagingFiles" else list(existing)),
            mock.patch.object(setup, "_drive_mb", lambda d: drives.get(d)),
            mock.patch.object(setup, "_memory_status", lambda: ms)]


def on_windows(fn):
    def call(*a):
        with mock.patch.object(setup, "WIN", True):
            return fn(*a)
    return call


class Parse(unittest.TestCase):
    def test_entries(self):
        self.assertEqual(setup.parse_paging_file("c:\\pagefile.sys 64000 64000"), ("C", 64000, 64000))
        self.assertEqual(setup.parse_paging_file("D:\\pagefile.sys 1024 32768"), ("D", 1024, 32768))
        for managed in ("C:\\pagefile.sys 0 0", "C:\\pagefile.sys", "C:\\pagefile.sys x y"):
            self.assertEqual(setup.parse_paging_file(managed), ("C", -1, -1), managed)
        self.assertEqual(setup.parse_paging_file("?:\\pagefile.sys"), ("?", -1, -1))      # every drive automatic
        for junk in ("", "x", "pagefile.sys 1 2"):
            self.assertIsNone(setup.parse_paging_file(junk), junk)


class Reading(unittest.TestCase):
    def read(self, paging, **k):
        with contextlib.ExitStack() as st:
            for p in windows(paging, **k):
                st.enter_context(p)
            files = on_windows(PAGE_FILE_SETTING)()
            return on_windows(PAGE_FILE_GB)(), files

    def test_two_fixed_files(self):                   # this PR's author: 2 x 64000 MB on C: and D:
        gb, files = self.read(["c:\\pagefile.sys 64000 64000", "d:\\pagefile.sys 64000 64000"], ram=127.2,
                              now_mb=128000, drives={"C": (334983, 64000), "D": (178396, 64000)})
        self.assertEqual(gb, 125.0)
        self.assertFalse(setup.page_file_grows(files))

    def test_a_growing_file_counts_its_size_now(self):
        for paging in (["?:\\pagefile.sys"], ["C:\\pagefile.sys 0 0"], ["C:\\pagefile.sys"]):   # system-managed
            with self.subTest(paging):
                gb, files = self.read(paging, now_mb=4083)
                self.assertAlmostEqual(gb, 4083 / 1024)      # not 3 x RAM: it may not grow in time (#60)
                self.assertTrue(setup.page_file_grows(files))
        gb, files = self.read(["C:\\pagefile.sys 1024 32768"], now_mb=1024)
        self.assertEqual(gb, 1.0)                            # its initial size, not its maximum
        self.assertTrue(setup.page_file_grows(files))
        _, files = self.read(["?:\\pagefile.sys"], now_mb=4083, existing=("\\??\\D:\\pagefile.sys",),
                             drives={"D": (100000, 4083)})
        self.assertEqual(files, [("D", -1, -1, 4083, 100000)])    # "?:" is where Windows keeps it

    def test_an_initial_size_larger_than_the_file_now_counts(self):
        self.assertEqual(self.read(["C:\\pagefile.sys 8192 8192"], now_mb=2048, drives={"C": (400000, 2048)})[0], 8.0)
        gb, _ = self.read(["C:\\pagefile.sys 8192 8192"], now_mb=2048, drives={"C": (1000, 2048)})   # a full drive
        self.assertAlmostEqual(gb, 3048 / 1024)

    def test_off_and_unread(self):
        gb, files = self.read([])
        self.assertEqual((gb, files), (0.0, []))
        self.assertFalse(setup.page_file_grows(files))
        gb, files = self.read(None, now_mb=4083)             # the setting cannot be read: the commit limit alone
        self.assertAlmostEqual(gb, 4083 / 1024)
        self.assertIsNone(files)
        self.assertFalse(setup.page_file_grows(files))

    def test_not_windows(self):
        with mock.patch.object(setup, "WIN", False):
            self.assertIsNone(setup.page_file_gb())
            self.assertIsNone(setup.page_file_setting())


class Advice(unittest.TestCase):
    def test_fixed_size_always_and_why_when_it_grows(self):
        fixed = setup.page_file_advice(2.0, [("C", 2048, 2048, 2048, 400000)])
        self.assertIn("Windows' page file is 2.0 GB: the graphics card's memory needs room there too", fixed)
        self.assertIn(FIXED, fixed)
        self.assertNotIn(GROWS, fixed)
        self.assertNotIn("Set it to \"System managed\"", fixed)
        for files in ([("C", -1, -1, 4083, 400000)], [("C", 1024, 32768, 1024, 400000)],
                      [("C", 2048, 2048, 2048, 400000), ("D", -1, -1, 1024, 400000)]):
            with self.subTest(files):
                text = setup.page_file_advice(3.99, files)
                self.assertIn(GROWS, text)
                self.assertIn("a fixed 64 GB worked", text)
                self.assertIn(FIXED, text)
        self.assertNotIn(GROWS, setup.page_file_advice(0.0, []))
        self.assertNotIn(GROWS, setup.page_file_advice(1.0, None))   # the setting unread


class StepOne(unittest.TestCase):
    """setup --check on the harness's 32 GB PC with the page-file reading of a Windows PC: warned below 4 GB."""
    WARN = "Windows' page file is"

    def check(self, paging, now_mb, ram=None):
        ram0, found = PROFILES["32GB-2x24GB"]
        ram = ram or ram0
        extra = [mock.patch.object(setup, "page_file_gb", on_windows(PAGE_FILE_GB)),
                 mock.patch.object(setup, "page_file_setting", on_windows(PAGE_FILE_SETTING)),
                 *windows(paging, ram=ram, now_mb=now_mb)]
        code, out, _, _ = install(ram, found, ["--check"], extra=extra)
        self.assertEqual(code, 0, out[-3000:])
        return next((x for x in out.splitlines() if self.WARN in x), None)

    def test_a_system_managed_file_is_warned_with_the_reason(self):
        line = self.check(["?:\\pagefile.sys"], 4083)
        self.assertIn("Windows' page file is 4.0 GB", line)
        self.assertIn(GROWS, line)
        self.assertIn(FIXED, line)

    def test_a_small_fixed_file_and_none(self):
        for paging, now in ((["C:\\pagefile.sys 2048 2048"], 2048), ([], 0)):
            with self.subTest(paging):
                line = self.check(paging, now)
                self.assertIn(f"Windows' page file is {now / 1024:.1f} GB", line)
                self.assertIn(FIXED, line)
                self.assertNotIn(GROWS, line)

    def test_the_threshold_is_unchanged(self):
        self.assertIsNone(self.check(["?:\\pagefile.sys"], 8192))          # 8 GB now: not warned, as before
        self.assertIsNone(self.check(["C:\\pagefile.sys 64000 64000"], 64000))


if __name__ == "__main__":
    unittest.main()
