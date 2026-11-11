"""Windows MAX_PATH (260) and the engine installer.

A fresh Windows PC has long paths off. `strata launch` on an AMD PC failed with "local file operation failed
(FileNotFoundError)" because the staging folder (components\\.updates\\staging\\<12 hex>\\engine\\<folder>) plus the longest file in
the hip packages (149 characters) went past 259 under an ordinary user name. This pins the path arithmetic for the real 0.2.1
package file lists (tests/terminal/data/windows_package_paths.json, made from the release tarballs) and the behaviour of the fix."""

import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path, PureWindowsPath
from unittest import mock

from strata_control import engine_component as ec, longpath, updater

DATA = json.loads((Path(__file__).parent / "data" / "windows_package_paths.json").read_text(encoding="utf-8"))
LIMIT = longpath.PLAIN_LIMIT
USER = "u" * 20
COMPONENTS = PureWindowsPath(rf"C:\Users\{USER}\AppData\Local\Strata\components")      # what components_dir() gives on Windows
LEGACY_STAGE = COMPONENTS / ".updates" / "staging" / ("0123456789ab")                    # what 0.2.1 used
NEW_STAGE = COMPONENTS / updater.STAGE_DIR / "0123"
WINDOWS = os.name == "nt"


def longest(entry) -> str:
    return max(entry["longest"], key=len)


class PathArithmetic(unittest.TestCase):
    def test_the_old_staging_folder_was_past_the_limit(self):
        """The reproduction: with the 0.2.1 layout the AMD packages do not fit under a 20-character user name."""
        too_long = {}
        for e in DATA["engines"]:
            n = len(str(LEGACY_STAGE / "engine" / e["folder"] / longest(e)))
            if n > LIMIT:
                too_long[e["folder"]] = n
        self.assertEqual(sorted(too_long), ["0.1.46-windows-x86_64-hip-rdna3", "0.1.46-windows-x86_64-hip-rx9070"], too_long)

    def test_every_staged_path_fits_for_a_20_character_user_name(self):
        for e in DATA["engines"]:
            n = len(str(NEW_STAGE / "engine" / e["folder"] / longest(e)))
            self.assertLessEqual(n, LIMIT, f"{e['folder']}: staged path is {n} characters")

    def test_every_final_path_fits_for_a_20_character_user_name(self):
        for e in DATA["engines"]:
            n = len(str(COMPONENTS / "engine" / e["folder"] / longest(e)))
            self.assertLessEqual(n, LIMIT, f"{e['folder']}: installed path is {n} characters")
        for c in DATA["control"]:
            for base in (COMPONENTS / "control" / c["version"], COMPONENTS / updater.STAGE_DIR / "c0123"):
                n = len(str(base / longest(c)))
                self.assertLessEqual(n, LIMIT, f"control {c['version']}: path is {n} characters")

    def test_staging_is_no_deeper_than_nine_characters_over_the_install(self):
        live = COMPONENTS / "engine" / "x"
        staged = NEW_STAGE / "engine" / "x"
        self.assertLessEqual(len(str(staged)) - len(str(live)), 9)

    def test_the_hip_estimate_covers_the_real_packages(self):
        for e in DATA["engines"]:
            if "-hip-" in e["folder"]:
                self.assertGreaterEqual(longpath.LONGEST_IN_PACKAGE["hip"], len(longest(e)))


class Messages(unittest.TestCase):
    def test_a_missing_file_is_named(self):
        words = longpath.file_words(FileNotFoundError(2, "No such file or directory", r"C:\x\y\z.dat"))
        self.assertIn(r"C:\x\y\z.dat", words)
        self.assertIn("FileNotFoundError", words)

    def test_something_without_a_name_still_says_what_kind(self):
        self.assertIn("(OSError)", longpath.file_words(OSError("boom")))

    @unittest.skipUnless(WINDOWS, "the limit is a Windows one")
    def test_a_path_that_is_too_long_says_so(self):
        name = "C:\\" + "\\".join(["d" * 60] * 5) + "\\file.dat"
        words = longpath.file_words(FileNotFoundError(2, "No such file or directory", name))
        self.assertIn("longer than Windows allows", words)
        self.assertIn("STRATA_COMPONENTS_DIR", words)
        self.assertIn(name, words)

    @unittest.skipUnless(WINDOWS, "the limit is a Windows one")
    def test_no_warning_for_a_short_folder_and_one_for_a_deep_one(self):
        self.assertIsNone(longpath.warning_for(Path(r"C:\Strata\engine\0.1.46-windows-x86_64-hip-rdna3"), 150))
        deep = Path("C:/Users/" + USER + "/AppData/Local/Strata/components/engine/0.1.46-windows-x86_64-hip-rx9070/" + "q" * 30)
        words = longpath.warning_for(deep, 150)
        self.assertIn("long paths", words)
        self.assertIn("carries on", words)


@unittest.skipUnless(WINDOWS, "extended-length paths are a Windows thing")
class DeepFolders(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-lp-")
        self.addCleanup(lambda: __import__("shutil").rmtree(longpath.long_path(self.temp.name), ignore_errors=True))
        self.deep = Path(self.temp.name) / ("p" * 120) / ("q" * 120)

    def archive(self, member, data=b"engine bytes"):
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w:gz") as tar:
            info = tarfile.TarInfo(member)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        path = Path(self.temp.name) / "pkg.tar.gz"
        path.write_bytes(raw.getvalue())
        return path

    def test_long_path_adds_the_prefix_once(self):
        p = longpath.long_path(r"C:\a\b")
        self.assertTrue(str(p).startswith("\\\\?\\C:\\a"))
        self.assertEqual(longpath.long_path(p), p)
        self.assertEqual(longpath.plain(p), r"C:\a\b")

    def test_extract_goes_through_the_extended_form(self):
        member = "rocm/bin/hipblaslt/library/gfx1151/" + "T" * 137 + ".dat.zlib"
        archive = self.archive(member)
        seen = []
        real = updater.components.safe_extract

        def spy(archive_, destination, **kw):
            seen.append(str(destination))
            return real(archive_, destination, **kw)

        with mock.patch.object(updater.components, "safe_extract", spy):
            updater.extract(archive, self.deep / "engine" / "x", "tar.gz", 1 << 20)
        self.assertTrue(seen[0].startswith("\\\\?\\"), seen)
        target = longpath.long_path(self.deep / "engine" / "x") / member
        self.assertGreater(len(longpath.plain(target)), LIMIT)
        self.assertEqual(target.read_bytes(), b"engine bytes")

    def test_a_failed_unpack_names_the_file(self):
        archive = self.archive("a/b.dat")
        with mock.patch.object(updater.components, "safe_extract", side_effect=FileNotFoundError(3, "The system cannot find the path", r"C:\n\o.dat")):
            with self.assertRaises(updater.UpdateError) as cm:
                updater.extract(archive, self.deep, "tar.gz", 1 << 20)
        self.assertIn(r"C:\n\o.dat", str(cm.exception))
        self.assertEqual(cm.exception.exit_code, 6)

    def test_verify_reads_files_that_sit_past_the_limit(self):
        from tests.terminal import update_fixture as fx
        root = Path(self.temp.name)
        built = fx.build_package(root / "b", "0.1.45")
        # a folder deep enough that the stub's files are past 259 characters
        base = longpath.long_path(root / ("m" * 110) / ("n" * 110))
        import shutil
        shutil.copytree(longpath.long_path(built), base / built.name)
        component = ec.read_manifest(base / built.name)
        self.assertGreater(len(longpath.plain(base / built.name / component.entrypoint)), 0)
        ec.verify(component, full=True)                                  # raises IntegrityProblem if a file looks missing

    def test_the_probe_runs_and_cleans_up(self):
        longpath.plain_long_paths_work(Path(self.temp.name))
        self.assertEqual([p.name for p in Path(self.temp.name).iterdir() if p.name.startswith(".lp-")], [])


class StagingFolder(unittest.TestCase):
    def test_new_stage_is_short_and_sweeps_old_staging(self):
        with tempfile.TemporaryDirectory(prefix="strata-stage-") as tmp:
            root = Path(tmp)
            old = root / ".updates" / "staging" / "abc"
            old.mkdir(parents=True)
            stale = root / updater.STAGE_DIR / "dead"
            stale.mkdir(parents=True)
            stage = updater.new_stage(root)
            self.assertEqual(stage.parent, root / updater.STAGE_DIR)
            self.assertEqual(len(stage.name), 4)
            self.assertFalse(old.exists() or stale.exists())
            self.assertTrue(updater.new_stage(root, "c").name.startswith("c"))


if __name__ == "__main__":
    unittest.main()
