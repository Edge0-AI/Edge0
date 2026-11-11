"""H100 fresh-install items for DeepSeek in the Terminal: the screens' packages, the Advanced list's own flags, the disk in "fits"."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from strata_cli.launch import cmd, params as P  # noqa: E402
from strata_cli.launch.catalog import TIGHT, verdict  # noqa: E402

CARD = {"index": 0, "name": "RTX 5070", "vram_gb": 12.0, "vendor": "nvidia", "usable": True}


class Screens(unittest.TestCase):
    def test_missing_package_is_named_not_a_traceback(self):
        import importlib.util as u
        real = u.find_spec
        with mock.patch.object(u, "find_spec", lambda n, *a, **k: None if n == "rich" else real(n, *a, **k)):
            self.assertEqual(cmd._screens_missing(), "rich")
        with mock.patch.object(u, "find_spec", lambda n, *a, **k: None if n == "textual" else real(n, *a, **k)):
            self.assertEqual(cmd._screens_missing(), "textual")

    def test_setup_installs_the_screens_packages(self):
        import setup
        names = [r.split("==")[0].split(";")[0].strip() for r in setup.runtime_packages()]
        self.assertIn("textual", names)
        self.assertIn("numpy", names)


class Advanced(unittest.TestCase):
    ARGS = ["--pack", "P", "--max-context", "8192", "--eos", "1", "--dspark", "auto", "--dspark-conf", "0.65"]

    def test_deepseeks_own_flags_are_known_before_the_engine_is_installed(self):
        rows = P.build_rows(P.dsv41_help_text(None), self.ARGS, None, {}, {}, {})
        mine = {r.name: r for r in rows if r.name in self.ARGS}
        self.assertEqual(set(mine), {"--pack", "--max-context", "--eos", "--dspark", "--dspark-conf"})
        for r in mine.values():
            self.assertNotIn("not in this engine's list", r.description)
            self.assertTrue(r.known)

    def test_a_real_typing_mistake_is_still_flagged(self):
        rows = P.build_rows(P.dsv41_help_text(None), self.ARGS + ["--dsparkk", "1"], None, {}, {}, {})
        self.assertIn("not in this engine's list", next(r for r in rows if r.name == "--dsparkk").description)


class Presets(unittest.TestCase):
    def test_a_saved_preset_is_read_from_where_it_is_written_whatever_the_data_dir(self):
        import tempfile
        from strata_cli.launch import settings as S, store
        with tempfile.TemporaryDirectory() as d, mock.patch.dict("os.environ", {"STRATA_LAUNCH_PRESET_DIR": d}):
            s = S.LaunchSettings(family="deepseek41", model="IST-DAS-2.00bpw", data_dir=str(Path(d) / "elsewhere"), name="DS 8K")
            store.save(s, "DS 8K")
            self.assertEqual([p.name for p in store.list_presets()], ["DS 8K"])
            self.assertEqual(store.preset_dir(), Path(d))          # (the folder does not depend on the launch's --data-dir)


class PlanEngine(unittest.TestCase):
    def test_an_installed_deepseek_engine_is_named_in_the_plan_not_downloaded(self):
        import tempfile
        from strata_cli.launch import runner as R
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d) / "engine-dsv41" / "0.1.49-windows-x86_64-cuda-cuda13"
            pkg.mkdir(parents=True)
            exe = pkg / ("strata-dsv41.exe" if sys.platform == "win32" else "strata-dsv41")
            exe.write_bytes(b"x")
            fake = mock.MagicMock()
            fake.s.data_dir = d
            fake.catalog.data_dir.return_value = Path(d)
            fake.catalog.strata.setup_module.side_effect = RuntimeError("no setup")
            with mock.patch.dict("os.environ", {"STRATA_DSV41_EXE": ""}),                     mock.patch("strata_control.engine_component.status", return_value={"usable": False}):
                found, known = R.Runner.dsv41_engine_found(fake)
            self.assertEqual(found, exe)
            self.assertTrue(known)


class Disk(unittest.TestCase):
    def setUp(self):
        import strata_mcp as M
        self.s = M.Strata(ROOT)

    def hw(self, disk):
        return {"ram_gb": 64.0, "gpus": [CARD], "engine_vendor": "nvidia", "disk_free_gb": disk}

    def test_fits_in_memory_but_not_on_the_disk_is_tight_with_the_numbers(self):
        w = self.s.dsv41_words(self.hw(37.0), "IST-DAS-2.25bpw")
        self.assertTrue(w.startswith("tight"), w)
        self.assertIn("free disk space", w)
        self.assertIn("37 GB", w)
        self.assertEqual(verdict(w), (TIGHT, "tight: needs more disk space"))

    def test_enough_disk_or_already_installed_is_unchanged(self):
        self.assertNotIn("free disk space", self.s.dsv41_words(self.hw(900.0), "IST-DAS-2.25bpw"))
        self.assertNotIn("free disk space", self.s.dsv41_words(self.hw(37.0), "IST-DAS-2.25bpw", installed=True))
        self.assertNotIn("free disk space", self.s.dsv41_words({**self.hw(None)}, "IST-DAS-2.25bpw"))


if __name__ == "__main__":
    unittest.main()


class FinalHint(unittest.TestCase):
    def test_no_preset_means_the_plain_stop_command_and_a_way_to_keep_the_settings(self):
        from strata_cli.launch import settings as S
        s = S.LaunchSettings(name="IST-DAS-2.00bpw 8K")
        lines = cmd.final_hint_lines(s, False)
        text = "\n".join(lines)
        self.assertIn("strata launch --stop", text)
        self.assertNotIn("--preset", text.replace("--save-preset", ""))
        self.assertIn("--save-preset NAME", text)

    def test_a_saved_preset_keeps_its_name_in_the_hint(self):
        from strata_cli.launch import settings as S
        s = S.LaunchSettings(name="Office")
        self.assertIn('--stop --preset "Office"', "\n".join(cmd.final_hint_lines(s, True)))


class OwnEngine(unittest.TestCase):
    """The run config of an installed DeepSeek model may name another install's (or an older) engine folder: the launch uses this install's package."""

    def fake(self, own_root, found):
        from strata_cli.launch import runner as R
        f = mock.MagicMock()
        f.notices = []
        f.dsv41_engine_found.return_value = (found, True)
        f.upgrade_dsv41_engine.side_effect = lambda path: path          # (the upgrade has its own tests: tests/terminal/test_dsv41_upgrade.py)
        return R.Runner.use_own_dsv41_engine, f

    def test_a_stale_path_in_the_config_is_replaced_by_this_installs_package(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            own = Path(d) / "mine" / "engine-dsv41" / "0.1.49-x" / "strata-dsv41.exe"
            own.parent.mkdir(parents=True)
            own.write_bytes(b"x")
            fn, f = self.fake(d, own)
            cfg = {"arch": "deepseek41", "exe": str(Path(d) / "other" / "engine-dsv41" / "0.1.48-x" / "strata-dsv41.exe"),
                   "lib_dirs": [str(Path(d) / "other" / "engine-dsv41" / "0.1.48-x"), "C:/cuda"]}
            fn(f, cfg)
            self.assertEqual(cfg["exe"], str(own))
            self.assertEqual(cfg["lib_dirs"], ["C:/cuda", str(own.parent)])
            self.assertTrue(f.notices)

    def test_nothing_changes_for_qwen_or_when_this_install_has_no_package(self):
        fn, f = self.fake("x", Path("/p/strata-dsv41.exe"))
        cfg = {"arch": "qwen4exp", "exe": "component:strata-engine", "lib_dirs": ["a"]}
        before = dict(cfg)
        fn(f, cfg)
        self.assertEqual(cfg, before)
        fn2, f2 = self.fake("x", None)
        cfg2 = {"arch": "deepseek41", "exe": "/old/strata-dsv41.exe", "lib_dirs": []}
        fn2(f2, cfg2)
        self.assertEqual(cfg2["exe"], "/old/strata-dsv41.exe")


class MissingEngine(unittest.TestCase):
    """This install has no DeepSeek engine package and the model config's exe is gone: not 'already installed'; setup decides (feed or a plain line)."""

    def check(self, cfg, found):
        from strata_cli.launch import runner as R
        f = mock.MagicMock()
        f.dsv41_engine_found.return_value = (found, True)
        return R.Runner.dsv41_engine_missing(f, cfg)

    def test_gone_exe_and_no_package_means_setup_runs(self):
        self.assertTrue(self.check({"arch": "deepseek41", "exe": "C:/gone/engine-dsv41/0.1.48/strata-dsv41.exe"}, None))
        self.assertTrue(self.check({"arch": "deepseek41"}, None))

    def test_an_existing_exe_or_an_own_package_or_qwen_is_left_alone(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "strata-dsv41.exe"
            exe.write_bytes(b"x")
            self.assertFalse(self.check({"arch": "deepseek41", "exe": str(exe)}, None))
            self.assertFalse(self.check({"arch": "deepseek41", "exe": "C:/gone/x.exe"}, exe))
        self.assertFalse(self.check({"arch": "qwen4exp", "exe": "component:strata-engine"}, None))


class WhichInstallsEngine(unittest.TestCase):
    """The DeepSeek engine package that runs: the newest of the running Terminal's own install and the install the model's settings came from."""

    def runner(self, own, root):
        f = mock.MagicMock()
        f.catalog.src = Path(own)
        f.catalog.root = Path(root)
        f.catalog.strata.setup_module.side_effect = RuntimeError("none")
        f.catalog.data_dir.return_value = Path(root) / "data"
        f.s.data_dir = None
        return f

    def make(self, base, version):
        d = Path(base) / "engine-dsv41" / f"{version}-windows-x86_64-cuda-cuda13"
        d.mkdir(parents=True)
        exe = d / ("strata-dsv41.exe" if sys.platform == "win32" else "strata-dsv41")
        exe.write_bytes(b"x")
        return exe

    def found(self, own, root):
        from strata_cli.launch import runner as R
        f = self.runner(own, root)
        with mock.patch.dict("os.environ", {"STRATA_DSV41_EXE": "", "STRATA_INSTALL_DIR": ""}), \
                mock.patch("strata_control.engine_component.status", return_value={"usable": False}):
            out = R.Runner.dsv41_engine_found(f)
        return out[0], getattr(f, "dsv41_choice", None)

    def test_the_running_terminals_newer_package_beats_an_older_first_install(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            old = self.make(Path(d) / "inst5", "0.1.48")
            new = self.make(Path(d) / "inst8", "0.1.49")
            exe, choice = self.found(Path(d) / "inst8", Path(d) / "inst5")
            self.assertEqual(exe, new)
            self.assertIn("own folder", choice[1])
            self.assertIn("newest of 2", choice[1])

    def test_an_older_package_in_the_running_terminal_does_not_beat_a_newer_one_in_the_settings_install(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self.make(Path(d) / "inst8", "0.1.48")
            new = self.make(Path(d) / "inst5", "0.1.49")
            exe, _ = self.found(Path(d) / "inst8", Path(d) / "inst5")
            self.assertEqual(exe, new)

    def test_the_same_version_in_both_means_the_running_terminals_own(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self.make(Path(d) / "inst5", "0.1.49")
            own = self.make(Path(d) / "inst8", "0.1.49")
            exe, choice = self.found(Path(d) / "inst8", Path(d) / "inst5")
            self.assertEqual(exe, own)
            self.assertIn("own folder", choice[1])

    def test_only_the_settings_install_has_one(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            only = self.make(Path(d) / "inst5", "0.1.49")
            (Path(d) / "inst8").mkdir()
            exe, choice = self.found(Path(d) / "inst8", Path(d) / "inst5")
            self.assertEqual(exe, only)
            self.assertIn("settings came from", choice[1])

    def test_versions_compare_as_numbers(self):
        from strata_control import dsv41_engine
        self.assertGreater(dsv41_engine.version_key("0.1.49-x"), dsv41_engine.version_key("0.1.9-x"))
        self.assertGreater(dsv41_engine.version_key("0.1.49.1-x"), dsv41_engine.version_key("0.1.49-x"))


class DiskCountsOnlyTheRest(unittest.TestCase):
    """The menu's disk check asks only for what is still to be downloaded: files already in the models folder count as done (P2003)."""

    def setUp(self):
        import strata_mcp as M
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = M.Strata(ROOT)
        self.s.data_dir = lambda: Path(self.tmp.name)
        self.hw = {"ram_gb": 64.0, "gpus": [CARD], "engine_vendor": "nvidia", "disk_free_gb": 176.0}

    def put_files(self, model="IST-DAS-2.00bpw", legacy=False):
        S = self.s.setup_module()
        fam = S.MULTI_FILE_FAMILIES["deepseek41"]
        md = fam["models"][model]
        models = Path(self.tmp.name) / "models"
        sizes = {}
        for where, names in ((models / (fam["tag"] + model), md["shards"]), (models / (fam["tag"] + "shared"), md["shared"])):
            where.mkdir(parents=True, exist_ok=True)
            for n in names:
                name = S.DSV41_LEGACY_NAMES.get(n, n) if legacy and n in S.DSV41_LEGACY_NAMES else n
                real = S.DSV41_ALL_FILES[name][0]
                (where / name).write_bytes(b"x" * 8)
                sizes[name] = (8, S.DSV41_ALL_FILES[name][1])
        return sizes

    def test_with_every_file_in_place_a_small_disk_is_not_tight(self):
        S = self.s.setup_module()
        with mock.patch.dict(S.DSV41_ALL_FILES, self.put_files()):
            words = self.s.dsv41_words(self.hw, "IST-DAS-2.00bpw")
        self.assertNotIn("free disk space", words)

    def test_the_first_releases_names_count_too(self):
        S = self.s.setup_module()
        with mock.patch.dict(S.DSV41_ALL_FILES, self.put_files(legacy=True)):
            self.assertNotIn("free disk space", self.s.dsv41_words(self.hw, "IST-DAS-2.00bpw"))

    def test_with_nothing_in_place_176_gb_is_tight_for_the_257_gb_set(self):
        words = self.s.dsv41_words(self.hw, "IST-DAS-2.00bpw")
        self.assertIn("free disk space", words)
        self.assertTrue(words.startswith("tight"))

    def test_only_part_of_the_files_means_only_the_rest_is_counted(self):
        S = self.s.setup_module()
        sizes = self.put_files()
        first = S.MULTI_FILE_FAMILIES["deepseek41"]["models"]["IST-DAS-2.00bpw"]["shards"][0]
        part = dict(sizes)
        part[first] = (S.DSV41_ALL_FILES[first][0], sizes[first][1])          # shard 1 is not here at its real size: the rest of the files are
        (Path(self.tmp.name) / "models" / ("deepseek-v4.1-flash-IST-DAS-2.00bpw") / first).unlink()
        with mock.patch.dict(S.DSV41_ALL_FILES, part):
            hw = {**self.hw, "disk_free_gb": 46.0}                               # shard 1 is 39.1 GB: 46 GB is enough for it plus the 6 GB of room
            self.assertNotIn("free disk space", self.s.dsv41_words(hw, "IST-DAS-2.00bpw"))
            hw = {**self.hw, "disk_free_gb": 44.0}
            self.assertIn("free disk space", self.s.dsv41_words(hw, "IST-DAS-2.00bpw"))
