"""The hipBLASLt tuning tables in the AMD engine packages: what a package holds per card family, and the one rule that decides at a start
whether a table is used (strata_control/hipblaslt_tuning.py). Folders are throwaways; no GPU tool is run.

The packages' hipBLASLt versions below are the ones the 0.2.x release packages bundle (docs/AMD_HIP.md, Ready-made packages): Windows ROCm
10.2.0a20260930 carries 1.5.0 (100500), the Linux 7.10.0a20251120 wheels 1.2.0 (100200), Linux Strix Halo's 7.14.1 1.4.1 (100401)."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import hipblaslt_tuning as ht

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "tools" / "hip"
sys.path.insert(0, str(ROOT / "packaging"))
import package_engine  # noqa: E402
import stage_tuning  # noqa: E402

#: (platform, variant) -> (archs of the package, hipBLASLt version it bundles, architectures that must carry a table)
PACKAGES = {
    ("windows-x86_64", "rx9070"): (("gfx1200", "gfx1201"), 100500, {"gfx1201"}),
    ("windows-x86_64", "rdna3"): (("gfx1100", "gfx1101", "gfx1102", "gfx1151"), 100500, {"gfx1100", "gfx1151"}),
    ("linux-x86_64", "rx9070"): (("gfx1200", "gfx1201"), 100200, set()),                    # no table for 1.2.0 on RDNA4 (docs/AMD_HIP.md)
    ("linux-x86_64", "rx7900"): (("gfx1100", "gfx1101", "gfx1102"), 100200, {"gfx1100"}),
    ("linux-x86_64", "halo"): (("gfx1151",), 100401, {"gfx1151"}),
    # 0.2.2 RX 6000 packages: hipBLASLt ships no gfx1030 / gfx1031 kernels at any version, so no table, plain hipBLAS
    ("windows-x86_64", "rdna2"): (("gfx1030", "gfx1031"), 100500, set()),
    ("linux-x86_64", "rx6000"): (("gfx1030", "gfx1031"), 100400, set()),
}


ON = {ht.SWITCH: "on"}            # the switch that uses a matching table whatever "auto" does
AUTO = {}


def fake_table(folder: Path, arch: str, version: int, *, header: str | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ht.table_name(arch, version)
    path.write_text(f"# solution ids are scoped to this hipBLASLt version and device architecture\n"
                    f"{header or f'{ht.HEADER} {arch} {version}'}\nbf16 128 2560 128 4096 73754\n", encoding="utf-8")
    return path


def fake_package(root: Path, *, version: int | None = 100500, tables=(("gfx1201", 100500),), build=True) -> Path:
    folder = root / "0.2.1-pkg"
    folder.mkdir(parents=True)
    meta = {"version": "0.2.1", "backend": "hip"}
    if version is not None:
        meta["hipblaslt_version"] = version
    if build:
        (folder / "BUILD.json").write_text(json.dumps(meta), encoding="utf-8")
    for arch, ver in tables:
        fake_table(folder / ht.TABLE_DIR, arch, ver)
    return folder


class TempDir(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)


class Decision(TempDir):
    """Match, mismatch, switch, the person's own choice."""

    def test_a_table_for_this_card_and_this_hipblaslt_is_used(self):
        pkg = fake_package(self.root)
        d = ht.resolve(pkg, ["gfx1201"], environ=ON)
        self.assertEqual(d.path, pkg / "tuning" / "gfx1201-hipblaslt-100500.txt")
        self.assertIn("gfx1201-hipblaslt-100500.txt", d.message)
        self.assertIn(ht.SWITCH, d.message)               # the line tells how to turn it off
        self.assertEqual(d.level, "info")

    def test_a_table_for_another_hipblaslt_version_is_not_used_and_the_log_says_why(self):
        pkg = fake_package(self.root, version=100200, tables=(("gfx1201", 100500),))
        d = ht.resolve(pkg, ["gfx1201"], environ=ON)
        self.assertIsNone(d.path)
        self.assertEqual(d.level, "warn")
        self.assertIn("100200", d.message)
        self.assertIn("100500", d.message)
        self.assertIn("plain hipBLAS", d.message)

    def test_a_card_without_a_table_in_the_package_gets_plain_hipblas_and_one_line(self):
        pkg = fake_package(self.root)
        d = ht.resolve(pkg, ["gfx1200"], environ=ON)
        self.assertIsNone(d.path)
        self.assertIn("none for gfx1200", d.message)

    def test_the_card_that_has_a_table_is_found_among_several(self):
        pkg = fake_package(self.root, tables=(("gfx1100", 100500), ("gfx1151", 100500)))
        self.assertEqual(ht.resolve(pkg, ["gfx1101", "gfx1151"], environ=ON).path.name, "gfx1151-hipblaslt-100500.txt")

    def test_a_package_that_does_not_say_its_hipblaslt_is_not_guessed_at(self):
        pkg = fake_package(self.root, version=None)
        d = ht.resolve(pkg, ["gfx1201"], environ=ON)
        self.assertIsNone(d.path)
        self.assertEqual(d.level, "warn")
        self.assertIn("hipblaslt_version", d.message)

    def test_an_unreadable_card_is_a_note_not_a_guess(self):
        d = ht.resolve(fake_package(self.root), [], environ=ON)
        self.assertIsNone(d.path)
        self.assertIn("could not be read", d.message)

    def test_a_table_whose_header_disagrees_with_its_name_is_not_used(self):
        pkg = fake_package(self.root, tables=())
        fake_table(pkg / "tuning", "gfx1201", 100500, header=f"{ht.HEADER} gfx1201 100401")
        d = ht.resolve(pkg, ["gfx1201"], environ=ON)
        self.assertIsNone(d.path)
        self.assertIn("header", d.message)

    def test_the_switch_turns_it_off_in_the_environment_and_in_the_run_config(self):
        pkg = fake_package(self.root)
        for word in ("off", "OFF", "0", "no", "false", "none"):
            d = ht.resolve(pkg, ["gfx1201"], environ={ht.SWITCH: word})
            self.assertIsNone(d.path, word)
            self.assertIn("off", d.message)
        self.assertIsNone(ht.resolve(pkg, ["gfx1201"], environ=AUTO, cfg={"hipblaslt_table": False}).path)
        self.assertIsNotNone(ht.resolve(pkg, ["gfx1201"], environ=AUTO, cfg={"hipblaslt_table": True}).path)
        self.assertIsNone(ht.resolve(pkg, ["gfx1201"], environ={ht.SWITCH: "off"}, cfg={"hipblaslt_table": True}).path)   # the environment wins
        for word in ("on", "ON", "1", "yes", "true"):
            self.assertIsNotNone(ht.resolve(pkg, ["gfx1201"], environ={ht.SWITCH: word}).path, word)

    def test_auto_follows_the_one_constant_and_says_what_is_available(self):
        pkg = fake_package(self.root)
        with mock.patch.object(ht, "AUTO_USES_TABLE", False):
            d = ht.resolve(pkg, ["gfx1201"], environ=AUTO)
            self.assertIsNone(d.path)
            self.assertIn("gfx1201-hipblaslt-100500.txt", d.message)
            self.assertIn(f"{ht.SWITCH}=on", d.message)
            self.assertEqual(ht.mode(AUTO), "auto")
        with mock.patch.object(ht, "AUTO_USES_TABLE", True):
            d = ht.resolve(pkg, ["gfx1201"], environ=AUTO)
            self.assertEqual(d.path, pkg / "tuning" / "gfx1201-hipblaslt-100500.txt")
            self.assertIsNone(ht.resolve(pkg, ["gfx1201"], environ={ht.SWITCH: "off"}).path)

    def test_the_default_is_the_owners_decision(self):
        """A table changes the bits of the prompt's dense GEMMs (notes-021-hip.md); the owner chose: on by default for Strix Halo only."""
        self.assertFalse(ht.AUTO_USES_TABLE)
        self.assertEqual(ht.AUTO_ARCHS, ("gfx1151",))

    def test_auto_uses_the_table_on_strix_halo_only(self):
        for arch, used in (("gfx1151", True), ("gfx1201", False), ("gfx1100", False)):
            root = self.root / arch
            root.mkdir()
            pkg = fake_package(root, tables=((arch, 100500),))
            d = ht.resolve(pkg, [arch], environ=AUTO)
            self.assertEqual(d.path is not None, used, (arch, d.message))
            self.assertIsNone(ht.resolve(pkg, [arch], environ={ht.SWITCH: "off"}).path)

    def test_a_table_the_person_set_is_never_replaced(self):
        pkg = fake_package(self.root)
        mine = str(self.root / "mine.txt")
        for environ, cfg in (({ht.ENGINE_VAR: mine}, {}), ({}, {"env": {ht.ENGINE_VAR: mine}})):
            d = ht.resolve(pkg, ["gfx1201"], environ=environ, cfg=cfg)
            self.assertIsNone(d.path)
            self.assertIn(mine, d.message)
            out = ht.apply({"backend": "hip", **cfg}, pkg, ["gfx1201"], environ=environ, say=lambda line: None)
            self.assertEqual((out.get("env") or {}).get(ht.ENGINE_VAR, mine), mine)


class Apply(TempDir):
    def test_apply_puts_the_table_in_a_copy_of_the_config_and_says_so_once(self):
        pkg = fake_package(self.root)
        cfg = {"backend": "hip", "env": {"STRATA_RESIDENT_PIN": "0"}}
        lines = []
        out = ht.apply(cfg, pkg, ["gfx1201"], environ=ON, say=lines.append)
        self.assertEqual(out["env"][ht.ENGINE_VAR], str(pkg / "tuning" / "gfx1201-hipblaslt-100500.txt"))
        self.assertEqual(out["env"]["STRATA_RESIDENT_PIN"], "0")
        self.assertNotIn(ht.ENGINE_VAR, cfg["env"])        # the caller's config is untouched
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("[strata] hipBLASLt tuning table"))

    def test_apply_with_a_mismatch_leaves_the_config_alone_and_warns(self):
        pkg = fake_package(self.root, version=100200)
        cfg = {"backend": "hip"}
        lines = []
        self.assertIs(ht.apply(cfg, pkg, ["gfx1201"], environ=ON, say=lines.append), cfg)
        self.assertEqual(len(lines), 1)
        self.assertIn("note", lines[0])

    def test_other_backends_and_packages_without_tables_are_silent(self):
        lines = []
        pkg = fake_package(self.root)
        cfg = {"backend": "cuda"}
        self.assertIs(ht.apply(cfg, pkg, ["gfx1201"], environ=ON, say=lines.append), cfg)
        bare = self.root / "cuda-pkg"
        bare.mkdir()
        (bare / "BUILD.json").write_text(json.dumps({"version": "0.2.1"}), encoding="utf-8")
        out = ht.apply({"backend": "hip"}, bare, ["gfx1201"], environ=ON, say=lines.append)
        self.assertNotIn("env", out)
        self.assertEqual(lines, [])

    def test_the_server_start_applies_it_for_a_component_engine(self):
        from serve import server
        pkg = fake_package(self.root)
        (pkg / "strata").write_text("x", encoding="utf-8")
        cfg = {"exe": "component:strata-engine", "backend": "hip"}
        out = io.StringIO()
        with mock.patch.object(server.engine_component, "expand", return_value=str(pkg / "strata")), \
                mock.patch.object(ht, "present_archs", return_value=["gfx1201"]), \
                mock.patch.dict(os.environ, {}, clear=False), contextlib.redirect_stdout(out):
            os.environ.pop(ht.SWITCH, None)
            os.environ.pop(ht.ENGINE_VAR, None)
            self.assertNotIn(ht.ENGINE_VAR, server.child_env(server.with_component_engine(cfg)))     # "auto" is off by default
            os.environ[ht.SWITCH] = "on"
            started = server.with_component_engine(cfg)
            env = server.child_env(started)
            self.assertEqual(env[ht.ENGINE_VAR], str(pkg / "tuning" / "gfx1201-hipblaslt-100500.txt"))
            os.environ[ht.SWITCH] = "off"
            self.assertNotIn(ht.ENGINE_VAR, server.child_env(server.with_component_engine(cfg)))
        self.assertIn("hipBLASLt tuning table", out.getvalue())

    def test_present_archs_picks_the_configured_card_on_linux(self):
        with mock.patch("strata_control.engine_component.hip_archs_present", return_value=["gfx1100", "gfx1201"]):
            if os.name != "nt":
                self.assertEqual(ht.present_archs({"gpu": 1}), ["gfx1201"])
                self.assertEqual(ht.present_archs({"gpu": [1, 0]}), ["gfx1201", "gfx1100"])
            self.assertEqual(ht.present_archs({}), ["gfx1100", "gfx1201"])


class PackageContents(TempDir):
    """What each package of the release matrix carries, from the real tables of tools/hip."""

    def test_every_release_package_carries_the_tables_its_cards_have(self):
        for (platform, variant), (archs, version, wanted) in PACKAGES.items():
            stage = self.root / f"{platform}-{variant}"
            stage.mkdir()
            copied = ht.stage_tables(stage, archs, version, TABLES)
            self.assertEqual({n.split("-")[0] for n in copied}, wanted, (platform, variant))
            on_disk = {p.name for p in (stage / "tuning").glob("*.txt")} if (stage / "tuning").is_dir() else set()
            self.assertEqual(on_disk, set(copied), (platform, variant))
            for arch in wanted:                           # the file the engine start will pick is there and its own header agrees
                self.assertTrue(ht.header_ok(stage / "tuning" / ht.table_name(arch, version), arch, version), (platform, arch))

    def test_the_missing_report_names_the_architectures_without_a_table(self):
        left = ht.missing_report(("gfx1100", "gfx1101", "gfx1102"), 100200, TABLES)
        self.assertEqual(sorted(left), ["gfx1101", "gfx1102"])
        self.assertEqual(ht.missing_report(("gfx1201",), 100200, TABLES), {"gfx1201": [100202, 100500]})

    def test_every_table_in_the_repo_has_a_header_that_matches_its_name(self):
        tables = ht.tables_in(TABLES)
        self.assertGreater(len(tables), 8)
        for (arch, version), path in tables.items():
            self.assertTrue(ht.header_ok(path, arch, version), path.name)

    def test_version_from_header_reads_the_rocm_tree(self):
        inc = self.root / "include" / "hipblaslt"
        inc.mkdir(parents=True)
        (inc / "hipblaslt-version.h").write_text("#define HIPBLASLT_VERSION_MAJOR 1\n#define HIPBLASLT_VERSION_MINOR 5\n"
                                                 "#define HIPBLASLT_VERSION_PATCH 0\n", encoding="utf-8")
        self.assertEqual(ht.version_from_header(self.root), 100500)
        self.assertIsNone(ht.version_from_header(self.root / "nowhere"))

    def test_stage_tuning_writes_the_version_and_the_tables(self):
        stage = self.root / "stage"
        stage.mkdir()
        (stage / "BUILD.json").write_text(json.dumps({"version": "0.2.1", "archs": ["gfx1151"]}), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(stage_tuning.main(["--stage", str(stage), "--archs", "gfx1151", "--version", "100401"]), 0)
        meta = json.loads((stage / "BUILD.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["hipblaslt_version"], 100401)
        self.assertEqual(meta["archs"], ["gfx1151"])      # what was there stays
        self.assertTrue((stage / "tuning" / "gfx1151-hipblaslt-100401.txt").is_file())

    def test_stage_tuning_can_demand_a_table(self):
        stage = self.root / "stage"
        stage.mkdir()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as caught:
            stage_tuning.main(["--stage", str(stage), "--archs", "gfx1201", "--version", "100200", "--expect", "gfx1201"])
        self.assertIn("gfx1201", str(caught.exception))

    def test_stage_tuning_refuses_a_rocm_whose_version_it_cannot_read(self):
        stage = self.root / "stage"
        stage.mkdir()
        with self.assertRaises(SystemExit):
            stage_tuning.main(["--stage", str(stage), "--archs", "gfx1201", "--rocm", str(self.root / "no-rocm")])


class PackagedAndHashed(TempDir):
    """packaging/package_engine.py: the tables are files of the package (hashed, so signed with it) and a bad AMD stage is refused."""

    def stage(self, **kw):
        src = self.root / "src"
        src.mkdir()
        (src / "strata").write_bytes(b"engine" * 100)
        (src / "libamdhip64.so.7").write_bytes(b"hip" * 100)
        (src / "BUILD.json").write_text(json.dumps({"version": "0.2.1", **kw}), encoding="utf-8")
        return src

    def build(self, src, archs=("gfx1201",), tree=False):
        out = self.root / "out"
        out.mkdir(exist_ok=True)
        with mock.patch.object(package_engine, "_bundle_missing_libraries", return_value=[]):
            return package_engine.build(src, "hip", "0.2.1", "linux-x86_64", out, entry="strata", archs=archs, build_id="t", tree=tree,
                                        variant="rx9070")

    def test_the_tables_are_listed_and_hashed_in_the_manifest_even_without_tree(self):
        src = self.stage(hipblaslt_version=100500)
        fake_table(src / "tuning", "gfx1201", 100500)
        folder = self.build(src)
        listed = {f["path"]: f for f in json.loads((folder / "engine-component.json").read_text(encoding="utf-8"))["files"]}
        self.assertIn("tuning/gfx1201-hipblaslt-100500.txt", listed)
        self.assertEqual(len(listed["tuning/gfx1201-hipblaslt-100500.txt"]["sha256"]), 64)
        (folder / "tuning" / "gfx1201-hipblaslt-100500.txt").write_text("changed", encoding="utf-8")   # a swapped table fails the start's hash check
        from strata_control import engine_component as ec
        with self.assertRaises(ec.ComponentProblem):
            ec.verify(ec.read_manifest(folder), full=True)

    def test_an_amd_stage_without_a_hipblaslt_version_is_refused(self):
        src = self.stage()
        with self.assertRaises(SystemExit) as caught:
            self.build(src)
        self.assertIn("hipblaslt_version", str(caught.exception))

    def test_a_table_for_another_version_or_card_than_the_package_is_refused(self):
        src = self.stage(hipblaslt_version=100500)
        fake_table(src / "tuning", "gfx1201", 100202)
        with self.assertRaises(SystemExit) as caught:
            self.build(src)
        self.assertIn("100202", str(caught.exception))
        other = self.root / "other"
        other.mkdir()
        src2 = other / "src"
        src2.mkdir()
        (src2 / "strata").write_bytes(b"e")
        (src2 / "BUILD.json").write_text(json.dumps({"hipblaslt_version": 100500}), encoding="utf-8")
        fake_table(src2 / "tuning", "gfx1100", 100500)
        with mock.patch.object(package_engine, "_bundle_missing_libraries", return_value=[]), self.assertRaises(SystemExit) as caught:
            package_engine.build(src2, "hip", "0.2.1", "linux-x86_64", other / "out", entry="strata", archs=("gfx1201",), build_id="t")
        self.assertIn("gfx1100", str(caught.exception))

    def test_a_cuda_package_needs_no_hipblaslt_version(self):
        src = self.stage()
        out = self.root / "out"
        out.mkdir()
        with mock.patch.object(package_engine, "_bundle_missing_libraries", return_value=[]):
            package_engine.build(src, "cuda", "0.2.1", "linux-x86_64", out, entry="strata", archs=("sm_86",), build_id="t")


class SetupFindsThePackageTable(TempDir):
    """setup.py outside the app (a git checkout, or the old Windows zip that START-HERE unpacks): the engine folder's own tuning/ comes first."""

    def load_setup(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup_hipblaslt_copy", ROOT / "setup.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_the_engine_folders_table_wins_over_tools_hip_and_the_switch_turns_it_off(self):
        setup = self.load_setup()
        pkg = self.root / "engine"
        fake_table(pkg / "tuning", "gfx1201", 100500)
        seen = []
        with mock.patch.object(setup, "ok", seen.append), mock.patch.object(setup, "say", seen.append), \
                mock.patch.object(setup, "warn", seen.append), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ht.SWITCH, None)
            self.assertEqual(setup.hipblaslt_table("gfx1201", [], 100500, [pkg / "tuning"]), pkg / "tuning" / "gfx1201-hipblaslt-100500.txt")
            self.assertEqual(setup.hipblaslt_table("gfx1201", [], 100500), TABLES / "gfx1201-hipblaslt-100500.txt")
            self.assertIsNone(setup.hipblaslt_table("gfx1201", [], 100200, [pkg / "tuning"]))      # no table for 1.2.0: warned, plain
            os.environ[ht.SWITCH] = "off"
            self.assertIsNone(setup.hipblaslt_table("gfx1201", [], 100500, [pkg / "tuning"]))
        self.assertTrue(any("no hipBLASLt tuning table" in str(line) for line in seen))


if __name__ == "__main__":
    unittest.main()
