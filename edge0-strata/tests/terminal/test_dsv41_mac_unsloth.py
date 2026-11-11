"""Unsloth's DeepSeek sets on a Mac: UD-IQ1_S runs on the Metal engine (IQ1_S kernels, run on an M5 Max) and is offered like the IST-DAS sets;
UD-IQ1_M was never run on a Mac, so it is not offered there (model list, `--model`, `--model-path`, the app's "I already have the model"
check, setup), and the line says what does work.
On every other computer nothing changes."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.catalog import Catalog
from strata_cli.launch.runner import LaunchError, Runner
from strata_control import dsv41_fit, dsv41_sets, models
from tests.terminal.launch_support import HW_12, Home, catalog

LINE = "Unsloth's UD-IQ1_M set doesn't run on a Mac yet; the UD-IQ1_S, 2.00 and 2.25 sets do."
MAC = {**HW_12, "mac": True, "ram_gb": 128.0, "disk_free_gb": 900.0, "gpus": [], "engine_vendor": "apple"}
PC = {**HW_12, "ram_gb": 64.0, "disk_free_gb": 900.0, "engine_vendor": "nvidia"}
UD = ("UD-IQ1_M",)          # the set the gate holds back in these tests (MAC_NOT_YET is empty in the product: the owner offers UD-IQ1_M on a Mac, not run there)
RUNS = "UD-IQ1_S"            # the one that was run there
BOTH = ("UD-IQ1_S", "UD-IQ1_M")   # both ran on a DGX Spark (10-30): AARCH64_NOT_YET is empty; SparkGate patches it back to test the mechanism


def hidden(fn):
    """Run a test with the Mac gate as it was while UD-IQ1_M was kept off a Mac (MAC_NOT_YET = UD, and no Mac fit words for it)."""
    import functools

    @functools.wraps(fn)
    def run(*a, **k):
        pack = {m: v for m, v in dsv41_fit.MAC_SET_PACK.items() if m not in UD}
        with mock.patch.object(dsv41_fit, "MAC_NOT_YET", UD), mock.patch.object(dsv41_fit, "MAC_SET_PACK", pack):
            return fn(*a, **k)
    return run


def ud_files(model):
    return {**dsv41_sets.tables()["sets"][model]["shards"], **dsv41_sets.tables()["sets"][model]["engram_alt"]}


class Line(unittest.TestCase):
    @hidden
    def test_the_line_and_the_names(self):
        self.assertEqual(dsv41_fit.MAC_NOT_YET, UD)
        self.assertEqual(dsv41_fit.MAC_NOT_YET_LINE, LINE)
        self.assertFalse(set(UD) & set(dsv41_fit.MAC_SET_PACK))


class Lists(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.hw = dict(MAC)
        p = mock.patch.object(Catalog, "hardware", lambda s, fresh=False: dict(self.hw))
        p.start()
        self.addCleanup(p.stop)

    def models(self):
        return [r.model for r in catalog(self.hw).rows() if r.family == "deepseek41"]

    @hidden
    def test_a_mac_is_not_offered_the_unsloth_sets(self):
        got = self.models()
        for m in UD:
            self.assertNotIn(m, got)
        self.assertFalse(set(got) - set(dsv41_fit.MAC_SET_PACK), got)

    def test_a_mac_is_offered_ud_iq1_s(self):
        self.assertIn(RUNS, self.models())
        self.assertEqual(catalog(self.hw).resolve("ud-iq1_s"), ("deepseek41", RUNS))
        plan = dsv41_fit.plan([{"vendor": "apple", "unified_gb": 128.0, "name": "Apple M5 Max"}], 128.0, 900.0, model=RUNS)
        self.assertTrue(plan.get("mac"))
        self.assertIn("UD-IQ1_S", plan["words"])

    def test_a_pc_is_offered_them(self):
        self.hw = dict(PC)
        got = self.models()
        for m in UD + (RUNS,):
            self.assertIn(m, got)

    @hidden
    def test_model_flag_on_a_mac_says_what_works(self):
        with self.assertRaises(ValueError) as cm:
            catalog(self.hw).resolve("ud-iq1_m", None)
        self.assertIn(LINE, str(cm.exception))
        self.assertIn("ist-das-2.25bpw", str(cm.exception))
        self.assertEqual(catalog(self.hw).resolve("ist-das-2.25bpw"), ("deepseek41", "IST-DAS-2.25bpw"))
        self.assertEqual(catalog(self.hw).resolve("ud-iq1_s"), ("deepseek41", RUNS))

    def test_model_flag_on_a_pc_is_unchanged(self):
        self.hw = dict(PC)
        self.assertEqual(catalog(self.hw).resolve("ud-iq1_s"), ("deepseek41", "UD-IQ1_S"))


class OwnFiles(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.hw = dict(MAC)
        p = mock.patch.object(Catalog, "hardware", lambda s, fresh=False: dict(self.hw))
        p.start()
        self.addCleanup(p.stop)

    def folder(self, model):
        folder = Path(os.environ["STRATA_INSTALL_DIR"]) / model
        folder.mkdir(parents=True, exist_ok=True)
        for name in dsv41_sets.tables()["sets"][model]["shards"]:
            (folder / name).write_bytes(b"GGUF")
        return folder

    @hidden
    def test_model_path_of_a_ud_folder_on_a_mac_stops_with_the_line(self):
        for model in UD:
            folder = self.folder(model)
            with mock.patch.object(sys, "platform", "darwin"):
                with self.assertRaises(LaunchError) as cm:
                    Runner(S.LaunchSettings(model_path=str(folder)), catalog=catalog(self.hw)).check_own_files()
            self.assertIn(LINE, cm.exception.text)
            self.assertIn("2.25", cm.exception.hint if hasattr(cm.exception, "hint") else cm.exception.text)

    def test_model_path_of_the_ud_iq1_s_folder_on_a_mac_is_the_model(self):
        folder = self.folder(RUNS)
        with mock.patch.object(sys, "platform", "darwin"):
            r = Runner(S.LaunchSettings(model_path=str(folder)), catalog=catalog(self.hw))
            r.check_own_files()
        self.assertEqual((r.s.family, r.s.model), ("deepseek41", RUNS))

    def test_the_same_folder_on_another_computer_is_the_model(self):
        folder = self.folder("UD-IQ1_S")
        with mock.patch.object(sys, "platform", "linux"):
            r = Runner(S.LaunchSettings(model_path=str(folder)), catalog=catalog(self.hw))
            r.check_own_files()
        self.assertEqual((r.s.family, r.s.model), ("deepseek41", "UD-IQ1_S"))


class OfferedOnAMac(unittest.TestCase):
    """The product state: the Mac gate is empty, both Unsloth sets are offered; UD-IQ1_M says plainly that it was not run on a Mac."""
    PLAIN = "Not run on a Mac yet; UD-IQ1_S was. It keeps fewer experts in memory, so expect it to be slower."

    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.hw = dict(MAC)
        p = mock.patch.object(Catalog, "hardware", lambda s, fresh=False: dict(self.hw))
        p.start()
        self.addCleanup(p.stop)

    def test_the_gate_is_empty_and_both_sets_have_a_pack(self):
        self.assertEqual(dsv41_fit.MAC_NOT_YET, ())
        for m in ("UD-IQ1_S", "UD-IQ1_M"):
            self.assertIn(m, dsv41_fit.MAC_SET_PACK)
            self.assertIsNone(dsv41_fit.not_here(m))

    def test_the_list_and_the_flag(self):
        got = [r.model for r in catalog(self.hw).rows() if r.family == "deepseek41"]
        for m in ("UD-IQ1_S", "UD-IQ1_M"):
            self.assertIn(m, got)
        self.assertEqual(catalog(self.hw).resolve("ud-iq1_m"), ("deepseek41", "UD-IQ1_M"))

    def test_the_plan_of_ud_iq1_m_says_it_was_not_run_and_invents_no_numbers(self):
        plan = dsv41_fit.plan([{"vendor": "apple", "unified_gb": 128.0, "name": "Apple M5 Max"}], 128.0, 900.0, model="UD-IQ1_M")
        self.assertTrue(plan.get("mac"))
        self.assertIn(self.PLAIN, plan["words"])
        self.assertEqual(plan["measured"], self.PLAIN)
        self.assertNotIn("tokens/s", plan["words"])
        self.assertAlmostEqual(plan["download_gb"], 258.0, delta=1.0)        # with the drafter and the Engram
        s = dsv41_fit.plan([{"vendor": "apple", "unified_gb": 128.0, "name": "Apple M5 Max"}], 128.0, 900.0, model="UD-IQ1_S")
        self.assertIn("tokens/s", s["words"])

    def test_own_files_and_the_preview_on_a_mac(self):
        folder = Path(os.environ["STRATA_INSTALL_DIR"]) / "UD-IQ1_M"
        folder.mkdir(parents=True, exist_ok=True)
        for name in dsv41_sets.tables()["sets"]["UD-IQ1_M"]["shards"]:
            (folder / name).write_bytes(b"GGUF")
        with mock.patch.object(sys, "platform", "darwin"):
            r = Runner(S.LaunchSettings(model_path=str(folder)), catalog=catalog(self.hw))
            r.check_own_files()
            self.assertEqual((r.s.family, r.s.model), ("deepseek41", "UD-IQ1_M"))
        with mock.patch.object(models, "_dsv41_files", return_value=ud_files("UD-IQ1_M")), mock.patch.object(sys, "platform", "darwin"),                 __import__("tempfile").TemporaryDirectory() as tmp:
            plan = models.plan_import(tmp)
        self.assertEqual(plan["file_set_status"], "plausible", plan["issues"])
        self.assertEqual(plan["candidate_selection"]["model"], "UD-IQ1_M")


class AppCheck(unittest.TestCase):
    def preview(self, model, platform):
        with mock.patch.object(models, "_dsv41_files", return_value=ud_files(model)), mock.patch.object(sys, "platform", platform), \
                __import__("tempfile").TemporaryDirectory() as tmp:
            return models.plan_import(tmp)

    @hidden
    def test_on_a_mac_the_preview_says_so_and_selects_nothing(self):
        for model in UD:
            plan = self.preview(model, "darwin")
            self.assertIn(LINE, plan["messages"])
            self.assertIn("not_on_this_mac", plan["issues"])
            self.assertEqual(plan["file_set_status"], "invalid_or_incomplete")
            self.assertIsNone(plan["candidate_selection"])
            self.assertIn(LINE.rstrip("."), plan["plain_issues"])

    def test_on_a_mac_the_ud_iq1_s_preview_is_the_set(self):
        plan = self.preview(RUNS, "darwin")
        self.assertEqual(plan["file_set_status"], "plausible", plan["issues"])
        self.assertEqual(plan["candidate_selection"]["model"], RUNS)
        self.assertNotIn(LINE, plan["messages"])

    def test_on_a_pc_the_preview_is_the_set(self):
        for model in UD + (RUNS,):
            plan = self.preview(model, "linux")
            self.assertEqual(plan["file_set_status"], "plausible", plan["issues"])
            self.assertEqual(plan["candidate_selection"]["model"], model)
            self.assertNotIn(LINE, plan["messages"])

    def test_an_ist_das_set_on_a_mac_is_still_the_set(self):
        t = dsv41_sets.tables()["sets"]["IST-DAS-2.25bpw"]["shards"]
        eng = next(iter(dsv41_sets.tables()["engram"]))
        with mock.patch.object(models, "_dsv41_files", return_value={**t, eng: dsv41_sets.tables()["engram"][eng]}), \
                mock.patch.object(sys, "platform", "darwin"), __import__("tempfile").TemporaryDirectory() as tmp:
            plan = models.plan_import(tmp)
        self.assertEqual(plan["file_set_status"], "plausible", plan["issues"])
        self.assertEqual(plan["candidate_selection"]["model"], "IST-DAS-2.25bpw")


class Setup(unittest.TestCase):
    def runTest(self):                                    # (SparkGate borrows run_setup)
        pass

    """setup.py is the last stop: the line, before an engine, a download or a pack is touched."""

    def run_setup(self, model, platform, machine=None):
        import contextlib
        import importlib.util
        import io
        import types
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("setup_mac_ud", root / "setup.py")
        setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup)
        a = types.SimpleNamespace(model=model, models_dir="/nonexistent/models", yes=True, gguf_dir=None)
        out = io.StringIO()
        with mock.patch.object(sys, "platform", platform), mock.patch("platform.machine", return_value=machine or "x86_64"),                 contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                setup.install_dsv41(a, "deepseek41", {}, False, False, 13, True, 8080, 128.0, False)
            except BaseException as exc:                      # noqa: BLE001 - fail() exits; anything after the check would raise something else
                return type(exc).__name__, out.getvalue()
        return None, out.getvalue()

    @hidden
    def test_a_mac_is_stopped_before_anything_else(self):
        for model in UD:
            exc, text = self.run_setup(model, "darwin")
            self.assertEqual(exc, "SystemExit", (model, text[-400:]))
            self.assertIn(LINE, text)
            self.assertIn("IST-DAS-2.25bpw", text)
            self.assertIn("nothing has been downloaded", text)


SPARK_LINE = "Unsloth's 1-bit DeepSeek sets were not tested on a DGX Spark yet; the 2.00 and 2.25 sets were."


def spark():
    """A DGX Spark: linux-aarch64 (the two patches the one rule reads)."""
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(mock.patch.object(sys, "platform", "linux"))
    stack.enter_context(mock.patch("platform.machine", return_value="aarch64"))
    return stack


class SparkGate(unittest.TestCase):
    """The same gate for the DGX Spark (dsv41_fit.AARCH64_NOT_YET): the sets were never run there. Removing the names from the tuple lifts it."""

    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.hw = {**PC, "engine_vendor": "nvidia"}
        p = mock.patch.object(Catalog, "hardware", lambda s, fresh=False: dict(self.hw))
        p.start()
        self.addCleanup(p.stop)
        q = mock.patch.object(dsv41_fit, "AARCH64_NOT_YET", BOTH)     # the mechanism, as it was while the sets were kept off
        q.start()
        self.addCleanup(q.stop)

    def test_the_rule(self):
        self.assertEqual(dsv41_fit.AARCH64_NOT_YET, BOTH)
        self.assertEqual(dsv41_fit.AARCH64_NOT_YET_LINE, SPARK_LINE)
        with spark():
            for m in BOTH:
                self.assertEqual(dsv41_fit.not_here(m), SPARK_LINE)
            self.assertIsNone(dsv41_fit.not_here("IST-DAS-2.25bpw"))
        with mock.patch.object(sys, "platform", "linux"), mock.patch("platform.machine", return_value="x86_64"):
            self.assertIsNone(dsv41_fit.not_here("UD-IQ1_S"))
        with mock.patch.object(sys, "platform", "darwin"):
            self.assertIsNone(dsv41_fit.not_here("UD-IQ1_S"))
            self.assertIsNone(dsv41_fit.not_here("UD-IQ1_M"))      # offered on a Mac (not run there)

    def test_not_in_the_list_on_a_spark_but_on_a_pc(self):
        with spark():
            got = [r.model for r in catalog(self.hw).rows() if r.family == "deepseek41"]
        for m in BOTH:
            self.assertNotIn(m, got)
        self.assertIn("IST-DAS-2.25bpw", got)
        pc = [r.model for r in catalog(self.hw).rows() if r.family == "deepseek41"]
        for m in BOTH:
            self.assertIn(m, pc)

    def test_model_flag_on_a_spark_says_what_works(self):
        with spark():
            with self.assertRaises(ValueError) as cm:
                catalog(self.hw).resolve("ud-iq1_s", None)
        self.assertIn(SPARK_LINE, str(cm.exception))
        self.assertIn("ist-das-2.25bpw", str(cm.exception))

    def test_model_path_of_a_ud_folder_on_a_spark_stops(self):
        folder = Path(os.environ["STRATA_INSTALL_DIR"]) / "UD-IQ1_M"
        folder.mkdir(parents=True, exist_ok=True)
        for name in dsv41_sets.tables()["sets"]["UD-IQ1_M"]["shards"]:
            (folder / name).write_bytes(b"GGUF")
        with spark():
            with self.assertRaises(LaunchError) as cm:
                Runner(S.LaunchSettings(model_path=str(folder)), catalog=catalog(self.hw)).check_own_files()
        self.assertIn(SPARK_LINE, cm.exception.text)

    def test_app_check_on_a_spark(self):
        import tempfile
        for model in UD:
            with mock.patch.object(models, "_dsv41_files", return_value=ud_files(model)), tempfile.TemporaryDirectory() as tmp, spark():
                plan = models.plan_import(tmp)
            self.assertIn(SPARK_LINE, plan["messages"])
            self.assertIn("not_on_this_spark", plan["issues"])
            self.assertIsNone(plan["candidate_selection"])
            self.assertIn(SPARK_LINE.rstrip("."), plan["plain_issues"])

    def test_setup_stops_on_a_spark(self):
        for model in UD:
            exc, text = Setup().run_setup(model, "linux", machine="aarch64")
            self.assertEqual(exc, "SystemExit", (model, text[-300:]))
            self.assertIn(SPARK_LINE, text)
            self.assertIn("nothing has been downloaded", text)


class SparkOffered(unittest.TestCase):
    """Both Unsloth sets ran on a DGX Spark (2026-10-30): the production rule hides nothing there."""

    def test_nothing_is_hidden_on_a_spark(self):
        self.assertEqual(dsv41_fit.AARCH64_NOT_YET, ())
        with spark():
            for m in BOTH + ("IST-DAS-2.25bpw",):
                self.assertIsNone(dsv41_fit.not_here(m))


if __name__ == "__main__":
    unittest.main()
