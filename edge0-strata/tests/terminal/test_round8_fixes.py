"""Owner test round 8 (7 machines), the Control / TUI / setup fixes: F5, F3, N2 to N9, N11."""

import argparse
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from strata_cli.launch import adopt, library, runner as R
from strata_cli.launch import settings as S
from strata_cli.launch.runner import Runner
from tests.terminal.launch_support import HW_12, Home, catalog


def runner_for(**kw) -> Runner:
    return Runner(S.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768, vision="off", **kw), catalog=catalog(HW_12))


class EngineExitIsExplained(unittest.TestCase):
    """F5 (L3): an Arc engine died with 'No device of requested type available' and the person saw only 'server_exited'."""

    def config_with_log(self, lines):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        log = Path(folder.name) / "engine.log"
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        cfg = Path(folder.name) / "run.json"
        cfg.write_text(json.dumps({"log": str(log)}), encoding="utf-8")
        return cfg, log

    def test_arc_without_the_render_group_names_the_cause_the_next_step_and_the_log(self):
        cfg, log = self.config_with_log(["strata generate: memory guard: limit 61.6 GiB", "strata generate: SYCL init",
                                         "terminate called after throwing an instance of 'sycl::exception'",
                                         "  what():  No device of requested type available. -1 (PI_ERROR_DEVICE_NOT_FOUND)"])
        text, nxt = R.explain_engine_exit({"state": "failed", "error": "server_exited", "engine_exit_code": 134}, cfg)
        self.assertIn("stopped while it was loading the model", text)
        self.assertIn("No device of requested type available", text)
        self.assertNotIn("server_exited", text + nxt)
        self.assertIn("render group", nxt)
        self.assertIn(str(log), nxt)

    def test_a_gpu_runtime_start_failure_gets_a_plain_sentence_and_keeps_the_runtime_text_out_of_it(self):
        cfg, log = self.config_with_log(["strata generate: loading", "terminate called after throwing an instance of 'rocr::AMD::hsa_exception'",
                                         "  what():  "])
        text, nxt = R.explain_engine_exit({"state": "failed", "error": "server_exited", "engine_exit_code": 134}, cfg)
        self.assertIn("stopped while it was loading the model (exit code 134)", text)
        self.assertNotIn("hsa_exception", text + nxt)
        self.assertNotIn("terminate called", text + nxt)
        self.assertIn("graphics card was not ready yet", nxt)
        self.assertIn("wait a few seconds", nxt)
        self.assertIn("restart the PC", nxt)
        self.assertIn(str(log), nxt)
        text, nxt = R.explain_engine_exit({"error": "server_exited", "engine_exit_code": 134,
                                           "log_tail": "terminate called after throwing an instance of 'rocr::AMD::hsa_exception' | what():"}, None)
        self.assertNotIn("hsa_exception", text)
        self.assertIn("graphics card was not ready yet", nxt)

    def test_a_real_cause_in_the_same_log_wins_over_the_gpu_runtime_start_failure(self):
        cfg, log = self.config_with_log(["strata generate: CUDA error: out of memory",
                                         "terminate called after throwing an instance of 'rocr::AMD::hsa_exception'", "  what():  "])
        text, nxt = R.explain_engine_exit({"state": "failed", "error": "server_exited", "engine_exit_code": 134}, cfg)
        self.assertIn("out of memory", text)
        self.assertIn("--context 16k", nxt)
        self.assertNotIn("graphics card was not ready", nxt)
        cfg, _ = self.config_with_log(["Permission denied: /dev/dri/renderD128", "terminate called after throwing an instance of 'rocr::AMD::hsa_exception'"])
        self.assertIn("render group", R.explain_engine_exit({"error": "server_exited", "engine_exit_code": 134}, cfg)[1])
        cfg, _ = self.config_with_log(["hsaKmtOpenKFD ok", "strata generate: CUDA error: out of memory"])
        self.assertNotIn("graphics card was not ready", R.explain_engine_exit({"error": "server_exited"}, cfg)[1])

    def test_out_of_memory_and_old_engine_get_their_own_next_step(self):
        cfg, _ = self.config_with_log(["strata generate: CUDA error: out of memory"])
        self.assertIn("--context 16k", R.explain_engine_exit({"error": "server_exited"}, cfg)[1])
        cfg, _ = self.config_with_log(["strata generate: no kernel for TQK6 on this engine"])
        self.assertIn("strata update", R.explain_engine_exit({"error": "server_exited"}, cfg)[1])

    def test_an_unreadable_log_still_says_the_stage_and_where_to_look(self):
        text, nxt = R.explain_engine_exit({"state": "failed", "error": "server_exited", "log_tail": "boom | Error: cannot map the pack"}, None)
        self.assertIn("cannot map the pack", text)
        self.assertIn("strata logs", nxt)
        text, _ = R.explain_engine_exit({"state": "starting", "error": "startup_wait_expired"}, None)
        self.assertIn("did not finish loading in time", text)


class NoTuiNeverAsks(unittest.TestCase):
    def test_no_tui_keeps_the_models_and_says_so_without_input(self):
        with tempfile.TemporaryDirectory() as old, tempfile.TemporaryDirectory() as new:
            (Path(old) / "models").mkdir()
            (Path(old) / "models" / "a.gguf").write_bytes(b"x" * 2048)
            from strata_cli.launch import cmd
            a = argparse.Namespace(data_dir=new, move_data=False, yes=False, plan=False, no_tui=True)
            cat = mock.Mock()
            cat.data_dir.return_value = Path(old)
            out = io.StringIO()
            with mock.patch.object(library, "movable_library", return_value=40.0), mock.patch("builtins.input", side_effect=AssertionError("asked")), \
                    mock.patch.object(sys, "stdin") as stdin, redirect_stdout(out):
                stdin.isatty.return_value = True
                self.assertFalse(cmd._move_data_choice(a, S.LaunchSettings(), cat, False))
            self.assertIn("Keeping your 40 GB of models where they are", out.getvalue())
            self.assertIn("--move-data", out.getvalue())

    def test_a_run_without_a_keyboard_says_the_same(self):
        from strata_cli.launch import cmd
        a = argparse.Namespace(data_dir="/new", move_data=False, yes=False, plan=False, no_tui=False)
        out = io.StringIO()
        with mock.patch.object(library, "movable_library", return_value=3.0), mock.patch("builtins.input", side_effect=AssertionError("asked")), \
                mock.patch.object(sys, "stdin") as stdin, redirect_stdout(out):
            stdin.isatty.return_value = False
            self.assertFalse(cmd._move_data_choice(a, S.LaunchSettings(), mock.Mock(), False))
        self.assertIn("Keeping your 3.0 GB", out.getvalue())


class SetupArguments(unittest.TestCase):
    def installed(self, h: Home, **cfg):
        path = h.base / "install" / "strata-iq3_xxs.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"exe": "x", "args": [], **cfg}), encoding="utf-8")
        return path

    def test_a_context_change_forwards_the_installed_roles(self):
        # F3: 0=main,1=helper came back from a 32k to 16k change as 0=helper,1=main
        with Home() as h:
            path = self.installed(h, gpu=[1, 0], gpus_asked=True, gpu_roles={"0": "main", "1": "helper"})
            r = runner_for()
            with mock.patch.object(Runner, "installed_state", return_value=(path, None)):
                a = r.setup_args()
            self.assertEqual(a[a.index("--gpu-roles") + 1], "0=main,1=helper")

    def test_installed_cards_are_forwarded_and_the_automatic_pick_is_not(self):
        with Home() as h:
            path = self.installed(h, gpu=[1, 0], gpus_asked=True)
            with mock.patch.object(Runner, "installed_state", return_value=(path, None)):
                a = runner_for().setup_args()
            self.assertEqual(a[a.index("--gpus") + 1], "1,0")
            path = self.installed(h, gpu=0)                               # never asked: setup picks again
            with mock.patch.object(Runner, "installed_state", return_value=(path, None)):
                a = runner_for().setup_args()
            self.assertNotIn("--gpus", a)
            self.assertNotIn("--gpu", a)

    def test_cards_named_on_the_command_line_win_over_the_installed_ones(self):
        with Home() as h:
            path = self.installed(h, gpu=[1, 0], gpus_asked=True, gpu_roles={"0": "main", "1": "helper"})
            with mock.patch.object(Runner, "installed_state", return_value=(path, None)):
                a = runner_for(gpus=[0]).setup_args()
            self.assertEqual(a[a.index("--gpu") + 1], "0")
            self.assertNotIn("--gpu-roles", a)

    def test_mtp_off_reaches_setup(self):
        with Home():
            a = runner_for(mtp="off").setup_args()
            self.assertEqual(a[a.index("--mtp") + 1], "off")
            self.assertNotIn("--mtp", runner_for().setup_args())

    def test_setup_skips_the_draft_layer_download_for_mtp_off(self):
        # setup.py is a script; its two fetch sites must be guarded by the explicit --mtp off (N3: a 5.2 GB download happened anyway)
        text = (Path(__file__).resolve().parents[2] / "setup.py").read_text(encoding="utf-8")
        self.assertIn('if a.mtp == "off" and not (rt / "experts.bin").exists():', text)
        self.assertIn('if use_mtp and a.mtp == "off" and not (mtp / "mtp-q2_0.gguf").exists():', text)
        self.assertIn('"use --data-dir on a bigger drive (strata launch --data-dir)" if CONTROL', text)       # N2: a launch option, not --models-dir


class PlanPredictsEveryDownload(unittest.TestCase):
    def plan_text(self, h: Home, own: Path, **kw) -> dict:
        s = S.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768, vision="on", model_path=str(own), data_dir=str(h.base / "data"), **kw)
        r = Runner(s, catalog=catalog(HW_12))
        with mock.patch.object(Runner, "check_own_files"), mock.patch.object(Runner, "ready_pack_gap", return_value=None):
            return r.download_plan()

    def test_a_local_gguf_without_a_reader_shows_the_picture_reader_and_the_switch(self):
        with Home() as h:
            own = h.base / "own"
            own.mkdir()
            (own / "m-00001-of-00002.gguf").write_bytes(b"x")
            plan = self.plan_text(h, own)
            self.assertIn("picture reader", plan["text"])
            self.assertIn("--vision off", plan["text"])
            self.assertIn("draft layer", plan["text"])
            self.assertIn("--mtp off", plan["text"])

    def test_a_reader_beside_the_files_is_used_and_not_counted(self):
        with Home() as h:
            own = h.base / "own"
            own.mkdir()
            (own / "m-00001-of-00002.gguf").write_bytes(b"x")
            (own / "mmproj-Qwen3.8-Flash-Next-BF16.gguf").write_bytes(b"x")
            plan = self.plan_text(h, own)
            self.assertNotIn("picture reader", plan["text"])

    def test_mtp_off_is_not_counted(self):
        with Home() as h:
            own = h.base / "own"
            own.mkdir()
            (own / "m-00001-of-00002.gguf").write_bytes(b"x")
            self.assertNotIn("draft layer", self.plan_text(h, own, mtp="off")["text"])


class ReadyPackPlan(unittest.TestCase):
    def test_a_ready_made_pack_is_counted_and_the_hint_is_a_launch_option(self):
        with Home() as h:
            own = h.base / "own"
            own.mkdir()
            r = Runner(S.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768, vision="off", model_path=str(own)), catalog=catalog(HW_12))
            gap = {"gb": 129.0, "folder": str(h.base / "data" / "models"), "free_gb": 110.0}
            with mock.patch.object(Runner, "ready_pack_gap", return_value=gap):
                plan = r.download_plan()
            self.assertGreaterEqual(plan["gb"], 129.0)
            self.assertIn("ready-made pack", plan["text"])
            self.assertIn("--data-dir", plan["text"])
            self.assertNotIn("Nothing to download", plan["text"])


class EngineIsFreshEnough(unittest.TestCase):
    def test_versions_compare_by_numbers(self):
        self.assertEqual(R.version_tuple("0.1.45.3"), (0, 1, 45, 3))
        self.assertEqual(R.version_tuple("junk"), ())

    def test_an_old_engine_is_named_in_plain_words_and_a_current_one_is_not(self):
        with Home():
            r = runner_for()
            r.catalog.strata.setup_module()           # loads setup.py's tables (MIN_ENGINE)
            words = r.stale_engine("0.1.45.3")
            self.assertIn("older than this model needs", words)
            self.assertIn("0.1.45.3", words)
            self.assertIsNone(r.stale_engine("0.1.47"))
            self.assertIsNone(r.stale_engine("source"))

    def test_gyro_on_a_mac_needs_engine_0_1_47(self):
        with Home():
            r = Runner(S.LaunchSettings(family="gyro", model="Gyro-S", context=32768, vision="off"), catalog=catalog(HW_12))
            with mock.patch.object(sys, "platform", "darwin"):
                self.assertEqual(r.needed_engine(), (0, 1, 47))
                self.assertIn("older than this model needs (0.1.47", r.stale_engine("0.1.45.3"))
                self.assertIsNone(r.stale_engine("0.1.47"))
            with mock.patch.object(sys, "platform", "linux"):
                self.assertLess(r.needed_engine(), (0, 1, 47))               # CUDA, HIP and SYCL had the kernels earlier
                self.assertIsNone(r.stale_engine("0.1.46"))
            with mock.patch.object(sys, "platform", "darwin"):
                q = Runner(S.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768, vision="off"), catalog=catalog(HW_12))
                self.assertLess(q.needed_engine(), (0, 1, 47))              # only Gyro is held to it

    def test_a_stale_engine_is_updated_or_said_never_ignored(self):
        with Home():
            r = runner_for()
            events = []
            r.emit = lambda stage, text, pct=None, level="info", **kw: events.append((level, text))
            before = {"usable": True, "engine": {"version": "0.1.45.3", "backend": "cuda"}}
            proc = mock.Mock()
            proc.stdout = iter([])
            proc.wait.return_value = 0
            from strata_control import engine_component as ec
            with mock.patch.object(ec, "status", return_value=before), mock.patch.object(Runner, "_spawn", return_value=proc):
                r.ensure_engine()
            self.assertTrue(any(level == "warn" and "older than this model needs" in text for level, text in events), events)
            self.assertTrue(any("strata update" in n for n in r.notices), r.notices)


class InstallIsNotHijacked(unittest.TestCase):
    def test_an_earlier_install_outside_the_chosen_data_folder_is_named(self):
        with Home() as h:
            cat = mock.Mock()
            cat.places = [mock.Mock(path=h.base / "old" / "Strata", source="app", configs=2)]
            words = R.install_isolation_words(cat, str(h.base / "scratch"))
            self.assertEqual(len(words), 1)
            self.assertIn("earlier install", words[0])
            self.assertIn(str(h.base / "old" / "Strata"), words[0])
            cat.places = [mock.Mock(path=h.base / "scratch" / "install", source="app", configs=1)]
            self.assertEqual(R.install_isolation_words(cat, str(h.base / "scratch")), [])
            cat.places = [mock.Mock(path=h.base / "old", source="explicit", configs=1)]
            self.assertEqual(R.install_isolation_words(cat, str(h.base / "scratch")), [])
            self.assertEqual(R.install_isolation_words(cat, None), [])

    def test_an_install_inside_strata_data_dir_comes_first(self):
        from strata_control import discovery
        with tempfile.TemporaryDirectory() as data, tempfile.TemporaryDirectory() as other:
            for folder in (Path(data) / "install", Path(other) / "install"):
                folder.mkdir()
                (folder / "strata-q.json").write_text("{}", encoding="utf-8")
            env = {"STRATA_DATA_DIR": data}
            found = discovery.places(environ=env, here=Path(other) / "install", home=Path(other))
            self.assertEqual(Path(found[0].path).resolve(), (Path(data) / "install").resolve())


class IdentityAndAdoption(unittest.TestCase):
    def test_family_that_does_not_match_the_file_header_is_warned_about(self):
        with Home() as h:
            r = Runner(S.LaunchSettings(family="swift", model="IQ2_XS", model_path=str(h.base)), catalog=catalog(HW_12))
            plan = {"candidate_selection": {"family": "qwen", "model": "IQ3_XXS"}, "folder_holds": [], "issues": []}
            from strata_control import models
            with mock.patch.object(models, "own_path_problem", return_value=None), mock.patch.object(models, "plan_import", return_value=plan), \
                    mock.patch.object(models, "import_target", return_value=str(h.base)):
                r.resolve_own_files()
            self.assertEqual(len(r.identity_notes), 1)
            self.assertIn("look like qwen IQ3_XXS", r.identity_notes[0])
            self.assertIn("not swift IQ2_XS", r.identity_notes[0])
            self.assertEqual((r.s.family, r.s.model), ("swift", "IQ2_XS"))        # warned, never forced

    def test_gyro_is_adopted_like_the_others_and_the_text_has_no_double_period(self):
        with Home() as h:
            data = h.base / "data"
            model_dir = data / "models" / "gyro-Gyro-S"
            model_dir.mkdir(parents=True)
            (model_dir / "Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf").write_bytes(b"x")
            pack = data / "packs" / "gyro-gyro-s"
            (pack / "tokenizer").mkdir(parents=True)
            (pack / "index.txt").write_text("x", encoding="utf-8")
            cfg, problems = adopt.plan(family="gyro", model="Gyro-S", model_dir=model_dir, data_dir=data, src=Path(__file__).resolve().parents[2],
                                       context=8192, kv=None, model_name="gyro", family_def={}, log=h.base / "x.log", backend="sycl",
                                       pack_name="gyro-gyro-s")
            self.assertEqual(problems, [], problems)
            self.assertEqual(cfg["args"][cfg["args"].index("--ple-gguf") + 1], cfg["args"][cfg["args"].index("--native") + 1])
            _, refused = adopt.plan(family="nvfp4", model="NVFP4", model_dir=model_dir, data_dir=data, src=Path("."), context=8192, kv=None,
                                    model_name="x", family_def={}, log=h.base / "x.log", backend=None)
            self.assertTrue(refused[0].endswith("not for nvfp4"))                 # the caller adds the one full stop


class UpdateCheckNamesTheChoice(unittest.TestCase):
    def test_the_selected_package_is_printed_before_the_rejected_ones(self):
        from strata_cli import update_cmd
        report = {"feed": {"sequence": 107, "channel": "stable", "key_id": "k", "trusted_by": "built-in keys"},
                  "engine": {"status": "install", "installed": None, "download_bytes": 5 << 20,
                             "offered": {"version": "0.1.47", "platform": "linux-x86_64", "backend": "cuda", "variant": "cuda12"},
                             "reasons": ["0.1.47 linux-x86_64-cuda-cuda13 needs driver 580"], "note": None},
                  "control": {"status": "current", "installed": "0.1.47", "offered": None}, "app": {"status": "not-in-feed"}}
        out = io.StringIO()
        with redirect_stdout(out):
            update_cmd._print_check(report)
        text = out.getvalue()
        self.assertIn("selected for this PC: 0.1.47 linux-x86_64-cuda-cuda12", text)
        self.assertLess(text.index("selected for this PC"), text.index("not used for this PC"))


if __name__ == "__main__":
    unittest.main()
