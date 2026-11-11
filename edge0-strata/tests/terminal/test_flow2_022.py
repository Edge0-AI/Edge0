"""0.2.2 flow walk, second round (phaseA-tests/flow-findings-022.md): what `strata launch` says and does with your own model files, the
plan, the download figure, the long silent step and a tuning result that already exists."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.catalog import Catalog
from strata_cli.launch.runner import LaunchError, Runner
from tests.terminal.launch_support import HW_12, Home, run_cli
from tests.terminal.test_own_files import IQ3XXS, Q2, gguf_bytes


class Base(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
        p.start()
        self.addCleanup(p.stop)
        self.tmp = tempfile.TemporaryDirectory(prefix="strata-flow2-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def files(self, folder: Path, names: list[str], kind: str) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        for n in names:
            (folder / n).write_bytes(gguf_bytes([kind]))
        return folder

    def runner(self, **kw) -> Runner:
        return Runner(S.LaunchSettings(**kw), catalog=Catalog(hardware=HW_12))


class MissingOwnFiles(Base):
    """Finding 2: a --model-path that is not there, or holds no model, is said in plain words before anything starts."""

    def test_a_folder_that_does_not_exist_is_named_and_nothing_is_promised(self):
        missing = self.base / "models"
        code, out, err = run_cli(["launch", "--plan", "--model-path", str(missing)])
        self.assertEqual(code, 2)
        self.assertIn(f"This folder does not exist: {missing}", err)
        self.assertNotIn("Qwen", err)
        self.assertNotIn("continues where it stopped", err)
        self.assertNotIn("could not be installed", err)

    def test_a_missing_file_says_file(self):
        code, out, err = run_cli(["launch", "--plan", "--model-path", str(self.base / "x-00001-of-00002.gguf")])
        self.assertEqual(code, 2)
        self.assertIn("This file does not exist", err)

    def test_an_empty_folder_says_there_are_no_gguf_files(self):
        empty = self.base / "empty"
        empty.mkdir()
        code, out, err = run_cli(["launch", "--plan", "--model", "q2_0", "--model-path", str(empty)])
        self.assertEqual(code, 2)
        self.assertIn("no .gguf model files", err)
        self.assertIn("...-00001-of-00002.gguf", err)

    def test_machine_output_carries_the_same_words(self):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", "--model-path", str(self.base / "gone")])
        self.assertEqual(code, 2)
        self.assertIn("This folder does not exist", json.loads(out)["error"]["message"])

    def test_the_engine_is_not_touched_for_a_wrong_path(self):
        r = self.runner(model_path=str(self.base / "gone"))
        with mock.patch.object(Runner, "ensure_engine", side_effect=AssertionError("engine step must not start")):
            with self.assertRaises(LaunchError) as ctx:
                r.run()
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("does not exist", ctx.exception.text)

    def test_the_failure_hint_of_own_files_promises_no_download(self):
        r = self.runner(model_path=str(self.files(self.base / "m", Q2, "Q2_0")), family="qwen", model="Q2_0")
        proc = mock.Mock()
        proc.stdout = iter(["=== Step 5: Getting the model files ===", "[X] missing /m/x.gguf"])
        proc.wait.return_value = 1
        with mock.patch.object(Runner, "_spawn", return_value=proc), mock.patch.object(Runner, "adopt_prepared_files", return_value=None):
            with self.assertRaises(LaunchError) as ctx:
                r.ensure_model()
        self.assertNotIn("download", ctx.exception.next)
        self.assertIn("finished steps are skipped", ctx.exception.next)

    def test_the_screens_say_the_same(self):
        from strata_control import models
        text, nxt = models.own_path_problem(str(self.base / "nope"))
        self.assertIn("This folder does not exist", text)
        self.assertIn("Nothing was downloaded", nxt)
        self.assertIsNone(models.own_path_problem(str(self.files(self.base / "ok", IQ3XXS, "IQ3_XXS"))))

    def test_a_folder_one_level_above_the_files_is_not_called_empty(self):
        self.files(self.base / "repo" / "IQ3_XXS", IQ3XXS, "IQ3_XXS")
        from strata_control import models
        self.assertIsNone(models.own_path_problem(str(self.base / "repo")))


class LinkedShards(Base):
    """Finding 3: shards that are links are accepted with no --model, as with --model, and a problem is said once."""

    def links(self, names: list[str], real: bool = True) -> Path:
        blobs = self.base / "blobs"
        snap = self.base / "snap"
        blobs.mkdir(exist_ok=True)
        snap.mkdir(exist_ok=True)
        for i, n in enumerate(names):
            target = blobs / f"sha{i}"
            if real:
                target.write_bytes(gguf_bytes(["Q2_0"]))
            try:
                os.symlink(target, snap / n)
            except (OSError, NotImplementedError):
                self.skipTest("this account cannot create symbolic links")
        return snap

    def test_linked_shards_without_a_model_name_are_used(self):
        snap = self.links(Q2)
        code, out, err = run_cli(["launch", "--plan", "--output", "json", "--model-path", str(snap)])
        self.assertEqual(code, 0, err or out)
        s = json.loads(out)["plan"]["settings"]
        self.assertEqual((s["family"], s["model"]), ("qwen", "Q2_0"))
        self.assertEqual(Path(s["model_path"]), snap)                       # the link names stay (the shard names are what setup reads)

    def test_linked_shards_with_the_model_named_give_the_same_folder(self):
        snap = self.links(Q2)
        code, out, err = run_cli(["launch", "--plan", "--output", "json", "--model", "q2_0", "--model-path", str(snap)])
        self.assertEqual(code, 0, err or out)
        self.assertEqual(Path(json.loads(out)["plan"]["settings"]["model_path"]), snap)

    def test_links_to_nothing_are_one_sentence_not_one_per_file(self):
        snap = self.links(Q2, real=False)
        code, out, err = run_cli(["launch", "--plan", "--model-path", str(snap)])
        self.assertEqual(code, 5)
        self.assertEqual(err.count("shortcut"), 1, err)
        self.assertIn(Q2[0], err)
        self.assertIn(Q2[1], err)

    def test_the_same_problem_for_many_files_is_said_once(self):
        from strata_control import models
        issues = [f"m-0000{i}-of-00005.gguf: GGUF metadata is truncated" for i in range(1, 6)] + ["incomplete_or_invalid_shard_set"]
        words = models.problem_words(issues)
        self.assertEqual(words.count("the file is cut short"), 1)
        self.assertIn("m-00001-of-00005.gguf, m-00002-of-00005.gguf and 3 more files", words)
        self.assertEqual(models.problem_words(["a.gguf: GGUF metadata is truncated", "b.gguf: GGUF metadata is truncated"]),
                         "a.gguf and b.gguf: the file is cut short (a download or copy that did not finish)")
        self.assertEqual(models.problem_words(["multiple_model_groups", "unrecognized_or_unsupported_model_variant"]),
                         "the folder holds more than one model")
        many = [f"unsupported_or_missing_architecture", "incomplete_or_invalid_shard_set", "multiple_model_groups",
                "unsupported_or_incomplete_tokenizer_metadata", "missing_or_unexpected_expert_quantization"]
        self.assertTrue(models.problem_words(many).endswith("and 2 more problems"))



class PlanWords(Base):
    """Findings 1 and 7: --plan shows the real model, the picture setting, a copyable line, and what would be downloaded."""

    def plan_text(self, *args):
        code, out, err = run_cli(["launch", "--plan", *args])
        self.assertEqual(code, 0, err or out)
        return out

    def test_own_files_show_the_model_they_are(self):
        folder = self.files(self.base / "IQ3_XXS", IQ3XXS, "IQ3_XXS")
        out = self.plan_text("--model-path", str(folder))
        self.assertIn("Plan for 'IQ3_XXS", out)
        self.assertNotIn("model default", out)
        self.assertNotIn("None", out)
        self.assertIn("Qwen3.8-Flash-Next IQ3_XXS", out)
        self.assertIn(f"your files in {folder}", out)
        self.assertRegex(out, r"pictures (on|off)")
        self.assertIn("--model iq3_xxs", out.split("same as")[1])         # the copyable line names the model too
        self.assertIn(f'--model-path "{folder}"', out.split("same as")[1])

    def test_a_named_catalog_model_shows_its_size_and_the_download(self):
        out = self.plan_text("--model", "q2_0")
        self.assertIn("Qwen3.8-Flash-Next Q2_0, 66 GB", out)
        self.assertRegex(out, r"download\s+About 7\d\.\d GB to download: the model \(66\.4 GB\), the draft layer \(5\.2 GB\)")
        self.assertIn("The engine for your graphics card is downloaded first", out)

    def test_nothing_is_downloaded_when_everything_is_here(self):
        folder = self.files(self.base / "IQ3_XXS", IQ3XXS, "IQ3_XXS")
        data = self.base / "Strata" 
        (data / "models" / "x").mkdir(parents=True)
        (data / "mtp" / "rt").mkdir(parents=True)
        (data / "mtp" / "rt" / "experts.bin").write_bytes(b"x")
        (data / "models" / "mmproj-Qwen3.8-Flash-Next-BF16.gguf").write_bytes(b"x")
        # the files sit under <data>/models/<tag>: the draft layer and the picture reader are found beside them
        moved = data / "models" / "iq3_xxs"
        folder.rename(moved)
        out = self.plan_text("--model-path", str(moved))
        self.assertIn("download  Nothing to download: your model files are used where they are.", out)

    def test_own_files_with_no_draft_layer_say_what_is_fetched(self):
        folder = self.files(self.base / "IQ3_XXS", IQ3XXS, "IQ3_XXS")
        with mock.patch.object(Runner, "existing_data_dir", return_value=None),                 mock.patch.object(Catalog, "data_dir", return_value=self.base / "emptydata"):
            out = self.plan_text("--model-path", str(folder), "--vision", "off")
        self.assertIn("About 5.2 GB to download: the draft layer (5.2 GB). Your model files are used where they are.", out)

    def test_a_model_whose_files_are_all_on_disk_does_not_list_a_zero_download(self):
        from strata_cli.launch.modelstate import ModelState, Part
        data = self.base / "d"
        (data / "mtp" / "rt").mkdir(parents=True)
        (data / "mtp" / "rt" / "experts.bin").write_bytes(b"x")
        whole = ModelState(installed=False, parts=[Part("weights", "the model", 0.0)], on_disk_gb=66.4, total_gb=66.4)
        with mock.patch.object(Runner, "model_state", return_value=whole):
            out = self.plan_text("--model", "q2_0", "--data-dir", str(data), "--vision", "off")
            self.assertIn("download  Nothing to download: the model is already on this computer.", out)
            out = self.plan_text("--model", "q2_0", "--data-dir", str(self.base / "other"), "--vision", "off")
        self.assertIn("About 5.2 GB to download: the draft layer (5.2 GB). 66 GB of the model is already in its folder.", out)
        self.assertNotIn("0.0 GB", out)

    def test_json_plan_carries_the_numbers(self):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", "--model", "q2_0", "--mtp", "off", "--vision", "off"])
        self.assertEqual(code, 0, err)
        plan = json.loads(out)["plan"]
        self.assertEqual(plan["model"]["title"], "Qwen3.8-Flash-Next Q2_0")
        self.assertEqual(plan["model"]["model_gb"], 66.4)
        self.assertEqual(plan["download"]["gb"], 66.4)
        self.assertEqual([p["kind"] for p in plan["download"]["parts"]], ["weights"])



class DownloadTotal(Base):
    """Finding 4: the figure counts every shard (and the picture reader), in the folder the setup is really given."""

    def run_lines(self, runner: Runner, lines: list[str]) -> list[str]:
        seen: list[str] = []
        runner.emit_cb = lambda ev: seen.append(ev.text)
        proc = mock.Mock()
        proc.stdout = iter(lines)
        proc.wait.return_value = 1                               # stop after the lines: the failure path is not what is looked at
        with mock.patch.object(Runner, "_spawn", return_value=proc), mock.patch.object(Runner, "adopt_prepared_files", return_value=None):
            with self.assertRaises(LaunchError):
                runner.ensure_model()
        return seen

    def test_the_total_is_the_whole_model_in_the_chosen_data_folder(self):
        data = self.base / "mydata"
        folder = data / "models" / "Q2_0"
        folder.mkdir(parents=True)
        first = folder / (Q2[0] + ".part")
        first.write_bytes(bytes(100_000_000))
        r = self.runner(family="qwen", model="Q2_0", data_dir=str(data), vision="on", context=32768)
        job = r.progress_job()
        self.assertEqual(Path(job["folder"]), folder)
        self.assertEqual(len(job["files"]), 3)                                   # two shards and the picture reader
        seen = self.run_lines(r, ["=== Step 5: Getting the model files ===", f"{Q2[0]}: 0.10 / 37.60 GB (0%)"])
        text = [x for x in seen if "GB" in x]
        self.assertTrue(text, seen)
        self.assertIn("of 67.3 GB", text[-1])                                     # 66.4 GB of shards + 0.9 GB picture reader
        self.assertNotIn("37.6", " ".join(text))

    def test_the_default_data_folder_is_used_when_none_was_chosen(self):
        r = self.runner(family="qwen", model="Q2_0", vision="off", context=32768)
        self.assertEqual(Path(r.progress_job()["folder"]).parts[-2:], ("models", "Q2_0"))
        self.assertEqual(len(r.progress_job()["files"]), 2)



class QuietStep(Base):
    """Finding 5: setup step 6 (packing) prints nothing for minutes: the person sees the plain sentence and a 'still working' line."""

    def slow(self, steps):
        import time
        for delay, line in steps:
            time.sleep(delay)
            yield line + "\n"

    def run_steps(self, steps, heartbeat=0.3):
        r = self.runner(family="qwen", model="Q2_0", vision="off", context=32768)
        r.HEARTBEAT_S = heartbeat
        seen = []
        r.emit_cb = lambda ev: seen.append(ev.text)
        proc = mock.Mock()
        proc.stdout = self.slow(steps)
        proc.wait.return_value = 1
        with mock.patch.object(Runner, "_spawn", return_value=proc), mock.patch.object(Runner, "adopt_prepared_files", return_value=None):
            with self.assertRaises(LaunchError):
                r.ensure_model()
        return seen

    def test_a_silent_pack_step_says_what_it_does_and_that_it_is_still_going(self):
        seen = self.run_steps([(0, "=== Step 5: Getting the model files ==="), (0, "all model files present"),
                               (0, "=== Step 6: Preparing the model for Strata ==="),
                               (0, "> python tools/iq_pack.py --gguf x --out y --experts-bin"), (1.6, "[X] stopped for the test")])
        pack = [x for x in seen if x.startswith("Packing the experts into one file")]
        self.assertTrue(pack, seen)
        self.assertEqual(pack[0], "Packing the experts into one file for low-memory mode (one time)")
        beats = [x for x in pack if "still working" in x]
        self.assertGreaterEqual(len(beats), 2, seen)                    # a line every interval of silence
        self.assertNotIn("Setup step 6", " ".join(seen))

    def test_no_heartbeat_while_lines_keep_coming(self):
        seen = self.run_steps([(0, "=== Step 6: Preparing the model for Strata ==="), (0, "> python tools/iq_pack.py --gguf x --out y --experts-bin")] +
                              [(0.1, f"  layer {i}") for i in range(5)] + [(0, "[X] stopped")], heartbeat=2.0)
        self.assertFalse([x for x in seen if "still working" in x])

    def test_the_plain_printer_shows_each_heartbeat(self):
        import io
        from strata_cli.launch import cmd
        from strata_cli.launch.runner import Event
        buf = io.StringIO()
        with mock.patch.object(cmd.sys, "stdout", buf):
            emit = cmd._printer()
            emit(Event("model", "Packing (still working, 30 s so far)"))
            emit(Event("model", "Packing (still working, 60 s so far)"))
            emit(Event("model", "Packing (still working, 60 s so far)"))
        self.assertEqual(buf.getvalue().count("still working"), 2)



class StoredTuning(Base):
    """Finding 6: a run by flags (no saved preset) reuses the tuning result stored for the same model, choices and computer."""

    IDENT = {"hardware": {"hard": {"gpus": ["RTX 5070"], "cpu": "test", "logical": 16, "ram_gib": 64}, "soft": {}},
             "engine": {"backend": "cuda"}, "model": {"sha256_shard1": "abc"}}

    def setUp(self):
        super().setUp()
        from strata_control import presets
        self.presets = presets
        self.cfg_path = self.base / "launch.json"
        self.cfg = {"args": ["--native", "x/model-00001-of-00002.gguf", "--max-context", "32768", "--kv", "int8"], "model_name": "qwen"}
        self.cfg_path.write_text(json.dumps(self.cfg), encoding="utf-8")
        self.ident = json.loads(json.dumps(self.IDENT))
        p = mock.patch.object(presets, "current_identity", lambda cfg, **kw: self.ident)
        p.start()
        self.addCleanup(p.stop)

    def store(self):
        return self.presets.create({"settings": {"args": {}, "env": {}, "per_request": {}}, "outcome": "improved"},
                                   choices=self.presets.choices_from_config(self.cfg), identity=self.ident, name="Q2 32K - Full speed")

    def runner_for_tune(self, retune=False):
        r = self.runner(family="qwen", model="Q2_0", tune="quick")
        r.retune = retune
        r.config_path = self.cfg_path
        return r

    def run_tune(self, r):
        seen = []
        r.emit_cb = lambda ev: seen.append(ev.text)
        with mock.patch.object(Runner, "_spawn", side_effect=RuntimeError("the tuner started")) as spawn:
            try:
                r.tune()
                started = False
            except RuntimeError:
                started = True
        return seen, started

    def test_nothing_stored_means_it_is_measured(self):
        seen, started = self.run_tune(self.runner_for_tune())
        self.assertTrue(started)

    def test_a_big_model_gets_a_longer_start_wait_by_itself(self):                    # #3949 #3459
        r = self.runner_for_tune()
        row = type("Row", (), {"family": r.s.family, "model": r.s.model, "download_gb": 20.0})
        with mock.patch.object(r.catalog, "rows", return_value=[row]):
            self.assertEqual(r.start_wait(), 600)                                    # a small model keeps ten minutes
        row.download_gb = 250.0
        with mock.patch.object(r.catalog, "rows", return_value=[row]):
            self.assertEqual(r.start_wait(), 1800)                                   # 600 + 210 x 8, capped at 30 minutes
        row.download_gb = 83.0
        with mock.patch.object(r.catalog, "rows", return_value=[row]):
            self.assertEqual(r.start_wait(), 944)

    def test_a_stored_result_is_used_and_said(self):
        saved = self.store()
        r = self.runner_for_tune()
        seen, started = self.run_tune(r)
        self.assertFalse(started)
        text = seen[-1]
        self.assertIn("already tuned for this computer", text)
        self.assertIn("Add --retune to measure again", text)
        self.assertIn("Q2 32K - Full speed", text)
        self.assertIn("--retune", text)
        self.assertEqual(r.s.tuned_preset, saved["id"])
        self.assertTrue(r.s.is_tuned)

    def test_retune_measures_again(self):
        self.store()
        seen, started = self.run_tune(self.runner_for_tune(retune=True))
        self.assertTrue(started)

    def test_another_context_is_another_result(self):
        self.store()
        self.cfg["args"][3] = "65536"
        self.cfg_path.write_text(json.dumps(self.cfg), encoding="utf-8")
        seen, started = self.run_tune(self.runner_for_tune())
        self.assertTrue(started)

    def test_a_changed_graphics_card_is_measured_again(self):
        self.store()
        self.ident["hardware"]["hard"]["gpus"] = ["RTX 4090"]
        seen, started = self.run_tune(self.runner_for_tune())
        self.assertTrue(started)

    def test_the_flag_asks_for_a_new_measurement_and_implies_quick(self):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", "--model", "q2_0", "--retune"])
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["plan"]["settings"]["tune"], "quick")
        code, out, err = run_cli(["launch", "--help"])
        self.assertIn("--retune", out)



if __name__ == "__main__":
    unittest.main()
