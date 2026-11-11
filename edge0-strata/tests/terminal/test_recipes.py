"""Owner-editable typed operations; previews never run or write an operation."""

import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from strata_cli.cli import main
from strata_control import journal, profiles, recipes


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        environment = mock.patch.dict(os.environ, {"STRATA_PROFILE_DIR": str(self.root / "profiles"),
                                                "XDG_CONFIG_HOME": str(self.root / "settings")})
        environment.start()
        self.addCleanup(environment.stop)
        engine = self.root / "engine"
        engine.write_bytes(b"fake engine")
        engine.chmod(0o700)
        tokenizer = self.root / "tokenizer"
        tokenizer.mkdir()
        for name in ("vocab.json", "merges.txt", "token_type.json"):
            (tokenizer / name).write_text("{}", encoding="utf-8")
        self.config = self.root / "run.json"
        self.config.write_text(json.dumps({"exe": str(engine), "tokenizer": str(tokenizer),
                                           "args": [], "cwd": str(self.root), "model_name": "fake",
                                           "host": "127.0.0.1", "port": 0}), encoding="utf-8")
        profiles.create("local", str(self.config))

    def _cli(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["ops", *args, "--profile", "local", "--output", "json"])
        return code, json.loads(output.getvalue())

    def test_benchmark_job_receipt_is_readable_by_new_owner_bridge_process(self):
        _, revision = profiles.load("local")
        job_id = "a" * 32
        intent = {"path": str(self.root / "future-benchmark"), "suite": "serving-smoke",
                  "concurrency": 2, "repeat": 1, "request_timeout": 30,
                  "condition": "baseline", "if_profile_revision": revision,
                  "if_source_sha256": "b" * 64,
                  "if_owner_id": "c" * 32, "if_instance_id": "d" * 32}
        journal.begin_direct("local", "bench.run", revision, intent, operation_id=job_id)
        request = {"schema_version": 1, "action": "bench.job", "profile": "local",
                   "params": {"job_id": job_id}}
        process = subprocess.run([sys.executable, "-m", "strata_control.management"],
                                 input=json.dumps(request), text=True, capture_output=True,
                                 cwd=Path(__file__).resolve().parents[2], timeout=10, check=True)
        response = json.loads(process.stdout)
        self.assertTrue(response["ok"])
        self.assertEqual(response["job"]["job_id"], job_id)
        self.assertEqual(response["job"]["state"], "launching")
        self.assertIsNone(response["job"]["result"])

    def test_init_set_plan_and_run_are_typed_and_machine_clean(self):
        self.assertEqual(self._cli("list")[1]["recipes"], [])
        self.assertFalse((profiles.profile_dir() / ".ops").exists())
        code, initialized = self._cli("init")
        self.assertEqual(code, 0)
        self.assertEqual(initialized["created"], ["restart", "calibrate", "check"])
        self.assertEqual(self._cli("init")[1]["created"], [])
        self.assertEqual(len(self._cli("list")[1]["recipes"]), 3)
        before, revision = recipes.load("local", "restart")
        code, preview = self._cli("set", "restart", "startup_wait_s=45", "--plan")
        self.assertEqual(code, 0)
        self.assertTrue(preview["would_save"])
        self.assertEqual(recipes.load("local", "restart"), (before, revision))
        self.assertEqual(self._cli("set", "restart", "startup_wait_s=45")[0], 0)
        self.assertFalse(self._cli("set", "restart", "startup_wait_s=45", "--plan")[1]["would_save"])
        with mock.patch("strata_control.runtime.plan_restart", return_value={"ok": True,
                         "state": "ready", "would_drain": False}), \
                mock.patch("strata_control.runtime.restart", return_value={"ok": True,
                           "state": "ready", "instance_id": "next"}) as restart:
            code, planned = self._cli("plan", "restart")
            self.assertEqual(code, 0)
            self.assertFalse(planned["plan"]["would_execute"])
            self.assertEqual(planned["plan"]["operation"]["startup_wait_s"], 45)
            self.assertTrue(planned["plan"]["operation"]["run_effects"]["drain_current_instance"])
            restart.assert_not_called()
            code, ran = self._cli("run", "restart")
            self.assertEqual(code, 0)
            self.assertEqual(ran["operation"]["result"]["instance_id"], "next")
            restart.assert_called_once_with("local", startup_wait=45)
        self.assertEqual(self._cli("run", "check")[0], 0)
        with mock.patch("strata_control.runtime.restart") as restart:
            code, invalid = self._cli("run", "restart", "--exclusive")
            self.assertEqual(code, 2)
            self.assertFalse(invalid["ok"])
            restart.assert_not_called()

    def test_calibration_requires_fresh_exclusivity_and_uses_new_report_path(self):
        recipes.initialize("local")
        source = self.config.read_bytes()
        report_dir = self.root / "reports"
        self.assertEqual(self._cli("set", "calibrate", f"report_dir={report_dir}")[0], 0)
        code, result = self._cli("plan", "calibrate")
        self.assertEqual(code, 0)
        self.assertFalse(result["plan"]["would_execute"])
        self.assertTrue(result["plan"]["operation"]["run_effects"]["start_engine_for_decode_sweep"])
        self.assertFalse(result["plan"]["operation"]["run_effects"]["change_run_config"])
        self.assertFalse(report_dir.exists())
        code, denied = self._cli("run", "calibrate")
        self.assertEqual(code, 2)
        self.assertFalse(denied["ok"])

        def fake_runner(cfg, say):
            return {"settings": {"--pcie-frac": "0.20"},
                    "report": {"default": {"pcie_frac": 0.55},
                               "confirm": {"default": [10.0, 10.2, 10.1],
                                           "candidate": [12.0, 12.2, 12.1]},
                               "tok_s": 12.1, "seconds": 30}}

        with mock.patch("tools.calibrate.run", side_effect=fake_runner), \
                mock.patch("strata_control.calibration._hardware_key", return_value="fake-gpu"), \
                mock.patch("strata_control.calibration._port_active", return_value=False):
            first = self._cli("run", "calibrate", "--exclusive")
            second = self._cli("run", "calibrate", "--exclusive")
        self.assertEqual(first[0], 0)
        self.assertEqual(second[0], 0)
        first_path = Path(first[1]["operation"]["result"]["report_path"])
        second_path = Path(second[1]["operation"]["result"]["report_path"])
        self.assertNotEqual(first_path, second_path)
        self.assertTrue(first_path.is_file())
        self.assertTrue(second_path.is_file())
        self.assertEqual(self.config.read_bytes(), source)
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(first_path.stat().st_mode), 0o600)

    def test_component_recipe_can_enable_a_typed_live_card_check(self):
        cfg = json.loads(self.config.read_text(encoding="utf-8"))
        cfg["gpu"] = [0, 1, 2, 3]
        cfg["backend"] = "cuda"
        self.config.write_text(json.dumps(cfg), encoding="utf-8")
        recipes.initialize("local")
        code, proposed = self._cli("set", "check", "live=true", "--plan")
        self.assertEqual(code, 0)
        self.assertTrue(proposed["would_save"])
        self.assertEqual(recipes.load("local", "check")[0]["params"], {})
        self.assertEqual(self._cli("set", "check", "live=true")[0], 0)
        cards = "\n".join(f"{i}, RTX 6000 test, GPU-test-{i}, 49140, 580.95" for i in range(4))
        with mock.patch("strata_control.doctor.shutil.which", return_value="/fake/nvidia-smi"), \
                mock.patch("strata_control.doctor.subprocess.run",
                           return_value=mock.Mock(returncode=0, stdout=cards)) as probe:
            code, preview = self._cli("plan", "check")
            self.assertEqual(code, 0)
            self.assertTrue(preview["plan"]["operation"]["run_effects"]["query_host_gpus"])
            self.assertEqual(preview["plan"]["operation"]["preflight"]["live_gpu_probe"]["status"], "ok")
            code, executed = self._cli("run", "check", "--if-revision", preview["plan"]["recipe_revision"])
        self.assertEqual(code, 0)
        self.assertEqual(executed["operation"]["result"]["report"]["live_gpu_probe"]["status"], "ok")
        self.assertEqual(probe.call_count, 2)
        record = journal.load("local", executed["operation"]["operation_id"])
        self.assertEqual(record["intent"], {"live": True})
        with self.assertRaisesRegex(profiles.ProfileError, "parameter is not supported"):
            recipes.propose_value("local", "check", "live=maybe")

    def test_invalid_edit_keeps_draft_and_saved_recipe(self):
        recipes.initialize("local")
        before = recipes.load("local", "check")
        editor = self.root / "bad_editor.py"
        editor.write_text("import pathlib, sys\npathlib.Path(sys.argv[1]).write_text('{}')\n")
        with mock.patch.dict(os.environ, {"VISUAL": f"{sys.executable} {editor}"}):
            code, result = self._cli("edit", "check")
        self.assertEqual(code, 2)
        self.assertFalse(result["ok"])
        self.assertEqual(recipes.load("local", "check"), before)
        self.assertEqual(len(list((profiles.profile_dir() / ".ops" / "local").glob(".check-edit-*.json"))), 1)

    def test_history_rollback_and_stale_edit(self):
        recipes.initialize("local")
        original, first = recipes.load("local", "restart")
        proposed, revision = recipes.propose_value("local", "restart", "startup_wait_s=60")
        recipes.save("local", "restart", proposed, revision)
        with mock.patch("strata_control.runtime.restart") as restart:
            code, stale = self._cli("run", "restart", "--if-revision", first)
            self.assertEqual(code, 6)
            self.assertFalse(stale["ok"])
            restart.assert_not_called()
        with self.assertRaisesRegex(profiles.ProfileError, "changed during edit"):
            recipes.save("local", "restart", original, first)
        entries = recipes.history("local", "restart")
        self.assertEqual(entries[1]["revision"], first)
        code, restored = self._cli("rollback", "restart", "--to", first)
        self.assertEqual(code, 0)
        self.assertEqual(restored["recipe"]["params"]["startup_wait_s"], 300)
        self.assertEqual(recipes.load("local", "restart")[0], original)

    def test_recipe_file_cannot_inject_shell_kind_or_symlink(self):
        recipes.initialize("local")
        path = profiles.profile_dir() / ".ops" / "local" / "restart.json"
        original = path.read_bytes()
        path.write_text(json.dumps({"schema_version": 1, "profile": "local", "id": "restart",
                                    "kind": "shell.exec", "params": {"command": "rm -rf /"}}))
        with self.assertRaisesRegex(profiles.ProfileError, "unsupported"):
            recipes.load("local", "restart")
        path.unlink()
        outside = self.root / "outside.json"
        outside.write_bytes(original)
        path.symlink_to(outside)
        with self.assertRaisesRegex(profiles.ProfileError, "regular file"):
            recipes.load("local", "restart")

    def test_init_rejects_conflicting_default_name_before_other_writes(self):
        recipes.create("local", "check", "runtime.restart")
        with self.assertRaisesRegex(profiles.ProfileError, "different kind"):
            recipes.initialize("local")
        self.assertEqual([item["id"] for item in recipes.list_recipes("local")], ["check"])

    def test_operation_journal_records_compact_success_and_is_read_only_to_list(self):
        recipes.initialize("local")
        self.assertEqual(self._cli("runs")[1]["runs"], [])
        self.assertFalse((profiles.profile_dir() / ".ops" / "local" / ".runs").exists())
        code, executed = self._cli("run", "check")
        self.assertEqual(code, 0)
        operation_id = executed["operation"]["operation_id"]
        code, listed = self._cli("runs")
        self.assertEqual(code, 0)
        self.assertEqual(listed["runs"][0]["operation_id"], operation_id)
        code, shown = self._cli("result", operation_id)
        self.assertEqual(code, 0)
        record = shown["record"]
        self.assertEqual(record["state"], "succeeded")
        self.assertEqual(record["intent"], {})
        self.assertEqual([event["state"] for event in record["events"]], ["running", "succeeded"])
        self.assertEqual(set(record["result"]), {"ok", "issue_codes"})
        path = profiles.profile_dir() / ".ops" / "local" / ".runs" / f"{operation_id}.json"
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_failed_and_uncertain_runs_keep_distinct_journal_states_without_raw_result(self):
        recipes.initialize("local")
        with mock.patch("strata_control.runtime.restart", return_value={"ok": False,
                         "state": "ready", "error": "new_instance_failed", "exit_code": 6,
                         "private_trace": "TOPSECRET"}):
            code, executed = self._cli("run", "restart")
        self.assertEqual(code, 6)
        failed = journal.load("local", executed["operation"]["operation_id"])
        self.assertEqual(failed["state"], "failed")
        self.assertNotIn("TOPSECRET", json.dumps(failed))
        with mock.patch("strata_control.runtime.restart",
                        side_effect=profiles.ProfileError("runtime operation wait expired", 8)):
            code, timed_out = self._cli("run", "restart")
        self.assertEqual(code, 8)
        self.assertFalse(timed_out["ok"])
        newest = journal.list_runs("local")[0]
        uncertain = journal.load("local", newest["operation_id"])
        self.assertEqual(uncertain["state"], "completion_unknown")
        self.assertEqual(uncertain["result"]["exit_code"], 8)
        with mock.patch("strata_control.runtime.restart", side_effect=KeyboardInterrupt):
            code, interrupted = self._cli("run", "restart")
        self.assertEqual(code, 130)
        self.assertFalse(interrupted["ok"])
        newest = journal.list_runs("local")[0]
        self.assertEqual(journal.load("local", newest["operation_id"])["state"], "completion_unknown")

    def test_broken_machine_output_detaches_without_reporting_local_file_failure(self):
        recipes.initialize("local")

        class ClosedPipe(io.StringIO):
            def write(self, text):
                raise BrokenPipeError("consumer closed")

        with redirect_stdout(ClosedPipe()):
            self.assertEqual(main(["ops", "list", "--profile", "local", "--output", "json"]), 0)

    def test_journal_prunes_old_terminal_records_but_not_running_work(self):
        recipes.initialize("local")
        with mock.patch.object(journal, "MAX_RECORDS", 2):
            ids = []
            for _ in range(3):
                code, executed = self._cli("run", "check")
                self.assertEqual(code, 0)
                ids.append(executed["operation"]["operation_id"])
            self.assertEqual(len(journal.list_runs("local")), 2)
            with self.assertRaisesRegex(profiles.ProfileError, "not found"):
                journal.load("local", ids[0])
            # Two unfinished records exhaust the cap; neither may be discarded.
            directory = profiles.profile_dir() / ".ops" / "local" / ".runs"
            for path in directory.glob("*.json"):
                path.unlink()
            first = journal.begin("local", "check", "components.verify", "0" * 16, "0" * 16)
            second = journal.begin("local", "check", "components.verify", "0" * 16, "0" * 16)
            with self.assertRaisesRegex(profiles.ProfileError, "journal is full"):
                journal.begin("local", "check", "components.verify", "0" * 16, "0" * 16)
            self.assertEqual(journal.load("local", first["operation_id"])["state"], "running")
            self.assertTrue(journal.load("local", first["operation_id"])["completion_unverified"])
            self.assertEqual(journal.load("local", second["operation_id"])["state"], "running")

    def test_invalid_journal_event_is_reported_without_rendering_record_content(self):
        recipes.initialize("local")
        operation_id = self._cli("run", "check")[1]["operation"]["operation_id"]
        path = profiles.profile_dir() / ".ops" / "local" / ".runs" / f"{operation_id}.json"
        record = json.loads(path.read_text())
        record["events"][-1]["state"] = "\x1b[31mforged\x1b[0m"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(profiles.ProfileError, "invalid events"):
            journal.load("local", operation_id)
        code, listed = self._cli("runs")
        self.assertEqual(code, 0)
        self.assertIn("error", listed["runs"][0])
        self.assertNotIn("forged", json.dumps(listed))

    def test_direct_journal_is_typed_compact_and_keeps_uncertain_interruptions(self):
        revision = profiles.load("local")[1]
        operation_id, result = journal.run_direct("local", "runtime.restart", revision,
            {"wait_s": 20, "startup_wait_s": 10, "expected_profile_revision": revision},
            lambda: {"ok": False, "state": "ready", "error": "activation_failed", "exit_code": 6,
                     "private_trace": "TOPSECRET"})
        self.assertFalse(result["ok"])
        record = journal.load("local", operation_id)
        self.assertEqual(record["schema_version"], 2)
        self.assertEqual(record["origin"], "direct")
        self.assertEqual(record["state"], "failed")
        self.assertNotIn("TOPSECRET", json.dumps(record))
        deadline_id, _ = journal.run_direct("local", "runtime.stop", revision, {"wait_s": 12},
                                            lambda: {"ok": False, "state": "stopping", "exit_code": 8})
        self.assertEqual(journal.load("local", deadline_id)["state"], "completion_unknown")
        with self.assertRaises(KeyboardInterrupt):
            journal.run_direct("local", "runtime.stop", revision, {"wait_s": 12},
                               lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
        latest = journal.list_runs("local")[0]
        uncertain = journal.load("local", latest["operation_id"])
        self.assertEqual(uncertain["state"], "completion_unknown")
        with self.assertRaisesRegex(profiles.ProfileError, "unsupported fields"):
            journal.begin_direct("local", "runtime.stop", revision,
                                 {"wait_s": 12, "shell": "rm -rf /"})
        self.assertEqual(len(journal.list_runs("local")), 3)

    def test_direct_journal_rejects_tampered_origin(self):
        revision = profiles.load("local")[1]
        record = journal.begin_direct("local", "runtime.stop", revision, {"wait_s": 12})
        path = profiles.profile_dir() / ".ops" / "local" / ".runs" / f"{record['operation_id']}.json"
        forged = json.loads(path.read_text())
        forged["origin"] = "remote"
        path.write_text(json.dumps(forged))
        with self.assertRaisesRegex(profiles.ProfileError, "unsupported intent"):
            journal.load("local", record["operation_id"])


if __name__ == "__main__":
    unittest.main()
