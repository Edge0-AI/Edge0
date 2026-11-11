"""Calibration report binding, review and atomic apply without a GPU."""

import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from strata_cli.cli import main
from strata_control import calibration, journal, management, profiles, runtime
import setup


class CalibrationFlowTests(unittest.TestCase):
    def test_mac_ram_identity_is_available_without_proc_meminfo(self):
        with mock.patch.object(setup, "WIN", False), \
                mock.patch.object(setup.sys, "platform", "darwin"), \
                mock.patch.object(setup.subprocess, "run",
                                  return_value=SimpleNamespace(stdout="17179869184\n")):
            self.assertEqual(setup.ram_gb(), 16.0)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = mock.patch.dict(os.environ, {"STRATA_PROFILE_DIR": str(self.root / "profiles"),
                                           "STRATA_RUNTIME_DIR": str(self.root / "runtime"),
                                           "XDG_CONFIG_HOME": str(self.root / "settings"),
                                           "APPDATA": str(self.root / "settings"),
                                           "STRATA_API_KEY": ""})
        patcher.start()
        self.addCleanup(patcher.stop)
        # setup's own answer: <XDG_CONFIG_HOME>/strata/settings.json, or %APPDATA%/Strata/settings.json on Windows
        self.settings_file = setup.settings_path()
        self.assertTrue(str(self.settings_file).startswith(str(self.root)), "settings must stay in the temp tree")
        engine = self.root / "strata-engine"
        engine.write_bytes(b"fake engine")
        engine.chmod(0o700)
        tokenizer = self.root / "tokenizer"
        tokenizer.mkdir()
        for name in ("vocab.json", "merges.txt", "token_type.json"):
            (tokenizer / name).write_text("{}", encoding="utf-8")
        (self.root / "pack").mkdir()
        (self.root / "model.gguf").write_bytes(b"fake model")
        self.config = self.root / "run.json"
        self.config.write_text(json.dumps({"exe": str(engine), "tokenizer": str(tokenizer),
                                           "args": ["--pack", str(self.root / "pack"),
                                                    "--native", str(self.root / "model.gguf")],
                                           "cwd": str(self.root), "model_name": "test-model", "port": 8095}),
                               encoding="utf-8")
        profiles.create("local", str(self.config))
        hardware = mock.patch("strata_control.calibration._hardware_key", return_value="test-gpu|24GB|test-cpu")
        hardware.start()
        self.addCleanup(hardware.stop)
        port = mock.patch("strata_control.calibration._port_active", return_value=False)
        port.start()
        self.addCleanup(port.stop)

    def fake_result(self, cfg, say, **_):
        say("measured fake decode workload")
        return {"settings": {"--pcie-frac": "0.20"},
                "report": {"default": {"pcie_frac": 0.55}, "confirm": {"default": [10.0, 10.2, 10.1],
                                                                  "candidate": [12.0, 12.2, 12.1]},
                           "tok_s": 12.1, "seconds": 30}}

    def test_desktop_calibration_receipts_recover_measure_apply_and_rollback(self):
        def send(action, params):
            return management.dispatch({"schema_version": 1, "action": action,
                                        "profile": "local", "params": params})

        path = self.root / "desktop-calibration.json"
        plan = send("calibration.plan", {})["plan"]
        reviewed = {"path": str(path), "exclusive_ack": True,
                    "if_profile_revision": plan["profile_revision"],
                    "if_source_sha256": plan["source_sha256"]}
        measure_id = uuid.uuid4().hex
        created = send("calibration.job.create", {"job_id": measure_id, "action": "measure",
                                                  "if_report_sha256": None, **reviewed})
        self.assertEqual(created["job_id"], measure_id)
        self.assertEqual(send("calibration.job", {"job_id": measure_id})["job"]["state"], "launching")
        with mock.patch("tools.calibrate.run", side_effect=self.fake_result), \
                mock.patch("strata_control.calibration._smoke", return_value=2):
            measured = send("calibration.measure", {**reviewed, "job_id": measure_id})
            self.assertEqual(measured["operation_id"], measure_id)
            recovered = send("calibration.job", {"job_id": measure_id})["job"]
            self.assertEqual(recovered["state"], "succeeded")
            self.assertEqual(recovered["result"]["settings"], {"--pcie-frac": "0.20"})

            apply_plan = send("calibration.apply.plan", {"path": str(path)})
            apply = {**reviewed, "if_profile_revision": apply_plan["profile_revision"],
                     "if_source_sha256": apply_plan["source_sha256"],
                     "if_report_sha256": apply_plan["report_sha256"]}
            apply_id = uuid.uuid4().hex
            send("calibration.job.create", {"job_id": apply_id, "action": "apply", **apply})
            with self.assertRaisesRegex(profiles.ProfileError, "only calibration measurement"):
                send("calibration.cancel", {"job_id": apply_id})
            self.assertEqual(send("calibration.apply", {**apply, "job_id": apply_id})["operation_id"], apply_id)
            self.assertTrue(send("calibration.job", {"job_id": apply_id})["job"]["result"]["changed"])

            rollback_plan = send("calibration.rollback.plan", {"path": str(path)})
            rollback = {**reviewed, "if_profile_revision": rollback_plan["profile_revision"],
                        "if_source_sha256": rollback_plan["source_sha256"],
                        "if_report_sha256": rollback_plan["report_sha256"]}
            rollback_id = uuid.uuid4().hex
            send("calibration.job.create", {"job_id": rollback_id, "action": "rollback", **rollback})
            self.assertEqual(send("calibration.rollback", {**rollback, "job_id": rollback_id})["operation_id"],
                             rollback_id)
            self.assertTrue(send("calibration.job", {"job_id": rollback_id})["job"]["result"]["changed"])

    def test_saved_measurement_reconciles_exact_unfinished_job(self):
        def send(action, params):
            return management.dispatch({"schema_version": 1, "action": action,
                                        "profile": "local", "params": params})

        path = self.root / "recovered-calibration.json"
        plan = send("calibration.plan", {})["plan"]
        job_id = uuid.uuid4().hex
        reviewed = {"job_id": job_id, "action": "measure", "path": str(path),
                    "exclusive_ack": True, "if_profile_revision": plan["profile_revision"],
                    "if_source_sha256": plan["source_sha256"], "if_report_sha256": None}
        send("calibration.job.create", reviewed)
        with mock.patch("tools.calibrate.run", side_effect=self.fake_result):
            calibration.measure("local", path, exclusive_ack=True, runner=self.fake_result,
                                say=lambda _: None, job_id=job_id)
        recovered = send("calibration.job", {"job_id": job_id})["job"]
        self.assertEqual(recovered["state"], "succeeded")
        self.assertEqual(recovered["outcome"], "report_recovered")
        self.assertEqual(journal.load("local", job_id)["state"], "succeeded")
        result = subprocess.run([sys.executable, "-m", "strata_control.management"],
            input=json.dumps({"schema_version": 1, "action": "calibration.job", "profile": "local",
                              "params": {"job_id": job_id}}), text=True, capture_output=True,
            cwd=Path(__file__).resolve().parents[2], env=os.environ.copy(), check=True)
        self.assertEqual(json.loads(result.stdout)["job"]["result"]["report_id"],
                         recovered["result"]["report_id"])
        with self.assertRaisesRegex(profiles.ProfileError, "already finished"):
            journal.finish("local", job_id, "failed", {"error_type": "wrong"})

    def test_failed_desktop_calibration_preflight_finishes_receipt(self):
        def send(action, params):
            return management.dispatch({"schema_version": 1, "action": action,
                                        "profile": "local", "params": params})

        plan = send("calibration.plan", {})["plan"]
        job_id = uuid.uuid4().hex
        path = self.root / "failed-calibration.json"
        reviewed = {"path": str(path), "exclusive_ack": True,
                    "if_profile_revision": plan["profile_revision"],
                    "if_source_sha256": plan["source_sha256"]}
        send("calibration.job.create", {"job_id": job_id, "action": "measure",
                                        "if_report_sha256": None, **reviewed})
        self.config.write_text(self.config.read_text().replace("test-model", "changed-model"))
        with self.assertRaisesRegex(profiles.ProfileError, "source changed"):
            send("calibration.measure", {**reviewed, "job_id": job_id})
        self.assertEqual(journal.load("local", job_id)["state"], "failed")
        self.assertFalse(path.exists())

    def test_unrelated_report_cannot_finish_calibration_receipt(self):
        def send(action, params):
            return management.dispatch({"schema_version": 1, "action": action,
                                        "profile": "local", "params": params})

        plan = send("calibration.plan", {})["plan"]
        path = self.root / "unrelated-report.json"
        job_id = uuid.uuid4().hex
        send("calibration.job.create", {"job_id": job_id, "action": "measure",
            "path": str(path), "exclusive_ack": True,
            "if_profile_revision": plan["profile_revision"],
            "if_source_sha256": plan["source_sha256"], "if_report_sha256": None})
        calibration.measure("local", path, exclusive_ack=True, runner=self.fake_result,
                            say=lambda _: None, job_id=uuid.uuid4().hex)
        self.assertEqual(send("calibration.job", {"job_id": job_id})["job"]["state"], "launching")
        self.assertEqual(journal.load("local", job_id)["state"], "running")

    def test_active_measurement_progress_and_exact_cancel(self):
        from tools import calibrate as cal

        def send(action, params):
            return management.dispatch({"schema_version": 1, "action": action,
                                        "profile": "local", "params": params})

        path = self.root / "stopped-calibration.json"
        plan = send("calibration.plan", {})["plan"]
        job_id = uuid.uuid4().hex
        reviewed = {"path": str(path), "exclusive_ack": True,
                    "if_profile_revision": plan["profile_revision"],
                    "if_source_sha256": plan["source_sha256"]}
        send("calibration.job.create", {"job_id": job_id, "action": "measure",
                                        "if_report_sha256": None, **reviewed})
        started = threading.Event()
        outcome = []
        original = self.config.read_bytes()

        def waiting(cfg, say, *, progress, cancel_event):
            progress("pcie_share", 2, 21)
            started.set()
            if not cancel_event.wait(4):
                raise AssertionError("operator cancellation did not reach the worker")
            raise cal.CalibrationCancelled("operator requested stop")

        def worker():
            try:
                send("calibration.measure", {**reviewed, "job_id": job_id})
            except BaseException as exc:
                outcome.append(exc)

        def bridge(action):
            request = {"schema_version": 1, "action": action, "profile": "local",
                       "params": {"job_id": job_id}}
            child = subprocess.run([sys.executable, "-m", "strata_control.management"],
                input=json.dumps(request), text=True, capture_output=True,
                cwd=Path(__file__).resolve().parents[2], env=os.environ.copy(), check=True)
            return json.loads(child.stdout)

        with mock.patch("tools.calibrate.run", side_effect=waiting):
            thread = threading.Thread(target=worker)
            thread.start()
            try:
                self.assertTrue(started.wait(4))
                active = send("calibration.job", {"job_id": job_id})["job"]
                self.assertEqual(active["state"], "running")
                self.assertEqual(active["activity"], "active")
                self.assertEqual(active["progress"]["stage"], "pcie_share")
                self.assertEqual(active["progress"]["completed"], 2)
                fresh = bridge("calibration.job")["job"]
                self.assertEqual(fresh["state"], "running")
                self.assertEqual(fresh["progress"]["completed"], 2)
                with self.assertRaisesRegex(profiles.ProfileError, "not found"):
                    send("calibration.cancel", {"job_id": uuid.uuid4().hex})
                self.assertEqual(bridge("calibration.cancel")["state"], "cancel_requested")
            finally:
                thread.join(timeout=6)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], profiles.ProfileError)
        self.assertEqual(outcome[0].exit_code, 130)
        finished = send("calibration.job", {"job_id": job_id})["job"]
        self.assertEqual(finished["state"], "cancelled")
        self.assertTrue(finished["cancel_requested"])
        self.assertFalse(path.exists())
        self.assertEqual(self.config.read_bytes(), original)
        with self.assertRaisesRegex(profiles.ProfileError, "not active"):
            send("calibration.cancel", {"job_id": job_id})

    def test_pruning_finished_receipt_removes_progress_and_cancel_marker(self):
        from strata_control import calibration_jobs

        def send(action, params):
            return management.dispatch({"schema_version": 1, "action": action,
                                        "profile": "local", "params": params})

        plan = send("calibration.plan", {})["plan"]
        shared = {"action": "measure", "exclusive_ack": True,
                  "if_profile_revision": plan["profile_revision"],
                  "if_source_sha256": plan["source_sha256"], "if_report_sha256": None}
        first = uuid.uuid4().hex
        send("calibration.job.create", {**shared, "job_id": first,
                                        "path": str(self.root / "first.json")})
        calibration_jobs._save_progress("local", first, "pcie_share", 2, 21)
        marker = calibration_jobs._cancel_path("local", first)
        marker.write_bytes(b"1")
        marker.chmod(0o600)
        journal.finish("local", first, "failed", {"cancelled": True})
        with mock.patch.object(journal, "MAX_RECORDS", 1):
            send("calibration.job.create", {**shared, "job_id": uuid.uuid4().hex,
                                            "path": str(self.root / "second.json")})
        self.assertFalse(calibration_jobs._progress_path("local", first).exists())
        self.assertFalse(marker.exists())

    def test_management_bridge_reviews_and_journals_exclusive_calibration(self):
        def send(action, params):
            return management.dispatch({"schema_version": 1, "action": action,
                                        "profile": "local", "params": params})

        report_path = self.root / "bridge-calibration.json"
        initial = self.config.read_bytes()
        plan = send("calibration.plan", {})["plan"]
        self.assertEqual(plan["source_config"], str(self.config.resolve()))
        self.assertTrue(plan["component_preflight"]["ok"])
        source = {"if_profile_revision": plan["profile_revision"],
                  "if_source_sha256": plan["source_sha256"]}
        measure = {"path": str(report_path), "exclusive_ack": True, **source}
        with self.assertRaisesRegex(profiles.ProfileError, "exclusive-access"):
            send("calibration.measure", {**measure, "exclusive_ack": False})
        with self.assertRaisesRegex(profiles.ProfileError, "source changed"):
            send("calibration.measure", {**measure, "if_source_sha256": "0" * 64})
        self.assertEqual(journal.list_runs("local"), [])
        with mock.patch("tools.calibrate.run", side_effect=self.fake_result), \
                mock.patch("strata_control.calibration._smoke", return_value=3):
            measured = send("calibration.measure", measure)
            self.assertTrue(measured["ok"])
            self.assertTrue(measured["applicable"])
            self.assertEqual(self.config.read_bytes(), initial)
            self.assertEqual(journal.load("local", measured["operation_id"])["state"], "succeeded")

            apply_plan = send("calibration.apply.plan", {"path": str(report_path)})
            self.assertTrue(apply_plan["plan"]["applicable"])
            apply = {"path": str(report_path), "exclusive_ack": True,
                     "if_profile_revision": apply_plan["profile_revision"],
                     "if_source_sha256": apply_plan["source_sha256"],
                     "if_report_sha256": apply_plan["report_sha256"]}
            with self.assertRaisesRegex(profiles.ProfileError, "report changed"):
                send("calibration.apply", {**apply, "if_report_sha256": "0" * 64})
            self.assertEqual(len(journal.list_runs("local")), 1)
            applied = send("calibration.apply", apply)
            self.assertTrue(applied["application"]["changed"])
            self.assertTrue(Path(applied["application"]["application_record"]).is_file())

            rollback_plan = send("calibration.rollback.plan", {"path": str(report_path)})
            self.assertTrue(rollback_plan["plan"]["applicable"])
            rolled = send("calibration.rollback", {"path": str(report_path),
                "exclusive_ack": True, "if_profile_revision": rollback_plan["profile_revision"],
                "if_source_sha256": rollback_plan["source_sha256"],
                "if_report_sha256": rollback_plan["report_sha256"]})
            self.assertTrue(rolled["rollback"]["changed"])
        self.assertEqual(self.config.read_bytes(), initial)
        self.assertEqual([row["kind"] for row in journal.list_runs("local")],
                         ["engine.calibrate.rollback", "engine.calibrate.apply",
                          "engine.calibrate.measure"])

    def test_missing_report_folder_is_refused_before_the_sweep(self):
        # #3187: the sweep takes ~19 minutes; the folder check must not wait until after it
        for output in (self.root / "no-such-dir" / "report.json", self.config / "report.json"):
            with self.assertRaisesRegex(profiles.ProfileError, "report output directory does not exist"):
                calibration.measure("local", output, exclusive_ack=True,
                                    runner=lambda *_a, **_k: self.fail("the sweep must not start"))

    def test_calibration_rechecks_reviewed_digests_inside_operation_lock(self):
        report_path = self.root / "pinned-calibration.json"
        with self.assertRaisesRegex(profiles.ProfileError, "source changed after review"):
            calibration.measure("local", report_path, exclusive_ack=True,
                runner=lambda *_args, **_kwargs: self.fail("runner must not start"),
                expected_source_sha256="0" * 64)
        self.assertFalse(report_path.exists())
        report = calibration.measure("local", report_path, exclusive_ack=True,
                                     runner=self.fake_result, say=lambda _: None)
        source = self.config.read_bytes()
        with self.assertRaisesRegex(profiles.ProfileError, "report changed after review"):
            calibration.apply_result("local", report_path, exclusive_ack=True,
                expected_report_sha256="0" * 64,
                smoke=lambda *_args: self.fail("smoke must not start"))
        self.assertEqual(self.config.read_bytes(), source)
        calibration.apply_result("local", report_path, exclusive_ack=True,
                                 expected_report_sha256=report["body_sha256"], smoke=lambda *_args: 2)
        with self.assertRaisesRegex(profiles.ProfileError, "report changed after review"):
            calibration.rollback_result("local", report_path, exclusive_ack=True,
                expected_report_sha256="0" * 64,
                smoke=lambda *_args: self.fail("smoke must not start"))

    def test_measure_review_apply_and_setup_persistence(self):
        before = self.config.read_bytes()
        report_path = self.root / "calibration.json"
        report = calibration.measure("local", report_path, exclusive_ack=True, runner=self.fake_result, say=lambda _: None)
        self.assertEqual(self.config.read_bytes(), before)
        self.assertTrue(report["applicable"])
        self.assertEqual(report["settings"]["--pcie-frac"], "0.20")
        if os.name != "nt":
            self.assertEqual(stat.S_IMODE(report_path.stat().st_mode), 0o600)
        preview = calibration.review("local", report_path)
        self.assertTrue(preview["applicable"])
        self.assertIn("--pcie-frac", preview["after_args"])
        applied = calibration.apply_result("local", report_path, exclusive_ack=True, smoke=lambda cfg: 4)
        self.assertTrue(applied["changed"])
        self.assertEqual(applied["smoke_output_tokens"], 4)
        self.assertEqual(Path(applied["backup"]).read_bytes(), before)
        self.assertIn("--pcie-frac", json.loads(self.config.read_text())["args"])
        settings = json.loads((self.settings_file).read_text())
        self.assertEqual(settings["calibration"]["test-gpu|24GB|test-cpu"]["settings"], report["settings"])
        self.assertFalse(calibration.review("local", report_path)["applicable"])

    def test_review_and_rollback_restore_config_and_absent_settings(self):
        before = self.config.read_bytes()
        report_path = self.root / "calibration.json"
        calibration.measure("local", report_path, exclusive_ack=True, runner=self.fake_result, say=lambda _: None)
        applied = calibration.apply_result("local", report_path, exclusive_ack=True, smoke=lambda cfg: 3)
        self.assertTrue(Path(applied["application_record"]).is_file())
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(Path(applied["application_record"]).stat().st_mode), 0o600)
        preview = calibration.review_rollback("local", report_path)
        self.assertTrue(preview["applicable"], preview["reasons"])
        self.assertTrue(preview["setup_settings_would_be_removed"])
        restored = calibration.rollback_result("local", report_path, exclusive_ack=True, smoke=lambda cfg: 2)
        self.assertTrue(restored["changed"])
        self.assertEqual(restored["smoke_output_tokens"], 2)
        self.assertEqual(self.config.read_bytes(), before)
        self.assertFalse((self.settings_file).exists())
        self.assertFalse(calibration.review_rollback("local", report_path)["applicable"])

    @unittest.skipUnless(os.name == "posix", "managed activity lease is POSIX-only")
    def test_managed_activity_lease_blocks_apply_and_rollback_smoke(self):
        report_path = self.root / "calibration.json"

        def assert_exclusive():
            with self.assertRaisesRegex(profiles.ProfileError, "engine lease"):
                runtime.acquire_activity_lease(shared=True)

        def runner(cfg, say):
            assert_exclusive()
            return self.fake_result(cfg, say)

        def smoke(cfg):
            assert_exclusive()
            return 2

        calibration.measure("local", report_path, exclusive_ack=True,
                            runner=runner, say=lambda _: None)
        before = self.config.read_bytes()
        lease = runtime.acquire_activity_lease(shared=True)
        try:
            with self.assertRaisesRegex(profiles.ProfileError, "engine lease"):
                calibration.apply_result("local", report_path, exclusive_ack=True,
                                         smoke=lambda cfg: self.fail("apply smoke ran"))
        finally:
            os.close(lease)
        self.assertEqual(self.config.read_bytes(), before)
        calibration.apply_result("local", report_path, exclusive_ack=True, smoke=smoke)
        applied = self.config.read_bytes()
        lease = runtime.acquire_activity_lease(shared=True)
        try:
            with self.assertRaisesRegex(profiles.ProfileError, "engine lease"):
                calibration.rollback_result("local", report_path, exclusive_ack=True,
                                            smoke=lambda cfg: self.fail("rollback smoke ran"))
        finally:
            os.close(lease)
        self.assertEqual(self.config.read_bytes(), applied)
        with mock.patch("strata_control.calibration._port_active", return_value=True):
            with self.assertRaisesRegex(profiles.ProfileError, "server is listening"):
                calibration.rollback_result("local", report_path, exclusive_ack=True,
                                            smoke=lambda cfg: self.fail("rollback smoke ran"))
        self.assertEqual(self.config.read_bytes(), applied)
        self.assertTrue(calibration.rollback_result("local", report_path, exclusive_ack=True,
                                                    smoke=smoke)["changed"])

    def test_rollback_restores_existing_settings_exactly(self):
        settings_path = self.settings_file
        settings_path.parent.mkdir(parents=True)
        original_settings = b'{"unrelated": "keep", "calibration": {"other-device": {"settings": {}}}}\n'
        settings_path.write_bytes(original_settings)
        original_config = self.config.read_bytes()
        report_path = self.root / "calibration.json"
        calibration.measure("local", report_path, exclusive_ack=True, runner=self.fake_result, say=lambda _: None)
        applied = calibration.apply_result("local", report_path, exclusive_ack=True, smoke=lambda cfg: 2)
        self.assertEqual(Path(applied["settings_backup"]).read_bytes(), original_settings)
        calibration.rollback_result("local", report_path, exclusive_ack=True, smoke=lambda cfg: 2)
        self.assertEqual(settings_path.read_bytes(), original_settings)
        self.assertEqual(self.config.read_bytes(), original_config)

    def test_rollback_refuses_stale_state_and_failed_smoke(self):
        report_path = self.root / "calibration.json"
        calibration.measure("local", report_path, exclusive_ack=True, runner=self.fake_result, say=lambda _: None)
        calibration.apply_result("local", report_path, exclusive_ack=True, smoke=lambda cfg: 2)
        applied_config = self.config.read_bytes()
        settings_path = self.settings_file
        applied_settings = settings_path.read_bytes()
        with self.assertRaisesRegex(profiles.ProfileError, "rollback smoke generation failed"):
            calibration.rollback_result("local", report_path, exclusive_ack=True, smoke=lambda cfg: 1 / 0)
        self.assertEqual(self.config.read_bytes(), applied_config)
        settings_path.write_bytes(applied_settings + b" ")
        preview = calibration.review_rollback("local", report_path)
        self.assertIn("setup_settings_changed_after_apply", preview["reasons"])
        with self.assertRaisesRegex(profiles.ProfileError, "rollback is stale"):
            calibration.rollback_result("local", report_path, exclusive_ack=True, smoke=lambda cfg: 2)
        self.assertEqual(self.config.read_bytes(), applied_config)
        settings_path.write_bytes(applied_settings)
        self.config.write_bytes(applied_config + b" ")
        preview = calibration.review_rollback("local", report_path)
        self.assertIn("run_config_changed_after_apply", preview["reasons"])
        with self.assertRaisesRegex(profiles.ProfileError, "rollback is stale"):
            calibration.rollback_result("local", report_path, exclusive_ack=True, smoke=lambda cfg: 2)

    def test_rollback_detects_change_during_smoke(self):
        report_path = self.root / "calibration.json"
        calibration.measure("local", report_path, exclusive_ack=True, runner=self.fake_result, say=lambda _: None)
        calibration.apply_result("local", report_path, exclusive_ack=True, smoke=lambda cfg: 2)
        applied_config = self.config.read_bytes()
        settings_path = self.settings_file
        applied_settings = settings_path.read_bytes()

        def smoke(cfg):
            settings_path.write_bytes(applied_settings + b" ")
            return 2

        with self.assertRaisesRegex(profiles.ProfileError, "changed during rollback smoke"):
            calibration.rollback_result("local", report_path, exclusive_ack=True, smoke=smoke)
        self.assertEqual(self.config.read_bytes(), applied_config)

    def test_rollback_rejects_modified_backup(self):
        report_path = self.root / "calibration.json"
        calibration.measure("local", report_path, exclusive_ack=True, runner=self.fake_result, say=lambda _: None)
        applied = calibration.apply_result("local", report_path, exclusive_ack=True, smoke=lambda cfg: 2)
        Path(applied["backup"]).write_bytes(b"{}")
        preview = calibration.review_rollback("local", report_path)
        self.assertIn("config_backup_changed", preview["reasons"])
        with self.assertRaisesRegex(profiles.ProfileError, "rollback is stale"):
            calibration.rollback_result("local", report_path, exclusive_ack=True, smoke=lambda cfg: 2)

    def test_rollback_requires_exclusive_acknowledgement(self):
        report_path = self.root / "calibration.json"
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["engine", "calibrate", "--profile", "local", "--rollback-result",
                         str(report_path), "--output", "json"])
        self.assertEqual(code, 2)
        self.assertFalse(json.loads(output.getvalue())["ok"])
        with self.assertRaisesRegex(profiles.ProfileError, "exclusive-access"):
            calibration.rollback_result("local", report_path)

    @unittest.skipUnless(os.name == "posix", "symlink test requires POSIX")
    def test_apply_refuses_symlinked_backup_directory(self):
        report_path = self.root / "calibration.json"
        calibration.measure("local", report_path, exclusive_ack=True, runner=self.fake_result, say=lambda _: None)
        previous = self.config.read_bytes()
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        (self.root / ".strata-calibration-backups").symlink_to(elsewhere, target_is_directory=True)
        with self.assertRaisesRegex(profiles.ProfileError, "owner-only and not a symlink"):
            calibration.apply_result("local", report_path, exclusive_ack=True, smoke=lambda cfg: 2)
        self.assertEqual(self.config.read_bytes(), previous)
        self.assertEqual(list(elsewhere.iterdir()), [])

    def test_rollback_write_failure_restores_applied_state(self):
        report_path = self.root / "calibration.json"
        calibration.measure("local", report_path, exclusive_ack=True, runner=self.fake_result, say=lambda _: None)
        calibration.apply_result("local", report_path, exclusive_ack=True, smoke=lambda cfg: 2)
        current_config = self.config.read_bytes()
        settings_path = self.settings_file
        current_settings = settings_path.read_bytes()
        original_replace = calibration._replace_bytes
        failures = 0

        def fail_restore(path, payload):
            nonlocal failures
            if path == self.config.resolve() and failures == 0:
                failures += 1
                original_replace(path, payload)
                raise OSError("simulated failure after config replacement")
            return original_replace(path, payload)

        with mock.patch("strata_control.calibration._replace_bytes", side_effect=fail_restore):
            with self.assertRaisesRegex(profiles.ProfileError, "applied state restored"):
                calibration.rollback_result("local", report_path, exclusive_ack=True, smoke=lambda cfg: 2)
        self.assertEqual(self.config.read_bytes(), current_config)
        self.assertEqual(settings_path.read_bytes(), current_settings)

    def test_failed_smoke_or_changed_config_never_applies(self):
        report_path = self.root / "calibration.json"
        calibration.measure("local", report_path, exclusive_ack=True, runner=self.fake_result, say=lambda _: None)
        before = self.config.read_bytes()
        with self.assertRaisesRegex(profiles.ProfileError, "smoke generation failed"):
            calibration.apply_result("local", report_path, exclusive_ack=True, smoke=lambda cfg: 1 / 0)
        self.assertEqual(self.config.read_bytes(), before)
        self.config.write_bytes(before + b" ")
        with self.assertRaisesRegex(profiles.ProfileError, "stale"):
            calibration.apply_result("local", report_path, exclusive_ack=True, smoke=lambda cfg: 4)

    def test_report_id_and_settings_are_validated_before_apply(self):
        report_path = self.root / "calibration.json"
        calibration.measure("local", report_path, exclusive_ack=True, runner=self.fake_result, say=lambda _: None)
        report = json.loads(report_path.read_text())
        report["report_id"] = "../../outside"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(profiles.ProfileError, "unsupported calibration report"):
            calibration.review("local", report_path)
        report["report_id"] = "0" * 32
        report["settings"] = {"--pool-workers": "999999"}
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(profiles.ProfileError, "positive whole number"):
            calibration.review("local", report_path)
        report["settings"] = {"--pcie-frac": "0.75"}
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(profiles.ProfileError, "changed after measurement"):
            calibration.review("local", report_path)

    def test_machine_cli_measure_review_apply_emits_one_json_record(self):
        report_path = self.root / "calibration.json"
        with mock.patch("tools.calibrate.run", side_effect=self.fake_result), \
                mock.patch("strata_control.calibration._smoke", return_value=3):
            for args, expected_key in ((["--measure", "--out", str(report_path), "--exclusive"], "report"),
                                       (["--review-result", str(report_path)], "review"),
                                       (["--apply-result", str(report_path), "--exclusive"], "application"),
                                       (["--review-rollback", str(report_path)], "rollback_review"),
                                       (["--rollback-result", str(report_path), "--exclusive"], "rollback")):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = main(["engine", "calibrate", "--profile", "local", *args, "--output", "json"])
                self.assertEqual(code, 0)
                parsed = json.loads(output.getvalue())
                self.assertTrue(parsed["ok"])
                self.assertIn(expected_key, parsed)
                if expected_key in ("report", "application", "rollback"):
                    record = journal.load("local", parsed["operation_id"])
                    self.assertEqual(record["schema_version"], 2)
                    self.assertEqual(record["origin"], "direct")
                    self.assertEqual(record["state"], "succeeded")
                    self.assertNotIn("candidate", json.dumps(record))
                else:
                    self.assertNotIn("operation_id", parsed)
        self.assertEqual([row["kind"] for row in journal.list_runs("local")],
                         ["engine.calibrate.rollback", "engine.calibrate.apply", "engine.calibrate.measure"])

    def test_explicit_run_config_path_retains_legacy_nonjournaled_calibration(self):
        report_path = self.root / "calibration-path.json"
        output = io.StringIO()
        with mock.patch("tools.calibrate.run", side_effect=self.fake_result):
            with redirect_stdout(output):
                code = main(["engine", "calibrate", "--profile", str(self.config), "--measure",
                             "--exclusive", "--out", str(report_path), "--output", "json"])
        self.assertEqual(code, 0)
        self.assertNotIn("operation_id", json.loads(output.getvalue()))
        self.assertEqual(journal.list_runs("local"), [])

    def test_direct_apply_failure_is_a_compact_failed_operation(self):
        report_path = self.root / "calibration.json"
        calibration.measure("local", report_path, exclusive_ack=True, runner=self.fake_result, say=lambda _: None)
        output = io.StringIO()
        with mock.patch("strata_control.calibration._smoke", side_effect=OSError("private engine trace")):
            with redirect_stdout(output):
                code = main(["engine", "calibrate", "--profile", "local", "--apply-result",
                             str(report_path), "--exclusive", "--output", "json"])
        self.assertEqual(code, 6)
        self.assertFalse(json.loads(output.getvalue())["ok"])
        record = journal.load("local", journal.list_runs("local")[0]["operation_id"])
        self.assertEqual((record["kind"], record["state"]), ("engine.calibrate.apply", "failed"))
        self.assertNotIn("private engine trace", json.dumps(record))

    def test_measure_requires_exclusive_access_and_rejects_live_profile_port(self):
        report_path = self.root / "calibration.json"
        with self.assertRaisesRegex(profiles.ProfileError, "exclusive-access"):
            calibration.measure("local", report_path, runner=self.fake_result)
        with mock.patch("strata_control.calibration._port_active", return_value=True):
            with self.assertRaisesRegex(profiles.ProfileError, "server is listening"):
                calibration.measure("local", report_path, exclusive_ack=True, runner=self.fake_result)
        profiles.set_values("local", ["port=9123"])
        with mock.patch("strata_control.calibration._port_active",
                        side_effect=lambda cfg: cfg.get("port") == 9123) as checked:
            with self.assertRaisesRegex(profiles.ProfileError, "server is listening"):
                calibration.measure("local", report_path, exclusive_ack=True, runner=self.fake_result)
        self.assertEqual([call.args[0]["port"] for call in checked.call_args_list], [8095, 9123])
        self.assertFalse(report_path.exists())

    def test_measure_and_smoke_use_saved_engine_overrides_without_writing_them(self):
        profiles.set_values("local", ["parallel=8", "batch_groups=2"])
        source = self.config.read_bytes()
        report_path = self.root / "calibration.json"
        seen = {}

        def runner(cfg, say):
            seen["measured"] = cfg
            return self.fake_result(cfg, say)

        def smoke(cfg):
            seen["smoked"] = cfg
            return 2

        calibration.measure("local", report_path, exclusive_ack=True, runner=runner, say=lambda _: None)
        self.assertEqual(seen["measured"]["parallel"], 8)
        self.assertEqual(seen["measured"]["args"][-2:], ["--batch-groups", "2"])
        self.assertEqual(self.config.read_bytes(), source)
        calibration.apply_result("local", report_path, exclusive_ack=True, smoke=smoke)
        self.assertEqual(seen["smoked"]["parallel"], 8)
        self.assertIn("--batch-groups", seen["smoked"]["args"])
        saved_args = json.loads(self.config.read_text())["args"]
        self.assertNotIn("--batch-groups", saved_args)

    def test_profile_edit_during_measurement_marks_report_stale(self):
        report_path = self.root / "calibration.json"

        def runner(cfg, say):
            profiles.set_values("local", ["parallel=4"])
            return self.fake_result(cfg, say)

        report = calibration.measure("local", report_path, exclusive_ack=True, runner=runner, say=lambda _: None)
        self.assertFalse(report["applicable"])
        preview = calibration.review("local", report_path)
        self.assertFalse(preview["applicable"])
        self.assertIn("measurement_became_stale", preview["reasons"])

    def test_profile_edit_during_smoke_prevents_apply(self):
        report_path = self.root / "calibration.json"
        calibration.measure("local", report_path, exclusive_ack=True, runner=self.fake_result, say=lambda _: None)
        source = self.config.read_bytes()

        def smoke(cfg):
            profiles.set_values("local", ["parallel=4"])
            return 2

        with self.assertRaisesRegex(profiles.ProfileError, "changed during smoke"):
            calibration.apply_result("local", report_path, exclusive_ack=True, smoke=smoke)
        self.assertEqual(self.config.read_bytes(), source)

    def test_settings_write_failure_restores_previous_run_config(self):
        report_path = self.root / "calibration.json"
        calibration.measure("local", report_path, exclusive_ack=True, runner=self.fake_result, say=lambda _: None)
        source = self.config.read_bytes()
        original_replace = calibration._replace_bytes
        failures = 0

        def fail_settings(path, payload):
            nonlocal failures
            if path.name == "settings.json":
                failures += 1
                if failures == 1:
                    original_replace(path, payload)
                    raise OSError("simulated failure after settings replacement")
            return original_replace(path, payload)

        with mock.patch("strata_control.calibration._replace_bytes", side_effect=fail_settings):
            with self.assertRaisesRegex(profiles.ProfileError, "config restored"):
                calibration.apply_result("local", report_path, exclusive_ack=True, smoke=lambda cfg: 2)
        self.assertEqual(self.config.read_bytes(), source)
        self.assertFalse((self.settings_file).exists())


if __name__ == "__main__":
    unittest.main()
