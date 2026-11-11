"""Serving benchmark evidence with a live mock HTTP server, no model weights."""

import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, IM_END, MockEngine, Service, serve
from strata_cli import bench
from strata_cli.cli import main
from strata_cli.client import Endpoint
from strata_control import benchmarking, journal, management
from strata_control.private_files import is_unshared
from strata_control.profiles import ProfileError


class _NoGpu:
    note = None

    def ok(self):
        return False

    def name(self):
        return None

    def read(self):
        return {}


class BenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # The telemetry sampler reads the machine's real GPU on a timer; a card's first reading lands
        # between two benchmark runs and makes their reported environments differ. Use no GPU.
        patcher = mock.patch("serve.telemetry.gpu_reader", lambda index=0, amd=False: _NoGpu())
        patcher.start()
        self.addCleanup(patcher.stop)
        tok = ByteTokenizer()
        service = Service(MockEngine(tok, "Benchmark mock answer.", max_context=4096), tok,
                          ChatTemplate(Path(__file__).resolve().parents[2] / "serve/chat_template.jinja"))
        service.api_key = "inference-test-secret"
        service.telemetry_key = "monitor-test-secret"
        self.service = service
        self.httpd = serve(service, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.inference = Endpoint(self.url, service.api_key)
        self.telemetry = Endpoint(self.url, service.telemetry_key)

    def test_live_benchmark_writes_private_report_without_prompt_or_answer(self):
        out = self.root / "smoke"
        with redirect_stdout(io.StringIO()):
            report = bench.run(self.inference, self.telemetry, out, concurrency=2)
        self.assertTrue(report["successful"])
        self.assertEqual([p["kind"] for p in report["phases"]], ["warmup", "serial", "mixed"])
        self.assertEqual(report["phases"][2]["requested"], 2)
        self.assertEqual(report["phases"][2]["client_workers"], 2)
        self.assertEqual(report["instance_id"], report["after"]["instance_id"])
        self.assertTrue(all(r["ttft_s"] is not None and r["completion_tokens"] is not None
                            for p in report["phases"] for r in p["requests"]))
        contents = (out / "results.json").read_text(encoding="utf-8")
        for secret in ("inference-test-secret", "monitor-test-secret", "Benchmark mock answer.",
                       "Answer in one sentence"):
            self.assertNotIn(secret, contents)
        self.assertIn("prompt_sha256", contents)
        if os.name != "nt":
            self.assertEqual(stat.S_IMODE(out.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE((out / "results.json").stat().st_mode), 0o600)
        self.assertEqual(bench.summary(bench.load(out))["successful"], True)
        self.assertEqual(self.telemetry.snapshot()["work"]["client_disconnect_cancels"], 0)

    def test_phase_progress_is_private_and_recovers_from_saved_report(self):
        output = self.root / "progress"
        reached = threading.Event()
        release = threading.Event()
        original = bench._phase

        def held_phase(*args, **kwargs):
            if kwargs["kind"] == "mixed":
                reached.set()
                if not release.wait(10):
                    raise AssertionError("progress test did not release the mixed phase")
            return original(*args, **kwargs)

        with mock.patch.object(bench, "_phase", side_effect=held_phase), \
                ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(bench.run, self.inference, self.telemetry, output, concurrency=2)
            try:
                self.assertTrue(reached.wait(10))
                active = bench.progress(output)
                self.assertEqual(active["state"], "running")
                self.assertEqual(active["completed_phases"], 2)
                self.assertEqual(active["current_phase"], "mixed")
            finally:
                release.set()
            self.assertTrue(future.result(timeout=10)["successful"])
        saved = bench.progress(output / "results.json")
        self.assertEqual(saved["state"], "report_saved")
        self.assertEqual(saved["completed_phases"], 3)
        self.assertTrue(saved["successful"])
        raw = (output / "progress.json").read_text()
        self.assertNotIn("inference-test-secret", raw)
        self.assertNotIn("Benchmark mock answer", raw)
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE((output / "progress.json").stat().st_mode), 0o600)

    def test_saved_profile_benchmark_lease_refuses_overlap(self):
        with mock.patch.dict(os.environ, {"STRATA_PROFILE_DIR": str(self.root / "profiles")}):
            output = self.root / "leased-run"
            self.assertEqual(benchmarking.benchmark_activity("local", output), "untracked")
            with benchmarking.benchmark_lease("local", output):
                self.assertEqual(benchmarking.benchmark_activity("local", output), "active")
                self.assertEqual(benchmarking.benchmark_activity("local", self.root / "other-run"),
                                 "other_active")
                raw = (self.root / "profiles" / ".bench-local.lock").read_bytes()
                self.assertNotIn(os.fsencode(str(output)), raw)
                with self.assertRaisesRegex(ProfileError, "another benchmark is running"):
                    with benchmarking.benchmark_lease("local"):
                        self.fail("overlapping benchmark lease was accepted")
            self.assertEqual(benchmarking.benchmark_activity("local", output), "ended")
            self.assertEqual(benchmarking.benchmark_activity("local", self.root / "raw-run"),
                             "untracked")
            with benchmarking.benchmark_lease("local"):
                self.assertEqual(benchmarking.benchmark_activity("local", output), "untracked")

    def test_operator_cancel_stops_live_stream_and_saves_distinct_evidence(self):
        output = self.root / "operator-stopped"
        self.service.engine.delay = 0.15
        with mock.patch.dict(os.environ, {"STRATA_PROFILE_DIR": str(self.root / "profiles")}), \
                ThreadPoolExecutor(max_workers=1) as pool:
            def execute():
                with benchmarking.benchmark_lease("local", output):
                    return bench.run(self.inference, self.telemetry, output,
                                     concurrency=2, request_timeout=30)
            started = time.monotonic()
            future = pool.submit(execute)
            deadline = started + 10
            while time.monotonic() < deadline:
                if output.exists() and self.telemetry.snapshot()["work"]["active"] > 0:
                    break
                time.sleep(0.05)
            else:
                self.fail("benchmark did not enter an active request")
            wrong = self.root / "another-run"
            wrong.mkdir(mode=0o700)
            with self.assertRaisesRegex(ProfileError, "exact benchmark run is not active"):
                benchmarking.request_cancel("local", wrong)
            linked = self.root / "linked-run"
            linked.symlink_to(output, target_is_directory=True)
            with self.assertRaisesRegex(ProfileError, "owner-only"):
                benchmarking.request_cancel("local", linked)
            with mock.patch("strata_control.profiles.load", return_value=({"id": "local"}, "a" * 16)):
                request = management.dispatch({"schema_version": 1, "action": "bench.cancel",
                                               "profile": "local",
                                               "params": {"path": str(output / "results.json")}})
            self.assertEqual(request["state"], "cancel_requested")
            report = future.result(timeout=10)
            self.assertLess(time.monotonic() - started, 10)
            self.assertTrue(report["cancelled"])
            self.assertFalse(report["successful"])
            self.assertEqual(bench.progress(output)["state"], "cancelled")
            self.assertTrue(benchmarking.cancel_requested(output))
            self.assertTrue(bench.load(output)["cancelled"])
            self.assertIn("| Result | Cancelled |", bench.markdown_report(report))
            self.assertEqual(benchmarking.benchmark_activity("local", output), "ended")
            with self.assertRaisesRegex(ProfileError, "not active"):
                benchmarking.request_cancel("local", output)

    def test_progress_bridge_reports_exact_run_lease(self):
        output = self.root / "bridge-progress"
        with redirect_stdout(io.StringIO()):
            bench.run(self.inference, self.telemetry, output, concurrency=1)
        request = {"schema_version": 1, "action": "bench.progress", "profile": "local",
                   "params": {"path": str(output / "results.json")}}
        with mock.patch.dict(os.environ, {"STRATA_PROFILE_DIR": str(self.root / "profiles")}), \
                mock.patch("strata_control.profiles.load", return_value=({"id": "local"}, "a" * 16)):
            with benchmarking.benchmark_lease("local", output):
                active = management.dispatch(request)
                self.assertEqual(active["activity"], "active")
                self.assertEqual(active["progress"]["state"], "report_saved")
            self.assertEqual(management.dispatch(request)["activity"], "ended")

    def test_durable_benchmark_job_receipt_survives_host_state_loss(self):
        output = self.root / "durable-job"
        instance = self.telemetry.snapshot()["instance"]["id"]
        revision, source = "a" * 16, "b" * 64
        job_id = "c" * 32
        params = {"path": str(output), "suite": "serving-smoke", "concurrency": 2,
                  "repeat": 1, "request_timeout": 30, "condition": "baseline",
                  "if_profile_revision": revision, "if_source_sha256": source,
                  "if_owner_id": "d" * 32, "if_instance_id": instance}
        identity = {"profile": "local", "revision": revision, "source_sha256": source}
        settings = (None, identity, self.inference, self.telemetry, output)
        planned = {"profile_revision": revision, "source_sha256": source,
                   "owner_id": "d" * 32, "instance_id": instance}
        (self.root / "profiles").mkdir(mode=0o700)
        with mock.patch.dict(os.environ, {"STRATA_PROFILE_DIR": str(self.root / "profiles")}), \
                mock.patch("strata_control.profiles.load", return_value=({"id": "local"}, revision)), \
                mock.patch.object(benchmarking, "_settings", return_value=settings), \
                mock.patch.object(benchmarking, "plan", return_value=planned), \
                mock.patch.object(bench, "_managed_match", return_value={"matched": True,
                                                                          "instance_id": instance}):
            created = management.dispatch({"schema_version": 1, "action": "bench.job.create",
                                           "profile": "local", "params": {**params, "job_id": job_id}})
            self.assertEqual(created["job_id"], job_id)
            self.assertEqual(benchmarking.job_status("local", job_id)["job"]["state"],
                             "launching")
            real_run = bench.run

            def report_before_worker_receipt(*args, **kwargs):
                report = real_run(*args, **kwargs)
                self.assertEqual(benchmarking.job_status("local", job_id)["job"]["state"],
                                 "completed")
                return report

            with mock.patch.object(bench, "run", side_effect=report_before_worker_receipt), \
                    redirect_stdout(io.StringIO()):
                result = management.dispatch({"schema_version": 1, "action": "bench.run",
                                              "profile": "local", "params": {**params, "job_id": job_id}})
            self.assertTrue(result["successful"])
            recovered = management.dispatch({"schema_version": 1, "action": "bench.job",
                                             "profile": "local", "params": {"job_id": job_id}})["job"]
            self.assertEqual(recovered["state"], "completed")
            self.assertTrue(recovered["result"]["successful"])
            self.assertEqual(recovered["result"]["run_id"], bench.load(output)["run_id"])
            self.assertEqual(recovered["progress"]["state"], "report_saved")
            self.assertEqual(journal.load("local", job_id)["state"], "succeeded")
            receipt = self.root / "profiles" / ".ops" / "local" / ".runs" / f"{job_id}.json"
            if os.name == "nt":
                self.assertTrue(is_unshared(receipt))     # NTFS has no mode bits: assert the effective ACL
            else:
                self.assertEqual(stat.S_IMODE(receipt.stat().st_mode), 0o600)

            failed_path = self.root / "failed-job"
            failed_params = {**params, "path": str(failed_path)}
            failed_id = "e" * 32
            management.dispatch({"schema_version": 1, "action": "bench.job.create",
                                 "profile": "local", "params": {**failed_params, "job_id": failed_id}})
            with mock.patch.object(benchmarking, "plan", side_effect=ProfileError("stale benchmark", 6)):
                with self.assertRaisesRegex(ProfileError, "stale benchmark"):
                    management.dispatch({"schema_version": 1, "action": "bench.run",
                                         "profile": "local", "params": {**failed_params, "job_id": failed_id}})
            self.assertEqual(benchmarking.job_status("local", failed_id)["job"]["state"], "failed")

            recovered_path = self.root / "report-before-receipt"
            recovered_params = {**params, "path": str(recovered_path)}
            recovered_id = "f" * 32
            management.dispatch({"schema_version": 1, "action": "bench.job.create",
                                 "profile": "local", "params": {**recovered_params, "job_id": recovered_id}})
            with redirect_stdout(io.StringIO()):
                bench.run(self.inference, self.telemetry, recovered_path, concurrency=2,
                          request_timeout=30, condition="baseline", profile_identity=identity,
                          require_managed_match=True)
            self.assertEqual(journal.load("local", recovered_id)["state"], "running")
            self.assertEqual(benchmarking.job_status("local", recovered_id)["job"]["state"],
                             "completed")
            self.assertEqual(journal.load("local", recovered_id)["state"], "succeeded")

    def test_process_death_releases_exact_run_lease(self):
        output = self.root / "child-run"
        ready = self.root / "child-ready"
        environment = {**os.environ, "STRATA_PROFILE_DIR": str(self.root / "profiles")}
        code = ("from pathlib import Path\n"
                "from strata_control.benchmarking import benchmark_lease\n"
                "from time import sleep\n"
                f"with benchmark_lease('local', Path({str(output)!r})):\n"
                f"    Path({str(ready)!r}).write_text('ready')\n"
                "    sleep(30)\n")
        process = subprocess.Popen([sys.executable, "-c", code],
                                   cwd=Path(__file__).resolve().parents[2], env=environment,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 10
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            if not ready.exists():
                self.fail("benchmark lease child did not become ready: " +
                          (process.stderr.read().decode("utf-8", "replace") if process.poll() is not None else
                           "timed out"))
            with mock.patch.dict(os.environ, {"STRATA_PROFILE_DIR": environment["STRATA_PROFILE_DIR"]}):
                self.assertEqual(benchmarking.benchmark_activity("local", output), "active")
                if os.name == "nt":
                    # a venv's python.exe is a launcher that runs the real interpreter as its child
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
                else:
                    process.kill()
                process.wait(timeout=5)
                self.assertEqual(benchmarking.benchmark_activity("local", output), "ended")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stderr.close()

    def test_interrupted_phase_cannot_produce_successful_report(self):
        output = self.root / "interrupted"
        original = bench._phase

        def interrupted_phase(*args, **kwargs):
            if kwargs["kind"] == "serial":
                raise RuntimeError("synthetic benchmark interruption")
            return original(*args, **kwargs)

        with mock.patch.object(bench, "_phase", side_effect=interrupted_phase), \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "synthetic benchmark interruption"):
                bench.run(self.inference, self.telemetry, output, concurrency=2)
        report = bench.load(output)
        self.assertEqual([p["kind"] for p in report["phases"]], ["warmup"])
        self.assertEqual(report["planned_phases"], ["warmup", "serial", "mixed"])
        self.assertFalse(report["successful"])
        self.assertEqual(bench.summary(report)["completed_phase_count"], 1)
        self.assertEqual(bench.summary(report)["planned_phase_count"], 3)
        self.assertIn("| Result | Incomplete |", bench.markdown_report(report))
        record = bench.progress(output)
        self.assertEqual(record["state"], "report_saved")
        self.assertEqual(record["completed_phases"], 1)
        self.assertEqual(record["current_phase"], None)
        self.assertFalse(record["successful"])
        report["successful"] = True
        (output / "results.json").write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ProfileError, "unsupported benchmark report"):
            bench.load(output)
        del report["planned_phases"]  # Old reports must not claim success for a partial suite either.
        (output / "results.json").write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ProfileError, "unsupported benchmark report"):
            bench.load(output)

    def test_postflight_telemetry_parse_failure_keeps_failed_evidence(self):
        output = self.root / "postflight"
        original = bench._phase
        postflight = False

        def phase(*args, **kwargs):
            nonlocal postflight
            result = original(*args, **kwargs)
            if kwargs["kind"] == "mixed":
                postflight = True
            return result

        def snapshot():
            if postflight:
                raise ValueError("malformed postflight telemetry")
            return self.telemetry.snapshot()

        telemetry = mock.Mock(timeout=self.telemetry.timeout)
        telemetry.snapshot.side_effect = snapshot
        with mock.patch.object(bench, "_phase", side_effect=phase), redirect_stdout(io.StringIO()):
            report = bench.run(self.inference, telemetry, output, concurrency=2)
        self.assertFalse(report["successful"])
        self.assertEqual(len(report["phases"]), 3)
        self.assertEqual(report["after"]["instance_id"], None)
        self.assertFalse(bench.load(output)["successful"])
        self.assertEqual(bench.progress(output)["state"], "report_saved")

    def test_auth_failure_is_recorded_and_does_not_claim_success(self):
        out = self.root / "failed"
        with redirect_stdout(io.StringIO()):
            report = bench.run(Endpoint(self.url, "wrong-key"), self.telemetry, out, concurrency=2)
        self.assertFalse(report["successful"])
        self.assertTrue((out / "results.json").exists())
        self.assertTrue(all(r["error_code"] == "authentication_failed"
                            for p in report["phases"] for r in p["requests"]))

    def test_unloaded_model_is_rejected_before_creating_output(self):
        out = self.root / "unloaded"
        unloaded = mock.Mock()
        unloaded.snapshot.return_value = {"instance": {"id": "one"}, "model": {"loaded": False}}
        with self.assertRaisesRegex(ProfileError, "loaded Strata instance"):
            bench.run(self.inference, unloaded, out)
        self.assertFalse(out.exists())

    def test_invalid_condition_is_rejected_before_creating_output(self):
        out = self.root / "invalid-label"
        with self.assertRaisesRegex(ProfileError, "condition must be a lowercase name"):
            bench.run(self.inference, self.telemetry, out, condition="monitor\x1b[31m")
        self.assertFalse(out.exists())

    def test_run_cli_records_an_operator_condition(self):
        out = self.root / "cli-baseline"
        output = io.StringIO()
        with mock.patch.dict(os.environ, {"STRATA_API_KEY": self.service.api_key}), redirect_stdout(output):
            code = main(["bench", "run", "--suite", "serving-smoke", "--url", self.url, "--condition", "baseline",
                         "--concurrency", "1", "--out", str(out), "--output", "json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["summary"]["condition"], "baseline")
        report = bench.load(out)
        self.assertEqual(report["condition"], "baseline")
        self.assertTrue(any("monitor attachment is not verified" in note for note in report["limitations"]))

    def test_managed_benchmark_checks_owned_instance_before_and_after(self):
        identity = {"profile": "local", "revision": "a" * 16,
                    "source_sha256": "b" * 64}
        instance = self.telemetry.snapshot()["instance"]["id"]
        owner = {"ok": True, "state": "ready", "launch_attested": True,
                 "api_identity_verified": True, "generation_ready": True,
                 "instance_id": instance, "profile_revision": identity["revision"], "api_url": self.url}
        plan = {"ok": True, "changed": False, "current_instance_id": instance,
                "current_profile_revision": identity["revision"],
                "candidate_profile_revision": identity["revision"]}
        out = self.root / "managed"
        with mock.patch("strata_control.runtime.status", return_value=owner) as status, \
                mock.patch("strata_control.runtime.plan_restart", return_value=plan) as preview, \
                redirect_stdout(io.StringIO()):
            report = bench.run(self.inference, self.telemetry, out, concurrency=1,
                               profile_identity=identity, require_managed_match=True)
        self.assertTrue(report["successful"])
        self.assertEqual(report["profile_running_match"], "managed_launch_checked")
        self.assertTrue(report["profile_running_check"]["before"]["matched"])
        self.assertTrue(report["profile_running_check"]["after"]["matched"])
        self.assertEqual(bench.summary(report)["profile_running_match"], "managed_launch_checked")
        self.assertEqual(status.call_count, 2)
        self.assertEqual(preview.call_count, 2)

        with mock.patch("strata_control.runtime.status", return_value={**owner, "instance_id": "wrong"}), \
                mock.patch("strata_control.runtime.plan_restart", return_value=plan):
            rejected = self.root / "rejected"
            with self.assertRaisesRegex(ProfileError, "managed benchmark preflight failed: instance_mismatch"):
                bench.run(self.inference, self.telemetry, rejected, concurrency=1,
                          profile_identity=identity, require_managed_match=True)
        self.assertFalse(rejected.exists())

        with mock.patch("strata_control.runtime.status", return_value=owner), \
                mock.patch("strata_control.runtime.plan_restart", return_value={**plan, "changed": True}):
            stale = self.root / "stale"
            with self.assertRaisesRegex(ProfileError, "managed benchmark preflight failed: profile_or_source_changed"):
                bench.run(self.inference, self.telemetry, stale, concurrency=1,
                          profile_identity=identity, require_managed_match=True)
        self.assertFalse(stale.exists())

        with mock.patch("strata_control.runtime.status", side_effect=[owner, {**owner, "instance_id": "wrong"}]), \
                mock.patch("strata_control.runtime.plan_restart", return_value=plan), \
                redirect_stdout(io.StringIO()):
            changed = bench.run(self.inference, self.telemetry, self.root / "changed", concurrency=1,
                                profile_identity=identity, require_managed_match=True)
        self.assertFalse(changed["successful"])
        self.assertEqual(changed["profile_running_check"]["after"]["reason"], "instance_mismatch")
        self.assertTrue((self.root / "changed" / "results.json").is_file())

    def test_managed_match_flag_requires_a_saved_profile(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["bench", "run", "--suite", "serving-smoke", "--url", self.url, "--require-managed-match",
                         "--out", str(self.root / "unused"), "--output", "json"])
        self.assertEqual(code, 2)
        self.assertFalse(json.loads(output.getvalue())["ok"])
        self.assertFalse((self.root / "unused").exists())

    def test_desktop_review_runs_shared_benchmark_and_rejects_changed_owner(self):
        source = self.root / "run.json"
        source.write_text("{}", encoding="utf-8")
        instance = self.telemetry.snapshot()["instance"]["id"]
        revision = "a" * 16
        owner_id = ["b" * 32]
        owner = {"ok": True, "state": "ready", "launch_attested": True,
                 "api_identity_verified": True, "generation_ready": True,
                 "instance_id": instance, "profile_revision": revision, "api_url": self.url}
        restart = {"ok": True, "changed": False, "current_instance_id": instance,
                   "current_profile_revision": revision, "candidate_profile_revision": revision}
        launch = {"effective": {"host": "127.0.0.1", "port": self.httpd.server_address[1]}}
        settings = {"path": str(self.root / "desktop-run"), "suite": "serving-smoke",
                    "concurrency": 2, "repeat": 1, "request_timeout": 120,
                    "condition": "baseline"}
        def send(action, params):
            return management.dispatch({"schema_version": 1, "action": action,
                                        "profile": "local", "params": params})
        with mock.patch("strata_control.benchmarking._source", return_value=(
                source, {"id": "local"}, revision, {"model_name": "test", "gpu": "cpu"})), \
                mock.patch("strata_control.benchmarking._launch_plan", return_value=(launch, [])), \
                mock.patch("strata_control.runtime.status", side_effect=lambda _: {
                    **owner, "owner_id": owner_id[0]}), \
                mock.patch("strata_control.runtime.plan_restart", return_value=restart), \
                mock.patch.dict(os.environ, {"STRATA_API_KEY": self.service.api_key,
                                             "STRATA_TELEMETRY_KEY": self.service.telemetry_key}):
            reviewed = send("bench.run.plan", settings)["plan"]
            self.assertEqual(reviewed["planned_requests"], 5)
            self.assertEqual(reviewed["instance_id"], instance)
            pinned = {"if_profile_revision": reviewed["profile_revision"],
                      "if_source_sha256": reviewed["source_sha256"],
                      "if_owner_id": reviewed["owner_id"],
                      "if_instance_id": reviewed["instance_id"]}
            owner_id[0] = "c" * 32
            with self.assertRaisesRegex(ProfileError, "target changed after review"):
                send("bench.run", {**settings, **pinned})
            self.assertFalse(Path(settings["path"]).exists())
            owner_id[0] = "b" * 32
            with redirect_stdout(io.StringIO()):
                result = send("bench.run", {**settings, **pinned})
            self.assertTrue(result["successful"])
            self.assertTrue(Path(result["path"]).is_file())
            self.assertEqual(bench.load(Path(settings["path"]))["profile_running_match"],
                             "managed_launch_checked")

    def test_report_cli_json_and_malformed_report(self):
        out = self.root / "smoke"
        with redirect_stdout(io.StringIO()):
            bench.run(self.inference, self.telemetry, out, concurrency=1)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(["bench", "report", str(out), "--output", "json"]), 0)
        value = json.loads(stdout.getvalue())
        self.assertTrue(value["ok"])
        self.assertEqual(value["summary"]["suite"], "serving-smoke")
        report_file = out / "results.json"
        data = json.loads(report_file.read_text())
        for phase in data["phases"]:
            phase.pop("passed")  # reports written before the overload suite remain readable
        report_file.write_text(json.dumps(data), encoding="utf-8")
        self.assertTrue(all(phase["passed"] for phase in bench.summary(bench.load(out))["phases"]))
        data["phases"] = [{"kind": "mixed"}]
        report_file.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ProfileError, "unsupported benchmark report"):
            bench.load(out)

    def test_markdown_export_is_local_private_and_keeps_measurement_limits(self):
        first = self.root / "baseline"
        second = self.root / "candidate"
        with redirect_stdout(io.StringIO()):
            report = bench.run(self.inference, self.telemetry, first, concurrency=1)
        report["model_id"] = "model | <untrusted>"
        (first / "results.json").write_text(json.dumps(report), encoding="utf-8")
        other = json.loads(json.dumps(report))
        other["run_id"] = "f" * 32
        second.mkdir()
        (second / "results.json").write_text(json.dumps(other), encoding="utf-8")
        share = self.root / "share.md"
        with mock.patch.object(Endpoint, "snapshot", side_effect=AssertionError("export must stay offline")):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["bench", "report", str(first), "--output", "markdown",
                                       "--out", str(share)]), 0)
            comparison = io.StringIO()
            with redirect_stdout(comparison):
                self.assertEqual(main(["bench", "compare", str(first), str(second),
                                       "--output", "markdown"]), 0)
        exported = share.read_text(encoding="utf-8")
        self.assertIn("model \\| \\<untrusted\\>", exported)
        self.assertIn("Peak slots", exported)
        self.assertIn("Missing readings appear as `unavailable`", exported)
        self.assertNotIn(self.url, exported)
        for secret in ("inference-test-secret", "monitor-test-secret", "Benchmark mock answer.",
                       "Answer in one sentence"):
            self.assertNotIn(secret, exported)
            self.assertNotIn(secret, comparison.getvalue())
        self.assertIn("Provisional comparison", comparison.getvalue())
        self.assertIn("monitor attachment", comparison.getvalue())
        original = share.read_bytes()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["bench", "report", str(first), "--output", "markdown",
                                   "--out", str(share)]), 2)
        self.assertEqual(share.read_bytes(), original)
        if os.name != "nt":
            self.assertEqual(stat.S_IMODE(share.stat().st_mode), 0o600)

    def test_saved_run_comparison_is_read_only_and_provisional(self):
        baseline_path, candidate_path = self.root / "baseline", self.root / "monitored"
        with redirect_stdout(io.StringIO()):
            baseline = bench.run(self.inference, self.telemetry, baseline_path, concurrency=2, repeat=3,
                                 condition="baseline")
            candidate = bench.run(self.inference, self.telemetry, candidate_path, concurrency=2, repeat=3,
                                  condition="monitor-default")
        self.assertTrue(baseline["successful"] and candidate["successful"])
        before = [(path / "results.json").read_bytes() for path in (baseline_path, candidate_path)]
        with mock.patch.object(Endpoint, "snapshot", side_effect=AssertionError("compare must not contact server")):
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["bench", "compare", str(baseline_path), str(candidate_path),
                             "--output", "json"])
        self.assertEqual(code, 0)
        comparison = json.loads(output.getvalue())["comparison"]
        self.assertEqual(comparison["qualification"], "provisional")
        self.assertTrue(comparison["workload_match"])
        self.assertEqual(comparison["primary_phases"], 3)
        self.assertIsNotNone(comparison["median_primary_metrics"]["p95_ttft_s"]["baseline"])
        self.assertEqual(comparison["baseline"]["condition"], "baseline")
        self.assertEqual(comparison["candidate"]["condition"], "monitor-default")
        self.assertTrue(any("monitor attachment" in note for note in comparison["warnings"]))
        self.assertEqual(before, [(path / "results.json").read_bytes()
                                  for path in (baseline_path, candidate_path)])
        self.assertIn("provisional comparison", bench.format_comparison(comparison))
        missing = json.loads(json.dumps(candidate))
        next(phase for phase in missing["phases"] if phase["kind"] == "mixed")["summary"]["p95_ttft_s"] = None
        self.assertIsNone(bench.compare(baseline, missing)["median_primary_metrics"]["p95_ttft_s"]["candidate"])
        next(phase for phase in missing["phases"] if phase["kind"] == "mixed")["summary"]["p95_ttft_s"] = 10 ** 400
        self.assertIsNone(bench.compare(baseline, missing)["median_primary_metrics"]["p95_ttft_s"]["candidate"])

    def test_comparison_rejects_workload_and_identity_mismatch(self):
        first_path, second_path = self.root / "one", self.root / "two"
        with redirect_stdout(io.StringIO()):
            first = bench.run(self.inference, self.telemetry, first_path, concurrency=1)
            second = bench.run(self.inference, self.telemetry, second_path, concurrency=2)
        with self.assertRaisesRegex(ProfileError, "workload or concurrency differs"):
            bench.compare(first, second)
        second["phases"][2]["requested"] = first["phases"][2]["requested"]
        second["phases"][2]["client_workers"] = first["phases"][2]["client_workers"]
        second["phases"][2]["requests"] = first["phases"][2]["requests"]
        second["environment"]["slots_effective"] = 999
        with self.assertRaisesRegex(ProfileError, "reported environment differs: slots_effective"):
            bench.compare(first, second)
        second["environment"]["slots_effective"] = first["environment"]["slots_effective"]
        first["profile_identity"] = {"source_sha256": "0" * 64}
        second["profile_identity"] = {"source_sha256": "1" * 64}
        with self.assertRaisesRegex(ProfileError, "source config digests differ"):
            bench.compare(first, second)

    def test_overload_suite_requires_bounded_admission_before_creating_output(self):
        out = self.root / "overload"
        with self.assertRaisesRegex(ProfileError, "max_queue and queue_timeout"):
            bench.run(self.inference, self.telemetry, out, suite="parallel-overload-v1", concurrency=4)
        self.assertFalse(out.exists())
        self.service.max_queue = 1
        self.service.queue_timeout_s = 0.1
        with self.assertRaisesRegex(ProfileError, "must exceed the admission limit"):
            bench.run(self.inference, self.telemetry, out, suite="parallel-overload-v1", concurrency=2)
        self.assertFalse(out.exists())

    def test_cancel_suite_requires_queue_capacity_and_disconnect_telemetry(self):
        out = self.root / "cancel"
        with self.assertRaisesRegex(ProfileError, "cancellation suite requires"):
            bench.run(self.inference, self.telemetry, out, suite="parallel-cancel-v1", concurrency=1)
        self.assertFalse(out.exists())
        self.service.max_queue = 1
        self.service.queue_timeout_s = 0.1
        with self.assertRaisesRegex(ProfileError, "queue_timeout >= 2"):
            bench.run(self.inference, self.telemetry, out, suite="parallel-cancel-v1", concurrency=1)
        self.assertFalse(out.exists())

    def test_prefill_cancel_requires_parallel_engine(self):
        out = self.root / "prefill"
        with self.assertRaisesRegex(ProfileError, "at least two effective slots"):
            bench.run(self.inference, self.telemetry, out, suite="parallel-prefill-cancel-v1", concurrency=2)
        self.assertFalse(out.exists())

    def test_overload_suite_records_rejections_and_recovery_without_content(self):
        self.service.max_queue = 1
        self.service.queue_timeout_s = 0.1
        self.service.engine.delay = 0.02
        out = self.root / "overload"
        with redirect_stdout(io.StringIO()):
            report = bench.run(self.inference, self.telemetry, out, suite="parallel-overload-v1",
                               concurrency=6)
        self.assertTrue(report["successful"])
        self.assertEqual([phase["kind"] for phase in report["phases"]], ["warmup", "overload", "recovery"])
        overload = report["phases"][1]
        self.assertTrue(overload["passed"])
        self.assertEqual(overload["requested"], 6)
        self.assertGreaterEqual(overload["summary"]["expected_overload"], 1)
        self.assertEqual(overload["summary"]["expected_overload"],
                         overload["summary"]["admission_full"] + overload["summary"]["queue_wait_timeout"])
        self.assertEqual(overload["summary"]["failed"], 0)
        self.assertIsNotNone(overload["summary"]["launch_spread_s"])
        self.assertEqual(len([row["launch_offset_s"] for row in overload["requests"]]), 6)
        self.assertTrue(all(row["error_code"] in (None, "admission_full", "queue_wait_timeout")
                            for row in overload["requests"]))
        self.assertTrue(report["phases"][2]["passed"])
        data = (out / "results.json").read_text(encoding="utf-8")
        for secret in ("inference-test-secret", "monitor-test-secret", "Benchmark mock answer.",
                       "Answer in one sentence"):
            self.assertNotIn(secret, data)
        loaded = bench.load(out)
        self.assertEqual(bench.summary(loaded)["phases"][1]["expected_overload"],
                         overload["summary"]["expected_overload"])
        comparison_shape = json.loads(json.dumps(loaded))
        comparison_shape["run_id"] = "f" * 32
        same = bench.compare(loaded, comparison_shape)
        self.assertEqual(same["primary_kind"], "overload")
        self.assertIn("admission full", bench.format_comparison(same))
        human = bench.format_summary(bench.summary(loaded))
        self.assertIn("rejected", human)
        self.assertIn("timed out", human)
        self.assertIn("launch spread", human)
        self.assertNotIn("Benchmark mock answer", human)

    def test_cancel_suite_observes_active_and_queued_disconnect_and_recovery(self):
        self.service.max_queue = 1
        self.service.queue_timeout_s = 5
        engine = self.service.engine
        engine.script = engine.tok.encode("A" * 200, parse_special=True) + \
            engine.tok.encode(IM_END, parse_special=True)
        engine.delay = 0.01
        out = self.root / "cancel"
        with redirect_stdout(io.StringIO()):
            report = bench.run(self.inference, self.telemetry, out, suite="parallel-cancel-v1",
                               concurrency=1, request_timeout=10)
        self.assertTrue(report["successful"], bench.format_summary(bench.summary(report)))
        self.assertEqual([phase["kind"] for phase in report["phases"]],
                         ["warmup", "cancel_active", "cancel_queued", "recovery"])
        for phase in report["phases"][1:3]:
            self.assertTrue(phase["passed"])
            self.assertEqual(phase["summary"]["client_disconnect_delta"], 1)
            self.assertTrue(phase["summary"]["settled"])
            self.assertEqual(phase["summary"]["queue_timeouts_delta"], 0)
        self.assertEqual(report["phases"][2]["summary"]["companion_successful"], 1)
        data = (out / "results.json").read_text()
        self.assertNotIn("A" * 50, data)
        self.assertTrue(bench.load(out)["successful"])
        comparison_shape = json.loads(json.dumps(report))
        comparison_shape["run_id"] = "f" * 32
        comparison = bench.compare(report, comparison_shape)
        self.assertTrue(comparison["workload_match"])
        self.assertEqual(comparison["primary_kind"], "cancel_active")
        self.assertEqual(comparison["phases"][1]["metrics"]["client_disconnect_delta"]["baseline"], 1.0)
        self.assertIn("disconnects", bench.format_comparison(comparison))
        corrupt = json.loads(data)
        corrupt["phases"][1]["summary"].pop("client_disconnect_delta")
        (out / "results.json").write_text(json.dumps(corrupt))
        with self.assertRaisesRegex(ProfileError, "unsupported benchmark report"):
            bench.load(out)


if __name__ == "__main__":
    unittest.main()
