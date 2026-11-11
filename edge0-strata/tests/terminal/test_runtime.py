"""Managed local instance lifecycle with an owned fake engine process."""

import io
import hashlib
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from strata_cli.cli import main
from strata_cli import bench
from strata_cli.client import Endpoint
from strata_control import calibration, journal, management, owner_transport, procs, profiles, recipes, runtime


WINDOWS = os.name == "nt"


@unittest.skipUnless(os.name in ("posix", "nt"), "no qualified owner transport on this platform")
class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-runtime-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        env = mock.patch.dict(os.environ, {"STRATA_PROFILE_DIR": str(self.root / "profiles"),
                                       "STRATA_RUNTIME_DIR": str(self.root / "runtime")})
        env.start()
        self.addCleanup(env.stop)
        previous_key = os.environ.pop("STRATA_API_KEY", None)
        self.addCleanup(lambda: os.environ.__setitem__("STRATA_API_KEY", previous_key)
                        if previous_key is not None else os.environ.pop("STRATA_API_KEY", None))
        engine = self.root / "engine.py"
        engine.write_text("#!" + sys.executable + "\nimport sys\nfrom pathlib import Path\n"
                          "if '--batch' in sys.argv: print('INFO batch_slots=2', flush=True)\n"
                          "print('READY 4096 stop', flush=True)\n"
                          "for line in sys.stdin:\n"
                          "    if line.strip() == 'QUIT': break\n"
                          "    if line.startswith(('GEN ', 'BGEN ')):\n"
                          "        slot = int(line.split()[1]) if line.startswith('BGEN ') else None\n"
                          "        if Path('fail-generation').exists():\n"
                          "            print('ERR generation-failed', flush=True)\n"
                          "            continue\n"
                          "        if Path('empty-generation').exists():\n"
                          "            print('DONE 0 1 1.0 0.0 stop 0 0 0', flush=True)\n"
                          "            if slot is not None: print(f'BADM {slot} 0', flush=True)\n"
                          "            continue\n"
                          "        print('T 97', flush=True)\n"
                          "        print('DONE 1 1 1.0 1.0 length 0 0 0', flush=True)\n"
                          "        if slot is not None: print(f'BADM {slot} 0', flush=True)\n", encoding="utf-8")
        engine.chmod(0o700)
        if WINDOWS:
            # a script cannot be an engine exe on Windows: a .cmd that runs it stands in for the binary
            wrapper = self.root / "engine.cmd"
            wrapper.write_bytes(f'@echo off\r\n"{sys.executable}" "{engine}" %*\r\n'.encode())
            engine = wrapper
        from tools.strata_tokenizer import BYTE_TO_UNICODE
        tokenizer = self.root / "tokenizer"
        tokenizer.mkdir()
        tokens = [BYTE_TO_UNICODE[i] for i in range(256)] + [
            "ab", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
        (tokenizer / "vocab.json").write_text(json.dumps({token: i for i, token in enumerate(tokens)}))
        (tokenizer / "merges.txt").write_text("a b")
        (tokenizer / "token_type.json").write_text(json.dumps([0] * 257 + [3, 3, 3]))
        self.config = self.root / "run.json"
        self.config.write_text(json.dumps({"exe": str(engine), "tokenizer": str(tokenizer),
                                            "args": [], "cwd": str(self.root), "model_name": "fake",
                                            "host": "127.0.0.1", "port": 0}), encoding="utf-8")
        profiles.create("local", str(self.config))
        self.addCleanup(self._stop)

    def _stop(self):
        try:
            if runtime._read_state("local") is not None:
                runtime.stop("local", timeout=12)
        except (OSError, profiles.ProfileError):
            pass

    def _cli(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["runtime", *args, "--output", "json"])
        return code, json.loads(output.getvalue())

    def test_component_inspection_bridge_is_read_only_and_reports_missing_files(self):
        command = [sys.executable, "-m", "strata_control.management"]
        root = Path(__file__).resolve().parents[2]

        def send(live):
            request = {"schema_version": 1, "action": "components.inspect",
                       "profile": "local", "params": {"live": live}}
            completed = subprocess.run(command, input=json.dumps(request), text=True,
                                       capture_output=True, cwd=root, env=os.environ.copy(), timeout=15)
            return completed.returncode, json.loads(completed.stdout)

        code, result = send(False)
        self.assertEqual(code, 0)
        self.assertTrue(result["ok"])
        self.assertTrue(result["report"]["ok"])
        self.assertEqual(result["report"]["profile"], "local")
        self.assertEqual(Path(result["report"]["source_config"]), self.config.resolve())
        self.assertIsNone(result["report"]["live_gpu_probe"])
        self.assertEqual(journal.list_runs("local"), [])
        self.assertEqual(runtime.status("local")["state"], "stopped")

        code, invalid = send("true")
        self.assertEqual((code, invalid["ok"]), (2, False))
        self.assertIn("live must be true or false", invalid["error"])
        engine = self.root / ("engine.cmd" if WINDOWS else "engine.py")
        engine.unlink()
        code, missing = send(False)
        self.assertEqual((code, missing["ok"], missing["report"]["ok"]), (0, True, False))
        self.assertIn("engine", [row["name"] for row in missing["report"]["components"]
                      if row["status"] == "missing"])
        self.assertEqual(journal.list_runs("local"), [])

    def test_profile_change_bridge_reviews_revision_source_and_typed_overrides(self):
        command = [sys.executable, "-m", "strata_control.management"]
        root = Path(__file__).resolve().parents[2]

        def send(action, params):
            request = {"schema_version": 1, "action": action, "profile": "local", "params": params}
            completed = subprocess.run(command, input=json.dumps(request), text=True,
                                       capture_output=True, cwd=root, env=os.environ.copy(), timeout=15)
            return completed.returncode, json.loads(completed.stdout)

        other = self.root / "strata-other.json"
        other.write_text(self.config.read_text().replace('"fake"', '"other"'), encoding="utf-8")
        with mock.patch.dict(os.environ, {"STRATA_INSTALL_DIR": str(self.root)}):
            code, listed = send("models.installed", {})
            inspect_code, inspected = send("models.inspect", {"model_id": "other"})
        self.assertEqual(code, 0)
        self.assertIn("other", [item["id"] for item in listed["inventory"]["models"]])
        self.assertEqual((inspect_code, inspected["ok"]), (0, True))
        self.assertEqual(inspected["inspection"]["model"]["id"], "other")
        self.assertEqual(journal.list_runs("local"), [])
        candidate = self.root / "candidate.gguf"
        candidate.write_bytes(b"GGUF" + (3).to_bytes(4, "little"))
        code, import_plan = send("models.import.plan", {"path": str(candidate)})
        self.assertEqual((code, import_plan["ok"]), (0, True))
        self.assertFalse(import_plan["plan"]["would_write"])
        self.assertEqual(import_plan["plan"]["execution_compatibility"], "unverified")
        code, invalid_import = send("models.import.plan", {"path": "relative.gguf"})
        self.assertEqual((code, invalid_import["ok"]), (2, False))
        code, current = send("profile.inspect", {})
        self.assertEqual((code, current["ok"], current["profile"]["id"]), (0, True, "local"))
        self.assertEqual(current["profile"]["overrides"], {})
        edits = {"overrides": {"parallel": 4, "batch_groups": "auto", "max_queue": 8,
                               "queue_timeout": 2.5}, "source_config": str(other)}
        code, plan = send("profile.change.plan", {**edits, "if_revision": current["revision"]})
        self.assertEqual((code, plan["ok"], plan["plan"]["would_save"]), (0, True, True))
        self.assertEqual(plan["plan"]["revision"], current["revision"])
        self.assertEqual(plan["plan"]["effective"]["parallel"], 4)
        self.assertEqual(plan["plan"]["source_config_after"], str(other.resolve()))
        self.assertEqual(profiles.load("local")[0]["source_config"], str(self.config.resolve()))
        self.assertEqual(journal.list_runs("local"), [])

        execute = {**edits, "if_revision": plan["plan"]["revision"],
                   "if_source_sha256": plan["plan"]["source_sha256"]}
        profiles.set_values("local", ["max_queue=1"])
        code, stale = send("profile.change", execute)
        self.assertEqual((code, stale["ok"]), (6, False))
        self.assertIn("changed after review", stale["error"])
        code, stale_plan = send("profile.change.plan", {**edits, "if_revision": current["revision"]})
        self.assertEqual((code, stale_plan["ok"]), (6, False))
        code, plan = send("profile.change.plan", {**edits,
            "if_revision": profiles.load("local")[1]})
        execute.update(if_revision=plan["plan"]["revision"],
                       if_source_sha256=plan["plan"]["source_sha256"])
        other.write_text(other.read_text() + "\n", encoding="utf-8")
        code, changed_source = send("profile.change", execute)
        self.assertEqual((code, changed_source["ok"]), (6, False))
        code, plan = send("profile.change.plan", {**edits,
            "if_revision": profiles.load("local")[1]})
        execute.update(if_revision=plan["plan"]["revision"],
                       if_source_sha256=plan["plan"]["source_sha256"])
        code, saved = send("profile.change", execute)
        self.assertEqual((code, saved["ok"], saved["saved"]), (0, True, True))
        self.assertEqual(profiles.load("local")[0]["source_config"], str(other.resolve()))
        self.assertEqual(profiles.load("local")[0]["overrides"], edits["overrides"])
        self.assertEqual(runtime.status("local")["state"], "stopped")
        self.assertEqual(journal.list_runs("local"), [])

        code, invalid = send("profile.change.plan", {"overrides": {"parallel": True},
            "source_config": str(other), "if_revision": profiles.load("local")[1]})
        self.assertEqual((code, invalid["ok"]), (2, False))

        started = runtime.start("local", wait=12)
        self.assertTrue(started["ok"], started)
        running = runtime.status("local")
        self.assertEqual(running["state"], "ready")
        running_revision = running["profile_revision"]
        next_edit = {"overrides": {**edits["overrides"], "parallel": 6},
                     "source_config": str(other.resolve())}
        code, next_plan = send("profile.change.plan", {**next_edit,
            "if_revision": profiles.load("local")[1]})
        self.assertEqual((code, next_plan["ok"]), (0, True))
        code, next_saved = send("profile.change", {**next_edit,
            "if_revision": next_plan["plan"]["revision"],
            "if_source_sha256": next_plan["plan"]["source_sha256"]})
        self.assertEqual((code, next_saved["ok"]), (0, True))
        still_running = runtime.status("local")
        self.assertEqual(still_running["instance_id"], running["instance_id"])
        self.assertEqual(still_running["profile_revision"], running_revision)
        restart_plan = runtime.plan_restart("local")
        self.assertTrue(restart_plan["ok"], restart_plan)
        self.assertEqual(restart_plan["candidate_profile_revision"], next_saved["revision"])
        self.assertNotEqual(restart_plan["candidate_profile_revision"], running_revision)

    def test_local_management_bridge_and_cli_share_owner_and_journal(self):
        command = [sys.executable, "-m", "strata_control.management"]
        root = Path(__file__).resolve().parents[2]

        def send(action, params):
            request = {"schema_version": 1, "action": action, "profile": "local", "params": params}
            completed = subprocess.run(command, input=json.dumps(request), text=True,
                                       capture_output=True, cwd=root, env=os.environ.copy(), timeout=20)
            return completed.returncode, json.loads(completed.stdout)

        code, invalid = send("runtime.start", {"wait_s": True, "shutdown_timeout_s": 2,
                                               "if_profile_revision": None,
                                               "if_source_sha256": None})
        self.assertEqual((code, invalid["ok"]), (2, False))
        self.assertEqual(runtime.status("local")["state"], "stopped")
        self.assertEqual(journal.list_runs("local"), [])
        code, preview = send("runtime.start.plan", {"shutdown_timeout_s": 2})
        self.assertEqual(code, 0)
        self.assertFalse(preview["plan"]["would_start"])
        self.assertEqual(runtime.status("local")["state"], "stopped")
        start_params = {"wait_s": 10, "shutdown_timeout_s": 2,
                        "if_profile_revision": preview["plan"]["launch"]["profile_revision"],
                        "if_source_sha256": preview["plan"]["source_sha256"]}
        self.assertEqual(preview["plan"]["source_sha256"], hashlib.sha256(self.config.read_bytes()).hexdigest())
        code, missing = send("runtime.start", {**start_params, "if_source_sha256": None})
        self.assertEqual((code, missing["ok"]), (2, False))
        self.config.write_bytes(self.config.read_bytes() + b" ")
        code, stale_source = send("runtime.start", start_params)
        self.assertEqual((code, stale_source["error"]), (6, "source_changed_after_start_plan"))
        self.assertEqual(journal.list_runs("local"), [])
        self.config.write_bytes(self.config.read_bytes()[:-1])
        profiles.set_values("local", ["max_queue=32"])
        code, stale_profile = send("runtime.start", start_params)
        self.assertEqual((code, stale_profile["error"]), (6, "profile_changed_after_start_plan"))
        self.assertEqual(journal.list_runs("local"), [])
        code, preview = send("runtime.start.plan", {"shutdown_timeout_s": 2})
        self.assertEqual(code, 0)
        start_params.update(if_profile_revision=preview["plan"]["launch"]["profile_revision"],
                            if_source_sha256=preview["plan"]["source_sha256"])

        code, started = send("runtime.start", start_params)
        self.assertEqual(code, 0, started)
        self.assertTrue(started["ok"])
        start_record = journal.load("local", started["operation_id"])
        self.assertEqual(start_record["kind"], "runtime.start")
        self.assertEqual(start_record["intent"]["expected_source_sha256"], start_params["if_source_sha256"])
        cli_code, current = self._cli("status", "--profile", "local")
        self.assertEqual((cli_code, current["instance_id"]), (0, started["instance_id"]))
        code, verify_plan = send("runtime.verify.plan", {"load": False})
        self.assertEqual((code, verify_plan["ok"]), (0, True))
        self.assertEqual(verify_plan["plan"]["owner_id"], started["owner_id"])
        self.assertEqual(verify_plan["plan"]["instance_id"], started["instance_id"])
        self.assertFalse(verify_plan["plan"]["would_verify"])
        verify_params = {"wait_s": 10, "load": False,
                         "if_owner_id": verify_plan["plan"]["owner_id"],
                         "if_instance_id": verify_plan["plan"]["instance_id"]}
        count = len(journal.list_runs("local"))
        code, missing_verify = send("runtime.verify", {**verify_params, "if_owner_id": None})
        self.assertEqual((code, missing_verify["ok"]), (2, False))
        code, stale_verify_owner = send("runtime.verify", {**verify_params,
                                                           "if_owner_id": "0" * 32})
        self.assertEqual((code, stale_verify_owner["error"]),
                         (6, "owner_changed_after_verify_plan"))
        code, stale_verify_instance = send("runtime.verify", {**verify_params,
                                                              "if_instance_id": "0" * 32})
        self.assertEqual((code, stale_verify_instance["error"]),
                         (6, "instance_changed_after_verify_plan"))
        self.assertEqual(len(journal.list_runs("local")), count)
        host_denied = runtime._rpc("local", "verify", params={
            "wait": 10, "load": False, "expected_owner_id": "0" * 32,
            "expected_instance_id": started["instance_id"]})
        self.assertEqual(host_denied["error"], "owner_changed_after_verify_plan")
        host_denied = runtime._rpc("local", "verify", params={
            "wait": 10, "load": False, "expected_owner_id": started["owner_id"],
            "expected_instance_id": "0" * 32})
        self.assertEqual(host_denied["error"], "instance_changed_after_verify_plan")
        code, verified = send("runtime.verify", verify_params)
        self.assertEqual((code, verified["ok"], verified["verified_now"]), (0, True, True))
        self.assertEqual(verified["instance_id"], started["instance_id"])
        verify_record = journal.load("local", verified["operation_id"])
        self.assertEqual(verify_record["intent"]["expected_instance_id"], started["instance_id"])
        cli_code, cli_verify_plan = self._cli("verify", "--profile", "local", "--plan")
        self.assertEqual((cli_code, cli_verify_plan["plan"]["instance_id"]),
                         (0, started["instance_id"]))
        code, old_stop_plan = send("runtime.stop.plan", {})
        self.assertEqual((code, old_stop_plan["ok"]), (0, True))
        self.assertEqual(old_stop_plan["plan"]["instance_id"], started["instance_id"])
        code, restart_plan = send("runtime.restart.plan", {"allow_source_switch": False,
                                                             "allow_source_migration": False})
        self.assertEqual((code, restart_plan["ok"]), (0, True))
        self.assertEqual(restart_plan["current_owner_id"], started["owner_id"])
        self.assertEqual(restart_plan["current_instance_id"], started["instance_id"])
        restart_params = {"wait_s": 20, "startup_wait_s": 10,
                          "if_profile_revision": restart_plan["candidate_profile_revision"],
                          "if_owner_id": restart_plan["current_owner_id"],
                          "if_instance_id": restart_plan["current_instance_id"],
                          "if_source_sha256": restart_plan["candidate_source_sha256"],
                          "allow_source_switch": False, "allow_source_migration": False}
        code, missing_condition = send("runtime.restart", {**restart_params,
                                                             "if_source_sha256": None})
        self.assertEqual((code, missing_condition["ok"]), (2, False))
        code, missing_owner = send("runtime.restart", {**restart_params,
                                                         "if_owner_id": None})
        self.assertEqual((code, missing_owner["ok"]), (2, False))
        count = len(journal.list_runs("local"))
        code, stale_owner = send("runtime.restart", {**restart_params,
                                                       "if_owner_id": "0" * 32})
        self.assertEqual((code, stale_owner["error"]), (6, "owner_changed_after_restart_plan"))
        code, stale_instance = send("runtime.restart", {**restart_params,
                                                          "if_instance_id": "0" * 32})
        self.assertEqual((code, stale_instance["error"]), (6, "instance_changed_after_restart_plan"))
        self.assertEqual(len(journal.list_runs("local")), count)
        host_denied = runtime._rpc("local", "restart", params={
            "expected_owner_id": "0" * 32, "expected_instance_id": started["instance_id"]})
        self.assertEqual(host_denied["error"], "owner_changed_after_restart_plan")
        host_denied = runtime._rpc("local", "restart", params={
            "expected_owner_id": started["owner_id"], "expected_instance_id": "0" * 32})
        self.assertEqual(host_denied["error"], "instance_changed_after_restart_plan")
        self.assertEqual(runtime.status("local")["instance_id"], started["instance_id"])
        code, restarted = send("runtime.restart", restart_params)
        self.assertEqual((code, restarted["ok"]), (0, True), restarted)
        self.assertNotEqual(restarted["instance_id"], started["instance_id"])
        self.assertEqual(journal.load("local", restarted["operation_id"])["kind"], "runtime.restart")
        count = len(journal.list_runs("local"))
        code, stale_verify = send("runtime.verify", verify_params)
        self.assertEqual((code, stale_verify["error"]), (6, "instance_changed_after_verify_plan"))
        self.assertEqual(len(journal.list_runs("local")), count)
        code, stale_review = send("runtime.restart", restart_params)
        self.assertEqual((code, stale_review["error"]), (6, "instance_changed_after_restart_plan"))
        self.assertEqual(len(journal.list_runs("local")), count)
        self.assertEqual(runtime.status("local")["instance_id"], restarted["instance_id"])
        old_stop_params = {"wait_s": 12, "if_owner_id": old_stop_plan["plan"]["owner_id"],
                           "if_instance_id": old_stop_plan["plan"]["instance_id"]}
        count = len(journal.list_runs("local"))
        code, stale_stop = send("runtime.stop", old_stop_params)
        self.assertEqual((code, stale_stop["error"]), (6, "instance_changed_after_stop_plan"))
        self.assertEqual(len(journal.list_runs("local")), count)
        self.assertEqual(runtime.status("local")["instance_id"], restarted["instance_id"])
        code, stop_plan = send("runtime.stop.plan", {})
        self.assertEqual((code, stop_plan["ok"]), (0, True))
        stop_params = {"wait_s": 12, "if_owner_id": stop_plan["plan"]["owner_id"],
                       "if_instance_id": stop_plan["plan"]["instance_id"]}
        code, stale_owner = send("runtime.stop", {**stop_params, "if_owner_id": "0" * 32})
        self.assertEqual((code, stale_owner["error"]), (6, "owner_changed_after_stop_plan"))
        self.assertEqual(len(journal.list_runs("local")), count)
        host_denied = runtime._rpc("local", "stop", params={"expected_owner_id": "0" * 32,
                                                        "expected_instance_id": restarted["instance_id"]})
        self.assertEqual(host_denied["error"], "owner_changed_after_stop_plan")
        host_denied = runtime._rpc("local", "stop", params={"expected_owner_id": stop_params["if_owner_id"],
                                                        "expected_instance_id": started["instance_id"]})
        self.assertEqual(host_denied["error"], "instance_changed_after_stop_plan")
        self.assertEqual(runtime.status("local")["instance_id"], restarted["instance_id"])
        code, stopped = send("runtime.stop", stop_params)
        self.assertEqual((code, stopped["state"]), (0, "stopped"))
        self.assertEqual(journal.load("local", stopped["operation_id"])["kind"], "runtime.stop")
        self.assertEqual(runtime.status("local")["state"], "stopped")

    def test_local_management_bridge_rejects_ambiguous_requests(self):
        command = [sys.executable, "-m", "strata_control.management"]
        root = Path(__file__).resolve().parents[2]
        invalid = (
            '{"schema_version":1,"action":"runtime.status","action":"runtime.start",'
            '"profile":"local","params":{}}',
            '{"schema_version":1,"action":"runtime.start","profile":"local",'
            '"params":{"wait_s":NaN,"shutdown_timeout_s":2}}',
            '{"schema_version":1,"action":"runtime.start","profile":"local",'
            '"params":{"wait_s":10,"shutdown_timeout_s":2,"command":"rm -rf /"}}',
            '{"schema_version":2,"action":"runtime.status","profile":"local","params":{}}',
        )
        for payload in invalid:
            with self.subTest(payload=payload[:50]):
                completed = subprocess.run(command, input=payload, text=True, capture_output=True,
                                           cwd=root, env=os.environ.copy(), timeout=10)
                response = json.loads(completed.stdout)
                self.assertEqual((completed.returncode, response["ok"]), (2, False))
                self.assertEqual(runtime.status("local")["state"], "stopped")
        self.assertEqual(journal.list_runs("local"), [])

    def test_owner_host_refuses_source_changed_after_start_review(self):
        preview = management.runtime_action("start", "local", plan=True)["plan"]
        self.config.write_bytes(self.config.read_bytes() + b" ")
        host = subprocess.Popen([sys.executable, "-m", "strata_control.runtime", "--host", "local",
                                 "--shutdown-timeout", "2", "--startup-wait", "10",
                                 "--expected-profile-revision", preview["launch"]["profile_revision"],
                                 "--expected-source-sha256", preview["source_sha256"]],
                                cwd=Path(__file__).resolve().parents[2], env=os.environ.copy(),
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 8
            result = {"state": "stopped"}
            while time.monotonic() < deadline:
                result = runtime.status("local")
                if result.get("state") == "failed":
                    break
                self.assertIsNone(host.poll(), "owner host exited before reporting status")
                time.sleep(0.05)
            self.assertEqual(result.get("error"), "source_changed_after_start_plan")
            self.assertIsNone(result.get("child_pid"))
            self.assertEqual(list((self.root / "runtime").glob("*.ready.json")), [])
        finally:
            if host.poll() is None:
                try:
                    runtime.stop("local", timeout=8)
                finally:
                    host.wait(timeout=10)

    @unittest.skipIf(WINDOWS, "Windows console stop needs a platform-specific integration runner (no SIGTERM/SIGINT)")
    def test_foreground_jsonl_emits_ready_and_stop_without_human_stdout(self):
        for stop_signal, expected_code, expected_reason, quiet in (
                (signal.SIGTERM, 0, "sigterm", False),
                (signal.SIGINT, 130, "sigint", False),
                (signal.SIGTERM, 0, "sigterm", True)):
            with self.subTest(signal=expected_reason, quiet=quiet):
                rows = queue.Queue()
                stderr_path = self.root / f"serve-{expected_reason}-{'quiet' if quiet else 'normal'}.stderr"
                with stderr_path.open("w+", encoding="utf-8") as diagnostics:
                    process = subprocess.Popen(
                        [sys.executable, "strata.py", "serve", "--profile", "local", "--port", "0",
                         "--shutdown-timeout", "2", "--output", "jsonl", *(["--quiet"] if quiet else [])],
                        cwd=Path(__file__).resolve().parents[2], env=os.environ.copy(),
                        stdout=subprocess.PIPE, stderr=diagnostics, text=True, bufsize=1)
                    reader = threading.Thread(target=lambda: [rows.put(line) for line in process.stdout],
                                              daemon=True)
                    reader.start()
                    try:
                        collected = []
                        for _ in range(4):
                            collected.append(json.loads(rows.get(timeout=15)))
                            if collected[-1].get("event") == "ready":
                                break
                        self.assertEqual([row["event"] for row in collected], ["starting", "ready"])
                        ready = collected[-1]
                        self.assertGreater(ready["bound_port"], 0)
                        self.assertEqual(ready["model_state"], "ready")
                        self.assertEqual(ready["pid"], process.pid)
                        request = urllib.request.Request(
                            ready["api_url"] + "/v1/chat/completions", method="POST",
                            data=json.dumps({"messages": [{"role": "user", "content": "Hi"}],
                                             "max_tokens": 1}).encode(),
                            headers={"Content-Type": "application/json"})
                        with urllib.request.urlopen(request, timeout=5) as response:
                            self.assertEqual(response.status, 200)
                            self.assertIn("choices", json.load(response))
                        process.send_signal(stop_signal)
                        self.assertEqual(process.wait(timeout=15), expected_code)
                        reader.join(timeout=3)
                        self.assertFalse(reader.is_alive())
                        while not rows.empty():
                            collected.append(json.loads(rows.get_nowait()))
                        self.assertEqual([row["event"] for row in collected],
                                         ["starting", "ready", "stopping", "draining", "stopped"])
                        self.assertEqual(collected[-1]["reason"], expected_reason)
                        self.assertEqual(collected[-1]["exit_code"], expected_code)
                        self.assertTrue(all(row["schema_version"] == 1 for row in collected))
                        self.assertNotIn("api_key", json.dumps(collected).lower())
                        diagnostics.flush()
                        if quiet:
                            self.assertNotIn("ready:", stderr_path.read_text(encoding="utf-8"))
                        else:
                            self.assertIn("ready:", stderr_path.read_text(encoding="utf-8"))
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait(timeout=5)
                        reader.join(timeout=2)
                        process.stdout.close()

    def test_start_profile_revision_restart_and_stop(self):
        code, started = self._cli("start", "--profile", "local", "--wait", "10",
                                  "--shutdown-timeout", "2")
        self.assertEqual(code, 0)
        self.assertEqual(started["state"], "ready")
        self.assertTrue(started["api_identity_verified"])
        self.assertTrue(started["launch_attested"])
        self.assertTrue(started["generation_ready"])
        self.assertEqual(started["readiness_state"], "generation_ready")
        started_record = journal.load("local", started["operation_id"])
        self.assertEqual((started_record["schema_version"], started_record["kind"], started_record["state"]),
                         (2, "runtime.start", "succeeded"))
        self.assertTrue(started_record["result"]["launch_attested"])
        first_id = started["instance_id"]
        self.assertNotEqual(started["host_pid"], started["child_pid"])
        status = self._cli("status", "--profile", "local")[1]
        self.assertEqual(status["instance_id"], first_id)
        self.assertNotIn("operation_id", status)
        code, checked = self._cli("verify", "--profile", "local", "--wait", "10")
        self.assertEqual(code, 0)
        self.assertTrue(checked["verified_now"])
        self.assertTrue(checked["launch_attested"])
        self.assertTrue(checked["generation_ready"])
        checked_record = journal.load("local", checked["operation_id"])
        self.assertEqual(checked_record["kind"], "runtime.verify")
        self.assertTrue(checked_record["result"]["launch_attested"])
        profiles.set_values("local", ["parallel=2"])
        code, restarted = self._cli("restart", "--profile", "local", "--wait", "20")
        self.assertEqual(code, 0)
        self.assertTrue(restarted["restarted"])
        self.assertTrue(restarted["launch_attested"])
        self.assertTrue(restarted["generation_ready"])
        self.assertEqual(journal.load("local", restarted["operation_id"])["kind"], "runtime.restart")
        self.assertEqual(restarted["previous_instance_id"], first_id)
        self.assertNotEqual(restarted["instance_id"], first_id)
        code, stopped = self._cli("stop", "--profile", "local", "--wait", "12")
        self.assertEqual(code, 0)
        self.assertEqual(stopped["state"], "stopped")
        self.assertEqual(journal.load("local", stopped["operation_id"])["kind"], "runtime.stop")
        self.assertEqual([row["origin"] for row in journal.list_runs("local")], ["direct"] * 4)
        deadline = time.monotonic() + 3
        while runtime._read_state("local") is not None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertIsNone(runtime._read_state("local"))

    def test_cli_stop_plan_pins_the_reviewed_owner_and_instance(self):
        started = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(started["ok"])
        code, preview = self._cli("stop", "--profile", "local", "--plan")
        self.assertEqual(code, 0, preview)
        self.assertEqual(preview["plan"]["instance_id"], started["instance_id"])
        self.assertFalse(preview["plan"]["would_stop"])
        code, denied = self._cli("stop", "--profile", "local", "--wait", "12",
                                  "--if-owner-id", "0" * 32,
                                  "--if-instance-id", preview["plan"]["instance_id"])
        self.assertEqual((code, denied["error"]["message"]), (6, "owner_changed_after_stop_plan"))
        self.assertEqual(runtime.status("local")["instance_id"], started["instance_id"])
        code, stopped = self._cli("stop", "--profile", "local", "--wait", "12",
                                   "--if-owner-id", preview["plan"]["owner_id"],
                                   "--if-instance-id", preview["plan"]["instance_id"])
        self.assertEqual((code, stopped["state"]), (0, "stopped"))
        record = journal.load("local", stopped["operation_id"])
        self.assertEqual(record["intent"]["expected_owner_id"], preview["plan"]["owner_id"])

    def test_managed_benchmark_binds_to_owned_launch(self):
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
        cfg = json.loads(self.config.read_text(encoding="utf-8"))
        cfg["port"] = port
        self.config.write_text(json.dumps(cfg), encoding="utf-8")
        started = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(started["ok"])
        identity = {"profile": "local", "revision": profiles.load("local")[1],
                    "source_sha256": hashlib.sha256(self.config.read_bytes()).hexdigest()}
        endpoint = Endpoint(started["api_url"])
        output = self.root / "managed-bench"
        with redirect_stdout(io.StringIO()):
            result = bench.run(endpoint, endpoint, output, concurrency=1,
                               profile_identity=identity, require_managed_match=True)
        self.assertTrue(result["successful"])
        self.assertEqual(result["profile_running_match"], "managed_launch_checked")
        self.assertEqual(result["instance_id"], started["instance_id"])
        self.assertTrue(result["profile_running_check"]["after"]["matched"])
        output_json = io.StringIO()
        with redirect_stdout(output_json):
            code = main(["bench", "run", "--suite", "serving-smoke", "--profile", "local", "--require-managed-match",
                         "--concurrency", "1", "--out", str(self.root / "cli-managed"),
                         "--output", "json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output_json.getvalue())["summary"]["successful"], True)
        self.assertEqual(json.loads(output_json.getvalue())["summary"]["profile_running_match"],
                         "managed_launch_checked")
        self.assertEqual(bench.load(self.root / "cli-managed")["profile_running_match"],
                         "managed_launch_checked")
        comparison = bench.compare(result, bench.load(self.root / "cli-managed"))
        self.assertEqual(comparison["baseline"]["profile_running_match"], "managed_launch_checked")
        self.assertFalse(any("running server config" in warning for warning in comparison["warnings"]))

    def test_owned_runtime_and_calibration_exclude_each_other_even_on_port_zero(self):
        first = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(first["ok"])
        output = self.root / "blocked-calibration.json"
        with self.assertRaisesRegex(profiles.ProfileError, "engine lease"):
            calibration.measure("local", output, exclusive_ack=True,
                                runner=lambda cfg, say: self.fail("calibration runner was started"))
        self.assertFalse(output.exists())
        self.assertTrue(runtime.stop("local", timeout=12)["ok"])
        with calibration._managed_exclusive():
            with self.assertRaisesRegex(profiles.ProfileError, "runtime host failed to start"):
                runtime.start("local", wait=1, shutdown_timeout=2)
        self.assertTrue(runtime.start("local", wait=10, shutdown_timeout=2)["ok"])

    def test_failed_new_port_rolls_back_previous_owned_launch(self):
        first = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(first["ok"])
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen(1)
            profiles.set_values("local", [f"port={occupied.getsockname()[1]}"])
            result = runtime.restart("local", timeout=20)
        self.assertFalse(result["ok"])
        self.assertTrue(result["rolled_back"])
        self.assertNotEqual(result["instance_id"], first["instance_id"])
        self.assertEqual(runtime.status("local")["instance_id"], result["instance_id"])

    def test_changed_source_is_rejected_before_stopping_old_instance(self):
        first = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(first["ok"])
        self.config.write_bytes(self.config.read_bytes() + b" ")
        for allow in (False, True):
            result = runtime.restart("local", timeout=10, allow_source_switch=allow,
                                     expected_source_sha256=hashlib.sha256(self.config.read_bytes()).hexdigest()
                                     if allow else None)
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"], "source_config_changed_restart_requires_migration")
        self.assertEqual(runtime.status("local")["instance_id"], first["instance_id"])

    def test_edited_source_migration_is_previewed_digest_pinned_and_verified(self):
        first = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(first["ok"])
        edited = json.loads(self.config.read_text())
        edited["model_name"] = "edited-fake"
        self.config.write_text(json.dumps(edited), encoding="utf-8")
        denied = runtime.plan_restart("local")
        self.assertEqual(denied["error"], "source_config_changed_restart_requires_migration")
        self.assertTrue(denied["requires_allow_source_migration"])
        code, plan = self._cli("restart", "--profile", "local", "--allow-source-migration", "--plan")
        self.assertEqual(code, 0)
        self.assertTrue(plan["source_migration"])
        self.assertTrue(plan["rollback_snapshot_required"])
        self.assertFalse(plan["source_switch"])
        self.assertTrue(plan["changed"])
        self.assertEqual(runtime.status("local")["instance_id"], first["instance_id"])
        code, missing = self._cli("restart", "--profile", "local", "--allow-source-migration")
        self.assertEqual((code, missing["error"]["code"]), (2, 2))
        self.config.write_bytes(self.config.read_bytes() + b" ")
        code, stale = self._cli("restart", "--profile", "local", "--allow-source-migration",
                                "--if-source-sha256", plan["candidate_source_sha256"])
        self.assertEqual((code, stale["error"]), (6, "source_changed_after_preview"))
        self.assertEqual(runtime.status("local")["instance_id"], first["instance_id"])
        self.config.write_text(json.dumps(edited), encoding="utf-8")
        code, restarted = self._cli("restart", "--profile", "local", "--allow-source-migration",
                                    "--if-profile-revision", plan["candidate_profile_revision"],
                                    "--if-source-sha256", plan["candidate_source_sha256"],
                                    "--startup-wait", "10", "--wait", "20")
        self.assertEqual(code, 0, restarted)
        self.assertTrue(restarted["source_migrated"])
        self.assertFalse(restarted["source_switched"])
        self.assertTrue(restarted["launch_attested"])
        self.assertTrue(restarted["generation_ready"])
        self.assertNotEqual(restarted["instance_id"], first["instance_id"])
        self.assertTrue(journal.load("local", restarted["operation_id"])["result"]["source_migrated"])
        self.assertEqual(list((self.root / "runtime").glob("*.rollback.json")), [])

    def test_failed_edited_source_migration_uses_private_snapshot_for_rollback(self):
        secret = "migration-test-key"
        original = json.loads(self.config.read_text())
        original["api_key"] = secret
        self.config.write_text(json.dumps(original), encoding="utf-8")
        first = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(first["ok"])
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen(1)
            edited = json.loads(self.config.read_text())
            edited["port"] = occupied.getsockname()[1]
            self.config.write_text(json.dumps(edited), encoding="utf-8")
            plan = runtime.plan_restart("local", allow_source_migration=True)
            code, result = self._cli("restart", "--profile", "local", "--allow-source-migration",
                                     "--startup-wait", "30", "--wait", "20",
                                     "--if-source-sha256", plan["candidate_source_sha256"])
        self.assertEqual(code, 6)
        self.assertFalse(result["ok"])
        self.assertTrue(result["rolled_back"])
        self.assertEqual(result["state"], "ready")
        self.assertNotEqual(result["instance_id"], first["instance_id"])
        self.assertEqual(json.loads(self.config.read_text())["port"], edited["port"])
        snapshot = Path(result["rollback_source_config"])
        self.assertEqual(snapshot.parent, self.root / "runtime")
        if WINDOWS:
            from strata_control import private_files
            self.assertTrue(private_files.is_unshared(snapshot))      # NTFS has no mode bits: the ACL is the promise
        else:
            self.assertEqual(snapshot.stat().st_mode & 0o077, 0)
        self.assertEqual(json.loads(snapshot.read_text())["port"], 0)
        self.assertEqual(json.loads(snapshot.read_text())["api_key"], secret)
        self.assertNotIn(secret, json.dumps(result))
        record_result = journal.load("local", result["operation_id"])["result"]
        self.assertEqual(record_result["rollback_source_config"], str(snapshot))
        self.assertNotIn(secret, json.dumps(record_result))
        self.assertTrue(runtime.status("local")["generation_ready"])
        # The edited file can be retried as a distinct source after freeing its port.
        plan = runtime.plan_restart("local", allow_source_switch=True)
        self.assertTrue(plan["source_switch"])
        retried = runtime.restart("local", timeout=20, startup_wait=10,
                                  allow_source_switch=True,
                                  expected_source_sha256=plan["candidate_source_sha256"])
        self.assertTrue(retried["ok"], retried)
        self.assertFalse(snapshot.exists())

    def test_explicit_source_switch_previews_and_restarts_new_config(self):
        first = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(first["ok"])
        next_config = self.root / "next-run.json"
        changed = json.loads(self.config.read_text())
        changed["model_name"] = "fake-next"
        next_config.write_text(json.dumps(changed), encoding="utf-8")
        previous_revision = profiles.load("local")[1]
        source_output = io.StringIO()
        with redirect_stdout(source_output):
            source_code = main(["profile", "use-config", "local", "--from-config", str(next_config),
                                "--if-revision", previous_revision, "--output", "json"])
        self.assertEqual(source_code, 0)
        next_revision = json.loads(source_output.getvalue())["revision"]

        denied = runtime.plan_restart("local")
        self.assertFalse(denied["ok"])
        self.assertEqual(denied["error"], "source_switch_requires_opt_in")
        self.assertTrue(denied["requires_allow_source_switch"])
        code, preview = self._cli("restart", "--profile", "local", "--allow-source-switch", "--plan")
        self.assertEqual(code, 0)
        self.assertTrue(preview["source_switch"])
        self.assertTrue(preview["changed"])
        self.assertEqual(preview["current_source_config"], str(self.config.resolve()))
        self.assertEqual(preview["candidate_source_config"], str(next_config.resolve()))
        self.assertEqual(preview["candidate_profile_revision"], next_revision)
        self.assertEqual(runtime.status("local")["instance_id"], first["instance_id"])

        code, missing_digest = self._cli("restart", "--profile", "local", "--allow-source-switch")
        self.assertEqual(code, 2)
        self.assertFalse(missing_digest["ok"])
        self.assertEqual(runtime.status("local")["instance_id"], first["instance_id"])
        next_config.write_bytes(next_config.read_bytes() + b" ")
        code, stale = self._cli("restart", "--profile", "local", "--allow-source-switch",
                                "--if-source-sha256", preview["candidate_source_sha256"])
        self.assertEqual(code, 6)
        self.assertEqual(stale["error"], "source_changed_after_preview")
        self.assertEqual(runtime.status("local")["instance_id"], first["instance_id"])
        next_config.write_text(json.dumps(changed), encoding="utf-8")
        code, switched = self._cli("restart", "--profile", "local", "--allow-source-switch",
                                   "--if-profile-revision", next_revision, "--wait", "20",
                                   "--if-source-sha256", preview["candidate_source_sha256"],
                                   "--startup-wait", "10")
        self.assertEqual(code, 0, switched)
        self.assertTrue(switched["source_switched"])
        self.assertNotEqual(switched["instance_id"], first["instance_id"])
        self.assertEqual(switched["profile_revision"], next_revision)
        self.assertTrue(switched["launch_attested"])
        record = journal.load("local", switched["operation_id"])
        self.assertTrue(record["intent"]["allow_source_switch"])
        self.assertEqual(runtime.plan_restart("local")["source_switch"], False)

    def test_failed_source_switch_restores_prior_owned_launch(self):
        first = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(first["ok"])
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen(1)
            next_config = self.root / "bad-next-run.json"
            changed = json.loads(self.config.read_text())
            changed["port"] = occupied.getsockname()[1]
            next_config.write_text(json.dumps(changed), encoding="utf-8")
            proposed, revision = profiles.load("local")
            proposed["source_config"] = str(next_config)
            profiles.save("local", proposed, expected_revision=revision)
            digest = hashlib.sha256(next_config.read_bytes()).hexdigest()
            result = runtime.restart("local", timeout=20, startup_wait=30, allow_source_switch=True,
                                     expected_source_sha256=digest)
        self.assertFalse(result["ok"])
        self.assertTrue(result["rolled_back"])
        self.assertNotEqual(result["instance_id"], first["instance_id"])
        self.assertEqual(runtime.status("local")["instance_id"], result["instance_id"])
        self.assertEqual(runtime.status("local")["profile_revision"], revision)

    def test_source_switch_needs_unchanged_distinct_regular_rollback_file(self):
        first = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(first["ok"])
        hardlink = self.root / "same-inode.json"
        os.link(self.config, hardlink)
        proposed, revision = profiles.load("local")
        proposed["source_config"] = str(hardlink)
        profiles.save("local", proposed, expected_revision=revision)
        denied = runtime.plan_restart("local", allow_source_switch=True)
        self.assertEqual(denied["error"], "source_switch_requires_distinct_regular_files")

        next_config = self.root / "next-run.json"
        next_config.write_bytes(self.config.read_bytes())
        proposed, revision = profiles.load("local")
        proposed["source_config"] = str(next_config)
        profiles.save("local", proposed, expected_revision=revision)
        self.config.write_bytes(self.config.read_bytes() + b" ")
        denied = runtime.plan_restart("local", allow_source_switch=True)
        self.assertEqual(denied["error"], "rollback_source_changed")
        self.assertEqual(runtime.status("local")["instance_id"], first["instance_id"])
        allowed = runtime.plan_restart("local", allow_source_switch=True,
                                       allow_source_migration=True)
        self.assertTrue(allowed["source_switch"])
        self.assertTrue(allowed["source_migration"])
        result = runtime.restart("local", timeout=20, startup_wait=10,
                                 allow_source_switch=True, allow_source_migration=True,
                                 expected_source_sha256=allowed["candidate_source_sha256"])
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["source_switched"])
        self.assertTrue(result["source_migrated"])
        self.assertEqual(list((self.root / "runtime").glob("*.rollback.json")), [])

    def test_server_reading_changed_config_fails_launch_attestation(self):
        host = runtime.Host("local", shutdown_timeout=2, startup_wait=10)
        plan, args, source, original = host._plan()
        # Change valid JSON between the owner's snapshot and the server's read.
        self.config.write_bytes(original + b" ")
        try:
            host._launch(args, source, original, plan["profile_revision"])
            result = host._wait_ready(10)
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"], "probe_launch_attestation_mismatch")
        finally:
            host._stop_child()

    def test_authenticated_generation_probe_keeps_credential_out_of_status(self):
        secret = "runtime-probe-test-key"
        cfg = json.loads(self.config.read_text())
        cfg["api_key"] = secret
        self.config.write_text(json.dumps(cfg))
        result = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(result["generation_ready"])
        self.assertTrue(result["launch_attested"])
        self.assertNotIn(secret, json.dumps(result))
        self.assertNotIn(secret, json.dumps(runtime.status("local")))

    def test_remote_bind_preview_accepts_server_supported_key_list(self):
        cfg = json.loads(self.config.read_text())
        cfg.update(host="0.0.0.0", api_key=["first", "second"])
        self.config.write_text(json.dumps(cfg))
        from strata_cli.cli import _launch_plan
        plan, _ = _launch_plan("local")
        self.assertEqual(plan["effective"]["host"], "0.0.0.0")
        self.assertNotIn("first", json.dumps(plan))

    def test_remote_bind_without_a_key_is_planned_not_refused(self):
        # owner decision 2026-10-20: a key is recommended on a network, never required
        from strata_cli.cli import _launch_plan
        for key in (None, "none"):
            cfg = json.loads(self.config.read_text())
            cfg["host"] = "0.0.0.0"
            cfg.pop("api_key", None)
            if key:
                cfg["api_key"] = key
            self.config.write_text(json.dumps(cfg))
            with mock.patch.dict(os.environ, {"STRATA_API_KEY": ""}):
                plan, args = _launch_plan("local")
            self.assertEqual(plan["effective"]["host"], "0.0.0.0")

    def test_lazy_model_is_verified_as_unloaded_without_generation(self):
        cfg = json.loads(self.config.read_text())
        cfg["lazy_load"] = True
        self.config.write_text(json.dumps(cfg))
        result = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(result["ok"])
        self.assertTrue(result["api_identity_verified"])
        self.assertFalse(result["generation_ready"])
        self.assertEqual((result["model_state"], result["readiness_state"]), ("unloaded", "unloaded"))
        code, checked = self._cli("verify", "--profile", "local", "--wait", "10", "--load")
        self.assertEqual(code, 0)
        self.assertTrue(checked["generation_ready"])
        self.assertEqual(runtime.status("local")["model_state"], "ready")
        self.assertTrue(runtime.verify("local", timeout=10)["generation_ready"])

    def test_failed_generation_probe_does_not_report_ready(self):
        engine = self.root / "engine.py"
        engine.write_text(engine.read_text().replace("        print('T 97', flush=True)",
                                                      "        print('ERR probe-failed', flush=True)\n        continue"))
        result = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertFalse(result["ok"])
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["error"], "probe_http_400")

    def test_zero_token_completion_does_not_count_as_generation_ready(self):
        (self.root / "empty-generation").write_text("empty")
        result = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "probe_generation_no_tokens")

    def test_explicit_verify_reports_later_engine_failure_without_stopping_instance(self):
        first = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(first["generation_ready"])
        flag = self.root / "fail-generation"
        flag.write_text("fail")
        code, checked = self._cli("verify", "--profile", "local", "--wait", "10")
        self.assertEqual(code, 6)
        self.assertEqual(checked["error"], "probe_http_400")
        self.assertEqual(journal.load("local", checked["operation_id"])["state"], "failed")
        self.assertEqual(runtime.status("local")["instance_id"], first["instance_id"])
        flag.unlink()
        self.assertTrue(runtime.verify("local", timeout=10)["generation_ready"])

    def test_failed_generation_after_profile_edit_rolls_back_prior_launch(self):
        first = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(first["generation_ready"])
        engine = self.root / "engine.py"
        engine.write_text(engine.read_text().replace(
            "        print('T 97', flush=True)",
            "        if slot is not None:\n"
            "            print('ERR batch-probe-failed', flush=True)\n"
            "            continue\n"
            "        print('T 97', flush=True)"))
        profiles.set_values("local", ["parallel=2"])
        result = runtime.restart("local", timeout=30, startup_wait=10)
        self.assertFalse(result["ok"])
        self.assertTrue(result["rolled_back"])
        restored = runtime.status("local")
        self.assertTrue(restored["generation_ready"])
        self.assertEqual(restored["instance_id"], result["instance_id"])

    def test_start_plan_is_read_only(self):
        code, result = self._cli("start", "--profile", "local", "--plan")
        self.assertEqual(code, 0)
        self.assertFalse(result["plan"]["would_start"])
        self.assertIsNone(runtime._read_state("local"))
        self.assertEqual(journal.list_runs("local"), [])

    def test_restart_plan_compares_revision_without_stopping_instance(self):
        started = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(started["ok"])
        unchanged = runtime.plan_restart("local")
        self.assertTrue(unchanged["ok"])
        self.assertFalse(unchanged["changed"])
        profiles.set_values("local", ["parallel=2"])
        code, planned = self._cli("restart", "--profile", "local", "--plan")
        self.assertEqual(code, 0)
        self.assertTrue(planned["changed"])
        self.assertFalse(planned["would_drain"])
        self.assertFalse(planned["would_start"])
        count = len(journal.list_runs("local"))
        code, denied = self._cli("restart", "--profile", "local",
                                 "--if-profile-revision", planned["candidate_profile_revision"],
                                 "--if-source-sha256", planned["candidate_source_sha256"],
                                 "--if-owner-id", planned["current_owner_id"],
                                 "--if-instance-id", "0" * 32)
        self.assertEqual((code, denied["error"]["message"]),
                         (6, "instance_changed_after_restart_plan"))
        self.assertEqual(len(journal.list_runs("local")), count)
        self.assertEqual(runtime.status("local")["instance_id"], started["instance_id"])

    def test_editable_recipe_previews_and_reruns_owned_instance(self):
        recipes.initialize("local")
        started = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(started["ok"])
        profiles.set_values("local", ["parallel=2"])
        code, planned = self._ops("plan", "restart")
        self.assertEqual(code, 0)
        self.assertTrue(planned["plan"]["operation"]["preflight"]["changed"])
        self.assertEqual(runtime.status("local")["instance_id"], started["instance_id"])
        revision = planned["plan"]["recipe_revision"]
        preflight = planned["plan"]["operation"]["preflight"]
        code, ran = self._ops("run", "restart", "--if-revision", revision,
                              "--if-profile-revision", planned["plan"]["profile_revision"],
                              "--if-owner-id", preflight["current_owner_id"],
                              "--if-instance-id", preflight["current_instance_id"],
                              "--if-source-sha256", preflight["candidate_source_sha256"])
        self.assertEqual(code, 0)
        self.assertTrue(ran["operation"]["result"]["restarted"])
        self.assertNotEqual(ran["operation"]["result"]["instance_id"], started["instance_id"])
        record = journal.load("local", ran["operation"]["operation_id"])
        self.assertEqual(record["state"], "succeeded")
        self.assertEqual(record["result"]["instance_id"], ran["operation"]["result"]["instance_id"])
        count = len(journal.list_runs("local"))
        with self.assertRaisesRegex(profiles.ProfileError, "instance_changed_after_restart_plan"):
            recipes.run("local", "restart", expected_revision=revision,
                        expected_profile_revision=planned["plan"]["profile_revision"],
                        expected_owner_id=preflight["current_owner_id"],
                        expected_instance_id=preflight["current_instance_id"],
                        expected_source_sha256=preflight["candidate_source_sha256"])
        self.assertEqual(len(journal.list_runs("local")), count)

    def test_stale_profile_revision_does_not_drain_owned_instance(self):
        started = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(started["ok"])
        previous = profiles.load("local")[1]
        profiles.set_values("local", ["parallel=2"])
        result = runtime.restart("local", timeout=10, expected_profile_revision=previous)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "profile_changed_after_preview")
        self.assertEqual(runtime.status("local")["instance_id"], started["instance_id"])

    def _ops(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["ops", *args, "--profile", "local", "--output", "json"])
        return code, json.loads(output.getvalue())

    @unittest.skipIf(WINDOWS, "SIGTERM to a Windows process is TerminateProcess: the hard-kill test below covers it")
    def test_terminating_owner_host_stops_its_child_and_removes_state(self):
        started = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(started["ok"])
        os.kill(started["host_pid"], signal.SIGTERM)
        deadline = time.monotonic() + 8
        while runtime._read_state("local") is not None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertIsNone(runtime._read_state("local"))
        owned = runtime._STARTED_HOSTS.pop("local", None)
        if owned is not None:
            self.assertEqual(owned.wait(timeout=3), 0)

    def test_hard_killed_host_child_exits_and_next_start_recovers_private_state(self):
        started = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(started["ok"])
        old_child = started["child_pid"]
        state = runtime._read_state("local")
        orphan = self.root / "runtime" / f"local-{state['token']}-{'a' * 32}.rollback.json"
        orphan.write_bytes(self.config.read_bytes())
        orphan.chmod(0o600)
        owned = runtime._STARTED_HOSTS.pop("local")
        if WINDOWS:
            owned.kill()                          # TerminateProcess: the host cannot clean up after itself
            owned.wait(timeout=3)
            # the job object ends the server tree with the host, no matter how the host ended
            deadline = time.monotonic() + 15
            while runtime._pid_exists(old_child) and time.monotonic() < deadline:
                time.sleep(0.1)
            self.assertFalse(runtime._pid_exists(old_child))
        else:
            os.kill(started["host_pid"], signal.SIGKILL)
            self.assertEqual(owned.wait(timeout=3), -signal.SIGKILL)
        stale = runtime.status("local")
        self.assertEqual((stale["state"], stale["error"]), ("stale", "owner_host_lost"))
        # The next start waits for the inherited-pipe guard to stop the old
        # process group; it never sends a signal to the saved child PID.
        replacement = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(replacement["ok"])
        self.assertNotEqual(replacement["child_pid"], old_child)
        self.assertFalse(runtime._pid_exists(old_child))
        self.assertFalse(orphan.exists())

    def test_recovery_refuses_a_still_present_child(self):
        state_path, socket_path, _ = runtime._paths("local")
        token = "a" * 64
        state = {"schema_version": 1, "profile": "local",
                 "socket": "tcp:127.0.0.1:50000" if WINDOWS else str(socket_path),
                 "token": token, "host_pid": 999999}
        lease = runtime._write_new(state_path, (json.dumps(state) + "\n").encode())
        os.close(lease)
        marker = runtime._child_marker("local", token)
        marker.write_text(json.dumps({"host_token": token, "phase": "owned",
                                      "pid": os.getpid(),
                                      "ready": str(state_path.parent / "local-fake.ready.json")}), encoding="utf-8")
        marker.chmod(0o600)
        with self.assertRaisesRegex(profiles.ProfileError, "still present"):
            runtime.start("local", wait=0, shutdown_timeout=2)
        self.assertTrue(state_path.exists())

    # ---- owner transport, attach and identity (POSIX socket and Windows loopback alike)

    def _stale_host(self, child_pid, *, started=None, token="a" * 64):
        """A state file whose host is gone, and the ownership marker naming `child_pid` as its server."""
        state_path, socket_path, _ = runtime._paths("local")
        state = {"schema_version": 1, "profile": "local",
                 "socket": "tcp:127.0.0.1:50000" if WINDOWS else str(socket_path),
                 "token": token, "host_pid": 999999}
        lease = runtime._write_new(state_path, (json.dumps(state) + "\n").encode())
        runtime._unlock_file(lease)
        os.close(lease)
        marker = runtime._child_marker("local", token)
        record = {"host_token": token, "phase": "owned", "pid": child_pid,
                  "ready": str(state_path.parent / "local-fake.ready.json")}
        if started is not None:
            record["started"] = started
        marker.write_text(json.dumps(record), encoding="utf-8")
        marker.chmod(0o600)
        return state_path, marker

    def _send(self, state, token, command):
        client = owner_transport.connect(state["socket"], timeout=5)
        try:
            client.sendall((json.dumps({"token": token, "command": command}) + "\n").encode())
            return json.loads(client.recv(4096).split(b"\n")[0])
        finally:
            client.close()

    def test_a_second_client_attaches_from_another_process_and_shares_the_owner(self):
        started = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(started["ok"])
        state = runtime._read_state("local")
        if WINDOWS:
            self.assertRegex(state["socket"], r"^tcp:127\.0\.0\.1:[1-9][0-9]*$")
        root = Path(__file__).resolve().parents[2]
        done = subprocess.run([sys.executable, "strata.py", "runtime", "status", "--profile", "local",
                               "--output", "json"], cwd=root, env=os.environ.copy(), capture_output=True,
                              text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        other = json.loads(done.stdout)
        self.assertEqual((other["state"], other["owner_id"], other["instance_id"], other["host_pid"]),
                         ("ready", started["owner_id"], started["instance_id"], started["host_pid"]))
        self.assertEqual(runtime.status("local")["owner_id"], started["owner_id"])   # attaching took nothing over
        self.assertEqual(runtime.stop("local", timeout=12)["ok"], True)
        self.assertEqual(runtime.status("local")["state"], "stopped")

    def test_a_wrong_or_stale_token_cannot_stop_the_owner(self):
        started = runtime.start("local", wait=10, shutdown_timeout=2)
        self.assertTrue(started["ok"])
        state = runtime._read_state("local")
        wrong_last = "0" if state["token"][-1] != "0" else "1"
        for bad in ("0" * 64, state["token"][:-1] + wrong_last, "", "x" * 10):
            with self.subTest(token=bad[:6]):
                answer = self._send(state, bad, "stop")
                self.assertEqual((answer["ok"], answer["error"]), (False, "unauthorized"))
        self.assertEqual(runtime.status("local")["instance_id"], started["instance_id"])
        self.assertTrue(self._send(state, state["token"], "status")["ok"])     # the real token is what authorises

    def test_a_recorded_pid_of_an_unrelated_live_process_is_never_signalled(self):
        victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        try:
            state_path, marker = self._stale_host(victim.pid)
            with self.assertRaisesRegex(profiles.ProfileError, "still present"):
                runtime.start("local", wait=0, shutdown_timeout=2)
            with self.assertRaises(profiles.ProfileError):            # nothing answers on the recorded address
                runtime.stop("local", timeout=2)
            self.assertIsNone(victim.poll(), "an unrelated process was stopped")
            self.assertTrue(state_path.exists() and marker.exists())
        finally:
            victim.kill()
            victim.wait(timeout=10)

    @unittest.skipUnless(WINDOWS, "a creation time is recorded with the PID on Windows only")
    def test_a_reused_pid_with_another_creation_time_is_not_the_recorded_child(self):
        victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        try:
            real = runtime.winproc.process_started(victim.pid)
            self.assertTrue(runtime._pid_exists(victim.pid, real))
            self.assertFalse(runtime._pid_exists(victim.pid, real + 12345))
            self._stale_host(victim.pid, started=real + 12345)           # the number was reused by another process
            started = runtime.start("local", wait=10, shutdown_timeout=2)
            self.assertTrue(started["ok"])                               # recovered: it was not the recorded child
            self.assertIsNone(victim.poll(), "the process that reused the PID was stopped")
        finally:
            victim.kill()
            victim.wait(timeout=10)

    def test_marker_without_creation_time_ignores_a_reused_pid_of_a_foreign_program(self):
        # #3488: the marker had no `started`, the number was reused by another program, every load said "still present".
        child = {"pid": 38552}
        with mock.patch.object(runtime, "WINDOWS", True), \
                mock.patch.object(runtime, "_pid_exists", return_value=True), \
                mock.patch.object(runtime.winproc, "process_image", create=True) as image:
            image.return_value = r"C:\Windows\System32\svchost.exe"
            self.assertFalse(runtime._child_present(child))
            image.return_value = r"C:\Users\a\AppData\Local\Strata\python\python.exe"
            self.assertTrue(runtime._child_present(child))
            image.return_value = None                                    # unreadable: keep refusing, never guess
            self.assertTrue(runtime._child_present(child))
            image.return_value = r"C:\Windows\System32\svchost.exe"
            self.assertTrue(runtime._child_present({"pid": 1, "started": 5}))   # a creation time decides by itself

    def test_a_process_created_after_the_marker_was_written_is_not_the_recorded_child(self):
        # 2026-10-22: the app was force-killed, its marker had no creation time, the number now belonged to another Python program, and every
        # load said "lost host child is still present" until the marker was deleted by hand. The child was started BEFORE its marker was written.
        child = {"pid": 4242}
        with mock.patch.object(runtime, "_pid_exists", return_value=True), \
                mock.patch.object(runtime.winproc, "process_image", create=True, return_value="C:\\Python\\python.exe"), \
                mock.patch.object(runtime.procs, "started_epoch") as created:
            created.return_value = 1000.0 + 60
            self.assertFalse(runtime._child_present(child, written=1000.0), "created a minute after the marker: another process")
            created.return_value = 1000.0 - 5
            self.assertTrue(runtime._child_present(child, written=1000.0), "created before the marker: could be the child, keep refusing")
            created.return_value = 1000.0 + 1.5
            self.assertTrue(runtime._child_present(child, written=1000.0), "inside the clock slack: keep refusing")
            created.return_value = None
            self.assertTrue(runtime._child_present(child, written=1000.0), "creation time unreadable: never guess")
            created.return_value = 1000.0 + 60
            self.assertTrue(runtime._child_present({"pid": 4242, "started": 7}, written=1000.0) is True, "a recorded creation time decides by itself")
            self.assertTrue(runtime._child_present(child), "no marker time: unchanged")

    def test_a_marker_left_by_a_killed_app_is_cleaned_up_when_its_number_belongs_to_a_later_program(self):
        victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        try:
            state_path, marker = self._stale_host(victim.pid)             # a marker with no creation time (Windows could not read it)
            old = time.time() - 3600
            os.utime(marker, (old, old))                                  # written an hour ago: the victim started after it
            started = runtime.start("local", wait=10, shutdown_timeout=2)
            self.assertTrue(started["ok"], started)
            self.assertFalse(marker.exists(), "the stale marker was removed")
            self.assertIsNone(victim.poll(), "the program that reused the number was left alone")
        finally:
            victim.kill()
            victim.wait(timeout=10)

    def test_started_epoch_reads_a_live_process_and_nothing_for_a_dead_number(self):
        victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        try:
            now = time.time()
            created = procs.started_epoch(victim.pid)
            self.assertIsNotNone(created)
            self.assertLess(abs(created - now), 60)
        finally:
            victim.kill()
            victim.wait(timeout=10)
        self.assertIsNone(procs.started_epoch(victim.pid))
        self.assertIsNone(procs.started_epoch(0))
        self.assertIsNone(procs.started_epoch(-5))

    @unittest.skipUnless(WINDOWS, "os.kill(pid, 0) ends a process on Windows; liveness is a handle query")
    def test_liveness_check_does_not_end_the_process_it_asks_about(self):
        victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        try:
            for _ in range(5):
                self.assertTrue(runtime._pid_exists(victim.pid))
            self.assertIsNone(victim.poll())
        finally:
            victim.kill()
            victim.wait(timeout=10)
        self.assertFalse(runtime._pid_exists(victim.pid))

    @unittest.skipUnless(WINDOWS, "ACLs stand in for mode 0700 on Windows")
    def test_runtime_directory_is_owner_only_and_a_shared_one_is_refused(self):
        from strata_control import private_files
        directory = runtime.runtime_dir()
        self.assertTrue(private_files.is_private(directory))
        shared = Path(self.temp.name) / "shared-runtime"
        shared.mkdir()
        done = subprocess.run(["icacls", str(shared), "/grant", "*S-1-1-0:(OI)(CI)R"], capture_output=True)
        if done.returncode != 0:
            self.skipTest("icacls could not widen the test directory")
        with mock.patch.dict(os.environ, {"STRATA_RUNTIME_DIR": str(shared)}):
            # a folder that names Everyone is the user's own state folder: it is locked down once and used (never a stop)
            self.assertEqual(runtime.runtime_dir(), shared)
            self.assertTrue(private_files.is_unshared(shared))
        again = Path(self.temp.name) / "shared-runtime-2"
        again.mkdir()
        subprocess.run(["icacls", str(again), "/grant", "*S-1-1-0:(OI)(CI)R"], capture_output=True)
        with mock.patch.dict(os.environ, {"STRATA_RUNTIME_DIR": str(again)}), \
                mock.patch.object(private_files, "restrict_to_current_user"), \
                mock.patch.object(private_files, "repair_access", return_value=False):
            # when the lock-down does not happen it still refuses, in plain words and naming who can open it
            with self.assertRaisesRegex(profiles.ProfileError, "Everyone"):
                runtime.runtime_dir()

    def test_ready_pid_may_be_a_child_of_the_launcher_on_windows_only(self):
        # #3683: a Windows venv's python.exe starts the real interpreter as its child, so the server that writes the ready file
        # is one process below the pid the host started. It must still match; a stranger or a distant process must not.
        parents = {200: 100, 300: 200, 400: 300, 500: 400}
        with mock.patch.object(runtime.procs, "parent_of", side_effect=lambda pid: parents.get(pid)):
            with mock.patch.object(runtime, "WINDOWS", True):
                self.assertTrue(runtime._ready_pid_is_child(100, 100))
                self.assertTrue(runtime._ready_pid_is_child(200, 100))
                self.assertTrue(runtime._ready_pid_is_child(300, 100))
                self.assertTrue(runtime._ready_pid_is_child(400, 100))
                self.assertFalse(runtime._ready_pid_is_child(500, 100))      # too far below
                self.assertFalse(runtime._ready_pid_is_child(999, 100))      # unrelated
                self.assertFalse(runtime._ready_pid_is_child(None, 100))
                self.assertFalse(runtime._ready_pid_is_child("200", 100))
                self.assertFalse(runtime._ready_pid_is_child(0, 100))
            with mock.patch.object(runtime, "WINDOWS", False):
                self.assertTrue(runtime._ready_pid_is_child(100, 100))
                self.assertFalse(runtime._ready_pid_is_child(200, 100))      # POSIX has no launcher in between

    @unittest.skipUnless(WINDOWS, "the venv launcher is a Windows thing")
    def test_server_started_through_a_launcher_process_becomes_ready(self):
        launcher = self.root / "launcher.cmd"
        launcher.write_bytes(f'@echo off\r\n"{sys.executable}" %*\r\n'.encode())
        host = runtime.Host("local", shutdown_timeout=2, startup_wait=20)
        plan, args, source, original = host._plan()
        with mock.patch.object(sys, "executable", str(launcher)):
            host._launch(args, source, original, plan["profile_revision"])
        try:
            result = host._wait_ready(30)
            self.assertTrue(result["ok"], result)
            self.assertTrue(result["generation_ready"])
            ready = json.loads(host.ready_path.read_text(encoding="utf-8"))
            self.assertNotEqual(ready["pid"], host.child.pid)               # the server is below the launcher
        finally:
            host._stop_child()

    def test_a_ready_file_that_is_not_ours_is_named_in_the_status_phase(self):
        host = runtime.Host("local", shutdown_timeout=2, startup_wait=10)
        host.child = mock.Mock(pid=111)
        host.child.poll.return_value = None
        host.ready_path = self.root / "ready.json"
        host.ready_path.write_text(json.dumps({"pid": 222, "instance_id": "a" * 32}), encoding="utf-8")
        with mock.patch.object(runtime.procs, "parent_of", return_value=None):
            self.assertEqual(host._status()["phase"], "ready_file_not_matched")
        host.ready_path.unlink()
        self.assertEqual(host._status()["phase"], "loading")

    def test_exclusive_lease_waits_for_shared_holders_and_shared_holders_coexist(self):
        first = runtime.acquire_activity_lease(shared=True)
        second = runtime.acquire_activity_lease(shared=True)
        try:
            with self.assertRaisesRegex(profiles.ProfileError, "engine lease"):
                runtime.acquire_activity_lease(shared=False)
        finally:
            for descriptor in (first, second):
                runtime._unlock_file(descriptor)
                os.close(descriptor)
        exclusive = runtime.acquire_activity_lease(shared=False)
        try:
            with self.assertRaisesRegex(profiles.ProfileError, "engine lease"):
                runtime.acquire_activity_lease(shared=True)
        finally:
            runtime._unlock_file(exclusive)
            os.close(exclusive)


class StartWaitLimit(unittest.TestCase):
    def setUp(self):
        # A start writes an operation record before the engine runs, so these tests keep their records (and any profile
        # lookup) in a scratch folder rather than in the person's own state.
        temp = tempfile.TemporaryDirectory(prefix="strata-start-wait-test-")
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / "profiles").mkdir()
        env = mock.patch.dict(os.environ, {"STRATA_PROFILE_DIR": str(root / "profiles"),
                                           "STRATA_RUNTIME_DIR": str(root / "runtime")})
        env.start()
        self.addCleanup(env.stop)

    def test_a_start_may_wait_up_to_thirty_minutes_for_a_big_model(self):                    # #3949
        from strata_control import runtime as rt, profiles as pr
        self.assertEqual(rt.MAX_START_WAIT, 1800)
        with self.assertRaisesRegex(pr.ProfileError, "start wait must be 0..1800"):
            rt.start("x", wait=1801)
        with mock.patch.object(rt, "_name", side_effect=pr.ProfileError("reached")):
            with self.assertRaisesRegex(pr.ProfileError, "reached"):
                rt.start("x", wait=1800)

    def test_a_long_start_wait_reaches_the_runtime_through_the_cli(self):                     # #3459 #4191
        # The desktop app starts models through `strata runtime start --wait`, and that wait is checked three times before
        # the engine is reached: by management, by the operation record, and by the runtime itself. When only the runtime's
        # ceiling was raised (#3949), a long start died at each of the first two with
        # "start wait must be finite seconds in 0..600" / "direct runtime intent has an invalid timeout". So this walks the
        # whole way with only the engine itself faked: any layer that still carries the old 600 s shows up here.
        from strata_control import journal as jn, management as mg, profiles as pr, runtime as rt
        revision = "0123456789abcdef"
        for wait in (300, 600, 601, rt.MAX_START_WAIT):
            reached = {}
            with mock.patch.object(rt, "_name", lambda name: name), \
                    mock.patch.object(mg.profiles, "load", lambda name: ("src", revision)), \
                    mock.patch.object(rt, "start", lambda name, **kw: (reached.update(kw),
                            {"ok": True, "state": "ready", "profile": name})[1]):
                result = mg.runtime_action("start", "demo", wait=wait, shutdown_timeout=15)
                self.assertEqual(result["state"], "ready")
                self.assertEqual(reached["wait"], float(wait),
                                 f"a start asked for {wait} s did not reach runtime.start with it")

    def test_a_start_wait_above_the_runtime_ceiling_is_named_by_the_runtime(self):             # #3459 #4191
        from strata_control import management as mg, profiles as pr, runtime as rt
        revision = "0123456789abcdef"
        with mock.patch.object(rt, "_name", lambda name: name), \
                mock.patch.object(mg.profiles, "load", lambda name: ("src", revision)):
            with self.assertRaisesRegex(pr.ProfileError, f"start wait must be finite seconds in 0..{rt.MAX_START_WAIT}"):
                mg.runtime_action("start", "demo", wait=rt.MAX_START_WAIT + 1, shutdown_timeout=15)
            with self.assertRaisesRegex(pr.ProfileError, f"startup wait must be finite seconds in 1..{rt.MAX_START_WAIT}"):
                mg.runtime_action("restart", "demo", startup_wait=rt.MAX_START_WAIT + 1,
                                  if_owner_id="0" * 32, if_instance_id="0" * 32)

    def test_the_operation_record_takes_the_runtimes_own_start_ceiling(self):
        # The record is written before the start runs, so a ceiling kept here would stop a long wait no matter what the
        # two validators above allow. Waits that are not a start or a restart keep their own 600 s.
        from strata_control import journal as jn, profiles as pr, runtime as rt
        for kind, intent, allowed in (
                ("runtime.start", {"wait_s": rt.MAX_START_WAIT, "shutdown_timeout_s": 15}, True),
                ("runtime.start", {"wait_s": rt.MAX_START_WAIT + 1, "shutdown_timeout_s": 15}, False),
                ("runtime.restart", {"wait_s": 1000, "startup_wait_s": rt.MAX_START_WAIT,
                                     "expected_profile_revision": "0" * 16,
                                     "expected_owner_id": "0" * 32, "expected_instance_id": "0" * 32,
                                     "allow_source_switch": False, "allow_source_migration": False,
                                     "expected_source_sha256": "0" * 64}, True),
                ("runtime.stop", {"wait_s": rt.MAX_START_WAIT}, False),
                ("runtime.verify", {"wait_s": rt.MAX_START_WAIT, "load": False}, False)):
            with self.subTest(kind=kind, wait=intent["wait_s"]):
                if allowed:
                    jn._validate_direct_intent(kind, dict(intent))
                else:
                    with self.assertRaisesRegex(pr.ProfileError, "invalid timeout"):
                        jn._validate_direct_intent(kind, dict(intent))


if __name__ == "__main__":
    unittest.main()
