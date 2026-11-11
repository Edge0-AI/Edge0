"""Local profile and component workflows without a GPU or network service."""

import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from strata_cli.cli import main
from strata_control import doctor, profiles, runtime


class OperatorProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = mock.patch.dict(os.environ, {"STRATA_PROFILE_DIR": str(self.root / "profiles"),
                                             "STRATA_RUNTIME_DIR": str(self.root / "runtime"),
                                             "STRATA_API_KEY": ""})
        self.env.start()
        self.addCleanup(self.env.stop)
        engine = self.root / "engine"
        engine.write_text("engine placeholder", encoding="utf-8")
        engine.chmod(0o700)
        tokenizer = self.root / "tokenizer"
        tokenizer.mkdir()
        for name in ("vocab.json", "merges.txt", "token_type.json"):
            (tokenizer / name).write_text("{}", encoding="utf-8")
        pack = self.root / "pack"
        pack.mkdir()
        shard = self.root / "model.gguf"
        shard.write_bytes(b"model placeholder")
        self.shard = shard
        self.config = self.root / "run.json"
        self.config.write_text(json.dumps({"exe": str(engine), "args": ["--pack", str(pack), "--native", str(shard)],
                                           "cwd": str(self.root), "tokenizer": str(tokenizer),
                                           "model_name": "test-model", "port": 8095}), encoding="utf-8")

    def command(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(list(args))
        return code, json.loads(output.getvalue())

    def test_saved_overrides_drive_serve_plan_without_rewriting_run_config_or_starting(self):
        before = self.config.read_bytes()
        code, created = self.command("profile", "create", "four-gpu", "--from-config", str(self.config),
                                     "--output", "json")
        self.assertEqual(code, 0)
        self.assertEqual(created["profile"]["overrides"], {})
        code, changed = self.command("profile", "set", "four-gpu", "parallel=8", "batch_groups=2",
                                     "max_queue=32", "queue_timeout=12.5", "batch_telemetry=true",
                                     "--output", "json")
        self.assertEqual(code, 0)
        self.assertNotEqual(changed["revision"], created["revision"])
        with mock.patch("serve.server.main", side_effect=AssertionError("server started")):
            code, planned = self.command("serve", "--profile", "four-gpu", "--plan", "--output", "json")
        self.assertEqual(code, 0)
        plan = planned["plan"]
        self.assertEqual(plan["effective"]["parallel"], 8)
        self.assertEqual(plan["effective"]["batch_groups"], 2)
        self.assertEqual(plan["effective"]["max_queue"], 32)
        self.assertEqual(plan["effective"]["queue_timeout"], 12.5)
        self.assertTrue(plan["effective"]["batch_telemetry"])
        self.assertEqual(plan["origin"]["parallel"], "saved_profile")
        self.assertEqual(plan["engine_argv"][-4:], ["--batch-groups", "2", "--batch", "8"])
        self.assertIn("--max-queue", plan["server_argv"])
        self.assertIn("--batch-telemetry", plan["server_argv"])
        self.assertEqual(plan["server_argv"][plan["server_argv"].index("--queue-timeout") + 1], "12.5")
        self.assertEqual(self.config.read_bytes(), before)
        self.assertEqual(len(list((self.root / "profiles" / ".history" / "four-gpu").glob("*.json"))), 1)

    def test_saved_profile_is_used_by_actual_foreground_serve_call(self):
        profiles.create("local", str(self.config))
        profiles.set_values("local", ["parallel=4", "max_queue=7"])
        with mock.patch("serve.server.main", return_value=0) as server:
            self.assertEqual(main(["serve", "--profile", "local"]), 0)
        argv = server.call_args.args[0]
        self.assertEqual(server.call_args.kwargs["interrupt_exit_code"], 130)
        self.assertIn(str(self.config.resolve()), argv)
        self.assertEqual(argv[argv.index("--parallel") + 1], "4")
        self.assertEqual(argv[argv.index("--max-queue") + 1], "7")

    def test_json_alias_is_one_shot_and_keeps_machine_errors(self):
        code, listed = self.command("profile", "list", "--json")
        self.assertEqual(code, 0)
        self.assertTrue(listed["ok"])
        self.assertEqual(listed["profiles"], [])
        with mock.patch("serve.server.main", side_effect=AssertionError("server started")):
            code, planned = self.command("serve", "--profile", str(self.config), "--plan", "--json")
        self.assertEqual(code, 0)
        self.assertTrue(planned["ok"])
        self.assertIn("plan", planned)
        code, no_plan = self.command("serve", "--profile", str(self.config), "--json")
        self.assertEqual(code, 2)
        self.assertEqual(no_plan["error"]["code"], 2)
        code, stream = self.command("serve", "--profile", str(self.config), "--json",
                                    "--output", "jsonl")
        self.assertEqual(code, 2)
        self.assertEqual(stream["event"], "failed")
        code, harness = self.command("code", "sessions", "list", "--json")
        self.assertEqual(code, 5)
        self.assertEqual(harness["error"]["code"], 5)

    def test_quiet_suppresses_human_success_but_preserves_errors_and_json(self):
        output, diagnostics = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(diagnostics):
            self.assertEqual(main(["profile", "list", "--quiet"]), 0)
            self.assertEqual(main(["serve", "--profile", str(self.config), "--plan", "--quiet"]), 0)
        self.assertEqual((output.getvalue(), diagnostics.getvalue()), ("", ""))
        with redirect_stdout(output), redirect_stderr(diagnostics):
            self.assertEqual(main(["profile", "list", "--quiet", "--json"]), 0)
        self.assertTrue(json.loads(output.getvalue())["ok"])
        output, diagnostics = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(diagnostics):
            self.assertNotEqual(main(["profile", "show", "missing", "--quiet"]), 0)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("strata:", diagnostics.getvalue())
        output, diagnostics = io.StringIO(), io.StringIO()
        def failed_operation(_a):
            print("x" * 70_000)
            print("failure detail")
            return 5
        with mock.patch("strata_cli.cli._profile_command", side_effect=failed_operation):
            with redirect_stdout(output), redirect_stderr(diagnostics):
                self.assertEqual(main(["profile", "list", "--quiet"]), 5)
        self.assertEqual(output.getvalue(), "")
        self.assertLess(len(diagnostics.getvalue()), 66_000)
        self.assertIn("earlier human output omitted", diagnostics.getvalue())
        self.assertIn("failure detail", diagnostics.getvalue())

    def test_quiet_foreground_serve_hides_human_banner(self):
        output = io.StringIO()
        with mock.patch("serve.server.main", side_effect=lambda *_args, **_kwargs: print("ready banner") or 0):
            with redirect_stdout(output):
                self.assertEqual(main(["serve", "--profile", str(self.config), "--quiet"]), 0)
        self.assertEqual(output.getvalue(), "")
        output, diagnostics = io.StringIO(), io.StringIO()
        with mock.patch("serve.server.main", return_value=8):
            with redirect_stdout(output), redirect_stderr(diagnostics):
                self.assertEqual(main(["serve", "--profile", str(self.config), "--quiet"]), 8)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("drain deadline expired", diagnostics.getvalue())

    def test_foreground_jsonl_rejects_plan_and_bad_profile_as_events(self):
        code, invalid = self.command("serve", "--profile", str(self.config), "--plan",
                                     "--output", "jsonl")
        self.assertEqual(code, 2)
        self.assertEqual(invalid["event"], "failed")
        self.assertEqual(invalid["exit_code"], 2)
        code, missing = self.command("serve", "--profile", "missing-profile",
                                     "--output", "jsonl")
        self.assertNotEqual(code, 0)
        self.assertEqual(missing["event"], "failed")
        self.assertEqual(missing["exit_code"], code)

    def test_foreground_jsonl_reports_startup_failure_without_human_stdout(self):
        output, diagnostics = io.StringIO(), io.StringIO()
        with mock.patch("serve.server.main", side_effect=SystemExit("engine refused to start")):
            with redirect_stdout(output), redirect_stderr(diagnostics):
                code = main(["serve", "--profile", str(self.config), "--output", "jsonl"])
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(code, 6)
        self.assertEqual([row["event"] for row in rows], ["starting", "failed"])
        self.assertEqual(rows[-1]["phase"], "startup")
        self.assertNotIn("engine refused to start", output.getvalue())
        self.assertIn("startup failed", diagnostics.getvalue())

    @unittest.skipUnless(os.name == "posix", "managed activity lease is POSIX-only")
    def test_foreground_serve_holds_shared_lease_until_server_returns(self):
        profiles.create("local", str(self.config))

        def running_server(argv, **_kwargs):
            with self.assertRaisesRegex(profiles.ProfileError, "engine lease"):
                runtime.acquire_activity_lease(shared=False)
            return 0

        with mock.patch("serve.server.main", side_effect=running_server):
            self.assertEqual(main(["serve", "--profile", "local"]), 0)
        exclusive = runtime.acquire_activity_lease(shared=False)
        os.close(exclusive)

    def test_batch_telemetry_precedence_can_disable_a_run_config(self):
        cfg = json.loads(self.config.read_text())
        cfg["batch_telemetry"] = True
        self.config.write_text(json.dumps(cfg))
        profiles.create("local", str(self.config))
        code, inherited = self.command("serve", "--profile", "local", "--plan", "--output", "json")
        self.assertEqual(code, 0)
        self.assertTrue(inherited["plan"]["effective"]["batch_telemetry"])
        self.assertEqual(inherited["plan"]["origin"]["batch_telemetry"], "run_config")
        profiles.set_values("local", ["batch_telemetry=false"])
        code, disabled = self.command("serve", "--profile", "local", "--plan", "--output", "json")
        self.assertEqual(code, 0)
        self.assertFalse(disabled["plan"]["effective"]["batch_telemetry"])
        self.assertIn("--no-batch-telemetry", disabled["plan"]["server_argv"])
        code, enabled = self.command("serve", "--profile", "local", "--batch-telemetry",
                                     "--plan", "--output", "json")
        self.assertEqual(code, 0)
        self.assertEqual(enabled["plan"]["origin"]["batch_telemetry"], "command")
        self.assertIn("--batch-telemetry", enabled["plan"]["server_argv"])

    def test_serve_plan_validates_ready_file_without_writing_it(self):
        profiles.create("local", str(self.config))
        ready = self.root / "ready.json"
        code, result = self.command("serve", "--profile", "local", "--ready-file", str(ready),
                                    "--plan", "--output", "json")
        self.assertEqual(code, 0)
        self.assertEqual(result["plan"]["ready_file"], str(ready))
        self.assertFalse(ready.exists())
        ready.write_text("stale", encoding="utf-8")
        code, result = self.command("serve", "--profile", "local", "--ready-file", str(ready),
                                    "--plan", "--output", "json")
        self.assertEqual(code, 2)
        self.assertFalse(result["ok"])

    def test_stale_edit_rejected_and_bad_editor_content_keeps_saved_revision(self):
        profiles.create("local", str(self.config))
        current, old_revision = profiles.load("local")
        profiles.set_values("local", ["parallel=4"])
        with self.assertRaisesRegex(profiles.ProfileError, "changed during edit"):
            profiles.save("local", current, expected_revision=old_revision)
        _, current_revision = profiles.load("local")

        def corrupt(command, check):
            Path(command[-1]).write_text("{invalid", encoding="utf-8")
            return mock.Mock(returncode=0)

        with mock.patch.dict(os.environ, {"EDITOR": "fake-editor", "VISUAL": ""}), \
                mock.patch("strata_control.profiles.subprocess.run", side_effect=corrupt):
            code, result = self.command("profile", "edit", "local", "--output", "json")
        self.assertEqual(code, 2)
        self.assertFalse(result["ok"])
        draft_path = Path(result["error"]["message"].split("draft kept at ", 1)[1])
        self.assertTrue(draft_path.is_file())
        self.assertEqual(draft_path.read_text(encoding="utf-8"), "{invalid")
        self.assertEqual(profiles.load("local")[1], current_revision)

    def test_plan_then_history_and_rollback_restore_saved_settings_only(self):
        before = self.config.read_bytes()
        profiles.create("local", str(self.config))
        _, first = profiles.load("local")
        code, planned = self.command("profile", "set", "local", "parallel=8", "--plan", "--output", "json")
        self.assertEqual(code, 0)
        self.assertTrue(planned["would_save"])
        self.assertEqual(planned["changes"]["parallel"], {"before": None, "after": 8})
        self.assertEqual(profiles.load("local")[1], first)
        code, changed = self.command("profile", "set", "local", "parallel=8", "--output", "json")
        self.assertEqual(code, 0)
        code, listing = self.command("profile", "history", "local", "--output", "json")
        self.assertEqual(code, 0)
        self.assertEqual(listing["history"][0]["revision"], changed["revision"])
        self.assertIn(first, [item["revision"] for item in listing["history"]])
        code, restored = self.command("profile", "rollback", "local", "--to", first, "--output", "json")
        self.assertEqual(code, 0)
        self.assertEqual(restored["revision"], first)
        self.assertEqual(restored["profile"]["overrides"], {})
        self.assertEqual(self.config.read_bytes(), before)

    def test_use_config_previews_and_saves_a_new_model_source_without_editing_either_config(self):
        profiles.create("local", str(self.config))
        old_profile, old_revision = profiles.load("local")
        next_config = self.root / "next.json"
        next_value = json.loads(self.config.read_text(encoding="utf-8"))
        next_value["model_name"] = "next-model"
        next_config.write_text(json.dumps(next_value), encoding="utf-8")
        before = (self.config.read_bytes(), next_config.read_bytes())
        code, preview = self.command("profile", "use-config", "local", "--from-config",
                                     str(next_config), "--plan", "--output", "json")
        self.assertEqual(code, 0)
        self.assertTrue(preview["would_save"])
        self.assertEqual(preview["revision"], old_revision)
        self.assertEqual(preview["model_name"], "next-model")
        self.assertEqual(profiles.load("local"), (old_profile, old_revision))
        code, unavailable = self.command("profile", "use-config", "local", "--from-config",
                                         str(self.root / "missing.json"), "--output", "json")
        self.assertEqual(code, 5)
        self.assertFalse(unavailable["ok"])
        self.assertEqual(profiles.load("local"), (old_profile, old_revision))
        code, stale = self.command("profile", "use-config", "local", "--from-config",
                                   str(next_config), "--if-revision", "0" * 16, "--output", "json")
        self.assertEqual(code, 6)
        self.assertFalse(stale["ok"])
        code, saved = self.command("profile", "use-config", "local", "--from-config",
                                   str(next_config), "--if-revision", old_revision, "--output", "json")
        self.assertEqual(code, 0)
        self.assertEqual(saved["profile"]["source_config"], str(next_config.resolve()))
        self.assertNotEqual(saved["revision"], old_revision)
        self.assertEqual((self.config.read_bytes(), next_config.read_bytes()), before)
        self.assertIn(old_revision, [row["revision"] for row in profiles.history("local")])

    def test_doctor_distinguishes_required_missing_from_unverified_optional(self):
        profiles.create("local", str(self.config))
        code, result = self.command("doctor", "--profile", "local", "--components", "--output", "json")
        self.assertEqual(code, 0)
        self.assertTrue(result["report"]["ok"])
        self.assertEqual(result["report"]["integrity"], "unverified_no_manifest_hashes")
        self.shard.unlink()
        code, result = self.command("components", "verify", "--profile", "local", "--output", "json")
        self.assertEqual(code, 5)
        self.assertFalse(result["report"]["ok"])
        self.assertTrue(any(i["component"] == "native" for i in result["report"]["issues"]))

    def test_monitor_inventory_checks_the_optional_pinned_version(self):
        pinned = (Path(__file__).resolve().parents[2] / "requirements-monitor.txt").read_text()
        self.assertIn(f"textual=={doctor.MONITOR_TEXTUAL_VERSION}", pinned)
        with mock.patch.object(doctor.util, "find_spec", return_value=None):
            missing = doctor._monitor_component()
        self.assertEqual(missing["status"], "optional_missing")
        with mock.patch.object(doctor.util, "find_spec", return_value=object()), \
                mock.patch.object(doctor.metadata, "version", return_value="8.2.8"):
            present = doctor._monitor_component()
        self.assertEqual(present["status"], "ok")
        self.assertEqual(present["check"], "package_version_only")
        with mock.patch.object(doctor.util, "find_spec", return_value=object()), \
                mock.patch.object(doctor.metadata, "version", return_value="9.0.0"):
            changed = doctor._monitor_component()
        self.assertEqual(changed["status"], "unverified")
        self.assertEqual(changed["version"], "9.0.0")
        with mock.patch.object(doctor.util, "find_spec", return_value=object()), \
                mock.patch.object(doctor.metadata, "version",
                                  side_effect=doctor.metadata.PackageNotFoundError("textual")):
            unknown = doctor._monitor_component()
        self.assertEqual(unknown["status"], "unverified")

        output = io.StringIO()
        with mock.patch.object(doctor, "_monitor_component", return_value=changed), redirect_stdout(output):
            code = main(["doctor"])
        self.assertEqual(code, 0)  # an optional version does not stop serving
        self.assertIn("interactive monitor was qualified with Textual 8.2.8", output.getvalue())

    def test_live_doctor_checks_four_selected_nvidia_cards_without_starting_engine(self):
        config = json.loads(self.config.read_text(encoding="utf-8"))
        config["gpu"] = [0, 1, 2, 3]
        config["backend"] = "cuda"
        self.config.write_text(json.dumps(config), encoding="utf-8")
        profiles.create("four-gpu", str(self.config))
        cards = "\n".join(f"{i}, RTX 6000 test {i}, GPU-test-{i}, 49140, 580.95" for i in range(4))
        with mock.patch("strata_control.doctor.shutil.which", return_value="/fake/nvidia-smi"), \
                mock.patch("strata_control.doctor.subprocess.run",
                           return_value=mock.Mock(returncode=0, stdout=cards)) as probe:
            code, result = self.command("doctor", "--profile", "four-gpu", "--live", "--output", "json")
        self.assertEqual(code, 0)
        self.assertEqual(result["report"]["live_gpu_probe"]["status"], "ok")
        self.assertEqual(result["report"]["live_gpu_probe"]["selected_indices"], [0, 1, 2, 3])
        self.assertEqual(len(result["report"]["live_gpu_probe"]["cards"]), 4)
        self.assertEqual(probe.call_args.kwargs["timeout"], 5)
        self.assertFalse(probe.call_args.kwargs.get("shell", False))

        with mock.patch("strata_control.doctor.shutil.which", return_value="/fake/nvidia-smi"), \
                mock.patch("strata_control.doctor.subprocess.run",
                           return_value=mock.Mock(returncode=0, stdout=cards.rsplit("\n", 1)[0])):
            code, result = self.command("components", "verify", "--profile", "four-gpu", "--live",
                                        "--output", "json")
        self.assertEqual(code, 5)
        self.assertEqual(result["report"]["live_gpu_probe"]["status"], "missing_selected")
        self.assertTrue(any(i["code"] == "GPU_NOT_VISIBLE" for i in result["report"]["issues"]))

    def test_live_probe_failure_and_unsupported_hip_are_explicit(self):
        config = json.loads(self.config.read_text(encoding="utf-8"))
        config["gpu"] = [0, 1]
        config["backend"] = "cuda"
        self.config.write_text(json.dumps(config), encoding="utf-8")
        profiles.create("local", str(self.config))
        with mock.patch("strata_control.doctor.shutil.which", return_value=None):
            code, result = self.command("doctor", "--profile", "local", "--live", "--output", "json")
        self.assertEqual(code, 5)
        self.assertEqual(result["report"]["live_gpu_probe"]["status"], "unavailable")
        self.assertTrue(any(i["code"] == "LIVE_GPU_PROBE_UNAVAILABLE" for i in result["report"]["issues"]))

        config["backend"] = "hip"
        self.config.write_text(json.dumps(config), encoding="utf-8")
        with mock.patch("strata_control.doctor.subprocess.run",
                        side_effect=AssertionError("NVIDIA probe must not run for HIP")):
            code, result = self.command("doctor", "--profile", "local", "--live", "--output", "json")
        self.assertEqual(code, 5)
        self.assertEqual(result["report"]["live_gpu_probe"]["status"], "unverified")

    def test_doctor_default_does_not_run_host_probe(self):
        profiles.create("local", str(self.config))
        with mock.patch("strata_control.doctor.subprocess.run",
                        side_effect=AssertionError("default doctor must be offline")):
            code, result = self.command("doctor", "--profile", "local", "--output", "json")
        self.assertEqual(code, 0)
        self.assertIsNone(result["report"]["live_gpu_probe"])
        with mock.patch("strata_control.doctor.subprocess.run",
                        side_effect=AssertionError("a profile is required for live probes")):
            code, missing = self.command("doctor", "--live", "--output", "json")
        self.assertEqual(code, 5)
        self.assertTrue(any(i["code"] == "PROFILE_NOT_SELECTED_FOR_LIVE_PROBE"
                            for i in missing["report"]["issues"]))

    def test_live_probe_rejects_untrusted_card_output_and_timeouts(self):
        config = json.loads(self.config.read_text(encoding="utf-8"))
        config["gpu"] = 0
        config["backend"] = "cuda"
        self.config.write_text(json.dumps(config), encoding="utf-8")
        profiles.create("local", str(self.config))
        for output in ("0, GPU\x1b[31m, GPU-0, 1000, 580.95",
                       "0, GPU, GPU-0, 1000, 580.95\n0, GPU, GPU-0, 1000, 580.95",
                       "0, GPU, GPU-0, not-a-number, 580.95"):
            with self.subTest(output=output), \
                    mock.patch("strata_control.doctor.shutil.which", return_value="/fake/nvidia-smi"), \
                    mock.patch("strata_control.doctor.subprocess.run",
                               return_value=mock.Mock(returncode=0, stdout=output)):
                code, result = self.command("doctor", "--profile", "local", "--live", "--output", "json")
            self.assertEqual(code, 5)
            self.assertEqual(result["report"]["live_gpu_probe"]["status"], "unavailable")
        with mock.patch("strata_control.doctor.shutil.which", return_value="/fake/nvidia-smi"), \
                mock.patch("strata_control.doctor.subprocess.run",
                           side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)):
            code, result = self.command("doctor", "--profile", "local", "--live", "--output", "json")
        self.assertEqual(code, 5)
        self.assertEqual(result["report"]["live_gpu_probe"]["status"], "unavailable")

    def test_calibration_preview_has_no_engine_or_config_side_effects(self):
        profiles.create("local", str(self.config))
        before = self.config.read_bytes()
        with mock.patch("serve.server.StrataEngine", side_effect=AssertionError("engine started")), \
                mock.patch("tools.calibrate.run", side_effect=AssertionError("measurement started")):
            code, result = self.command("engine", "calibrate", "--profile", "local", "--plan", "--output", "json")
        self.assertEqual(code, 0)
        self.assertFalse(result["plan"]["would_start_engine"])
        self.assertFalse(result["plan"]["would_write_config"])
        self.assertEqual(self.config.read_bytes(), before)

    def test_invalid_profile_names_and_types_do_not_escape_store(self):
        for name in ("../outside", "UPPER", "a/b", ".hidden"):
            with self.assertRaises(profiles.ProfileError):
                profiles.create(name, str(self.config))
        profiles.create("local", str(self.config))
        for assignment in ("parallel=0", "max_queue=99999", "queue_timeout=nan",
                           "queue_timeout=0", "offline=yes", "batch_telemetry=yes", "host=bad host"):
            with self.assertRaises(profiles.ProfileError):
                profiles.set_values("local", [assignment])
        profiles.set_values("local", ["queue_timeout=5"])
        with self.assertRaisesRegex(profiles.ProfileError, "requires max_queue"):
            from strata_cli.cli import _launch_plan
            _launch_plan("local")


if __name__ == "__main__":
    unittest.main()
