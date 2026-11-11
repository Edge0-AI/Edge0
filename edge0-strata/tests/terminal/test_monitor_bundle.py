"""Packaged client smoke test without site packages, engine files or model weights."""

import io
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from strata_cli.cli import main as source_main


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "packaging/monitor_bundle.py"
SPEC = importlib.util.spec_from_file_location("strata_monitor_bundle_builder", BUILDER)
assert SPEC is not None and SPEC.loader is not None
BUILDER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER_MODULE)
build = BUILDER_MODULE.build
install = BUILDER_MODULE.install
uninstall = BUILDER_MODULE.uninstall
update = BUILDER_MODULE.update
rollback = BUILDER_MODULE.rollback
commit = BUILDER_MODULE.commit
verify = BUILDER_MODULE.verify


class MonitorBundleTests(unittest.TestCase):
    @staticmethod
    def _candidate(source: Path, destination: Path) -> Path:
        destination.parent.mkdir()
        destination.write_bytes(source.read_bytes())
        with zipfile.ZipFile(destination, "a") as archive:
            archive.comment = b"candidate client revision"
        report = json.loads(source.with_name(source.name + ".manifest.json").read_text())
        report["bundle"] = destination.name
        report["sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        destination.with_name(destination.name + ".manifest.json").write_text(json.dumps(report))
        verify(destination)
        return destination

    def test_crlf_license_checkout_still_verifies(self):
        """A Windows checkout (autocrlf) ships LICENSE with CRLF; the bundle must still verify."""
        real_copy = BUILDER_MODULE.shutil.copyfile

        def crlf_license(src, dst, *args, **kwargs):
            if Path(src).name != "LICENSE":
                return real_copy(src, dst, *args, **kwargs)
            text = Path(src).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            Path(dst).write_bytes(text)
            return dst

        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "monitor.pyz"
            with mock.patch.object(BUILDER_MODULE.shutil, "copyfile", crlf_license):
                build(bundle)
            with zipfile.ZipFile(bundle) as archive:
                self.assertIn(b"MIT License\r\n", archive.read("LICENSE"))
            self.assertEqual(verify(bundle)["schema_version"], 6)

    def test_update_rollback_and_commit_keep_one_verified_previous_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first" / "monitor.pyz"
            first.parent.mkdir()
            old = build(first)
            candidate = self._candidate(first, root / "second" / "next-monitor.pyz")
            next_sha = verify(candidate)["sha256"]
            target = root / "optional-monitor"
            install(first, target, if_candidate_sha256=old["sha256"])
            preview_run = subprocess.run([sys.executable, str(BUILDER), "update", str(candidate),
                                          "--target", str(target), "--plan"],
                                         capture_output=True, text=True, check=True)
            preview = json.loads(preview_run.stdout)
            self.assertEqual((preview["current_sha256"], preview["candidate_sha256"]),
                             (old["sha256"], next_sha))
            self.assertTrue(preview["would_update"])
            with self.assertRaisesRegex(ValueError, "preview again"):
                update(candidate, target, if_current_sha256=old["sha256"],
                       if_candidate_sha256="0" * 64)
            self.assertFalse((root / "optional-monitor.previous").exists())
            changed = subprocess.run([sys.executable, str(BUILDER), "update", str(candidate),
                                      "--target", str(target), "--if-current-sha256", old["sha256"],
                                      "--if-candidate-sha256", next_sha],
                                     capture_output=True, text=True, check=True)
            result = json.loads(changed.stdout)
            self.assertTrue(result["updated"])
            previous = root / "optional-monitor.previous"
            self.assertEqual(verify(target / "next-monitor.pyz")["sha256"], next_sha)
            self.assertEqual(verify(previous / "monitor.pyz")["sha256"], old["sha256"])
            with self.assertRaisesRegex(FileExistsError, "previous"):
                update(candidate, target, plan=True)
            preview_run = subprocess.run([sys.executable, str(BUILDER), "rollback", "--target",
                                          str(target), "--plan"], capture_output=True, text=True, check=True)
            preview = json.loads(preview_run.stdout)
            self.assertTrue(preview["would_rollback"])
            with self.assertRaisesRegex(ValueError, "preview again"):
                rollback(target, if_active_sha256=old["sha256"], if_rollback_sha256=old["sha256"])
            original = os.rename
            calls = 0

            def fail_last_exchange(source, destination):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise OSError("final exchange failed")
                return original(source, destination)

            with mock.patch.object(BUILDER_MODULE.os, "rename", side_effect=fail_last_exchange):
                with self.assertRaisesRegex(OSError, "final exchange failed"):
                    rollback(target, if_active_sha256=next_sha, if_rollback_sha256=old["sha256"])
            self.assertEqual(verify(target / "next-monitor.pyz")["sha256"], next_sha)
            self.assertEqual(verify(previous / "monitor.pyz")["sha256"], old["sha256"])
            self.assertFalse(list(root.glob(".strata-monitor-swap-*")))
            restored = subprocess.run([sys.executable, str(BUILDER), "rollback", "--target", str(target),
                                       "--if-active-sha256", next_sha,
                                       "--if-rollback-sha256", old["sha256"]],
                                      capture_output=True, text=True, check=True)
            result = json.loads(restored.stdout)
            self.assertTrue(result["rolled_back"])
            self.assertEqual(verify(target / "monitor.pyz")["sha256"], old["sha256"])
            self.assertEqual(verify(previous / "next-monitor.pyz")["sha256"], next_sha)
            with self.assertRaisesRegex(ValueError, "commit or rollback"):
                uninstall(target)
            planned = commit(target, plan=True)
            self.assertTrue(planned["would_commit"])
            accepted = subprocess.run([sys.executable, str(BUILDER), "commit", "--target", str(target),
                                       "--if-active-sha256", old["sha256"],
                                       "--if-previous-sha256", next_sha],
                                      capture_output=True, text=True, check=True)
            self.assertTrue(json.loads(accepted.stdout)["committed"])
            self.assertFalse(previous.exists())
            self.assertTrue(uninstall(target, if_active_sha256=old["sha256"])["removed"])

    def test_update_activation_failure_restores_installed_client(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first" / "monitor.pyz"
            first.parent.mkdir()
            old = build(first)
            candidate = self._candidate(first, root / "second" / "monitor.pyz")
            next_sha = verify(candidate)["sha256"]
            target = root / "optional-monitor"
            install(first, target, if_candidate_sha256=old["sha256"])
            original = os.rename
            calls = 0

            def fail_activation(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("activation failed")
                return original(source, destination)

            with mock.patch.object(BUILDER_MODULE.os, "rename", side_effect=fail_activation):
                with self.assertRaisesRegex(OSError, "activation failed"):
                    update(candidate, target, if_current_sha256=old["sha256"],
                           if_candidate_sha256=next_sha)
            self.assertEqual(verify(target / "monitor.pyz")["sha256"], old["sha256"])
            self.assertFalse((root / "optional-monitor.previous").exists())
            self.assertFalse(list(root.glob(".strata-monitor-install-*")))

            original_stage = BUILDER_MODULE._stage_copy

            def change_active_during_copy(bundle, report, destination):
                stage = original_stage(bundle, report, destination)
                active_bundle = destination / "monitor.pyz"
                with zipfile.ZipFile(active_bundle, "a") as archive:
                    archive.comment = b"external owner edit"
                manifest = active_bundle.with_name(active_bundle.name + ".manifest.json")
                current = json.loads(manifest.read_text())
                current["sha256"] = hashlib.sha256(active_bundle.read_bytes()).hexdigest()
                manifest.write_text(json.dumps(current))
                return stage

            with mock.patch.object(BUILDER_MODULE, "_stage_copy", side_effect=change_active_during_copy):
                with self.assertRaisesRegex(ValueError, "changed during copy"):
                    update(candidate, target, if_current_sha256=old["sha256"],
                           if_candidate_sha256=next_sha)
            self.assertNotEqual(verify(target / "monitor.pyz")["sha256"], old["sha256"])
            self.assertFalse((root / "optional-monitor.previous").exists())
            self.assertFalse(list(root.glob(".strata-monitor-install-*")))

    def test_offline_install_plan_and_remove_touch_only_monitor_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "portable-client.pyz"
            built = build(bundle)
            target = root / "optional-monitor"
            preview = subprocess.run([sys.executable, str(BUILDER), "install", str(bundle),
                                      "--target", str(target), "--plan"],
                                     capture_output=True, text=True, check=True)
            planned = json.loads(preview.stdout)
            self.assertTrue(planned["would_install"])
            self.assertFalse(target.exists())
            with self.assertRaisesRegex(ValueError, "preview again"):
                install(bundle, target)
            with self.assertRaisesRegex(ValueError, "preview again"):
                install(bundle, target, if_candidate_sha256="0" * 64)
            self.assertFalse(target.exists())
            installed_run = subprocess.run([sys.executable, str(BUILDER), "install", str(bundle),
                                            "--target", str(target),
                                            "--if-candidate-sha256", planned["sha256"]],
                                           capture_output=True, text=True, check=True)
            installed = json.loads(installed_run.stdout)
            self.assertTrue(installed["installed"])
            self.assertEqual(verify(target / bundle.name)["sha256"], built["sha256"])
            self.assertEqual({p.name for p in target.iterdir()},
                             {bundle.name, bundle.name + ".manifest.json"})
            if os.name == "posix":
                self.assertEqual(target.stat().st_mode & 0o077, 0)
                self.assertEqual((target / bundle.name).stat().st_mode & 0o077, 0)
            portable = subprocess.run([sys.executable, "-I", str(target / bundle.name), "logs", "--help"],
                                      cwd=directory, capture_output=True, text=True, check=True)
            self.assertIn("--follow", portable.stdout)
            extra = target / "user-notes.txt"
            extra.write_text("keep me")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                uninstall(target)
            self.assertTrue(extra.exists())
            extra.unlink()
            preview_remove = subprocess.run([sys.executable, str(BUILDER), "uninstall",
                                             "--target", str(target), "--plan"],
                                            capture_output=True, text=True, check=True)
            self.assertTrue(json.loads(preview_remove.stdout)["would_remove"])
            self.assertTrue(target.exists())
            with self.assertRaisesRegex(ValueError, "preview again"):
                uninstall(target)
            with self.assertRaisesRegex(ValueError, "preview again"):
                uninstall(target, if_active_sha256="0" * 64)
            self.assertTrue(target.exists())
            removal = subprocess.run([sys.executable, str(BUILDER), "uninstall", "--target",
                                      str(target), "--if-active-sha256",
                                      json.loads(preview_remove.stdout)["sha256"]],
                                     capture_output=True, text=True, check=True)
            self.assertTrue(json.loads(removal.stdout)["removed"])
            self.assertFalse(target.exists())

    def test_failed_copy_leaves_no_partial_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "monitor.pyz"
            built = build(bundle)
            target = root / "optional-monitor"
            with mock.patch.object(BUILDER_MODULE.shutil, "copyfile", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    install(bundle, target, if_candidate_sha256=built["sha256"])
            self.assertFalse(target.exists())
            self.assertFalse(list(root.glob(".strata-monitor-install-*")))

    def test_bundle_verification_rejects_large_or_linked_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "monitor.pyz"
            build(bundle)
            linked = root / "linked.pyz"
            linked.symlink_to(bundle)
            with self.assertRaisesRegex(ValueError, "bounded regular"):
                verify(linked)
            with bundle.open("ab") as stream:
                stream.write(b"x" * (1024 * 1024))
            with self.assertRaisesRegex(ValueError, "bounded regular"):
                verify(bundle)
            bundle = root / "unsigned.pyz"
            build(bundle)
            manifest = bundle.with_name(bundle.name + ".manifest.json")
            forged = json.loads(manifest.read_text())
            forged["publisher_signature"] = "pretend-signed"
            manifest.write_text(json.dumps(forged))
            with self.assertRaisesRegex(ValueError, "manifest or SHA-256"):
                verify(bundle)
            manifest.write_text("[]")
            with self.assertRaisesRegex(ValueError, "manifest or SHA-256"):
                verify(bundle)

    def test_bundle_carries_every_module_its_entry_point_imports(self):
        """The import closure of strata_cli.monitor_commands (lazy imports included) minus the launch package must be in the bundle."""
        import ast
        needed, todo = set(), ["strata_cli/monitor_commands.py"]
        while todo:
            name = todo.pop()
            if name in needed:
                continue
            needed.add(name)
            for node in ast.walk(ast.parse((ROOT / name).read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "strata_cli":
                    names = [node.module] + [node.module + "." + a.name for a in node.names]
                elif isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                else:
                    continue
                for module in names:
                    path = module.replace(".", "/") + ".py"
                    if module.startswith("strata_cli") and (ROOT / path).is_file() and "/launch/" not in path:
                        todo.append(path)
        self.assertLessEqual(needed, set(BUILDER_MODULE.SOURCES))

    def test_plain_monitor_runs_when_the_launch_package_is_not_bundled(self):
        """The pyz has no strata_cli.launch: interactive `monitor` must fall back to the plain Monitor, not fail."""
        from strata_cli import monitor as monitor_module
        from strata_cli import monitor_commands
        ran = []
        blocked = {"strata_cli.launch.monitor_ext": None}       # None in sys.modules makes the import raise ImportError
        with mock.patch.dict(sys.modules, blocked),                 mock.patch.object(monitor_module.MonitorApp, "run", lambda self: ran.append(self.endpoint.base_url)),                 mock.patch.object(sys.stdout, "isatty", return_value=True):
            code = monitor_commands.main(["monitor", "--url", "http://127.0.0.1:9"])
        self.assertEqual((code, ran), (0, ["http://127.0.0.1:9"]))

    def test_versioned_manifest_keeps_legacy_update_and_rollback_available(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "monitor.pyz"
            modern = root / "next-monitor.pyz"
            previous = root / "previous-monitor.pyz"
            build(modern)
            old_visual = root / "visual-monitor.pyz"
            for old_bundle, version in ((current, 1), (previous, 2), (old_visual, 3)):
                with zipfile.ZipFile(modern) as source, zipfile.ZipFile(old_bundle, "w", zipfile.ZIP_DEFLATED) as legacy:
                    for item in source.infolist():
                        if item.filename not in ("strata_cli/detail_dashboard.py", "strata_cli/nav.py",
                                                 "strata_cli/consoleinfo.py", "strata_cli/overview.py") and \
                                (version >= 3 or item.filename != "strata_cli/dashboard.py") and \
                                (version != 1 or item.filename != "LICENSE"):
                            legacy.writestr(item, source.read(item.filename))
                report = json.loads((root / "next-monitor.pyz.manifest.json").read_text())
                report["schema_version"] = version
                report["bundle"] = old_bundle.name
                report["sha256"] = hashlib.sha256(old_bundle.read_bytes()).hexdigest()
                del report["files"]["strata_cli/detail_dashboard.py"]
                del report["files"]["strata_cli/nav.py"]
                del report["files"]["strata_cli/consoleinfo.py"]
                del report["files"]["strata_cli/overview.py"]
                if version < 3:
                    del report["files"]["strata_cli/dashboard.py"]
                if version == 1:
                    del report["files"]["LICENSE"]
                    for field in ("compatibility", "license", "license_file"):
                        del report[field]
                old_bundle.with_name(old_bundle.name + ".manifest.json").write_text(json.dumps(report))
            self.assertEqual(verify(current)["schema_version"], 1)
            self.assertEqual(verify(previous)["schema_version"], 2)
            self.assertEqual(verify(old_visual)["schema_version"], 3)
            self.assertEqual(verify(modern)["schema_version"], 6)
            with zipfile.ZipFile(modern) as archive:
                self.assertIn(b"MIT License", archive.read("LICENSE"))
                self.assertIn(b"render_dashboard", archive.read("strata_cli/dashboard.py"))
                self.assertIn(b"render_detail_dashboard", archive.read("strata_cli/detail_dashboard.py"))
            target = root / "optional-monitor"
            install(current, target, if_candidate_sha256=verify(current)["sha256"])
            preview = update(modern, target, plan=True)
            self.assertTrue(preview["would_update"])
            self.assertTrue(update(modern, target, if_current_sha256=preview["current_sha256"],
                                   if_candidate_sha256=preview["candidate_sha256"])["updated"])
            self.assertEqual(verify(target / modern.name)["schema_version"], 6)
            self.assertEqual(verify(root / "optional-monitor.previous" / current.name)["schema_version"], 1)
            planned = rollback(target, plan=True)
            self.assertTrue(rollback(target, if_active_sha256=planned["active_sha256"],
                                     if_rollback_sha256=planned["rollback_sha256"])["rolled_back"])
            self.assertEqual(verify(target / current.name)["schema_version"], 1)
            second_target = root / "optional-monitor-from-v2"
            install(previous, second_target, if_candidate_sha256=verify(previous)["sha256"])
            preview = update(modern, second_target, plan=True)
            self.assertTrue(update(modern, second_target,
                                   if_current_sha256=preview["current_sha256"],
                                   if_candidate_sha256=preview["candidate_sha256"])["updated"])
            self.assertEqual(verify(second_target / modern.name)["schema_version"], 6)
            planned = rollback(second_target, plan=True)
            self.assertTrue(rollback(second_target, if_active_sha256=planned["active_sha256"],
                                     if_rollback_sha256=planned["rollback_sha256"])["rolled_back"])
            self.assertEqual(verify(second_target / previous.name)["schema_version"], 2)
            third_target = root / "optional-monitor-from-v3"
            install(old_visual, third_target, if_candidate_sha256=verify(old_visual)["sha256"])
            preview = update(modern, third_target, plan=True)
            self.assertTrue(update(modern, third_target,
                                   if_current_sha256=preview["current_sha256"],
                                   if_candidate_sha256=preview["candidate_sha256"])["updated"])
            planned = rollback(third_target, plan=True)
            self.assertTrue(rollback(third_target, if_active_sha256=planned["active_sha256"],
                                     if_rollback_sha256=planned["rollback_sha256"])["rolled_back"])
            self.assertEqual(verify(third_target / old_visual.name)["schema_version"], 3)

    def test_manifest_compatibility_and_license_notice_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "monitor.pyz"
            built = build(bundle)
            self.assertEqual(built["compatibility"], {"python_min": "3.10", "telemetry_schema": [1]})
            self.assertEqual(built["license_file"], "LICENSE")
            manifest = bundle.with_name(bundle.name + ".manifest.json")
            edited = json.loads(manifest.read_text())
            edited["compatibility"]["telemetry_schema"] = [2]
            manifest.write_text(json.dumps(edited))
            with self.assertRaisesRegex(ValueError, "compatibility or license"):
                verify(bundle)
            edited["schema_version"] = True
            manifest.write_text(json.dumps(edited))
            with self.assertRaisesRegex(ValueError, "manifest or SHA-256"):
                verify(bundle)

    def test_thin_bundle_and_source_share_one_shot_contract(self):
        snapshot = {"schema_version": 1, "source": "test-server", "sampled_at": 1_790_000_000,
                    "instance": {"id": "test-instance", "state": "ready"},
                    "model": {"id": "fixture-model", "loaded": True},
                    "work": {"active": 2, "queued": 1, "slots_effective": 4,
                             "requests": [{"id": "r1", "state": "reading", "slot": 0,
                                           "prompt_tokens": 200, "prompt_read": 50}]}}
        paths = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                paths.append(self.path)
                body = json.dumps(snapshot).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                bundle = Path(directory) / "strata-monitor.pyz"
                subprocess.run([sys.executable, str(BUILDER), "build", str(bundle)], check=True,
                               capture_output=True, text=True)
                subprocess.run([sys.executable, str(BUILDER), "verify", str(bundle)], check=True,
                               capture_output=True, text=True)
                self.assertLess(bundle.stat().st_size, 96 * 1024)
                with zipfile.ZipFile(bundle) as archive:
                    self.assertIn(b"MIT License", archive.read("LICENSE"))
                    self.assertIn("strata_cli/dashboard.py", archive.namelist())
                if importlib.util.find_spec("textual") is not None:
                    import_check = subprocess.run(
                        [sys.executable, "-I", "-c",
                         "import sys; sys.path.insert(0, " + repr(str(bundle)) + "); "
                         "from strata_cli.monitor import MonitorApp; print(MonitorApp.__name__)"],
                        cwd=directory, capture_output=True, text=True, check=True)
                    self.assertEqual(import_check.stdout.strip(), "MonitorApp")
                url = f"http://127.0.0.1:{server.server_port}"
                source_output = io.StringIO()
                with redirect_stdout(source_output):
                    self.assertEqual(source_main(["monitor", "--once", "--json", "--url", url]), 0)
                clean_env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
                clean_env["STRATA_CONNECTION_DIR"] = str(Path(directory) / "connections")
                portable = subprocess.run([sys.executable, "-S", str(bundle), "monitor", "--once", "--json",
                                           "--url", url], cwd=directory, env=clean_env, check=True,
                                          capture_output=True, text=True)
                self.assertEqual(json.loads(portable.stdout), json.loads(source_output.getvalue()))
                bare_source = subprocess.run([sys.executable, "-S", str(ROOT / "strata.py"), "monitor",
                                              "--once", "--json", "--url", url], cwd=directory, env=clean_env,
                                             check=True, capture_output=True, text=True)
                self.assertEqual(json.loads(bare_source.stdout), json.loads(portable.stdout))
                status = subprocess.run([sys.executable, "-S", str(bundle), "status", "--json", "--url", url],
                                        cwd=directory, env=clean_env, check=True, capture_output=True, text=True)
                self.assertEqual(json.loads(status.stdout)["snapshot"]["work"]["active"], 2)
                saved = subprocess.run([sys.executable, "-S", str(bundle), "connection", "list", "--json"],
                                       cwd=directory, env=clean_env, check=True, capture_output=True, text=True)
                self.assertEqual(json.loads(saved.stdout)["connections"], [])
                quiet_saved = subprocess.run([sys.executable, "-S", str(bundle), "connection", "list", "--quiet"],
                                             cwd=directory, env=clean_env, check=True,
                                             capture_output=True, text=True)
                self.assertEqual((quiet_saved.stdout, quiet_saved.stderr), ("", ""))
                quiet_json = subprocess.run([sys.executable, "-S", str(bundle), "connection", "list",
                                             "--quiet", "--json"], cwd=directory, env=clean_env,
                                            check=True, capture_output=True, text=True)
                self.assertEqual(json.loads(quiet_json.stdout)["connections"], [])
                self.assertEqual(paths, ["/v1/telemetry"] * 4)
                data = bytearray(bundle.read_bytes())
                data[-1] ^= 1
                bundle.write_bytes(data)
                broken = subprocess.run([sys.executable, str(BUILDER), "verify", str(bundle)],
                                        capture_output=True, text=True)
                self.assertEqual(broken.returncode, 1)
                self.assertIn("SHA-256 mismatch", broken.stderr)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
