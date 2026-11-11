"""Saved read-only connections work in both source and portable terminal clients."""

import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from strata_cli.cli import main
from strata_cli import connections
from strata_cli.client import ClientError, Endpoint, server_url
from strata_cli.render import render_view


ROOT = Path(__file__).resolve().parents[2]


class ConnectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-connections-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "connections"
        env = mock.patch.dict(os.environ, {"STRATA_CONNECTION_DIR": str(self.directory)}, clear=False)
        env.start()
        self.addCleanup(env.stop)

    def _cli(self, *argv):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(list(argv))
        return code, json.loads(output.getvalue())

    def test_create_edit_precedence_and_private_storage(self):
        code, created = self._cli("connection", "create", "edge", "--url", "https://Example.COM:443",
                                  "--prefix", "/proxy/strata", "--credential-env", "STRATA_READ_KEY",
                                  "--output", "json")
        self.assertEqual(code, 0)
        self.assertEqual(created["connection"]["url"], "https://example.com:443")
        self.assertEqual(created["connection"]["prefix"], "/proxy/strata")
        saved = self.directory / "edge.json"
        if os.name == "posix":
            self.assertEqual(self.directory.stat().st_mode & 0o077, 0)
            self.assertEqual(saved.stat().st_mode & 0o077, 0)
        self.assertNotIn("test-secret", saved.read_text())
        revision = created["revision"]
        code, plan = self._cli("connection", "set", "edge", "prefix=/new", "--plan",
                                "--if-revision", revision, "--output", "json")
        self.assertEqual(code, 0)
        self.assertTrue(plan["would_save"])
        self.assertEqual(connections.load("edge")[1], revision)
        code, changed = self._cli("connection", "set", "edge", "prefix=/new",
                                   "--if-revision", revision, "--output", "json")
        self.assertEqual(code, 0)
        self.assertNotEqual(changed["revision"], revision)
        code, stale = self._cli("connection", "set", "edge", "prefix=/other",
                                 "--if-revision", revision, "--output", "json")
        self.assertEqual((code, stale["error"]["code"]), (6, 6))
        self.assertEqual(connections.load("edge")[0]["prefix"], "/new")
        with mock.patch.dict(os.environ, {"STRATA_CONNECTION": "edge", "STRATA_URL": "https://ignored.example",
                                       "STRATA_READ_KEY": "test-secret"}):
            code, effective = self._cli("config", "show", "--effective", "--output", "json")
            self.assertEqual(code, 0)
            self.assertEqual(effective["effective"]["url"], "https://example.com:443")
            self.assertEqual(effective["effective"]["prefix"], "/new")
            self.assertEqual(effective["effective"]["base_url"], "https://example.com:443/new")
            self.assertEqual(effective["effective"]["origins"]["url"], "connection")
            self.assertEqual(effective["effective"]["timeout_s"], 3.0)
            self.assertEqual(effective["effective"]["origins"]["timeout_s"], "default")
            self.assertTrue(effective["effective"]["credential_present"])
            self.assertNotIn("test-secret", json.dumps(effective))
            code, override = self._cli("config", "show", "--url", "https://other.example",
                                       "--credential-env", "OTHER_KEY", "--timeout", "4.5",
                                       "--output", "json")
            self.assertEqual(code, 0)
            self.assertEqual(override["effective"]["url"], "https://other.example")
            self.assertEqual(override["effective"]["prefix"], "")
            self.assertEqual(override["effective"]["credential_env"], "OTHER_KEY")
            self.assertEqual(override["effective"]["origins"]["url"], "flag")
            self.assertEqual(override["effective"]["origins"]["timeout_s"], "flag")
            self.assertEqual(override["effective"]["timeout_s"], 4.5)
            code, conflict = self._cli("config", "show", "--url", "https://other.example",
                                       "--connection", "edge", "--output", "json")
            self.assertEqual((code, conflict["error"]["code"]), (2, 2))

    def test_connection_and_config_json_aliases(self):
        code, listed = self._cli("connection", "list", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(listed["connections"], [])
        code, effective = self._cli("config", "show", "--url", "http://127.0.0.1:8095", "--json")
        self.assertEqual(code, 0)
        self.assertTrue(effective["ok"])
        self.assertEqual(effective["effective"]["url"], "http://127.0.0.1:8095")

    def test_interactive_monitor_rejects_json_output_before_connecting(self):
        for output_flag in (("--output", "json"), ("--json",)):
            with self.subTest(output_flag=output_flag), mock.patch(
                    "strata_cli.monitor_commands.Endpoint") as endpoint:
                code, response = self._cli("monitor", "--url", "http://127.0.0.1:1",
                                           *output_flag)
                self.assertEqual((code, response["error"]["code"]), (2, 2))
                self.assertIn("requires --once", response["error"]["message"])
                endpoint.assert_not_called()
        bundle = Path(self.temp.name) / "strata-monitor.pyz"
        subprocess.run([sys.executable, str(ROOT / "packaging/monitor_bundle.py"),
                        "build", str(bundle)], capture_output=True, text=True, check=True)
        portable = subprocess.run([sys.executable, "-S", str(bundle), "monitor", "--url",
                                   "http://127.0.0.1:1", "--output", "json"],
                                  cwd=self.temp.name, capture_output=True, text=True)
        self.assertEqual(portable.returncode, 2)
        self.assertIn("requires --once", json.loads(portable.stdout)["error"]["message"])
        self.assertEqual(portable.stderr, "")

    def test_quiet_connection_and_status_keep_machine_records(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["connection", "list", "--quiet"]), 0)
        self.assertEqual(output.getvalue(), "")
        code, listed = self._cli("connection", "list", "--quiet", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(listed["connections"], [])
        with mock.patch("strata_cli.monitor_commands.Endpoint") as endpoint:
            endpoint.return_value.snapshot.return_value = {"schema_version": 1}
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["status", "--url", "http://127.0.0.1:8095", "--quiet"]), 0)
            self.assertEqual(output.getvalue(), "")
            with redirect_stdout(output):
                self.assertEqual(main(["status", "--url", "http://127.0.0.1:8095",
                                       "--quiet", "--json"]), 0)
            self.assertTrue(json.loads(output.getvalue())["ok"])

    def test_invalid_prefix_credentials_and_file_permissions_fail_closed(self):
        for prefix in ("//proxy", "/a/../b", "/a?key=secret", "/a//b", "https://host/path"):
            with self.assertRaises(connections.ConnectionError):
                connections.prefix_path(prefix)
        with self.assertRaises(ValueError):
            server_url("http://@127.0.0.1:8095")
        with self.assertRaises(connections.ConnectionError):
            connections.credential_name("BAD-NAME")
        connections.create("local", "http://127.0.0.1:8095")
        with self.assertRaises(connections.ConnectionError):
            connections.create("local", "http://127.0.0.1:8095")
        if os.name == "posix":
            record = self.directory / "local.json"
            record.chmod(0o644)                          # our own file with drifted modes is put back, not refused
            connections.load("local")
            self.assertEqual(stat.S_IMODE(record.stat().st_mode), 0o600)
            with mock.patch("os.getuid", return_value=os.getuid() + 1):      # a file that belongs to another account is refused
                with self.assertRaisesRegex(connections.ConnectionError, "belongs to .*sudo chown"):
                    connections.load("local")
        record = self.directory / "local.json"
        corrupted = json.loads(record.read_text())
        corrupted["url"] = 42
        record.write_text(json.dumps(corrupted))
        code, response = self._cli("config", "show", "--connection", "local", "--output", "json")
        self.assertEqual((code, response["error"]["code"]), (5, 5))

    def test_proxy_prefix_and_credential_reference_work_in_portable_bundle(self):
        seen = []
        snapshot = {"schema_version": 1, "instance": {"id": "from-proxy"},
                    "model": {"id": "fixture", "loaded": True}, "work": {"active": 3}}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append((self.path, self.headers.get("Authorization")))
                if self.path != "/proxy/v1/telemetry":
                    self.send_error(404)
                    return
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
            root = f"http://127.0.0.1:{server.server_port}"
            connections.create("proxied", root, prefix="/proxy", credential_env="STRATA_READ_KEY")
            with mock.patch.dict(os.environ, {"STRATA_READ_KEY": "test-secret"}):
                code, source = self._cli("status", "--connection", "proxied", "--json")
                self.assertEqual(code, 0)
                self.assertEqual(source["snapshot"]["work"]["active"], 3)
                bundle = Path(self.temp.name) / "strata-monitor.pyz"
                subprocess.run([sys.executable, str(ROOT / "packaging/monitor_bundle.py"),
                                "build", str(bundle)], capture_output=True, text=True, check=True)
                portable = subprocess.run([sys.executable, "-S", str(bundle), "status", "--connection",
                                           "proxied", "--json"], env=os.environ.copy(), cwd=self.temp.name,
                                          capture_output=True, text=True, check=True)
                self.assertEqual(json.loads(portable.stdout), source)
                portable_config = subprocess.run([sys.executable, "-S", str(bundle), "config", "show",
                                                  "--connection", "proxied", "--output", "json"],
                                                 env=os.environ.copy(), cwd=self.temp.name,
                                                 capture_output=True, text=True, check=True)
                self.assertEqual(json.loads(portable_config.stdout)["effective"]["prefix"], "/proxy")
                self.assertNotIn("test-secret", portable_config.stdout)
                subprocess.run([sys.executable, "-S", str(bundle), "connection", "create", "second",
                                "--url", root, "--output", "json"], env=os.environ.copy(),
                               cwd=self.temp.name, capture_output=True, text=True, check=True)
                self.assertEqual(connections.load("second")[0]["url"], root)
            self.assertEqual(seen, [("/proxy/v1/telemetry", "Bearer test-secret")] * 2)
            self.assertEqual(Endpoint(root, prefix="/proxy").base_url, root + "/proxy")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_deeply_nested_telemetry_returns_one_redacted_error(self):
        depth = 20000  # Python 3.13 parses 1100 levels; the recursion limit is reached at a few thousand on every version
        body = b"[" * depth + b"0" + b"]" * depth

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
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
            root = f"http://127.0.0.1:{server.server_port}"
            with self.assertRaisesRegex(ClientError, "malformed JSON telemetry"):
                Endpoint(root).snapshot()
            code, response = self._cli("status", "--url", root, "--json")
            self.assertEqual(code, 3)
            self.assertEqual(response["error"]["message"], "server returned malformed JSON telemetry")
            bundle = Path(self.temp.name) / "deep-json-monitor.pyz"
            subprocess.run([sys.executable, str(ROOT / "packaging/monitor_bundle.py"),
                            "build", str(bundle)], capture_output=True, text=True, check=True)
            portable = subprocess.run([sys.executable, "-I", "-S", str(bundle), "status",
                                       "--url", root, "--json"], cwd=self.temp.name,
                                      capture_output=True, text=True, check=False)
            self.assertEqual(portable.returncode, 3)
            self.assertFalse(portable.stderr)
            self.assertEqual(json.loads(portable.stdout)["error"]["message"],
                             "server returned malformed JSON telemetry")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_status_only_fallback_labels_missing_telemetry_without_inventing_work(self):
        paths = []
        metrics_status = [404]
        status = {"service": "strata", "model": "loaded-model", "loaded": True,
                  "started": 1_790_000_000, "concurrency": {"serving": 8}}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                paths.append(self.path)
                if self.path == "/metrics" and metrics_status[0] == 401:
                    self.send_error(401)
                    return
                if self.path != "/v1/status":
                    self.send_error(404)
                    return
                body = json.dumps(status).encode()
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
            root = f"http://127.0.0.1:{server.server_port}"
            snapshot = Endpoint(root).snapshot()
            self.assertEqual(paths, ["/v1/telemetry", "/metrics", "/v1/status"])
            self.assertEqual(snapshot["source"], "status-only")
            self.assertEqual(snapshot["quality"]["telemetry"], "unavailable")
            self.assertEqual(snapshot["model"], {"id": "loaded-model", "loaded": True, "context": None})
            self.assertEqual(snapshot["work"]["slots_effective"], 8)
            self.assertIsNone(snapshot["work"]["active"])
            self.assertIsNone(snapshot["rates"]["answer_tok_s"])
            display = render_view(snapshot, "overview", 80, 24)
            self.assertIn("[NO TELEMETRY]", display)
            self.assertIn("Telemetry unavailable; status only", display)
            self.assertIn("Sample age", render_view(snapshot, "overview", 40, 12))
            self.assertIn("In flight --", display)
            code, source = self._cli("status", "--url", root, "--json")
            self.assertEqual(code, 0)
            self.assertEqual(source["snapshot"]["quality"]["telemetry"], "unavailable")
            bundle = Path(self.temp.name) / "status-only.pyz"
            subprocess.run([sys.executable, str(ROOT / "packaging/monitor_bundle.py"),
                            "build", str(bundle)], capture_output=True, text=True, check=True)
            portable = subprocess.run([sys.executable, "-S", str(bundle), "status", "--url", root,
                                       "--json"], capture_output=True, text=True, check=True)
            portable_snapshot = json.loads(portable.stdout)["snapshot"]
            self.assertEqual(portable_snapshot["source"], "status-only")
            self.assertIsNone(portable_snapshot["work"]["active"])
            metrics_status[0] = 401
            paths.clear()
            with self.assertRaises(ClientError) as denied:
                Endpoint(root).snapshot()
            self.assertEqual(denied.exception.exit_code, 4)
            self.assertEqual(paths, ["/v1/telemetry", "/metrics"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
