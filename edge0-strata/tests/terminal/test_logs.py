"""Read-only activity CLI contract, including retention gaps and portable parity."""

import io
import json
import os
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
from strata_cli.client import sanitize_telemetry
from strata_cli.logs import ActivityContinuity, activity_delta


ROOT = Path(__file__).resolve().parents[2]


def sample(instance, last, seqs):
    raw = {"schema_version": 1, "instance": {"id": instance, "state": "ready"},
           "activity": {"last_seq": last, "events": [
               {"seq": seq, "at": 1_790_000_000.0, "request_id": f"{seq:012x}",
                "event": "queued", "prompt": "private"} for seq in seqs]}}
    return sanitize_telemetry(raw)


class LogsTests(unittest.TestCase):
    def test_monitor_continuity_reports_only_observed_gaps_and_resets_on_restart(self):
        tracker = ActivityContinuity()
        tracker.add(sample("instance-a", 35, [33, 34, 35]))
        self.assertEqual(tracker.status(), {"gap": None, "restarted": False, "regressed": False})
        tracker.add(sample("instance-a", 38, [34, 37, 38]))
        self.assertEqual(tracker.status()["gap"], {"from_seq": 36, "to_seq": 36})
        tracker.add(sample("instance-a", 39, [38, 39]))
        self.assertEqual(tracker.status()["gap"], {"from_seq": 36, "to_seq": 36})
        tracker.add(sample("instance-b", 2, [1, 2]))
        self.assertEqual(tracker.status(), {"gap": None, "restarted": True, "regressed": False})
        tracker.add(sample("instance-b", 40, [38, 39, 40]))
        self.assertEqual(tracker.status()["gap"], {"from_seq": 3, "to_seq": 37})
        tracker.add(sample("instance-b", 39, [38, 39]))
        self.assertTrue(tracker.status()["regressed"])

    def test_quiet_logs_keep_activity_rows_and_json(self):
        endpoint = mock.Mock()
        endpoint.snapshot.return_value = sample("quiet-instance", 1, [1])
        output = io.StringIO()
        with mock.patch("strata_cli.monitor_commands.Endpoint", return_value=endpoint), \
             redirect_stdout(output):
            self.assertEqual(main(["logs", "--quiet"]), 0)
        self.assertNotIn("Recent request activity", output.getvalue())
        self.assertIn("queued", output.getvalue())
        output = io.StringIO()
        with mock.patch("strata_cli.monitor_commands.Endpoint", return_value=endpoint), \
             redirect_stdout(output):
            self.assertEqual(main(["logs", "--quiet", "--json"]), 0)
        self.assertEqual(json.loads(output.getvalue())["events"][0]["event"], "queued")

    def test_delta_reports_missing_sequences_and_restart_without_duplicates(self):
        first = sample("instance-a", 35, [33, 34, 35])
        rows, instance, cursor = activity_delta(first, None, None)
        self.assertEqual([(r["type"], r.get("from_seq"), r.get("to_seq")) for r in rows],
                         [("gap", 1, 32), ("activity", None, None),
                          ("activity", None, None), ("activity", None, None)])
        self.assertEqual(activity_delta(first, instance, cursor)[0], [])
        rows, instance, cursor = activity_delta(sample("instance-a", 39, [35, 38, 39]), instance, cursor)
        self.assertEqual([(r["type"], r.get("from_seq"), r.get("to_seq")) for r in rows],
                         [("gap", 36, 37), ("activity", None, None), ("activity", None, None)])
        rows, instance, cursor = activity_delta(sample("instance-b", 2, [1, 2]), instance, cursor)
        self.assertEqual([r["type"] for r in rows], ["instance_change", "activity", "activity"])
        self.assertEqual((instance, cursor), ("instance-b", 2))

    def test_follow_jsonl_deduplicates_and_reports_gaps(self):
        snapshots = [sample("a", 2, [1, 2]), sample("a", 2, [1, 2]),
                     sample("a", 5, [2, 4, 5]), sample("b", 1, [1])]
        endpoint = mock.Mock()
        endpoint.snapshot.side_effect = snapshots
        stdout = io.StringIO()
        with mock.patch("strata_cli.monitor_commands.Endpoint", return_value=endpoint), \
             mock.patch("strata_cli.logs.time.sleep", side_effect=[None, None, None, KeyboardInterrupt]), \
             redirect_stdout(stdout):
            self.assertEqual(main(["logs", "--follow", "--output", "jsonl"]), 0)
        records = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual([r["type"] for r in records],
                         ["activity", "activity", "gap", "activity", "activity", "instance_change", "activity"])
        self.assertEqual((records[2]["from_seq"], records[2]["to_seq"]), (3, 3))
        self.assertTrue(all(r["schema_version"] == 1 for r in records))
        self.assertNotIn("private", stdout.getvalue())

    def test_unsupported_sources_modes_and_absent_activity(self):
        for args, expected in ((["logs", "--source", "harness", "--output", "json"], 5),
                               (["logs", "--follow", "--output", "json"], 2),
                               (["logs", "--follow", "--json"], 2),
                               (["logs", "--output", "jsonl"], 2),
                               (["logs", "--interval", "0.1", "--output", "json"], 2)):
            with self.subTest(args=args), redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(main(args), expected)
                self.assertEqual(json.loads(stream.getvalue())["error"]["code"], expected)
        endpoint = mock.Mock()
        endpoint.snapshot.return_value = sanitize_telemetry({"schema_version": 1, "instance": {"id": "old"}})
        with mock.patch("strata_cli.monitor_commands.Endpoint", return_value=endpoint), \
             redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(main(["logs", "--output", "json"]), 5)
            self.assertIn("unavailable", json.loads(stream.getvalue())["error"]["message"])

    def test_one_shot_source_and_portable_use_read_only_key(self):
        requests = []
        raw = {"schema_version": 1, "instance": {"id": "fixture"},
               "activity": {"last_seq": 1, "events": [
                   {"seq": 1, "at": 1_790_000_000, "request_id": "000000000001",
                    "event": "finished", "outcome": "stop", "prompt": "private"}]}}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append((self.path, self.headers.get("Authorization")))
                if self.headers.get("Authorization") != "Bearer read-key":
                    self.send_error(403)
                    return
                body = json.dumps(raw).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            with tempfile.TemporaryDirectory() as directory:
                bundle = Path(directory) / "strata-monitor.pyz"
                subprocess.run([sys.executable, str(ROOT / "packaging/monitor_bundle.py"), "build", str(bundle)],
                               check=True, capture_output=True, text=True)
                clean_env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
                clean_env["STRATA_TELEMETRY_KEY"] = "read-key"
                with mock.patch.dict(os.environ, {"STRATA_TELEMETRY_KEY": "read-key"}), \
                     redirect_stdout(io.StringIO()) as source:
                    self.assertEqual(main(["logs", "--url", url, "--credential-env",
                                           "STRATA_TELEMETRY_KEY", "--json"]), 0)
                portable = subprocess.run([sys.executable, "-S", str(bundle), "logs", "--url", url,
                                           "--credential-env", "STRATA_TELEMETRY_KEY", "--output", "json"],
                                          cwd=directory, env=clean_env, check=True, capture_output=True, text=True)
                self.assertEqual(json.loads(source.getvalue()), json.loads(portable.stdout))
                self.assertEqual(json.loads(portable.stdout)["events"][0]["event"], "finished")
                self.assertNotIn("private", portable.stdout)
                self.assertEqual(requests, [("/v1/telemetry", "Bearer read-key")] * 2)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)
