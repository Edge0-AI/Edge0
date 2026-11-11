"""Ready-file lifecycle checked against a live mock Strata API process."""

import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from serve.ready import preflight, publish, remove_if_owned


class ReadyFileTests(unittest.TestCase):
    def test_owner_only_file_is_never_removed_for_another_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ready.json"
            self.assertEqual(preflight(str(path)), path)
            publish(path, {"schema_version": 1, "instance_id": "one", "pid": os.getpid()})
            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaisesRegex(ValueError, "already exists"):
                preflight(str(path))
            remove_if_owned(path, "other")
            self.assertTrue(path.exists())
            remove_if_owned(path, "one")
            self.assertFalse(path.exists())

    @unittest.skipIf(os.name == "nt", "Windows console stop needs a platform-specific integration runner")
    def test_mock_server_publishes_bound_port_and_cleans_on_sigterm(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ready.json"
            env = os.environ.copy()
            env.pop("STRATA_API_KEY", None)
            env.pop("STRATA_TELEMETRY_KEY", None)
            process = subprocess.Popen([sys.executable, "-m", "serve.server", "--engine", "mock", "--port", "0",
                                        "--ready-file", str(path)], cwd=root, env=env,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 10
                record = None
                while time.monotonic() < deadline and process.poll() is None:
                    try:
                        record = json.loads(path.read_text(encoding="utf-8"))
                        break
                    except (FileNotFoundError, json.JSONDecodeError):
                        time.sleep(0.05)
                self.assertIsNotNone(record, process.stderr.read().decode("utf-8", "replace") if process.poll() is not None else "")
                self.assertEqual(record["pid"], process.pid)
                self.assertEqual(record["api_state"], "ready")
                self.assertEqual(record["model_state"], "ready")
                self.assertGreater(record["bound_port"], 0)
                self.assertNotIn("key", json.dumps(record).lower())
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=10), 0)
                self.assertFalse(path.exists())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                process.stderr.close()


if __name__ == "__main__":
    unittest.main()
