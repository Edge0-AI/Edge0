"""An owned server attests the bytes and arguments it actually received."""

import json
import unittest
from unittest import mock

from serve.launch_identity import attestation
from strata_control import readiness


class LaunchIdentityTests(unittest.TestCase):
    def test_attestation_changes_with_source_args_and_key(self):
        key = bytes(range(32))
        cfg = b'{"api_key":"a private setting","model_name":"fake"}'
        args = ["--config", "/tmp/run.json", "--parallel", "4"]
        value = attestation(key, cfg, args)
        self.assertEqual(len(value), 64)
        self.assertEqual(value, attestation(key, cfg, args))
        self.assertNotEqual(value, attestation(key, cfg + b" ", args))
        self.assertNotEqual(value, attestation(key, cfg, [*args, "--batch-telemetry"]))
        self.assertNotEqual(value, attestation(bytes(reversed(key)), cfg, args))
        self.assertNotIn("private", value)

    def test_wrong_or_absent_server_attestation_rejects_readiness_before_generation(self):
        ready = {"bound_host": "127.0.0.1", "bound_port": 1234,
                 "model_id": "fake", "model_state": "ready"}
        cfg = json.dumps({"model_name": "fake"}).encode()
        for reported in (None, "0" * 64):
            with self.subTest(reported=reported), mock.patch.object(readiness, "_request") as request:
                request.return_value = {"instance": {"id": "one", "launch_attestation": reported},
                                        "model": {"id": "fake", "loaded": True}}
                with self.assertRaisesRegex(readiness.ReadinessError,
                                            "probe_launch_attestation_mismatch"):
                    readiness.verify(ready, cfg, expected_host="127.0.0.1",
                                     expected_instance="one", expected_attestation="f" * 64,
                                     timeout=1)
                self.assertEqual(request.call_count, 1)

    def test_attested_lazy_model_reports_unloaded_without_generation(self):
        ready = {"bound_host": "127.0.0.1", "bound_port": 1234,
                 "model_id": "fake", "model_state": "unloaded"}
        with mock.patch.object(readiness, "_request", return_value={
                "instance": {"id": "one", "launch_attestation": "f" * 64},
                "model": {"id": "fake", "loaded": False}}) as request:
            result = readiness.verify(ready, b'{"model_name":"fake"}',
                                      expected_host="127.0.0.1", expected_instance="one",
                                      expected_attestation="f" * 64, timeout=1)
        self.assertTrue(result["launch_attested"])
        self.assertFalse(result["generation_ready"])
        self.assertEqual(request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
