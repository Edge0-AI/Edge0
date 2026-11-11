"""Read-only local inventory of setup-written model run configs."""

import io
import json
import os
import struct
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from strata_cli.cli import main


def gguf_fixture(*, architecture="qwen4exp", tokenizer=True, expert_type=42,
                 overlapping_tensors=False) -> bytes:
    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    pairs = [("general.architecture", 8, string(architecture))]
    if tokenizer:
        pairs += [("tokenizer.ggml.model", 8, string("gpt2")),
                  ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 1) + string("a")),
                  ("tokenizer.ggml.merges", 9, struct.pack("<IQ", 8, 1) + string("a b"))]
    header = (b"GGUF" + struct.pack("<IQQ", 3, 2 if overlapping_tensors else 1, len(pairs)) +
              b"".join(string(key) + struct.pack("<I", kind) + value for key, kind, value in pairs))
    elements, payload_bytes = (64, 18) if expert_type == 42 else (256, 144)
    header += string("blk.0.ffn_gate_exps.weight") + struct.pack(
        "<IQIQ", 1, elements, expert_type, 0)
    if overlapping_tensors:
        header += string("blk.1.ffn_gate_exps.weight") + struct.pack(
            "<IQIQ", 1, elements, expert_type, 0)
    return header + b"\x00" * (-len(header) % 32) + b"\x00" * payload_bytes


class ModelInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        engine = self.root / "engine"
        engine.write_bytes(b"fake engine")
        engine.chmod(0o700)
        tokenizer = self.root / "tokenizer"
        tokenizer.mkdir()
        for name in ("vocab.json", "merges.txt", "token_type.json"):
            (tokenizer / name).write_bytes(b"placeholder")
        pack = self.root / "pack"
        pack.mkdir()
        self.config = self.root / "strata-demo.json"
        self.config.write_text(json.dumps({
            "exe": str(engine), "tokenizer": str(tokenizer),
            "args": ["--pack", str(pack), "--max-context", "32768"],
            "backend": "cuda", "model_name": "Demo model", "api_key": "do-not-disclose",
        }), encoding="utf-8")

    def command(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["models", *args, "--install-dir", str(self.root), "--output", "json"])
        raw = output.getvalue()
        self.assertNotIn("do-not-disclose", raw)
        return code, json.loads(raw)

    def test_list_and_inspect_are_read_only_and_do_not_claim_compatibility(self):
        (self.root / "strata-demo.shared-settings.json").write_text("{}", encoding="utf-8")
        before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in self.root.iterdir() if path.is_file()}
        code, listed = self.command("list", "--installed")
        self.assertEqual(code, 0)
        self.assertEqual(listed["model_count"], 1)
        item = listed["models"][0]
        self.assertEqual(item["id"], "demo")
        self.assertEqual(item["component_status"], "present")
        self.assertEqual(item["context_tokens"], 32768)
        self.assertEqual(item["execution_compatibility"], "unverified")
        self.assertNotIn("args", item)
        code, inspected = self.command("inspect", "demo")
        self.assertEqual(code, 0)
        self.assertEqual(inspected["model"]["source_sha256"], item["source_sha256"])
        self.assertTrue(inspected["components"]["ok"])
        after = {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                 for path in self.root.iterdir() if path.is_file()}
        self.assertEqual(after, before)

    def test_missing_and_malformed_components_are_distinguished(self):
        (self.root / "tokenizer" / "vocab.json").unlink()
        code, listed = self.command("list", "--installed")
        self.assertEqual(code, 0)
        self.assertEqual(listed["models"][0]["component_status"], "missing_or_invalid")
        self.assertIn("COMPONENT_MISSING", listed["models"][0]["issue_codes"])
        # #3478: the list names what is missing, so the app can say which files instead of only "some files are missing"
        self.assertEqual([(m["part"], m["file"], m["status"]) for m in listed["models"][0]["missing"]],
                         [("tokenizer/vocab.json", "vocab.json", "missing")])
        (self.root / "strata-broken.json").write_text("{broken", encoding="utf-8")
        code, listed = self.command("list", "--installed")
        self.assertEqual(code, 0)
        self.assertEqual(listed["model_count"], 2)
        self.assertEqual(listed["models"][0]["component_status"], "invalid")
        code, error = self.command("inspect", "broken")
        self.assertEqual(code, 5)
        self.assertFalse(error["ok"])

    def test_invalid_ids_and_symlink_configs_are_rejected(self):
        code, error = self.command("inspect", "../demo")
        self.assertEqual(code, 2)
        self.assertFalse(error["ok"])
        if os.name == "posix":
            (self.root / "strata-link.json").symlink_to(self.config)
            code, listed = self.command("list", "--installed")
            self.assertEqual(code, 0)
            self.assertEqual(next(row for row in listed["models"] if row["id"] == "link")
                             ["component_status"], "invalid")
            code, error = self.command("inspect", "link")
            self.assertEqual(code, 5)
            self.assertFalse(error["ok"])

    def test_install_dir_must_be_absolute(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["models", "list", "--installed", "--install-dir", "relative", "--output", "json"])
        self.assertEqual(code, 5)
        self.assertFalse(json.loads(output.getvalue())["ok"])

    def test_served_inventory_uses_read_only_telemetry_connection(self):
        snapshot = {"model": {"id": "served-demo", "loaded": True},
                    "instance": {"id": "instance-123"}}
        output = io.StringIO()
        with mock.patch("strata_cli.client.Endpoint.snapshot", return_value=snapshot) as fetch, \
                redirect_stdout(output):
            code = main(["models", "list", "--served", "--url", "http://127.0.0.1:8095",
                         "--output", "json"])
        result = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(result["scope"], "served_current_instance")
        self.assertEqual(result["models"], [{"id": "served-demo", "loaded": True,
                                              "instance_id": "instance-123",
                                              "execution_compatibility": "unverified"}])
        fetch.assert_called_once_with()
        output = io.StringIO()
        with mock.patch("strata_cli.client.Endpoint.snapshot", return_value={
                "model": {"id": None, "loaded": None}, "instance": {"id": "i"}}), \
                redirect_stdout(output):
            code = main(["models", "list", "--served", "--url", "http://127.0.0.1:8095",
                         "--output", "json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["models"], [])

    def test_served_inventory_reaches_telemetry_with_read_only_key(self):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append((self.path, self.headers.get("Authorization")))
                if self.path != "/v1/telemetry" or self.headers.get("Authorization") != "Bearer read-only":
                    self.send_error(403)
                    return
                body = json.dumps({"schema_version": 1,
                                   "instance": {"id": "mock-1", "state": "ready"},
                                   "model": {"id": "example", "loaded": False}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            output = io.StringIO()
            with mock.patch.dict(os.environ, {"STRATA_TELEMETRY_KEY": "read-only"}), \
                    redirect_stdout(output):
                code = main(["models", "list", "--served", "--url",
                             f"http://127.0.0.1:{server.server_port}",
                             "--credential-env", "STRATA_TELEMETRY_KEY", "--output", "json"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue())["models"][0]["loaded"], False)
            self.assertEqual(seen, [("/v1/telemetry", "Bearer read-only")])
            self.assertNotIn("read-only", output.getvalue())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_connection_flags_require_served_scope(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["models", "list", "--installed", "--url", "https://example.invalid",
                         "--output", "json"])
        self.assertEqual(code, 2)
        self.assertFalse(json.loads(output.getvalue())["ok"])

    def test_import_plan_inspects_shards_without_adopting_or_copying(self):
        source = self.root / "source"
        source.mkdir()
        first = source / "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
        second = source / "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"
        for path in (first, second):
            path.write_bytes(gguf_fixture(tokenizer=path == first))
        (source / "mmproj-Qwen3.8-Flash-Next-BF16.gguf").write_bytes(b"auxiliary")
        before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in source.iterdir()}
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["models", "import", str(source), "--plan", "--output", "json"])
        self.assertEqual(code, 0)
        plan = json.loads(output.getvalue())["plan"]
        self.assertEqual(plan["file_count"], 2)
        self.assertEqual(plan["file_set_status"], "plausible")
        self.assertFalse(plan["would_write"])
        self.assertEqual(plan["execution_compatibility"], "unverified")
        self.assertEqual(plan["metadata"]["architecture"], "qwen4exp")
        self.assertEqual(plan["metadata"]["tokenizer_model"], "gpt2")
        self.assertTrue(plan["metadata"]["payload_extent_valid"])
        self.assertEqual(plan["metadata"]["tensor_types"], {"Q2_0": 1})
        self.assertEqual(plan["observed_expert_types"], ["Q2_0"])
        self.assertEqual(plan["candidate_selection"]["family"], "qwen")
        self.assertEqual(plan["candidate_selection"]["model"], "Q2_0")
        self.assertEqual(len(plan["ignored_auxiliary_files"]), 1)
        self.assertEqual({path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                          for path in source.iterdir()}, before)
        second.unlink()
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["models", "import", str(first), "--plan", "--output", "json"])
        self.assertEqual(code, 5)
        self.assertFalse(json.loads(output.getvalue())["ok"])
        self.assertEqual(json.loads(output.getvalue())["plan"]["file_set_status"],
                         "invalid_or_incomplete")

    def test_import_plan_rejects_symlink_and_wrong_magic(self):
        wrong = self.root / "bad.gguf"
        wrong.write_bytes(b"NOTG\x03\x00\x00\x00")
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["models", "import", str(wrong), "--plan", "--output", "json"])
        self.assertEqual(code, 5)
        self.assertFalse(json.loads(output.getvalue())["ok"])
        self.assertEqual(json.loads(output.getvalue())["plan"]["file_set_status"],
                         "invalid_or_incomplete")
        if os.name == "posix":
            link = self.root / "link.gguf"
            link.symlink_to(wrong)
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["models", "import", str(link), "--plan", "--output", "json"])
            self.assertEqual(code, 5)
            self.assertFalse(json.loads(output.getvalue())["ok"])

    def test_import_plan_rejects_other_architecture_and_bounded_metadata(self):
        candidate = self.root / "other.gguf"
        candidate.write_bytes(gguf_fixture(architecture="llama"))
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["models", "import", str(candidate), "--plan", "--output", "json"])
        self.assertEqual(code, 5)
        self.assertIn("unsupported_or_missing_architecture", json.loads(output.getvalue())["plan"]["issues"])
        # A malicious metadata count must fail before allocating or reading tensor data.
        candidate.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1_000_000_000))
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["models", "import", str(candidate), "--plan", "--output", "json"])
        self.assertEqual(code, 5)
        self.assertEqual(json.loads(output.getvalue())["plan"]["file_set_status"], "invalid_or_incomplete")
        key = b"tokenizer.ggml.tokens"
        candidate.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1) +
                              struct.pack("<Q", len(key)) + key +
                              struct.pack("<IIQ", 9, 8, 500_001))
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["models", "import", str(candidate), "--plan", "--output", "json"])
        self.assertEqual(code, 5)
        self.assertIn("inspection limit", json.dumps(json.loads(output.getvalue())["plan"]["issues"]))

    def test_import_plan_rejects_truncated_tensor_payload(self):
        candidate = self.root / "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
        candidate.write_bytes(gguf_fixture()[:-4])
        other = self.root / "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"
        other.write_bytes(gguf_fixture(tokenizer=False))
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["models", "import", str(candidate), "--plan", "--output", "json"])
        self.assertEqual(code, 5)
        self.assertIn("payload is shorter", json.dumps(json.loads(output.getvalue())["plan"]["issues"]))

    def test_import_plan_rejects_unexpected_expert_quantization(self):
        source = self.root / "quant"
        source.mkdir()
        names = [source / f"Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-{i:05d}-of-00002.gguf" for i in (1, 2)]
        names[0].write_bytes(gguf_fixture(expert_type=12))  # Q4_K under a Q2_0 filename
        names[1].write_bytes(gguf_fixture(tokenizer=False))
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["models", "import", str(source), "--plan", "--output", "json"])
        self.assertEqual(code, 5)
        self.assertIn("missing_or_unexpected_expert_quantization",
                      json.loads(output.getvalue())["plan"]["issues"])

    def test_import_plan_rejects_overlapping_tensor_ranges(self):
        source = self.root / "overlap"
        source.mkdir()
        first = source / "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
        second = source / "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"
        first.write_bytes(gguf_fixture(overlapping_tensors=True))
        second.write_bytes(gguf_fixture(tokenizer=False))
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["models", "import", str(source), "--plan", "--output", "json"])
        self.assertEqual(code, 5)
        self.assertIn("byte ranges overlap", json.dumps(json.loads(output.getvalue())["plan"]["issues"]))

    def test_import_plan_does_not_advertise_setup_unsupported_variant(self):
        source = self.root / "unsupported"
        source.mkdir()
        for index in (1, 2):
            (source / f"Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-{index:05d}-of-00002.gguf").write_bytes(
                gguf_fixture(tokenizer=index == 1))
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["models", "import", str(source), "--plan", "--output", "json"])
        self.assertEqual(code, 5)
        plan = json.loads(output.getvalue())["plan"]
        self.assertIsNone(plan["candidate_selection"])
        self.assertIn("unrecognized_or_unsupported_model_variant", plan["issues"])
