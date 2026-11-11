"""Importing a model together with its picture reader (mmproj), and the plain words when a model type is not ours."""

import io
import json
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from unittest import mock

from strata_cli.cli import main
from strata_control import models
from tests.terminal.test_models import gguf_fixture


def projector_fixture(projector_type="qwen3vl_merger", architecture="clip") -> bytes:
    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    pairs = [("general.architecture", 8, string(architecture)), ("clip.projector_type", 8, string(projector_type))]
    header = (b"GGUF" + struct.pack("<IQQ", 3, 1, len(pairs)) +
              b"".join(string(key) + struct.pack("<I", kind) + value for key, kind, value in pairs))
    header += string("v.blk.0.attn_q.weight") + struct.pack("<IQIQ", 1, 64, 1, 0)
    return header + b"\x00" * (-len(header) % 32) + b"\x00" * 128


class ImportProjectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "model"
        self.source.mkdir()

    def plan(self, path):
        output = io.StringIO()
        with redirect_stdout(output):
            main(["models", "import", str(path), "--plan", "--output", "json"])
        return json.loads(output.getvalue())["plan"]

    def model(self, **kwargs):
        first = self.source / "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00001.gguf"
        first.write_bytes(gguf_fixture(**kwargs))
        return first

    def test_pinned_projector_next_to_a_model_means_pictures(self):
        self.model()
        file = self.source / "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
        file.write_bytes(projector_fixture())
        with mock.patch.dict(models.PINNED_PROJECTORS, {file.name.lower(): file.stat().st_size}):
            plan = self.plan(self.source)
        self.assertTrue(plan["reads_pictures"])
        self.assertEqual(plan["projectors"][0]["status"], "ok")
        self.assertIn("pictures will work", plan["messages"][0])

    def test_same_name_but_another_size_is_not_called_official(self):
        self.model()
        (self.source / "mmproj-Qwen3.8-Flash-Next-BF16.gguf").write_bytes(projector_fixture())
        plan = self.plan(self.source)
        self.assertEqual(plan["projectors"][0]["status"], "other_name")

    def test_projector_beside_a_chosen_model_file_is_found(self):
        first = self.model()
        (self.source / "mmproj-F16.gguf").write_bytes(projector_fixture())
        plan = self.plan(first)
        self.assertEqual([p["name"] for p in plan["projectors"]], ["mmproj-F16.gguf"])
        self.assertTrue(plan["reads_pictures"])

    def test_projector_of_another_kind_is_not_used(self):
        self.model()
        (self.source / "mmproj-other.gguf").write_bytes(projector_fixture("mlp"))
        plan = self.plan(self.source)
        self.assertFalse(plan["reads_pictures"])
        self.assertEqual(plan["projectors"][0]["status"], "unsupported")
        self.assertIn("Custom model API", plan["projectors"][0]["plain"])

    def test_other_name_projector_of_our_kind_is_explained(self):
        self.model()
        (self.source / "mmproj-mine.gguf").write_bytes(projector_fixture())
        plan = self.plan(self.source)
        self.assertEqual(plan["projectors"][0]["status"], "other_name")
        self.assertTrue(plan["reads_pictures"])

    def test_unsupported_model_type_says_custom_api(self):
        first = self.source / "SmolVLM-256M-Q8_0.gguf"
        first.write_bytes(gguf_fixture(architecture="llama"))
        (self.source / "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf").write_bytes(projector_fixture("idefics3"))
        plan = self.plan(self.source)
        self.assertEqual(plan["file_set_status"], "invalid_or_incomplete")
        self.assertFalse(plan["reads_pictures"])
        self.assertEqual(plan["messages"], ["This model type isn't supported by the Strata engine. "
                                            "Add it as a Custom model API (llama-server) instead."])

    def test_choosing_the_projector_itself_says_so(self):
        projector = self.source / "mmproj-F16.gguf"
        projector.write_bytes(projector_fixture())
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["models", "import", str(projector), "--plan", "--output", "json"])
        self.assertEqual(code, 5)
        self.assertIn("picture reader", json.dumps(json.loads(output.getvalue())))


if __name__ == "__main__":
    unittest.main()
