"""Check the actual CLI process output and exit status."""
import argparse
import json
import os
import tempfile
from pathlib import Path
import subprocess
import shlex
import sys
import unittest

parser = argparse.ArgumentParser()
parser.add_argument("--cli", required=True)
parser.add_argument("--origin", required=True)
parser.add_argument("--metal", choices=["0", "1"], required=True)
parser.add_argument("--device", action="store_true")
options = parser.parse_args()


def run(*args):
    return subprocess.run([options.cli, *args], text=True, capture_output=True)


class CliTests(unittest.TestCase):
    def test_info(self):
        result = run("info")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = dict(line.split(": ", 1) for line in result.stdout.splitlines())
        for key in ["engine.version", "ggml.origin", "ggml.version", "backend.cpu", "backend.metal"]:
            self.assertIn(key, lines)
        commit = dict(line.split(": ", 1) for line in Path(options.origin).read_text().splitlines())["commit"]
        self.assertEqual(lines["ggml.origin"], "llama.cpp " + commit + " ggml/")
        self.assertRegex(lines["engine.source_commit"], r"^([0-9a-f]{40})?$")
        modified = lines["engine.source_modified"]
        self.assertRegex(modified, r"^(yes|no|unknown)( reason=\".+\")?$")
        if lines["engine.source_commit"] == "":
            self.assertTrue(modified.startswith("unknown reason="), modified)
        else:
            self.assertRegex(modified, r"^(yes|no)$")
        self.assertIn("initializable=unknown verified=unknown", lines["backend.cpu"])
        self.assertEqual(lines["model.0"], "edge0_35b")
        self.assertEqual([key for key in lines if key.startswith("model.")], ["model.0"])
        if options.metal == "0":
            self.assertIn("compiled=no", lines["backend.metal"])
            self.assertFalse(any("backend=metal" in line for key, line in lines.items() if key.startswith("device.")))

    def test_mixed_configuration(self):
        model = Path(os.environ.get("MOE_TINY", "/nonexistent")) / "tiny-a"
        if not (model / "assets.tsv").is_file():
            self.skipTest("MOE_TINY tiny-a is unavailable")
        base = ("run", "--model", str(model), "--device", "cpu", "--user", "Hello",
                "--temperature", "0", "--presence-penalty", "0", "--max-tokens", "2")
        original = run(*base)
        explicit = run(*base, "--prefill-device", "cpu", "--stage-policy", "fixed")
        self.assertEqual(original.returncode, 0, original.stderr)
        self.assertEqual(explicit.returncode, 0, explicit.stderr)
        self.assertEqual(original.stdout, explicit.stdout)
        summary = json.loads(explicit.stderr.strip().splitlines()[-1])
        self.assertEqual(summary["request_summary"]["handoffs"], 0)
        self.assertEqual(summary["request_summary"]["handoff_bytes"], 0)
        if options.metal == "0":
            unavailable = run(*base, "--prefill-device", "metal")
            self.assertNotEqual(unavailable.returncode, 0)
            self.assertIn("unsupported", unavailable.stderr)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay_base = base[:5] + base[7:]
            turns = root / "turns.json"
            turns.write_text(json.dumps({"turns": [{"input": [1, 2, 3, 4, 5]}]}))
            generated = run(*replay_base, "--turns", str(turns), "--out", str(root / "original"))
            self.assertEqual(generated.returncode, 0, generated.stderr)
            blocks_file = root / "original" / "blocks.json"
            blocks = json.loads(blocks_file.read_text())
            self.assertTrue(all(block["device"] == "CPU" for block in blocks["blocks"]))
            recorded = run(*replay_base, "--replay", str(blocks_file), "--out", str(root / "recorded"))
            self.assertEqual(recorded.returncode, 0, recorded.stderr)
            for block in blocks["blocks"]:
                block.pop("device")
            legacy = root / "legacy.json"
            legacy.write_text(json.dumps(blocks))
            fallback = run(*replay_base, "--replay", str(legacy), "--out", str(root / "fallback"))
            self.assertEqual(fallback.returncode, 0, fallback.stderr)
            self.assertEqual((root / "recorded" / "logits.f32").read_bytes(),
                             (root / "fallback" / "logits.f32").read_bytes())

    def test_verbose(self):
        for flags in [("--verbose",), ("--probe", "--verbose"), ("--verbose", "--probe")]:
            result = run("info", *flags)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("engine.version:", result.stdout)

    def test_probe(self):
        result = run("info", "--probe")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("backend.cpu: compiled=yes initializable=yes verified=unknown", result.stdout)

    def test_unknown(self):
        result = run("unknown")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("Usage:", result.stderr)

    def test_bad_arguments(self):
        for args in [(), ("info", "--bad"), ("info", "--probe", "extra")]:
            with self.subTest(args=args):
                result = run(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn("Usage:", result.stderr)


class MetalCliTests(unittest.TestCase):
    def test_metal_info_and_probe(self):
        info = run("info")
        self.assertEqual(info.returncode, 0, info.stderr)
        self.assertIn("backend.metal: compiled=yes initializable=unknown verified=unknown", info.stdout)
        self.assertTrue(any(line.startswith("device.") and "backend=metal" in line for line in info.stdout.splitlines()))
        self.assertFalse(any(line.startswith("ggml_metal_") for line in info.stderr.splitlines()), info.stderr)
        verbose = run("info", "--verbose")
        self.assertEqual(verbose.returncode, 0, verbose.stderr)
        self.assertTrue(any(line.startswith("ggml_metal_") for line in verbose.stderr.splitlines()), verbose.stderr)
        probe = run("info", "--probe")
        self.assertEqual(probe.returncode, 0, probe.stderr)
        self.assertIn("backend.metal: compiled=yes initializable=yes verified=unknown", probe.stdout)


if options.device:
    info = run("info")
    if info.returncode == 0 and not any(line.startswith("device.") and "backend=metal" in line
            and any(token.startswith("name=") and token != "name=" for token in shlex.split(line))
            for line in info.stdout.splitlines()):
        print("Not executed: no Metal device")
        sys.exit(77)
suite = unittest.defaultTestLoader.loadTestsFromTestCase(MetalCliTests if options.device else CliTests)
sys.exit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
