"""Model-term boundaries, registration exceptions and source selection."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


class ModelTermsTests(unittest.TestCase):
    def check(self, path, text, rejected):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(text, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-I", str(ROOT / "tools/checks/check_model_terms.py"), str(root)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, int(rejected), result.stdout + result.stderr)
            if rejected:
                self.assertIn(f"model-terms: {path}:1:", result.stdout)

    def test_terms(self):
        for term in ("edge0_35b", "EDGE0-35B", "edge0_8b", "EDGE0-8B", "35B", "8b",
                     "qwen3next", "DeltaNet", "PREROUTER", "LoRA"):
            for path in ("src/app/x.cpp", "src/models/api/x.h", "include/edge0/x.h", "tools/cli/x.cpp"):
                with self.subTest(term=term, path=path):
                    self.check(path, term, True)

    def test_exceptions(self):
        for path in ("src/app/adapters.cpp", "src/models/edge0_35b/x.cpp",
                     "src/models/another/x.h", "tests/unit/app/x.cpp",
                     "src/app/CMakeLists.txt", "include/edge0/notes.md"):
            self.check(path, "edge0_35b deltanet LoRA", False)

    def test_word_boundaries_and_pre_tokenizer(self):
        self.check("src/app/x.cpp", "qwen35 flora 135b 18b", False)

    def test_source_extensions(self):
        for suffix in (".cc", ".cxx", ".c++", ".hh", ".hpp", ".hxx", ".mm"):
            self.check("src/app/x" + suffix, "deltanet", True)
