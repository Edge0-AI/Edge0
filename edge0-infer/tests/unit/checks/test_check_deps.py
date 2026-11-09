"""Dependency whitelist positives and seeded violations."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[3] / "tools/checks/check_deps.py"


class DependencyTests(unittest.TestCase):
    def check(self, path, text, rule=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            file = root / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(text)
            result = subprocess.run([sys.executable, "-I", str(SCRIPT), str(root)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 1 if rule else 0, result.stdout + result.stderr)
            if rule:
                self.assertIn(rule, result.stdout)
                self.assertIn(("third_party/foo:1:" if path.startswith("third_party/foo/") else path + ":1:"), result.stdout)

    def test_clean(self):
        self.check("CMakeLists.txt", "add_subdirectory(third_party/ggml)\n")

    def test_fetch(self):
        self.check("CMakeLists.txt", "include(FetchContent)\n", "dependency-source")

    def test_third_party(self):
        self.check("third_party/foo/x.c", "", "third-party")

    def test_llama_link(self):
        self.check("CMakeLists.txt", "target_link_libraries(x llama)\n", "llama-library")

    def test_llama_link_in_any_directory(self):
        # No directory is exempt any more, including a standalone tool build.
        for path in ("tools/oracle/CMakeLists.txt", "tests/x/CMakeLists.txt", "tools/cli/CMakeLists.txt"):
            with self.subTest(path=path):
                self.check(path, "target_link_libraries(x llama)\n", "llama-library")
                self.check(path, "find_package(llama REQUIRED)\n", "llama-library")

    def test_gguf_import(self):
        self.check("tools/x.py", "import gguf\n", "gguf-py")


    def test_c_gguf_allowed(self):
        self.check("src/model/x.cpp", '#include "gguf.h"\n')

    def test_gguf_from(self):
        self.check("tools/oracle/x.py", "from gguf import Reader\n", "gguf-py")

    def test_external(self):
        self.check("CMakeLists.txt", "include(ExternalProject)\n", "dependency-source")

    def test_package(self):
        self.check("CMakeLists.txt", "find_package(llama REQUIRED)\n", "llama-library")

    def test_generated_dirs_skipped(self):
        self.check("tools/assets/.venv/lib/x.py", "import gguf\n")
        self.check("tools/__pycache__/x.py", "import gguf\n")
        self.check("tools/venv/x.py", "import gguf\n", "gguf-py")

    def test_cmake_fragment(self):
        self.check("cmake/x.cmake", "target_link_libraries(x llama)\n", "llama-library")
        self.check("cmake/x.cmake", "target_link_libraries(x ggml)\n")

    def test_source_extensions(self):
        for suffix in (".cc", ".cpp", ".cxx", ".c++", ".hh", ".hpp", ".hxx"):
            with self.subTest(suffix=suffix):
                self.check("src/app/x" + suffix, 'const char* dependency = "gguf-py/gguf";\n', "gguf-py")
                self.check("src/app/x" + suffix, '#include "gguf.h"\n')

    def test_build_directories(self):
        for directory in ("build", "build-debug"):
            with self.subTest(directory=directory):
                self.check(f"tools/{directory}/x.py", "import gguf\n", "gguf-py")
                self.check(f"tools/{directory}/x.py", "import pathlib\n")
                self.check(f"{directory}/x.py", "import gguf\n")

    def test_tests_root(self):
        self.check("tests/x.cxx", 'const char* dependency = "gguf-py/gguf";\n', "gguf-py")
        self.check("tests/x.cxx", '#include "gguf.h"\n')

    def test_toml_quotes(self):
        for quote in ("'", '"'):
            for package in ("gguf", "gguf>=0.1"):
                with self.subTest(quote=quote, package=package):
                    self.check("pyproject.toml", f"dependencies=[{quote}{package}{quote}]\n", "gguf-py")
                    self.check("pyproject.toml", f"dependencies=[{quote}other-{package}{quote}]\n")

    def test_requirements(self):
        for name in ("requirements.txt", "requirements-dev.txt"):
            with self.subTest(name=name):
                self.check(name, "gguf>=0.1\n", "gguf-py")
                self.check(name, "other-gguf>=0.1\n")

    def test_manifest(self):
        import hashlib
        for case in ('allowed', 'test_allowed', 'src', 'include', 'other', 'extra', 'hash', 'missing'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / 'tools/checks').mkdir(parents=True)
                library = root / 'third_party/example'
                library.mkdir(parents=True)
                (library / 'header.h').write_text('original')
                digest = hashlib.sha256(b'original').hexdigest()
                (root / 'tools/checks/third_party.toml').write_text(
                    '[libraries.example]\nheaders=["httplib.h", "nlohmann/json.hpp"]\n'
                    'allowed_prefixes=["tools/cli/serve", "tests/unit/cli/"]\n'
                    '[libraries.example.files."header.h"]\nsha256="' + digest + '"\n')
                path = {'allowed':'tools/cli/serve.cpp', 'test_allowed':'tests/unit/cli/test.cpp',
                        'src':'src/app/test.cpp', 'include':'include/edge0/test.h', 'other':'tools/cli/chat.cpp'}.get(case)
                if path:
                    file = root / path
                    file.parent.mkdir(parents=True, exist_ok=True)
                    file.write_text('#include <httplib.h>\n#include <nlohmann/json.hpp>\n')
                if case == 'extra': (library / 'extra.h').write_text('')
                if case == 'hash': (library / 'header.h').write_text('changed')
                if case == 'missing': (library / 'header.h').unlink()
                result = subprocess.run([sys.executable, '-I', str(SCRIPT), str(root)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0 if case in ('allowed', 'test_allowed') else 1, result.stdout + result.stderr)
