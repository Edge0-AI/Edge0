"""A partial jinja2 install must fail with the missing piece named (#217).

The Windows standalone app shipped a Python whose jinja2 package had no
``jinja2.sandbox`` module, so ``serve`` crashed at startup with a bare
``ModuleNotFoundError``. These tests run the real import machinery against a
stub jinja2 (package present, sandbox module absent) and check every guard
that verifies the server's packages names the submodule.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _stub_jinja2(path: Path) -> Path:
    """A jinja2 package directory with no sandbox module, as in #217."""
    package = path / "jinja2"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(
        '"""Partial jinja2: the package imports, sandbox.py is absent."""\n'
        '__version__ = "3.1.6"\n',
        encoding="utf-8",
    )
    return path


def _run(argv: list, stub: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(stub) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, *argv],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )


class PartialJinja2Test(unittest.TestCase):
    def test_frontend_import_names_missing_sandbox(self):
        with tempfile.TemporaryDirectory(prefix="partial-jinja2-") as tmp:
            proc = _run(["-c", "from serve.frontend import ChatTemplate"], _stub_jinja2(Path(tmp)))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("jinja2.sandbox", proc.stderr)
        self.assertIn("requirements.txt", proc.stderr)

    def test_strata_preflight_names_missing_sandbox(self):
        with tempfile.TemporaryDirectory(prefix="partial-jinja2-") as tmp:
            proc = _run(["strata.py", "serve"], _stub_jinja2(Path(tmp)))
        self.assertEqual(proc.returncode, 5)
        self.assertIn("jinja2.sandbox", proc.stderr)

    def test_control_missing_packages_names_missing_sandbox(self):
        with tempfile.TemporaryDirectory(prefix="partial-jinja2-") as tmp:
            proc = _run(
                ["-c", "import setup; print(setup.control_missing_packages())"],
                _stub_jinja2(Path(tmp)),
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("jinja2.sandbox", proc.stdout)


if __name__ == "__main__":
    unittest.main()
