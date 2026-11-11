"""The Terminal / Control bundle holds every tools/ module that setup.py can import (#3613, #3467).

0.1.47 shipped without tools/calibrate.py: setup.py imports it after a tune, so every launch after a tune ended in
ModuleNotFoundError. The bundle is staged here the way packaging/package_terminal.py stages it (the payload, the read-only tools,
then add_installer), and setup.py's tool imports are resolved against tools/ as the bundle has it, not against the source tree.
"""

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))
import package_terminal as pt  # noqa: E402


def imports_of(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


class BundleImports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="strata-bundle-")
        cls.root = Path(cls.temp.name) / "strata-terminal"
        cls.root.mkdir()
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        for item in pt.PAYLOAD:
            src, dst = ROOT / item, cls.root / item
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, ignore=ignore)
            else:
                shutil.copy2(src, dst)
        (cls.root / "tools").mkdir(exist_ok=True)
        for item in pt.TOOLS_SUBSET:
            shutil.copy2(ROOT / "tools" / item, cls.root / "tools" / item)
        cls.installer = pt.add_installer(cls.root, ROOT)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_calibrate_is_in_the_bundle(self):
        # the exact file #3613 and #3467 lacked
        self.assertTrue((self.root / "tools" / "calibrate.py").is_file())
        self.assertIn("calibrate", pt.INSTALLER_TOOLS)

    def test_every_tools_module_setup_py_imports_is_in_the_bundle(self):
        in_bundle = {p.stem for p in (self.root / "tools").glob("*.py")}
        in_source = {p.stem for p in (ROOT / "tools").glob("*.py")}
        wanted = imports_of(self.root / "setup.py") & in_source
        self.assertIn("calibrate", wanted, "setup.py no longer imports calibrate: update this test")
        self.assertEqual(sorted(wanted - in_bundle), [], "setup.py imports tools/ modules that the bundle does not carry")

    def test_the_bundled_tools_import_nothing_the_bundle_lacks(self):
        in_bundle = {p.stem for p in (self.root / "tools").glob("*.py")}
        in_source = {p.stem for p in (ROOT / "tools").glob("*.py")}
        for module in sorted(in_bundle):
            missing = sorted((imports_of(self.root / "tools" / f"{module}.py") & in_source) - in_bundle)
            self.assertEqual(missing, [], f"tools/{module}.py imports tools/ modules the bundle does not carry")

    def test_setup_py_and_each_of_its_tools_import_with_tools_limited_to_the_bundle(self):
        modules = sorted(set(pt.setup_tool_imports(self.root)) | set(pt.INSTALLER_TOOLS))
        code = ("import sys, os, importlib\n"
                f"sys.path[:0] = [{str(self.root)!r}, {str(self.root / 'tools')!r}]\n"
                "import setup\n"
                f"for name in {modules!r}:\n"
                "    module = importlib.import_module(name)\n"
                f"    assert os.path.samefile(os.path.dirname(module.__file__), {str(self.root / 'tools')!r}), (name, module.__file__)\n"
                "print('imports ok')\n")
        env = {**os.environ, "STRATA_CONTROL_INSTALL": "1", "PYTHONNOUSERSITE": "1"}
        for key in ("PYTHONPATH", "PYTHONHOME"):
            env.pop(key, None)
        done = subprocess.run([sys.executable, "-I", "-c", code], cwd=str(self.root.parent), env=env,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, (done.stdout or "") + (done.stderr or "")[-1500:])
        self.assertIn("imports ok", done.stdout)

    def test_a_tool_setup_starts_to_import_is_added_without_editing_the_list(self):
        # setup_tool_imports reads setup.py, so INSTALLER_TOOLS cannot go stale the way it did for calibrate
        self.assertTrue(set(pt.setup_tool_imports(ROOT)) <= set(pt.local_imports(ROOT, tuple(sorted(
            set(pt.INSTALLER_TOOLS) | set(pt.setup_tool_imports(ROOT)))))))
        without = tuple(x for x in pt.INSTALLER_TOOLS if x != "calibrate")
        self.assertNotIn("calibrate", pt.local_imports(ROOT, without))      # the old list really lacked it
        self.assertIn("calibrate", pt.setup_tool_imports(ROOT))             # and the scan finds it anyway


if __name__ == "__main__":
    unittest.main()
