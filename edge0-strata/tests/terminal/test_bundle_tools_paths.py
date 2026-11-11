"""Every tools/*.py that setup.py, strata_cli or strata_control runs BY PATH is in the packaged Terminal (and so in the Control bundle).

0.2.4 shipped without tools/dsv41/pack_gguf.py: the tests ran from the checkout, where the file exists, and the packaged Terminal failed at
"Preparing your model" with "can't open file .../tools/dsv41/pack_gguf.py".  This test reads the by-path references out of the source (an
AST scan: `ROOT / "tools" / "x" / "y.py"` and the plain string "tools/x/y.py"), stages the bundle the way package_terminal does, and checks that
each referenced file is in it.  It then runs the DeepSeek scripts from the BUNDLE with --help in a folder that holds nothing else, so an import
they need (the sibling smoke.py, strata_tokenizer.py) must come from the bundle too."""
from __future__ import annotations

import ast
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))
import package_terminal as pt  # noqa: E402

#: by-path references that are NOT part of the packaged Terminal on purpose (engine build and install code the bundle never runs). None today.
NOT_BUNDLED: set[str] = set()


def _flatten(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _flatten(node.left) + _flatten(node.right)
    return [node]


def referenced_tools(root: Path = ROOT) -> dict[str, set[str]]:
    """{'tools/dsv41/pack_gguf.py': {files that name it}} for setup.py, strata_cli/**, strata_control/**."""
    files = [root / "setup.py"] + sorted((root / "strata_cli").rglob("*.py")) + sorted((root / "strata_control").rglob("*.py"))
    found: dict[str, set[str]] = {}
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                vals = [p.value if isinstance(p, ast.Constant) and isinstance(p.value, str) else None for p in _flatten(node)]
                if "tools" in vals:
                    rest = []
                    for v in vals[vals.index("tools") + 1:]:
                        if v is None:
                            break
                        rest.append(v)
                    if rest and rest[-1].endswith(".py"):
                        found.setdefault("tools/" + "/".join(rest), set()).add(f.relative_to(root).as_posix())
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                m = re.fullmatch(r"tools[/\\]((?:\w+[/\\])*\w+\.py)", node.value)
                if m:
                    found.setdefault("tools/" + m.group(1).replace("\\", "/"), set()).add(f.relative_to(root).as_posix())
    return found


class ByPathToolsAreBundled(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name) / "bundle"
        cls.root.mkdir()
        (cls.root / "tools").mkdir()
        for item in pt.TOOLS_SUBSET:
            shutil.copy2(ROOT / "tools" / item, cls.root / "tools" / item)
        pt.add_installer(cls.root, ROOT)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_the_scan_finds_the_deepseek_scripts(self):
        # (a scan that finds nothing would make the next test pass for the wrong reason)
        refs = referenced_tools()
        for name in ("tools/dsv41/pack_gguf.py", "tools/dsv41/smoke.py", "tools/dsv41/tune.py", "tools/iq_pack.py", "tools/mtp_fetch.py"):
            self.assertIn(name, refs)
        self.assertGreaterEqual(len(refs), 12)

    def test_every_file_run_by_path_is_in_the_bundle(self):
        missing = {p: sorted(by) for p, by in referenced_tools().items() if p not in NOT_BUNDLED and not (self.root / p).is_file()}
        self.assertEqual(missing, {}, f"run by path, absent from the packaged Terminal: {missing}")

    def test_the_deepseek_scripts_start_from_the_bundle_alone(self):
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
        for rel in ("tools/dsv41/pack_gguf.py", "tools/dsv41/smoke.py", "tools/dsv41/tune.py"):
            with tempfile.TemporaryDirectory() as cwd:
                done = subprocess.run([sys.executable, str(self.root / rel), "--help"], capture_output=True, text=True, cwd=cwd, env=env, timeout=120)
            self.assertEqual(done.returncode, 0, f"{rel} --help from the bundle: {done.stderr[-400:]}")


if __name__ == "__main__":
    unittest.main()
