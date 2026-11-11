"""#3202: a Windows 11 with Russian localization (cp1251 / cp866 code pages): a Cyrillic folder name must reach the engine,
the run config and the tools unchanged."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORD = "".join(chr(c) for c in (0x414, 0x43e, 0x43a, 0x443, 0x43c, 0x435, 0x43d, 0x442, 0x44b))   # "Документы"


class CyrillicFolder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix=WORD + "-")
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_config_in_a_cyrillic_folder_is_found_and_read(self):
        from strata_control import discovery
        cfg = self.dir / "strata-q2_0.json"
        cfg.write_text(json.dumps({"exe": "x", "args": ["--native", str(self.dir / "m.gguf")]}, ensure_ascii=False), encoding="utf-8")
        place = discovery.places(explicit=self.dir)[0]
        self.assertEqual(place.configs, 1)
        back = json.loads(discovery.config_files(self.dir)[0].read_text(encoding="utf-8"))
        self.assertIn(WORD, back["args"][1])

    def test_a_child_python_gets_the_path_as_an_argument_and_prints_it_back(self):
        # the server starts the engine through serve.server.popen; the child reads the argument and its own file listing
        from serve import server
        (self.dir / "m.gguf").write_bytes(b"x")
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        code = "import sys, os; print(sys.argv[1]); print(os.path.exists(os.path.join(sys.argv[1], 'm.gguf')))"
        proc = server.popen("test child", [sys.executable, "-c", code, str(self.dir)], role="test",
                            stdout=subprocess.PIPE, env=env)
        out = proc.stdout.read().decode("utf-8").splitlines()
        proc.wait(timeout=30)
        self.assertEqual(out[0], str(self.dir))
        self.assertEqual(out[1], "True")

    def test_windows_launchers_set_utf8_mode(self):
        for name in ("START-HERE.bat", "bin/strata.cmd", "bin/strata.ps1"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("PYTHONUTF8", text, name)


if __name__ == "__main__":
    unittest.main()
