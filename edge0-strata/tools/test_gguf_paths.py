"""#3269: a tool run by hand from the Control package finds gguf-py in the install folder setup.py wrote it to."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths  # noqa: E402


class GgufPyLookup(unittest.TestCase):
    def test_install_dir_is_searched(self):
        with tempfile.TemporaryDirectory() as d:
            gp = Path(d) / "third_party" / "llama.cpp" / "gguf-py"
            (gp / "gguf").mkdir(parents=True)
            env = {k: v for k, v in os.environ.items() if k != "STRATA_GGUF_PY"}
            env["STRATA_INSTALL_DIR"] = d
            with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(_paths, "ENGINE", Path(d) / "nowhere" / "pkg"):
                self.assertEqual(_paths.gguf_py(), str(gp))

    def test_env_wins_and_message_names_it(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "gguf").mkdir()
            with mock.patch.dict(os.environ, {"STRATA_GGUF_PY": d}):
                self.assertEqual(_paths.gguf_py(), d)
        env = {k: v for k, v in os.environ.items() if k not in ("STRATA_GGUF_PY", "STRATA_INSTALL_DIR")}
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(_paths, "ENGINE", Path(d) / "a" / "b"), \
                mock.patch.object(_paths, "_install_dirs", return_value=[]):
            with self.assertRaises(SystemExit) as cm:
                _paths.gguf_py()
            self.assertIn("STRATA_GGUF_PY", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
