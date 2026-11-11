"""#3370: the encoder's work folder is made again when a temp cleaner removed it while the encoder was unloaded.

    python -m serve.test_vision_workdir
"""
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import serve.server as server


class WorkDir(unittest.TestCase):
    def test_start_makes_a_removed_work_folder_again(self):
        proc = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO("READY\n"), kill=lambda: None,
                               wait=lambda timeout=None: 0, poll=lambda: None)
        with tempfile.TemporaryDirectory() as tmp:
            v = server.Vision.__new__(server.Vision)
            v.spawn = (["strata-vision"], None, None)
            v.dir = Path(tmp) / "strata-vision-gone"
            v.cache = {"k": (v.dir / "k.sve", 4)}
            seen = []

            def fake_popen(*a, **k):
                seen.append(Path(k["cwd"]).is_dir())      # the real popen raises WinError 267 when it is not
                return proc

            with mock.patch.object(server, "popen", fake_popen):
                v._start()
            self.assertEqual(seen, [True])
            self.assertTrue(v.dir.is_dir())
            self.assertEqual(v.cache, {})                 # its cached embeddings went with the folder

    def test_an_existing_folder_and_its_cache_are_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = server.Vision.__new__(server.Vision)
            v.dir = Path(tmp)
            v.cache = {"k": (v.dir / "k.sve", 4)}
            v._ensure_dir()
            self.assertEqual(len(v.cache), 1)


if __name__ == "__main__":
    unittest.main()
