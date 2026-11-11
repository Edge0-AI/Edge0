"""#1445: a vision encoder that cannot start because the GPU is busy/unavailable (compute mode Exclusive_Process) says
so, with the nvidia-smi check, instead of only the first line of the encoder's output.

    python -m serve.test_vision_busy
"""
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import serve.server as server


class Busy(unittest.TestCase):
    def test_message_names_the_cause_from_the_line(self):
        m = server.vision_start_error("ggml_backend_cuda_device_get_memory: cudaMemGetInfo failed "
                                      "(CUDA-capable device(s) is/are busy or unavailable), returning 0/0")
        self.assertIn("Exclusive_Process", m)
        self.assertIn("nvidia-smi", m)
        self.assertIn("did not start", m)

    def test_message_names_the_cause_from_the_log_tail(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False) as f:
            f.write("x" * 40000 + "\ncudaMalloc failed: CUDA-capable device(s) is/are busy or unavailable\n")
            name = f.name
        try:
            with open(name, "a") as log:
                m = server.vision_start_error("load_hparams: something", log)
            self.assertIn("Exclusive_Process", m)
        finally:
            Path(name).unlink()

    def test_other_failures_are_unchanged(self):
        self.assertEqual(server.vision_start_error("ERR no mmproj\n"), "the vision encoder did not start: ERR no mmproj")

    def test_start_raises_it(self):
        line = "cudaMalloc failed: CUDA-capable device(s) is/are busy or unavailable\n"
        proc = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(line), kill=lambda: None, wait=lambda timeout=None: 0,
                               poll=lambda: None)
        v = server.Vision.__new__(server.Vision)
        v.spawn = (["strata-vision"], None, None)
        v.dir = Path(tempfile.gettempdir())
        with mock.patch.object(server, "popen", lambda *a, **k: proc):
            with self.assertRaises(RuntimeError) as cm:
                v._start()
        self.assertIn("Exclusive_Process", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
