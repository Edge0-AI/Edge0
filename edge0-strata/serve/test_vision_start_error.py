"""I3 (tools and images review, 2026-10-22): a picture reader that dies before it prints used to say only "the vision
encoder did not start: " (#3645, #3444, #3697, #3669).  The message now carries the tail of its stderr, its exit status
and, for a kernel-image error, the card/architecture hint.

    python -m serve.test_vision_start_error
"""
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import serve.server as server


def log_with(text):
    f = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
    f.write(text)
    f.flush()
    return f


class StartError(unittest.TestCase):
    def test_an_empty_line_is_never_a_blank_sentence(self):
        m = server.vision_start_error("")
        self.assertNotEqual(m.strip(), "the vision encoder did not start:")
        self.assertIn("without saying anything", m)

    def test_exit_code_and_stderr_tail_come_with_it(self):
        f = log_with("loading mmproj\nggml_cuda_init: failed\nsome last words\n")
        self.addCleanup(lambda: (f.close(), Path(f.name).unlink()))
        m = server.vision_start_error("", f, 3)
        self.assertIn("exit code 3", m)
        self.assertIn("some last words", m)
        self.assertIn("ggml_cuda_init: failed", m)

    def test_a_windows_crash_status_is_named(self):
        m = server.vision_start_error("", None, 0xC0000135)
        self.assertIn("0xC0000135", m)

    def test_kernel_image_error_names_the_card_and_the_way_out(self):
        for sign in ("CUDA error: device kernel image is invalid", "IM2COL failed", "no kernel image is available for execution"):
            with self.subTest(sign=sign):
                f = log_with("warm-up\n" + sign + "\n")
                try:
                    m = server.vision_start_error("", f, 1)
                finally:
                    f.close()
                    Path(f.name).unlink()
                self.assertIn("no code for this graphics card", m)
                self.assertIn("CPU", m)
                self.assertIn(sign.split(":")[-1].strip()[:12], m)

    def test_a_timeout_names_the_seconds_and_the_last_log_line(self):
        f = log_with("loading weights\nstill loading the projector\n")
        self.addCleanup(lambda: (f.close(), Path(f.name).unlink()))
        m = server.vision_start_error("", f, waited=300)
        self.assertIn("said nothing for 300 s", m)
        self.assertIn("still loading the projector", m)

    def test_the_reader_that_ends_before_READY_raises_with_its_exit_code(self):
        proc = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(""), kill=lambda: None, wait=lambda timeout=None: 0,
                               poll=lambda: 7, returncode=7)
        v = server.Vision.__new__(server.Vision)
        v.spawn = (["strata-vision"], None, None)
        v.dir = Path(tempfile.gettempdir())
        with mock.patch.object(server, "popen", lambda *a, **k: proc):
            with self.assertRaises(RuntimeError) as cm:
                v._start()
        self.assertIn("exit code 7", str(cm.exception))

    def test_a_normal_message_is_unchanged(self):
        self.assertEqual(server.vision_start_error("ERR no mmproj\n"), "the vision encoder did not start: ERR no mmproj")


if __name__ == "__main__":
    unittest.main()
