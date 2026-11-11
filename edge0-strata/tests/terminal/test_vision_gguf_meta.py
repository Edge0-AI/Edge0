"""strata-vision opens a GGUF with tensor types llama.cpp does not know (Gyro's TQK6 / TQK7), for its vocabulary only.

The check is a C++ program (tools/vision/gguf_meta_test.cpp, CMake target strata-vision-gguf-test) built with the encoder.
Set STRATA_VISION_GGUF_TEST to the built binary to run it here; without it the test is skipped (no compiler needed).
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class VisionGgufMetaTest(unittest.TestCase):
    def test_unknown_tensor_types_do_not_stop_the_vocabulary_load(self):
        binary = os.environ.get("STRATA_VISION_GGUF_TEST", "")
        if not binary or not Path(binary).is_file():
            self.skipTest("STRATA_VISION_GGUF_TEST is not set (build the strata-vision-gguf-test target)")
        with tempfile.TemporaryDirectory() as folder:
            done = subprocess.run([binary, folder], capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("GGUF-META PASS", done.stdout)

    def test_the_gyro_sizes_read_pictures(self):
        import setup
        for model in ("Gyro-S", "Gyro-M"):
            self.assertIs(setup.MODELS[model].get("vision"), True)
        self.assertIs(setup.FAMILIES["gyro"].get("vision"), True)


if __name__ == "__main__":
    unittest.main()
