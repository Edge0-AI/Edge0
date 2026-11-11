"""When setup.py stops with a Python error, `strata launch` says what it was and where the rest of the output is (#3613, #3467)."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import runner
from strata_cli.launch.runner import LaunchError, Runner, missing_file, python_failure, setup_failure

TRACEBACK = [
    "  [..] Tuning Strata for this PC",
    "Traceback (most recent call last):",
    '  File "C:\\Strata\\setup.py", line 6414, in calibrate_config',
    "    import calibrate as CAL",
    "ModuleNotFoundError: No module named 'calibrate'",
]


class SetupFailureWords(unittest.TestCase):
    def test_an_option_setup_refuses_is_said_with_its_reason(self):                    # #3785
        lines = ["Preparing unsloth UD-Q6_K_XL from your model files...",
                 "setup.py: error: argument --kv: invalid choice: 'bf16' (choose from int8, q4_0, k8v4, fp16)"]
        error = setup_failure(lines, "", "", "/tmp/a.setup.log", True)
        self.assertIn("The model could not be installed. Argument --kv: invalid choice: 'bf16'", error.text)
        self.assertIn("/tmp/a.setup.log", error.next)
        self.assertEqual(setup_failure(["nothing useful"], "", "", None, True).text, "The model could not be installed.")

    def test_setup_accepts_the_kv_format_the_docker_docs_offer(self):
        import subprocess, sys
        root = Path(__file__).resolve().parents[2]
        done = subprocess.run([sys.executable, str(root / "setup.py"), "--kv", "bogus"], capture_output=True, text=True, cwd=str(root), timeout=120)
        self.assertIn("fp16", done.stderr.splitlines()[-1])                  # the choices line of the argparse error

    def test_the_last_line_of_a_traceback_is_found(self):
        self.assertEqual(python_failure(TRACEBACK), "ModuleNotFoundError: No module named 'calibrate'")
        self.assertIsNone(python_failure(["[X] Not enough disk space", "       Free some space"]))
        self.assertIsNone(python_failure(["Traceback (most recent call last):", '  File "x.py", line 1, in <module>']))
        # the last traceback wins, frames and blank lines are skipped
        two = TRACEBACK[:1] + ["Traceback (most recent call last):", "  frame", "ValueError: old"] + TRACEBACK[1:]
        self.assertEqual(python_failure(two), "ModuleNotFoundError: No module named 'calibrate'")

    def test_a_missing_module_is_said_as_a_missing_file_with_the_next_step_and_the_output_path(self):
        error = setup_failure(TRACEBACK, "", "", r"C:\Users\x\launch\a.setup.log", False)
        self.assertIsInstance(error, LaunchError)
        self.assertIn("The model could not be installed.", error.text)
        self.assertIn("missing from this Strata install", error.text)
        self.assertIn("No module named 'calibrate'", error.text)
        self.assertIn("strata update", error.next)
        self.assertIn(r"C:\Users\x\launch\a.setup.log", error.next)

    def test_another_python_error_keeps_its_line_and_the_path(self):
        error = setup_failure(["Traceback (most recent call last):", "  f", "KeyError: 'cuda'"], "", "", "/tmp/a.setup.log", True)
        self.assertIn("KeyError: 'cuda'", error.text)
        self.assertNotIn("missing", error.text)
        self.assertIn("/tmp/a.setup.log", error.next)

    def test_a_setup_error_line_still_wins_and_gets_the_details_path(self):
        error = setup_failure(TRACEBACK, "Not enough disk space", "Free some space.", "/tmp/a.setup.log", False)
        self.assertEqual(error.text, "The model could not be installed. Not enough disk space.")
        self.assertEqual(error.next, "Free some space. Details: /tmp/a.setup.log")
        plain = setup_failure([], "", "", None, False)
        self.assertEqual(plain.text, "The model could not be installed.")
        self.assertIn("a download continues where it stopped", plain.next)
        self.assertNotIn("Details", plain.next)

    def test_missing_file_kinds(self):
        self.assertTrue(missing_file("ModuleNotFoundError: No module named 'x'"))
        self.assertTrue(missing_file("ImportError: cannot import name 'y'"))
        self.assertFalse(missing_file("ValueError: bad"))

    def test_the_setup_output_is_kept_in_the_launch_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Runner.__new__(Runner)
            with mock.patch.object(Runner, "name", new_callable=mock.PropertyMock, return_value="gyro-s-32k"),                     mock.patch.object(runner, "launch_dir", return_value=Path(folder)):
                kept = run.keep_setup_output([f"line {i}" for i in range(1000)] + TRACEBACK)
            self.assertEqual(Path(kept), Path(folder) / "gyro-s-32k.setup.log")
            text = Path(kept).read_text(encoding="utf-8")
            self.assertIn("ModuleNotFoundError: No module named 'calibrate'", text)
            self.assertEqual(len(text.splitlines()), 400)                 # the tail, not a runaway file
            with mock.patch.object(Runner, "name", new_callable=mock.PropertyMock, return_value="gyro-s-32k"),                     mock.patch.object(runner, "launch_dir", side_effect=OSError("read-only")):
                self.assertIsNone(run.keep_setup_output(["x"]))


    def test_a_message_that_ends_with_a_full_stop_does_not_get_a_second_one(self):
        error = setup_failure(TRACEBACK, "Strata can't reach GitHub to download the DeepSeek engine.", "Check the internet connection.", "/tmp/a.setup.log", False)
        self.assertIn("DeepSeek engine. Check", error.text + " " + error.next)
        self.assertNotIn("engine..", error.text)


if __name__ == "__main__":
    unittest.main()
