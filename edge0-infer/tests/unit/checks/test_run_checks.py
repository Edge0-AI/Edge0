"""Check JUnit classification and summary exit codes."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("run_checks", Path(__file__).resolve().parents[3] / "tools/run_checks.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class ResultsTests(unittest.TestCase):
    def parse(self, xml):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.xml"
            path.write_text(xml)
            return RUNNER.parse_results(path)

    def test_mixed(self):
        results = self.parse('''<testsuite>
          <testcase name="ok" status="run"/>
          <testcase name="bad"><failure message="failure"/><system-out>file.cpp:3: invalid</system-out></testcase>
          <testcase name="gpu" status="notrun"><skipped message="SKIP_RETURN_CODE"/><system-out>no Metal device</system-out></testcase>
        </testsuite>''')
        self.assertEqual([len(value) for value in results.values()], [1, 1, 1])
        self.assertEqual(results["not executed"], [("gpu", "no Metal device")])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(RUNNER.report(results), 1)
        self.assertIn("file.cpp:3: invalid", output.getvalue())

    def test_skip_is_not_pass(self):
        results = self.parse('<testsuites><testsuite><testcase name="git"><skipped message="git not found"/></testcase></testsuite></testsuites>')
        self.assertEqual(results["passed"], [])
        self.assertEqual(results["not executed"], [("git", "git not found")])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(RUNNER.report(results), 0)

    def test_error_and_empty_report(self):
        results = self.parse('<testsuite><testcase name="broken"><error message="crash"/></testcase></testsuite>')
        self.assertEqual(results["failed"], [("broken", "crash")])
        with self.assertRaises(ValueError):
            self.parse('<testsuite/>')


if __name__ == "__main__":
    unittest.main()
