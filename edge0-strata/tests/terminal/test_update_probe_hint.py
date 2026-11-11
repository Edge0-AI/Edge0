"""#4692: a failed one-token test start says where the kept engine log is (it said "report it with the log" and named none)."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from strata_control import updater


class ProbeHint(unittest.TestCase):
    def test_the_message_names_the_kept_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = SimpleNamespace(test_timeout=600, components=Path(tmp))
            component = SimpleNamespace(version="0.1.54")
            exc = updater._test_start_failure(component, {"error": "process_exited", "state": "failed"}, ctx, " | boom")
        text = str(exc)
        self.assertIn("test-start-0.1.54.log", text)
        self.assertIn("strata update --force", text)
        self.assertNotIn("boom", text)                      # the engine's own words stay out of the sentence


if __name__ == "__main__":
    unittest.main()
