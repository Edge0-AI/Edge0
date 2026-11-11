"""Windows token defaults must permit identifying children, not just opening files."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PROBE = r"""
import json, subprocess, sys
from pathlib import Path
from strata_control import private_files, procs

assert private_files.adopt_user_defaults()
child = subprocess.Popen(
    [sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(30)"],
    stdout=subprocess.PIPE, text=True,
)
try:
    assert child.stdout.readline().strip() == "ready"
    assert child.poll() is None
    token = procs.start_token(child.pid)
    assert token is not None, "live child became unreadable after adopting user defaults"
    recorded = procs.write_record(child.pid, "engine")
    assert recorded == token, "child identity could not be recorded"
    print(json.dumps({"identified": True, "recorded": True}), flush=True)
finally:
    child.kill()
    child.wait(timeout=10)
    child.stdout.close()
"""


@unittest.skipUnless(os.name == "nt", "Windows token default DACL")
class DefaultObjectAccess(unittest.TestCase):
    def test_adopting_user_defaults_preserves_child_identity(self):
        # Token defaults affect every subsequent object creation in this process.
        # Exercise them in a disposable helper, not in the test runner itself.
        with tempfile.TemporaryDirectory(prefix="strata-default-access-") as folder:
            env = dict(os.environ, LOCALAPPDATA=folder, PYTHONPATH=str(ROOT))
            result = subprocess.run(
                [sys.executable, "-c", PROBE], cwd=ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=25,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {"identified": True, "recorded": True})


if __name__ == "__main__":
    unittest.main()
