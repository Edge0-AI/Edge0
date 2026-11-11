"""serve/winjob.py: a contained child - and whatever it starts - ends when the process that contained it ends, even
when that process is killed outright (the console window closed, Task Manager), with no cleanup of its own.

    python -m unittest serve.test_winjob      (Windows only; skipped elsewhere)
"""
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# stands in for the server: contains a child that starts a grandchild, prints both pids, then waits to be killed
PARENT = f"""
import subprocess, sys, time
sys.path.insert(0, {str(ROOT)!r})
from serve.winjob import contain
code = ("import subprocess, sys, time; g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
        "print(g.pid, flush=True); time.sleep(120)")
child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
ok = contain(child)
print(ok, child.pid, child.stdout.readline().strip(), flush=True)
time.sleep(120)
"""


def running(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


@unittest.skipUnless(os.name == "nt", "job objects are Windows-only")
class TestWinJob(unittest.TestCase):
    def test_children_end_with_a_killed_server(self):
        # a venv's python.exe is a launcher that runs the real interpreter as its child: kill() must reach the latter
        python = getattr(sys, "_base_executable", None) or sys.executable
        parent = subprocess.Popen([python, "-c", PARENT], stdout=subprocess.PIPE, text=True)
        child = grandchild = None
        try:
            ok, child, grandchild = parent.stdout.readline().split()
            child, grandchild = int(child), int(grandchild)
            self.assertEqual(ok, "True")
            self.assertTrue(running(child) and running(grandchild))
            parent.kill()                       # TerminateProcess: no atexit, no finally - like closing the window
            parent.wait(10)
            deadline = time.monotonic() + 10
            while (running(child) or running(grandchild)) and time.monotonic() < deadline:
                time.sleep(0.2)
            self.assertFalse(running(child), "the child outlived the server")
            self.assertFalse(running(grandchild), "the grandchild outlived the server")
        finally:
            if parent.poll() is None:
                parent.kill()
            for pid in (child, grandchild):              # on a failure, don't leave them sleeping
                if pid and running(pid):
                    subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)

    def test_elsewhere_or_without_a_process_it_is_a_no_op(self):
        from serve.winjob import contain
        self.assertFalse(contain(None))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(os.name == "nt", "power throttling is a Windows setting")
class TestNoThrottle(unittest.TestCase):
    def test_a_contained_child_is_opted_out_of_power_throttling(self):
        # #691: contain() also sets ProcessPowerThrottling (EcoQoS off), so a minimized server window does not move the
        # engine to the E-cores
        from serve import winjob
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            self.assertTrue(winjob.no_throttle(int(child._handle)))
            self.assertTrue(winjob.contain(child))
        finally:
            child.kill()
            child.wait(10)
