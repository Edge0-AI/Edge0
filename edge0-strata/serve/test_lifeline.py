"""serve/lifeline.py: a process Strata starts ends when the process that started it ends, however that ends.

    python -m unittest serve.test_lifeline

Each case runs a small "server" process (a fresh interpreter) that starts a sleeping child through
`lifeline.popen`, then ends that server the hard way (kill -9 / TerminateProcess), and checks that the child
is gone. The mechanism under test is Windows' job object, Linux's PR_SET_PDEATHSIG, or the portable watcher
(forced with STRATA_LIFELINE=watcher so it is proven on every POSIX box, not only on macOS).
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POSIX = os.name == "posix"

SERVER = textwrap.dedent("""
    import os, sys, time, threading, subprocess
    sys.path.insert(0, {root!r})
    from serve import lifeline
    mode = sys.argv[1]
    sleeper = [sys.executable, "-c", "import time; time.sleep(600)"]
    out = {{"mechanism": lifeline.mechanism()}}
    if mode == "thread":                       # started on a request thread that ends at once
        box = []
        t = threading.Thread(target=lambda: box.append(lifeline.popen(sleeper, "engine")))
        t.start(); t.join()
        child = box[0]
    else:
        child = lifeline.popen(sleeper, "engine")
    out["child"] = child.pid
    out["server"] = os.getpid()
    print(__import__("json").dumps(out), flush=True)
    if mode == "exit":                         # ends normally, the child still running
        sys.exit(0)
    time.sleep(600)
""")


def alive(pid: int) -> bool:
    from strata_control import procs
    return procs.start_token(pid) is not None


class Lifeline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-lifeline-")
        self.addCleanup(self.temp.cleanup)
        self.env = {**os.environ, "LOCALAPPDATA": self.temp.name, "XDG_STATE_HOME": self.temp.name,
                    "PYTHONPATH": str(ROOT)}
        self.script = Path(self.temp.name) / "server.py"
        self.script.write_text(SERVER.format(root=str(ROOT)), encoding="utf-8")
        self.spawned = []
        self.addCleanup(self._reap)

    def _reap(self):
        for pid in self.spawned:
            if alive(pid):
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass

    def launch(self, mode, **env):
        proc = subprocess.Popen([sys.executable, str(self.script), mode], stdout=subprocess.PIPE, text=True,
                                env={**self.env, **env})
        info = json.loads(proc.stdout.readline())
        self.spawned.append(info["child"])
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        return proc, info

    def wait_gone(self, pid, seconds=15):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not alive(pid):
                return True
            time.sleep(0.1)
        return not alive(pid)

    def hard_kill(self, proc):
        if POSIX:
            os.kill(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(timeout=10)

    def _killed_parent_takes_the_child(self, **env):
        proc, info = self.launch("sleep", **env)
        self.assertTrue(alive(info["child"]))
        self.hard_kill(proc)
        self.assertTrue(self.wait_gone(info["child"]), f"the child outlived a killed server ({info['mechanism']})")
        return info["mechanism"]

    def test_a_child_ends_when_its_server_is_killed(self):
        self.assertIn(self._killed_parent_takes_the_child(), ("job", "pdeathsig", "watcher"))

    @unittest.skipUnless(POSIX, "the portable watcher is the POSIX path")
    def test_the_portable_watcher_ends_the_child_when_the_server_is_killed(self):
        self.assertEqual(self._killed_parent_takes_the_child(STRATA_LIFELINE="watcher"), "watcher")

    @unittest.skipUnless(POSIX, "the portable watcher is the POSIX path")
    def test_the_watcher_token_does_not_change_with_the_child_state(self):
        """The watcher's token has to be the child's START TIME, and nothing that moves while it runs.

        The token used to come from `ps -o lstart=,stat=`, so it carried the state character as well.  A child
        is R (running) for the first milliseconds of its life and then settles to S (sleeping); the token is
        taken right after `popen`, so the watcher then compared 'S' against its 'R' token, decided the child
        was not the one it had been told about, and exited WITHOUT signalling it.  That is the "child outlived
        a killed server" failure, and it appeared or vanished depending on which state the spawn was sampled
        in - which is why it read as a flake.  Sampled while the child spins (R) and again while it sleeps
        (S), the token must be the same either way.
        """
        from serve import lifeline
        from unittest import mock
        # in-process popen() means procs.write_record() would use the REAL per-user state dir; setUp's env only
        # reaches subprocesses. Keep this test's launch record in the temp dir like the other cases.
        with mock.patch.dict(os.environ, {"XDG_STATE_HOME": self.temp.name, "LOCALAPPDATA": self.temp.name}):
            child = lifeline.popen([sys.executable, "-c",
                                    "import time\nend = time.time() + 1.5\nwhile time.time() < end:\n    pass\n"
                                    "time.sleep(30)"], "engine")
        self.spawned.append(child.pid)
        self.addCleanup(lambda: child.poll() is None and child.kill())

        def state():
            """The child's state character, read independently of the code under test ('R' running, 'S' asleep)."""
            out = subprocess.run(["ps", "-o", "stat=", "-p", str(child.pid)], capture_output=True, text=True).stdout
            return out.strip()[:1]

        while_spinning = lifeline._start_token(child.pid)
        spinning_state = state()
        time.sleep(2.0)                                        # asleep by now
        while_sleeping = lifeline._start_token(child.pid)
        sleeping_state = state()
        self.assertEqual((spinning_state, sleeping_state), ("R", "S"),
                         "the child did not go from running to sleeping, so this proves nothing about the token")
        self.assertTrue(while_spinning and while_sleeping, (while_spinning, while_sleeping))
        self.assertEqual(while_spinning, while_sleeping,
                         "the token moved with the child's state, so the watcher would skip the kill")

    @unittest.skipUnless(sys.platform.startswith("linux"), "PR_SET_PDEATHSIG is Linux")
    def test_a_child_started_on_a_request_thread_outlives_that_thread_but_not_the_server(self):
        proc, info = self.launch("thread")
        self.assertEqual(info["mechanism"], "pdeathsig")
        time.sleep(1.0)
        self.assertTrue(alive(info["child"]), "the request thread ended and took the child with it")
        self.hard_kill(proc)
        self.assertTrue(self.wait_gone(info["child"]))

    def test_a_server_that_exits_normally_leaves_nothing_behind(self):
        proc, info = self.launch("exit")
        proc.wait(timeout=15)
        self.assertTrue(self.wait_gone(info["child"]), "end_children at exit must stop what is still running")

    def test_the_child_is_recorded_while_it_runs_and_forgotten_after(self):
        from strata_control import procs
        proc, info = self.launch("sleep")
        base = Path(self.temp.name) / "Strata" / "procs" if not POSIX else Path(self.temp.name) / "strata" / "procs"
        record = base / f"{info['child']}.json"
        deadline = time.monotonic() + 5
        while not record.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        data = json.loads(record.read_text(encoding="utf-8"))
        self.assertEqual((data["pid"], data["role"], data["parent"]["pid"]), (info["child"], "engine", info["server"]))
        self.assertEqual(data["token"], procs.start_token(info["child"]))
        self.hard_kill(proc)
        self.assertTrue(self.wait_gone(info["child"]))


if __name__ == "__main__":
    unittest.main()


STUBBORN = (
    "import signal, sys, time\n"
    "if hasattr(signal, 'SIGTERM'): signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "print('READY 4096 stop', flush=True)\n"
    "time.sleep(600)\n"          # never reads QUIT, never answers STOP, ignores SIGTERM
)


class HardFallback(unittest.TestCase):
    """The clean stop has a deadline: an engine that ignores STOP, QUIT and SIGTERM is killed, and close() returns."""

    def test_an_engine_that_ignores_everything_is_killed(self):
        from unittest import mock
        from serve import server
        engine = server.StrataEngine("missing-executable", [], lazy=True)
        engine.proc = subprocess.Popen([sys.executable, "-c", STUBBORN], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        engine.proc.stdout.readline()
        victim = engine.proc
        t0 = time.monotonic()
        with mock.patch.object(server, "ENGINE_QUIT_WAIT_S", 1), mock.patch.object(server, "ENGINE_TERM_WAIT_S", 1), \
                mock.patch.object(server, "ENGINE_KILL_WAIT_S", 10):
            engine.close()
        self.assertLess(time.monotonic() - t0, 12)
        self.assertIsNotNone(victim.poll(), "the engine is still running after close()")
        self.assertIsNone(engine.proc)
        if os.name == "posix":
            self.assertEqual(victim.returncode, -signal.SIGKILL, "it ignored SIGTERM, so only the kill ended it")
