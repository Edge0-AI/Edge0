"""#1012 (the old #792): requests waiting for the engine's control lines when the engine is restarted.

restart() runs StrataEngine.__init__ again; it used to make a new condition variable, lock, wait counter and wait
list while the waiters still held the old ones: they hung for good, or failed with `list.remove(x): x not in
list`.  The admission state is now kept across a restart.  A waiter either goes on with the new engine or ends
with a clean 503; never a hang, never a ValueError.  The fake engine of test_parallel runs behind the real
StrataEngine and Service.

    python -m unittest serve.test_restart_waiters -v
"""
import json
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve import test_parallel

REAL_POPEN = subprocess.Popen

class RestartWaiters(unittest.TestCase):
    # the fake-engine helpers of test_parallel (its module, not its class, is imported: no duplicate tests)
    chat = test_parallel.ParallelService.chat
    _start, tearDown = test_parallel.ParallelService.start, test_parallel.ParallelService.tearDown

    def start(self, slots, fit=None):
        """The fake engine, and it stays the engine for the restarts that follow (Popen is patched for the test)."""
        import serve.server as server
        real = REAL_POPEN
        self._start(slots, fit=fit)
        script = Path(self.tmp.name) / "fake_strata.py"
        patch = mock.patch.object(server.subprocess, "Popen",
                                  lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw))
        patch.start()
        self.addCleanup(patch.stop)

    def post(self, text, out, i):
        try:
            out[i] = ("ok", self.chat(text)["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as e:
            out[i] = ("http", e.code, json.loads(e.read().decode())["error"]["message"])
        except Exception as e:      # noqa: BLE001 - reported by the assertions
            out[i] = ("exc", repr(e))

    def waiters(self, n):
        """n requests that wait for the control lines: the test holds them (as a prompt read would)."""
        assert self.engine.ctl.acquire(timeout=5)
        out, threads = {}, []
        for i in range(n):
            th = threading.Thread(target=self.post, args=(f"question {i}", out, i))
            th.start()
            threads.append(th)
        deadline = time.time() + 20
        while self.engine.waiting < n and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(self.engine.waiting, n)
        return out, threads

    def joined(self, threads):
        for th in threads:
            th.join(60)
        self.assertFalse([th for th in threads if th.is_alive()], "a waiter hung")

    def clean(self):
        e = self.engine
        self.assertEqual((e.waiting, e.wait_lens), (0, []))
        self.assertFalse(e.ctl.locked())

    def kill(self):
        self.engine.proc.kill()
        self.engine.proc.wait(timeout=10)

    def go_on(self, slots):
        self.start(slots)
        e = self.engine
        cv, ctl, old_proc = e.slot_cv, e.ctl, e.proc
        out, threads = self.waiters(3)
        self.kill()
        restarter = threading.Thread(target=e.restart)               # what ensure_loaded does for the next request
        restarter.start()
        time.sleep(0.2)
        e.ctl.release()                                              # the request that held the lines is gone
        restarter.join(120)
        self.joined(threads)
        self.assertEqual(out, {i: ("ok", "ok, done.") for i in range(3)})
        self.assertIsNot(e.proc, old_proc)
        self.assertIs(e.slot_cv, cv)                                 # the same condition variable and lock
        self.assertIs(e.ctl, ctl)
        self.clean()

    def test_waiters_go_on_with_the_new_engine_2_slots(self):
        self.go_on(2)

    def test_waiters_go_on_with_the_new_engine_4_slots(self):
        self.go_on(4)

    def stays_down(self, slots):
        self.start(slots)
        out, threads = self.waiters(3)
        self.kill()
        self.engine.ctl.release()
        self.joined(threads)
        for i in range(3):
            self.assertEqual(out[i][:2], ("http", 503), out[i])
            self.assertNotIn("list.remove", out[i][2])
            self.assertIn("engine", out[i][2])
        self.clean()
        # the next request starts the engine again and is served
        self.assertEqual(self.chat("after")["choices"][0]["message"]["content"], "ok, done.")
        self.clean()

    def test_waiters_get_a_clean_503_when_the_engine_stays_down_2_slots(self):
        self.stays_down(2)

    def test_waiters_get_a_clean_503_when_the_engine_stays_down_4_slots(self):
        self.stays_down(4)

    def test_a_restart_that_fails_wakes_the_waiters(self):
        self.start(2)
        e = self.engine
        out, threads = self.waiters(2)
        self.kill()

        def refuse(*a, **k):                    # as __init__ when the new engine exits before READY
            e.proc = REAL_POPEN([sys.executable, "-c", "pass"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                text=True)
            raise RuntimeError("the engine exited before it was ready")
        e.__init__ = refuse
        try:
            with self.assertRaises(RuntimeError):
                e.restart(tries=1)
        finally:
            del e.__init__
        e.ctl.release()
        self.joined(threads)
        self.assertEqual([out[i][:2] for i in range(2)], [("http", 503)] * 2, out)
        self.clean()

    def killed_mid_read(self, slots, others):
        """A long prompt is being read (alone, or with `others` short requests arriving meanwhile: it gives way and is
        admitted into a slot) when the engine dies: every request ends at once with a 503.  It used to wait out the
        300 s drain for an end line that had been read already."""
        self.start(slots)
        out, threads = {}, []
        threads.append(threading.Thread(target=self.post, args=("LONGREPLY " + "word " * 740, out, "long")))
        threads[0].start()
        time.sleep(0.25)
        for i in range(others):
            th = threading.Thread(target=self.post, args=(f"short {i}", out, i))
            th.start()
            threads.append(th)
            time.sleep(0.05)
        time.sleep(0.1)
        t0 = time.time()
        self.kill()
        self.joined(threads)
        self.assertLess(time.time() - t0, 20)
        self.assertTrue(all(v[0] == "ok" or v[:2] == ("http", 503) for v in out.values()), out)
        self.assertEqual(out["long"][:2], ("http", 503), out)
        self.clean()
        self.assertEqual(self.chat("after")["choices"][0]["message"]["content"], "ok, done.")

    def test_killed_while_reading_alone(self):
        self.killed_mid_read(2, 0)

    def test_killed_while_reading_with_others_arriving(self):
        self.killed_mid_read(2, 2)

    def test_an_old_request_meets_a_restart_in_progress(self):
        """A request that learns its engine died while restart() has already taken the process (self.proc is None):
        its death note and its sends answer with EngineDied, not AttributeError (the client got no reply at all)."""
        from serve.server import EngineDied
        self.start(2)
        self.kill()
        self.engine.close()                                           # what restart() does first
        self.assertIsNone(self.engine.proc)
        self.assertTrue(self.engine.death_note())                    # a sentence, no AttributeError
        self.assertFalse(self.engine.alive())
        with self.assertRaises(EngineDied):
            self.engine._send("STOP")

    def refused(self, slots):
        """#1059: the engine refuses a request up front (ERR, then idle).  The server answered after a 300 s drain of
        a DONE that never comes, and held the control lines meanwhile."""
        self.start(slots)
        out = {}
        t0 = time.time()
        self.post("REFUSEME please", out, 0)
        self.assertLess(time.time() - t0, 20)
        self.assertEqual(out[0][0], "http", out)
        self.assertIn("refused", out[0][2])
        self.clean()
        self.assertEqual(self.chat("after")["choices"][0]["message"]["content"], "ok, done.")

    def test_an_engine_err_answers_at_once_solo_loop(self):
        self.refused(2)

    def test_an_engine_err_answers_at_once_not_batched(self):
        self.refused(0)

    def test_solo_engine_is_unchanged(self):
        self.start(4, fit=0)                                          # the engine turns batching off
        self.kill()
        self.assertEqual(self.chat("hi")["choices"][0]["message"]["content"], "ok, done.")


if __name__ == "__main__":
    unittest.main()
