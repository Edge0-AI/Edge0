"""#1727: prevent_sleep.  The Windows call is mocked; the real call is tested on Windows by hand.

    python -m unittest serve.test_keepawake -v
"""
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.keepawake import ES_CONTINUOUS, ES_SYSTEM_REQUIRED, KeepAwake
from serve.server import ByteTokenizer, MockEngine, Service

ROOT = Path(__file__).resolve().parents[1]
ON = ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def wait_for(cond, t=3.0):
    end = time.time() + t
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


class Recorder:
    def __init__(self):
        self.calls, self.threads = [], set()

    def __call__(self, flags):
        self.calls.append(flags)
        self.threads.add(threading.get_ident())
        return 0


class KeepAwakeTests(unittest.TestCase):
    def test_held_while_busy_and_released_when_idle(self):
        rec = Recorder()
        ka = KeepAwake(rec)
        self.assertEqual(rec.calls, [])                       # nothing until a request
        ka.acquire()
        ka.acquire()                                          # two requests: still one hold
        self.assertTrue(wait_for(lambda: rec.calls == [ON]))
        ka.release()
        time.sleep(0.1)
        self.assertEqual(rec.calls, [ON])                     # one still running
        ka.release()
        self.assertTrue(wait_for(lambda: rec.calls == [ON, ES_CONTINUOUS]))
        ka.close()
        self.assertEqual(rec.calls[-1], ES_CONTINUOUS)
        self.assertEqual(len(rec.threads), 1)                 # set and cleared by the same thread
        self.assertFalse(ON & 0x2)                            # never ES_DISPLAY_REQUIRED

    def test_exit_clears_a_held_state(self):
        rec = Recorder()
        ka = KeepAwake(rec)
        ka.acquire()
        self.assertTrue(wait_for(lambda: rec.calls == [ON]))
        ka.close()
        self.assertEqual(rec.calls, [ON, ES_CONTINUOUS])

    def test_a_failing_call_never_breaks_a_request(self):
        def boom(flags):
            raise OSError("no")
        ka = KeepAwake(boom)
        ka.acquire()
        ka.release()
        ka.close()

    def test_off_and_non_windows(self):
        say = mock.Mock()
        self.assertIsNone(KeepAwake.create(False, say))
        say.assert_not_called()
        with mock.patch("serve.keepawake.sys.platform", "linux"):
            self.assertIsNone(KeepAwake.create(True, say))
        self.assertIn("Windows only", say.call_args[0][0])

    def test_service_holds_for_the_whole_request_including_early_close(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        rec = Recorder()
        svc.keep_awake = ka = KeepAwake(rec)
        kinds = [k for k, _ in svc.run(tok.encode("hi"), False, None, 8, {}, threading.Event())]
        self.assertIn("done", kinds)
        self.assertTrue(wait_for(lambda: rec.calls == [ON, ES_CONTINUOUS]))
        gen = svc.run(tok.encode("hi"), False, None, 8, {}, threading.Event())
        next(gen)
        self.assertTrue(wait_for(lambda: rec.calls[-1] == ON))
        gen.close()                                           # the client left
        self.assertTrue(wait_for(lambda: rec.calls[-1] == ES_CONTINUOUS))
        ka.close()

    def test_default_service_has_none(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.assertIsNone(svc.keep_awake)


if __name__ == "__main__":
    unittest.main()
