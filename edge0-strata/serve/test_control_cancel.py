"""#879: a Stop that arrives while the engine reads the prompt (PP lines, no tokens yet) is sent at once, not after the
first token."""
import os
import queue
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve import server  # noqa: E402


class ControlCancelTest(unittest.TestCase):
    def _engine(self, lines):
        eng = object.__new__(server.StrataEngine)
        eng.lines = queue.Queue()
        for l in lines:
            eng.lines.put(l)
        eng.sent = []
        eng._send = eng.sent.append
        eng._ctl_mode = "solo"
        eng._ctl_result = None
        eng.progress = None
        eng.prefill_tok_s_mean = None
        eng._last_done = None
        eng._parse_done = lambda line: None
        return eng

    def test_stop_during_prompt_read(self):
        cancel = threading.Event()
        eng = self._engine(["PP 256 4096 1 100.0", "PP 512 4096 2 100.0", "PP 768 4096 3 100.0", "DONE 0"])
        gen = eng._control(cancel, lambda t: None)
        self.assertIsNone(next(gen))               # first PP line, nothing cancelled yet
        self.assertEqual(eng.sent, [])
        cancel.set()
        self.assertIsNone(next(gen))               # the next PP line: STOP goes out now
        self.assertEqual(eng.sent, ["STOP"])
        self.assertIsNone(next(gen))
        self.assertEqual(eng.sent, ["STOP"])       # once

    def test_no_stop_without_cancel(self):
        eng = self._engine(["PP 256 4096 1 100.0", "DONE 0"])
        for _ in eng._control(threading.Event(), lambda t: None):
            pass
        self.assertEqual(eng.sent, [])


if __name__ == "__main__":
    unittest.main()
