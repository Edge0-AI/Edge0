"""#4051: an engine an earlier run left behind is waited for until it has given its memory back.

The new engine's arena is as big as the old one's; started beside it, a 60 GiB PC went into memory reclaim for good.  The
server ends the old engine before it starts the next (strata_control.procs) and now waits for it to be gone for as long as
a big engine needs; one that is still there after that is said in plain words, and the start goes on (nothing is refused).

    python -m unittest serve.test_leftover_engines -v
"""
import contextlib
import io
import unittest
from unittest import mock

from serve import server
from strata_control import procs


class LeftoverEngines(unittest.TestCase):
    def run_it(self, leftovers=None, orphans=None, env=None):
        out = io.StringIO()
        with mock.patch.object(procs, "end_leftovers", return_value=leftovers or {"stuck": []}) as ends, \
                mock.patch.object(procs, "end_orphans", return_value=orphans or []) as orph, \
                mock.patch.dict("os.environ", env or {}, clear=False), contextlib.redirect_stdout(out):
            server.end_leftover_engines("/x/strata")
        return out.getvalue(), ends, orph

    def test_the_wait_is_long_enough_for_a_big_engine(self):
        _, ends, orph = self.run_it()
        self.assertGreaterEqual(server.LEFTOVER_SETTLE_S, 60)
        self.assertEqual(ends.call_args.kwargs["settle"], server.LEFTOVER_SETTLE_S)
        self.assertEqual(orph.call_args.kwargs["settle"], server.LEFTOVER_SETTLE_S)

    def test_nothing_left_says_nothing(self):
        text, _, _ = self.run_it(orphans=[{"pid": 7, "outcome": "ended"}])
        self.assertEqual(text, "")

    def test_an_engine_that_is_still_there_is_said_and_the_start_goes_on(self):
        text, _, _ = self.run_it(leftovers={"stuck": [{"pid": 4242, "role": "engine"}]},
                                 orphans=[{"pid": 4343, "outcome": "stuck"}])
        self.assertIn("4242, 4343", text)
        self.assertIn("WARNING", text)
        self.assertIn("wait until that process is gone", text)

    def test_keep_leftovers_leaves_them_alone(self):
        _, ends, orph = self.run_it(env={"STRATA_KEEP_LEFTOVERS": "1"})
        ends.assert_not_called()
        orph.assert_not_called()

    def test_a_failed_check_never_stops_a_start(self):
        out = io.StringIO()
        with mock.patch.object(procs, "end_leftovers", side_effect=RuntimeError("x")), contextlib.redirect_stdout(out):
            server.end_leftover_engines("/x/strata")
        self.assertIn("could not check", out.getvalue())


if __name__ == "__main__":
    unittest.main()
