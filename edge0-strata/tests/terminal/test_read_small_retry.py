"""A feed read that hangs once is tried once more: about one request in 25 to github.com hung for the whole 30 s while the next answered in 0.4 s,
and the app's first-run card said "offline" a moment before the install's own read worked."""
import io
import unittest
import urllib.error
from unittest import mock

from strata_control import transfer


class Answer(io.BytesIO):
    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class ReadSmallRetry(unittest.TestCase):
    def read(self, side_effect):
        with mock.patch.object(transfer, "check_url_policy", lambda url: None), mock.patch.object(transfer._opener, "open", side_effect=side_effect) as op:
            try:
                return transfer.read_small("https://updates.example.org/feed.json", timeout=30), op
            except transfer.TransferError as exc:
                return exc, op

    def test_one_timeout_is_forgiven_and_the_first_wait_is_short(self):
        got, op = self.read([urllib.error.URLError(TimeoutError("timed out")), Answer(b"{}")])
        self.assertEqual(got, b"{}")
        self.assertEqual([c.kwargs["timeout"] for c in op.call_args_list], [transfer.FIRST_TRY_SECONDS, 30])

    def test_two_timeouts_still_say_so(self):
        got, op = self.read([urllib.error.URLError(TimeoutError("timed out"))] * 2)
        self.assertIsInstance(got, transfer.TransferError)
        self.assertEqual(op.call_count, 2)

    def test_an_answer_of_404_is_not_tried_again(self):
        err = urllib.error.HTTPError("https://x", 404, "no", {}, None)
        got, op = self.read([err])
        self.assertIsInstance(got, transfer.TransferError)
        self.assertEqual(op.call_count, 1)

    def test_a_good_first_answer_is_one_request(self):
        got, op = self.read([Answer(b"{}")])
        self.assertEqual((got, op.call_count), (b"{}", 1))


if __name__ == "__main__":
    unittest.main()
