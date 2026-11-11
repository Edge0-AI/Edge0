"""One engine, one size: the Terminal and the app both say decimal megabytes, rounded half up (apps/host/tests/size-label.test.mjs has the
same numbers). 606,513,890 bytes is 607 MB in both places, never 606 in one and 607 in another."""
import unittest

from strata_cli import update_cmd
from strata_control import updater

# the same samples as the app's test: (bytes, megabytes)
SAMPLES = [(606513890, 607), (581299508, 581), (1, 1), (0, 1), (999_999, 1), (1_500_000, 2), (2_499_999, 2)]


class SizeLabel(unittest.TestCase):
    def test_megabytes_match_the_apps_figures(self):
        for size, expected in SAMPLES:
            self.assertEqual(updater.megabytes(size), expected, size)

    def test_the_terminal_offer_line_uses_the_same_helper(self):
        self.assertEqual(update_cmd._mib(606513890), "607 MB")
        self.assertEqual(update_cmd._mib(581299508), "581 MB")


if __name__ == "__main__":
    unittest.main()
