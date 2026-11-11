"""Closing always ends the engine: the close-path matrix (tests/terminal/lifecycle_matrix.py) with the fake engine.

Each case starts `strata serve` or `strata runtime start` with an engine package whose entry point is a script,
ends it one way, and asserts that no process of the tree is left, the port is free and the runtime lease can be taken.
The cases that need a real console window on Windows run when STRATA_MATRIX_CONSOLE=1 (they open a minimised
console); everything else runs everywhere the platform has the mechanism.
"""
import os
import unittest

from tests.terminal import lifecycle_matrix as matrix

CONSOLE = {"serve-ctrl-c", "serve-console-close"}


@unittest.skipUnless(os.name in ("posix", "nt"), "no qualified owner transport on this platform")
class LifecycleMatrix(unittest.TestCase):
    pass


def make(name):
    def test(self):
        if matrix.WINDOWS and name in CONSOLE and not os.environ.get("STRATA_MATRIX_CONSOLE"):
            self.skipTest("needs a console window; set STRATA_MATRIX_CONSOLE=1")
        result = matrix.run([name])[0]
        self.assertTrue(result["ok"], result.get("problems"))
    return test


for _name in matrix.SCENARIOS:
    if matrix.applicable(_name):
        setattr(LifecycleMatrix, "test_" + _name.replace("-", "_"), make(_name))


if __name__ == "__main__":
    unittest.main()
