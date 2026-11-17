"""serve/test_web_badges.py - the Recent requests table has a badge for every way a request can end (#3991).

A request that waits past the queue limit ends with finish "timeout"; the Status cell showed the raw word. Reads the page source.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parent / "web" / "app.js"
SERVER = Path(__file__).resolve().parent / "server.py"


class FinishBadges(unittest.TestCase):
    def test_a_timed_out_request_has_a_badge(self):
        src = APP.read_text(encoding="utf-8")
        table = src[src.index("const badge = {stop:"):]
        table = table[:table.index("};")]
        self.assertRegex(table, r'timeout: \["st-badge--queued", "Timeout"\]')

    def test_every_finish_the_server_stores_has_a_badge(self):
        src = APP.read_text(encoding="utf-8")
        table = src[src.index("const badge = {stop:"):]
        table = table[:table.index("};")]
        for finish in ("stop", "length", "cancel", "disconnect", "error", "timeout"):
            self.assertRegex(table, rf"\b{finish}:", finish)
        self.assertIn('"timeout"', SERVER.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
