"""serve/test_web_load_unload.py - the web page's header carries the Load / Unload control (#3550, by sakuumomo).

Pins, as text: index.html holds the header button and the Monitor's "Unloaded" badge; app.js wires the button to POST /load
and POST /unload, holds the POST's state until a later poll agrees, answers a 401 with the key copy, and refits the header
row only when its text changed (the per-second poll does not force a layout). The behaviour itself, in a real browser, is in
tools/webtest/browser_test.mjs ("load/unload: ...").

    python -m unittest serve.test_web_load_unload -v
"""
from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"


class WebLoadUnloadMarkup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (WEB / "index.html").read_text(encoding="utf-8")
        cls.js = (WEB / "app.js").read_text(encoding="utf-8")
        cls.css = (WEB / "app.css").read_text(encoding="utf-8")

    def test_header_button_and_monitor_badge(self):
        self.assertIn('id="unload-btn"', self.html)
        self.assertIn('data-s="unloaded"', self.html)
        self.assertLess(self.html.index('id="unload-btn"'), self.html.index('id="theme-btn"'))   # in the header row

    def test_button_posts_load_and_unload(self):
        self.assertIn('$("unload-btn").onclick', self.js)
        self.assertIn('fetch(loadingNow ? "load" : "unload", {method: "POST", headers: headers(true), body: "{}"})', self.js)

    def test_unload_asks_first_and_never_interrupts_a_turn(self):
        self.assertIn("confirm(tr(\"model.confirmUnload\"", self.js)
        self.assertIn("turnBlocksUnload(live)", self.js)
        self.assertIn("function turnBlocksUnload(live)", self.js)

    def test_401_gets_the_key_copy(self):
        # The control POST answers a 401 the way poll and send do, not with a bare HTTP line.
        self.assertIn('r.status === 401 ? tr("toast.keyAbout2"', self.js)

    def test_poll_hold_until_a_poll_agrees(self):
        self.assertIn("metricsHold = {state, after: metricsEpoch};", self.js)
        self.assertIn("function metricsAgree(live, state)", self.js)

    def test_header_refits_only_on_changed_text(self):
        # The per-second poll re-sets the pill; the fit runs when its text actually changed.
        self.assertIn("if (text !== lastPillText) { lastPillText = text; fitHeader(); }", self.js)

    def test_the_header_row_keeps_the_buttons_on_a_phone(self):
        self.assertIn("#unload-btn", self.css)
        self.assertIn(".st-header--clip #pill-text", self.css)
        for width in ("955px", "457px", "382px"):
            self.assertIn(f"@media (max-width: {width})", self.css)


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class WebLoadUnloadParses(unittest.TestCase):
    def test_app_js_parses(self):
        r = subprocess.run(["node", "--check", str(WEB / "app.js")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
