"""The chat's svg/html preview: sandboxed, no network, no way to the page (serve/web/preview.js)."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import unittest
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import PAGE_CSP, ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).parent / "web"


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class PreviewInNode(unittest.TestCase):
    def test_documents_frames_and_flags(self):
        r = subprocess.run([shutil.which("node"), str(Path(__file__).with_name("test_web_preview.js"))],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


class ThePageSource(unittest.TestCase):
    def test_no_way_for_a_frame_to_reach_the_page(self):
        scripts = {p.name: p.read_text(encoding="utf-8") for p in WEB.glob("*.js")}
        for name, src in scripts.items():
            with self.subTest(name=name):
                self.assertNotIn("allow-same-origin", src.replace("no allow-same-origin", ""), "a frame with the page's origin is the page")
                self.assertNotRegex(src, r"allow-(top-navigation|popups|forms|modals|downloads)")
        # the page never listens for what a frame posts to it
        self.assertNotRegex(scripts["app.js"], r"addEventListener\(\s*[\"']message[\"']")
        self.assertNotRegex(scripts["app.js"], r"\bonmessage\b")
        # every iframe starts closed, and is only ever made by preview.js
        for tag in re.findall(r"<iframe\b[^>]*>", scripts["app.js"]):
            self.assertIn('sandbox=""', tag)
        self.assertNotIn("createElement(\"iframe\")", scripts["app.js"])
        self.assertIn("srcdoc", scripts["preview.js"])
        self.assertNotRegex(scripts["preview.js"], r"\.src\s*=")


class TheHeader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve" / "chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_the_page_forbids_a_frame_to_load_an_address(self):
        with urllib.request.urlopen(self.base + "/", timeout=10) as r:
            self.assertEqual(r.headers.get("Content-Security-Policy"), PAGE_CSP)
        self.assertEqual(PAGE_CSP, "frame-src 'none'",
                         "only this directive: a srcdoc frame inherits the page's policy, and the opt-in script preview needs its inline script")

    def test_other_files_do_not_carry_it(self):
        with urllib.request.urlopen(self.base + "/web/app.js", timeout=10) as r:
            self.assertIsNone(r.headers.get("Content-Security-Policy"))


@unittest.skipUnless(os.environ.get("WEBTEST_MODULES") and shutil.which("node"),
                     "set WEBTEST_MODULES (a folder with node_modules/puppeteer-core) to run the browser tests")
class InABrowser(unittest.TestCase):
    def test_sandbox_in_chrome(self):
        r = subprocess.run([shutil.which("node"), str(ROOT / "tools" / "webtest" / "browser_test.mjs"), "preview"],
                           capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
