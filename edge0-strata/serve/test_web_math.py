"""Chat math on the served page (serve/web/math.js, math-adapter.js, vendor/katex): opt-in, offline, no scripts from the model."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

WEB = Path(__file__).parent / "web"
KATEX = WEB / "vendor" / "katex"


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class MathInNode(unittest.TestCase):
    def test_delimiters_escaping_drawing_and_hostile_formulas(self):
        r = subprocess.run([shutil.which("node"), str(Path(__file__).with_name("test_web_math.js"))],
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


class Vendored(unittest.TestCase):
    def test_the_files_are_the_ones_in_provenance(self):
        prov = json.loads((KATEX / "PROVENANCE.json").read_text(encoding="utf-8"))
        self.assertEqual((prov["package"], prov["version"], prov["license"]), ("katex", "0.19.0", "MIT"))
        self.assertTrue(prov["integrity"].startswith("sha512-"))
        on_disk = {p.relative_to(KATEX).as_posix() for p in KATEX.rglob("*") if p.is_file() and p.name != "PROVENANCE.json"}
        self.assertEqual(on_disk, set(prov["sha256"]), "a file was added or removed without PROVENANCE.json")
        for name, digest in prov["sha256"].items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((KATEX / name).read_bytes()).hexdigest(), digest)
        self.assertTrue((KATEX / "LICENSE").read_text(encoding="utf-8").startswith("The MIT License"))

    def test_it_works_offline(self):
        css = (KATEX / "dist" / "katex.min.css").read_text(encoding="utf-8")
        fonts = set(re.findall(r"url\(fonts/([\w-]+\.woff2)\)", css))
        self.assertEqual(len(fonts), 20)
        for font in fonts:
            self.assertTrue((KATEX / "dist" / "fonts" / font).is_file(), font)
        # no address outside this server in anything that loads on the page
        for f in (WEB / "math.js", WEB / "math-adapter.js", KATEX / "dist" / "katex.min.css"):
            self.assertNotRegex(f.read_text(encoding="utf-8"), r"https?://(?!www\.w3\.org)", f.name)
        self.assertNotIn("@import", css)

    def test_the_page_loads_nothing_until_math_is_on(self):
        html = (WEB / "index.html").read_text(encoding="utf-8")
        js = (WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn('src="web/math.js"', html)
        self.assertIn('src="web/math-adapter.js"', html)
        self.assertNotIn("katex", html.lower().replace("math-adapter", ""), "KaTeX is not named in the page: the adapter loads it")
        self.assertIn("math: false", js, "off by default")
        self.assertIn('id="s-math"', html)
        self.assertIn('aria-checked="false"', re.search(r'<button[^>]*id="s-math"[^>]*>', html).group(0))
        adapter = re.sub(r"^\s*//.*$", "", (WEB / "math-adapter.js").read_text(encoding="utf-8"), flags=re.M)   # the code, not its comments
        self.assertIn("trust: false", adapter)
        self.assertNotIn("innerHTML", adapter)
        self.assertNotIn("renderToString", adapter, "KaTeX builds DOM nodes; no HTML string goes into the page")


class Served(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=10) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", ""), e.read()

    def test_scripts_style_and_fonts_are_served(self):
        for path, want in (("/web/math.js", "javascript"), ("/web/math-adapter.js", "javascript"),
                           ("/web/vendor/katex/dist/katex.min.js", "javascript"),
                           ("/web/vendor/katex/dist/katex.min.css", "text/css"),
                           ("/web/vendor/katex/dist/fonts/KaTeX_Main-Regular.woff2", "font/woff2")):
            with self.subTest(path=path):
                code, ctype, body = self.get(path)
                self.assertEqual(code, 200)
                self.assertIn(want, ctype)
                self.assertTrue(body)

    def test_nothing_else_below_vendor(self):
        for path in ("/web/vendor/katex/PROVENANCE.json", "/web/vendor/katex/LICENSE", "/web/vendor/katex/dist/fonts/x.ttf",
                     "/web/vendor/katex/../../app.js", "/web/vendor/", "/web/vendor/katex/dist/fonts",
                     "/web/vendor/katex/dist/missing.js"):
            with self.subTest(path=path):
                self.assertEqual(self.get(path)[0], 404)


if __name__ == "__main__":
    unittest.main()
