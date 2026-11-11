"""The web page's translations: the mechanism (node, serve/test_web_i18n.js), the files the server hands out for it
(serve/web/locales, serve/web/vendor), and the catalogs against the text in the code (tools/web_i18n.py)."""
from __future__ import annotations

import shutil
import subprocess
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import web_i18n  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402
from serve.webfiles import subtree_file  # noqa: E402


def run_node(script: str, *args: str) -> None:
    result = subprocess.run([shutil.which("node"), str(Path(__file__).with_name(script)), *args],
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise AssertionError((result.stdout + result.stderr).strip())


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class Mechanism(unittest.TestCase):
    def test_tr_apply_and_set_language(self):
        run_node("test_web_i18n.js")


class WhatTheServerHandsOut(unittest.TestCase):
    """/web/locales and /web/vendor: whitelisted names and types, nothing outside those folders."""

    def test_names_and_types(self):
        web = Path(__file__).parent / "web"
        locales = web / "locales"
        if (locales / "de.json").is_file():
            self.assertEqual(subtree_file("/web/locales/de.json")[1], "application/json; charset=utf-8")
        for bad in ("/web/locales/de.js", "/web/locales/DE.json", "/web/locales/../app.js", "/web/locales/x/y.json",
                    "/web/locales/%2e%2e%2fapp.json", "/web/locales/de.json%00.js", "/web/locales/missing.json",
                    "/web/locales/C:\\x.json", "/web/vendor/katex/../../app.js", "/web/vendor/katex/dist/katex.min.js.map",
                    "/web/vendor/other/dist/katex.min.js", "/web/vendor/katex/dist/fonts/x.ttf", "/web/vendor/katex/dist/"):
            with self.subTest(bad=bad):
                self.assertIsNone(subtree_file(bad))

    def test_a_link_out_of_the_folder_is_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            web = Path(d) / "web"
            (web / "vendor" / "katex").mkdir(parents=True)
            (web / "locales").mkdir()
            (web / "secret.js").write_text("secret")
            (web / "vendor" / "katex" / "ok.js").write_text("ok")
            self.assertEqual(subtree_file("/web/vendor/katex/ok.js", web)[1], "text/javascript; charset=utf-8")
            try:
                (web / "vendor" / "katex" / "link.js").symlink_to(web / "secret.js")
            except (OSError, NotImplementedError):
                self.skipTest("symlinks need a privilege here")
            self.assertIsNone(subtree_file("/web/vendor/katex/link.js", web))


class Served(unittest.TestCase):
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

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=10) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", ""), e.read()

    def test_i18n_script_and_refusals(self):
        code, ctype, body = self.get("/web/i18n.js")
        self.assertEqual(code, 200)
        self.assertIn(b"StrataI18n", body)
        for path in ("/web/locales/../server.py", "/web/locales/nope.json", "/web/vendor/katex/../../app.js",
                     "/web/vendor/katex/dist/missing.js", "/web/locales/de.json%00"):
            with self.subTest(path=path):
                self.assertEqual(self.get(path)[0], 404)


@unittest.skipUnless((Path(__file__).parent / "web" / "locales" / "en.json").is_file(), "no catalogs yet")
class Catalogs(unittest.TestCase):
    """Every language in serve/web/i18n.js has a catalog with exactly the keys and {placeholders} of the code."""

    def test_the_text_in_the_code_is_plain_literals(self):
        self.assertTrue(web_i18n.english())

    def test_catalogs_are_complete(self):
        self.assertEqual(web_i18n.check(), [])


if __name__ == "__main__":
    unittest.main()
