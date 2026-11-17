"""The web page's chat history: saved chats in the browser (#1787), edit / regenerate / rewind with an Undo (#2615), the rewind
choice between "here" and a branch (#3823), and sending while an answer is written (#4004). One design, opt-in, off by default.

The logic runs in Node (serve/test_web_history.js); what the page does in a real browser is in tools/webtest/browser_test.mjs
(run here when WEBTEST_MODULES is set). These checks read the sources for what must hold whatever the code looks like:
off is the page from before, and nothing about a chat goes to the server."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).parent / "web"


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class HistoryInNode(unittest.TestCase):
    def test_messages_titles_branches_and_the_browser_store(self):
        r = subprocess.run([shutil.which("node"), str(Path(__file__).with_name("test_web_history.js"))],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


class OffIsThePageFromBefore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (WEB / "index.html").read_text(encoding="utf-8")
        cls.app = (WEB / "app.js").read_text(encoding="utf-8")
        cls.ui = (WEB / "chats-ui.js").read_text(encoding="utf-8")
        cls.store = (WEB / "chatstore.js").read_text(encoding="utf-8")

    def test_the_three_switches_start_off(self):
        self.assertIn('chatList: false, chatEdit: false, whileBusy: "wait"', self.app)
        for el in ("s-list", "s-edit"):
            self.assertIn('aria-checked="false"', re.search(rf'<button[^>]*id="{el}"[^>]*>', self.html).group(0))
        self.assertIn('data-v="wait"', self.html)
        self.assertRegex(self.html, r'<aside class="chat-sidebar"[^>]*\bhidden>')              # no list until it is switched on

    def test_the_browser_store_is_opened_only_for_the_list(self):
        self.assertEqual(self.ui.count("chatStore.start()"), 1)
        self.assertRegex(self.ui, r"async function openChats\(fromScreen\) \{\s*await chatStore\.start\(\);")
        self.assertEqual(len(re.findall(r"\bopenChats\(", self.ui)), 2)       # its definition and the one call, in setChatList
        self.assertNotIn("indexedDB", self.app, "the page's main script never touches the store")
        self.assertIn("if (settings.chatList) await setChatList(true, false);", self.app)

    def test_with_the_list_off_one_chat_is_kept_as_before(self):
        self.assertIn('if (!listOn) { store.set("chat", H.cleanMessages(messages)); return; }', self.ui)
        self.assertIn('$("new-btn").hidden = on;', self.ui)                  # only the list's own New chat shows while it is on

    def test_pictures_and_files_are_kept_as_names_only(self):
        self.assertIn("chatStore.put(H.cleanChat(chat))", self.ui)
        self.assertNotIn("chatStore.put(chat)", self.ui)

    def test_nothing_about_a_chat_goes_to_the_server(self):
        for name, src in (("app.js", self.app), ("chats-ui.js", self.ui), ("chatstore.js", self.store)):
            for call in re.findall(r'fetch\(\s*([^,)]+)', src):
                if call != '"v1/chat/completions"':                    # the question itself, as ever
                    self.assertNotRegex(call, r"chat|histor", f"{name}: {call}")

    def test_the_edit_controls_wait_for_a_finished_answer(self):
        self.assertIn("const historyLocked = () => !!busy || !!panel || !!pending;", self.ui)
        self.assertIn("$(\"s-list\").disabled = $(\"s-edit\").disabled = !!busy || !!panel;", self.app)

    def test_an_interrupted_answer_keeps_what_was_written(self):
        self.assertIn("if (cut) m.interrupted = true;", self.app)
        self.assertIn("busy.cutAfterTool = true; return; }   // let the tool finish", self.app)
        self.assertIn("settlePending(cut, !m.stopped && !m.error);", self.app)

    def test_sending_while_busy_parks_nothing_unless_asked(self):
        self.assertIn('const parksWhileBusy = () => settings.whileBusy === "queue" || settings.whileBusy === "interject";', self.app)
        self.assertIn("if (!parksWhileBusy()) return;", self.app)


class NothingOnTheServer(unittest.TestCase):
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

    def test_the_scripts_are_served_and_there_is_no_chat_endpoint(self):
        for name in ("history.js", "chatstore.js", "chats-ui.js"):
            with urllib.request.urlopen(f"{self.base}/web/{name}", timeout=10) as r:
                self.assertEqual(r.status, 200)
        for path in ("/chats", "/api/chats", "/history", "/api/history"):
            with self.subTest(path=path):
                with self.assertRaises(urllib.error.HTTPError) as e:
                    urllib.request.urlopen(self.base + path, timeout=10)
                self.assertEqual(e.exception.code, 404)


@unittest.skipUnless(os.environ.get("WEBTEST_MODULES") and shutil.which("node"),
                     "set WEBTEST_MODULES (a folder with node_modules/puppeteer-core) to run the browser tests")
class InABrowser(unittest.TestCase):
    def test_history_in_chrome(self):
        for name in ("history", "list", "rewind", "branch", "regenerate", "edit answer", "while answering"):
            with self.subTest(name=name):
                r = subprocess.run([shutil.which("node"), str(ROOT / "tools" / "webtest" / "browser_test.mjs"), name],
                                   capture_output=True, text=True, timeout=300)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
