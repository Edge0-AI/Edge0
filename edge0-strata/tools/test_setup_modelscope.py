"""ModelScope as a download source (--source / STRATA_SOURCE): which source is used, the URL mapping, and that a file
from ModelScope is checked against the SHA-256 ModelScope publishes (a wrong one is deleted).  No network: a local
HTTP server plays ModelScope.

    python -m unittest tools.test_setup_modelscope
"""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
PATH = "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"
HF_URL = setup.hf(REPO) + PATH


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


class Base(unittest.TestCase):
    def setUp(self):
        for name in ("say", "ok", "warn"):
            p = mock.patch.object(setup, name, lambda *a, **k: None)
            p.start()
            self.addCleanup(p.stop)
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        for k in ("STRATA_SOURCE", "HF_ENDPOINT", "MODELSCOPE_ENDPOINT"):
            os.environ.pop(k, None)
        setup._sources.clear()
        setup._ms_files.clear()
        self.addCleanup(setup._sources.clear)


class Source(Base):
    def test_explicit(self):
        for value, want in (("modelscope", "modelscope"), ("ms", "modelscope"), ("huggingface", "huggingface"),
                            ("hf", "huggingface")):
            setup._sources.clear()
            os.environ["STRATA_SOURCE"] = value
            self.assertEqual(setup.model_source(), want, value)

    def test_auto_is_huggingface_and_never_asks_the_network(self):
        """#908: setup does not switch hosts by itself, whatever answers."""
        with mock.patch.object(setup, "reachable", side_effect=AssertionError("no probe")):
            self.assertEqual(setup.model_source(), "huggingface")

    def test_a_failed_download_recommends_modelscope(self):
        self.assertIn("--source modelscope", setup.source_hint(HF_URL))
        self.assertEqual(setup.source_hint(setup.LLAMA_CPP_ZIP), "")

    def test_hf_endpoint_is_a_choice(self):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        with mock.patch.object(setup, "reachable", return_value=True) as r:
            self.assertEqual(setup.model_source(), "huggingface")
            r.assert_not_called()

    def test_mapping(self):
        self.assertEqual(setup.ms_file(HF_URL), (REPO, PATH))
        self.assertEqual(setup.ms_url(REPO, PATH),
                         "https://www.modelscope.cn/models/" + REPO + "/resolve/master/" + PATH)
        self.assertIsNone(setup.ms_file("https://huggingface.co/someone/other/resolve/main/x.gguf"))
        self.assertIsNone(setup.ms_file(setup.LLAMA_CPP_ZIP))


class Download(Base):
    """download() through a local server laid out as ModelScope: /models/<repo>/resolve/master/<path> and
    /api/v1/models/<repo>/repo/files (the query string is ignored by the test server)."""

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.www = Path(self.tmp.name) / "www"
        self.data = os.urandom(300_000)
        f = self.www / "models" / REPO / "resolve" / "master" / PATH
        f.parent.mkdir(parents=True)
        f.write_bytes(self.data)
        self.server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), lambda *a, **k: Quiet(*a, directory=str(self.www), **k))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        os.environ["MODELSCOPE_ENDPOINT"] = "http://127.0.0.1:%d" % self.server.server_address[1]
        os.environ["STRATA_SOURCE"] = "modelscope"
        self.dst = Path(self.tmp.name) / "out" / Path(PATH).name
        # a file with no pin of its own (the hash ModelScope publishes is the check); the pinned case is below
        pins = {**setup.REPO_PINS, REPO: {k: v for k, v in setup.REPO_PINS[REPO].items() if k != Path(PATH).name}}
        patch = mock.patch.object(setup, "REPO_PINS", pins)
        patch.start()
        self.addCleanup(patch.stop)

    def api(self, sha):
        p = self.www / "api" / "v1" / "models" / REPO / "repo" / "files"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"Data": {"Files": [{"Path": PATH, "Size": len(self.data), "Sha256": sha}]}}))

    def test_right_hash_is_kept(self):
        sha = hashlib.sha256(self.data).hexdigest()
        self.api(sha)
        setup.download(HF_URL, self.dst)
        self.assertEqual(self.dst.read_bytes(), self.data)
        self.assertIn("sha256 " + sha, self.dst.with_name(self.dst.name + ".done").read_text())

    def test_wrong_hash_is_deleted(self):
        self.api("0" * 64)
        with mock.patch.object(setup, "say"), self.assertRaises(SystemExit):
            setup.download(HF_URL, self.dst)
        self.assertFalse(self.dst.exists())
        self.assertFalse(self.dst.with_name(self.dst.name + ".done").exists())

    def test_no_hash_published(self):
        setup.download(HF_URL, self.dst)              # no API file: the download is kept, as from Hugging Face
        self.assertEqual(self.dst.read_bytes(), self.data)
        self.assertTrue(self.dst.with_name(self.dst.name + ".done").exists())

    def test_the_pin_decides_over_what_modelscope_publishes(self):
        """A pinned file is checked against its fixed SHA-256 whatever the host says: ModelScope serves current files."""
        sha = hashlib.sha256(self.data).hexdigest()
        self.api(sha)                                  # ModelScope vouches for its own copy ...
        pins = {**setup.REPO_PINS, REPO: {**setup.REPO_PINS[REPO], Path(PATH).name: (len(self.data), "1" * 64)}}
        with mock.patch.object(setup, "REPO_PINS", pins), mock.patch.object(setup, "say"),                 self.assertRaises(SystemExit):         # ... but it is not the pinned file
            setup.download(HF_URL, self.dst)
        self.assertFalse(self.dst.exists())
        pins[REPO][Path(PATH).name] = (len(self.data), sha)
        with mock.patch.object(setup, "REPO_PINS", pins):
            setup.download(HF_URL, self.dst)
        self.assertIn("sha256 " + sha, self.dst.with_name(self.dst.name + ".done").read_text())

    def test_falls_back_when_modelscope_is_silent(self):
        calls = []
        with mock.patch.object(setup, "reachable", return_value=False), \
                mock.patch.object(setup.urllib.request, "urlopen",
                                  side_effect=lambda req, timeout=0: calls.append(req.full_url) or (_ for _ in ()).throw(
                                      OSError("offline"))), \
                mock.patch.object(setup.time, "sleep"), self.assertRaises(SystemExit):
            setup.download(HF_URL, self.dst)
        self.assertTrue(calls and all(u == HF_URL for u in calls), calls[:2])


if __name__ == "__main__":
    unittest.main()
