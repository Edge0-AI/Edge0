"""Tests for the fixed SHA-256 and size of every model file setup downloads (REPO_PINS): every file a choice can
download has one, a wrong file is deleted and setup stops, a pinned commit that answers 404 stops the download
(STRATA_ALLOW_UNPINNED=1 is the tester's opt-out), and an installed model needs no network.  Mocked network - nothing
is downloaded.

    python -m unittest tools.test_setup_model_pins
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import re
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
HEX = re.compile(r"^[0-9a-f]{64}$")


class Response(io.BytesIO):
    def __init__(self, body=b"", status=200, length=None):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body) if length is None else length)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def quiet(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*args, **kw), out.getvalue()


def pinned_url(name):
    return setup.hf(REPO) + name


def with_pin(name, data):
    """REPO_PINS with one more file of the repository: `data`'s size and SHA-256."""
    pins = dict(setup.REPO_PINS)
    pins[REPO] = {**pins[REPO], name: (len(data), hashlib.sha256(data).hexdigest())}
    return mock.patch.object(setup, "REPO_PINS", pins)


class EveryFileIsPinned(unittest.TestCase):
    def pinned(self, url):
        pin = setup.file_pin(url)
        self.assertIsNotNone(pin, url)
        self.assertGreater(pin[0], 0)
        self.assertRegex(pin[1], HEX)

    def test_every_shard_of_every_choice(self):
        for fam_name, fam in {**setup.FAMILIES, **setup.OWN_PATH_FAMILIES}.items():
            if fam.get("ready"):                       # ready-made packs: pinned file by file (ready_files)
                continue
            for model in (setup.MODELS if fam_name in setup.FAMILIES else fam["models"]):
                if fam_name in setup.FAMILIES and fam_name not in setup.MODELS[model].get("families", ("qwen", "swift")):
                    continue
                with self.subTest(family=fam_name, model=model):
                    if fam_name in setup.FAMILIES:
                        for i in range(1, setup.model_shards(fam, model) + 1):
                            self.pinned(fam["hf"].format(q=model) + setup.model_file(fam, model, i))
                    else:
                        self.pinned(fam["hf"] + fam["file"].format(q=model))

    def test_the_picture_reader_of_every_family(self):
        for name, fam in {**setup.FAMILIES, **setup.OWN_PATH_FAMILIES}.items():
            with self.subTest(family=name):
                self.pinned(fam["mmproj_hf"] + fam["mmproj"])

    def test_the_nvfp4_packs_still_pin_their_files(self):
        for model in setup.MODELS:
            if setup.MODELS[model].get("repo"):
                for f in setup.ready_files(model):
                    self.assertRegex(f[5], HEX)

    def test_every_pin_is_well_formed_and_every_repository_is_pinned(self):
        for repo, files in setup.REPO_PINS.items():
            self.assertIn(repo, setup.HF_REVISIONS)
            for name, (size, sha) in files.items():
                with self.subTest(repo=repo, file=name):
                    self.assertIsInstance(size, int)
                    self.assertGreater(size, 1000)
                    self.assertRegex(sha, HEX)

    def test_known_values(self):
        # the Hub's LFS pointers; the picture reader is the same file the NVFP4 family already pinned
        self.assertEqual(setup.file_pin(pinned_url("mmproj-Qwen3.8-Flash-Next-BF16.gguf")),
                         setup.FAMILIES["nvfp4"]["mmproj_sha256"])
        self.assertEqual(setup.file_pin(pinned_url("Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf")),
                         (37623740192, "69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720"))
        # the n-gram table is one file for the four sizes and the Coder
        shard2 = {setup.ISTA_FILES["Qwen3.8-Flash-Next-GSQ-RCO-%s-00002-of-00002.gguf" % q] for q in
                  ("Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S")}
        self.assertEqual(len(shard2), 1)
        self.assertEqual(setup.CODER_FILES["Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf"], shard2.pop())

    def test_an_unknown_file_has_no_pin(self):
        self.assertIsNone(setup.file_pin(pinned_url("nothing.gguf")))
        self.assertIsNone(setup.file_pin("https://example.com/a/b/resolve/main/x.gguf"))


class DownloadChecksThePin(unittest.TestCase):
    def serve(self, body, length=None):
        seen = []

        def urlopen(req, timeout=None):
            seen.append(req.get_method())
            if req.get_method() == "HEAD":
                return Response(b"", length=length)
            return Response(body, status=200)

        return seen, mock.patch.object(setup.urllib.request, "urlopen", urlopen)

    def run_download(self, name, served, pinned, length=None):
        seen, patch = self.serve(served, length if length is not None else len(served))
        with tempfile.TemporaryDirectory() as d, with_pin(name, pinned), patch, \
                mock.patch.dict(setup.os.environ, {"STRATA_SOURCE": "huggingface"}):
            setup.os.environ.pop("STRATA_ALLOW_UNPINNED", None)
            dst = Path(d) / name
            try:
                _, out = quiet(setup.download, pinned_url(name), dst)
                code = None
            except SystemExit as e:
                code, out = e.code, ""
            result = (code, dst.exists(), dst.with_name(dst.name + ".done").exists(),
                      dst.with_name(dst.name + ".part").exists(),
                      dst.with_name(dst.name + ".done").read_text() if dst.with_name(dst.name + ".done").exists() else "")
        return seen, result

    def test_the_right_file_is_kept_with_its_hash_in_the_mark(self):
        data = b"the model" * 100
        seen, (code, exists, marked, part, mark_text) = self.run_download("t.gguf", data, data)
        self.assertIsNone(code)
        self.assertTrue(exists and marked and not part)
        self.assertIn("sha256 " + hashlib.sha256(data).hexdigest(), mark_text)

    def test_a_wrong_file_is_deleted_and_setup_stops(self):
        data = b"the model" * 100
        wrong = b"THE MODEL" * 100                     # the pinned size, another content
        seen, (code, exists, marked, part, _) = self.run_download("t.gguf", wrong, data)
        self.assertEqual(code, 1)
        self.assertFalse(exists or marked or part)

    def test_a_wrong_size_stops_before_the_download(self):
        data = b"the model" * 100
        seen, (code, exists, marked, part, _) = self.run_download("t.gguf", data + b"x", data)
        self.assertEqual(code, 1)
        self.assertEqual(seen, ["HEAD"])               # no GET: nothing of the wrong file was fetched
        self.assertFalse(exists or marked or part)


class NotFoundOnThePinnedCommit(unittest.TestCase):
    def urlopen(self, seen):
        def urlopen(req, timeout=None):
            seen.append((req.get_method(), req.full_url))
            if "/resolve/main/" not in req.full_url:
                raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
            return Response(b"current file")
        return urlopen

    def test_it_stops_with_a_plain_message(self):
        seen = []
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup.urllib.request, "urlopen", self.urlopen(seen)), \
                mock.patch.dict(setup.os.environ, {"STRATA_SOURCE": "huggingface"}), contextlib.redirect_stdout(out):
            setup.os.environ.pop("STRATA_ALLOW_UNPINNED", None)
            dst = Path(d) / "t.gguf"
            with self.assertRaises(SystemExit):
                setup.download(pinned_url("Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"), dst, "the model")
            self.assertFalse(dst.exists())
        self.assertIn("the pinned version of the model is no longer on Hugging Face", out.getvalue())
        self.assertIn("update Strata", out.getvalue())
        self.assertEqual(len(seen), 1)                 # no second try on main

    def test_the_opt_out_takes_the_current_file_and_says_so(self):
        seen = []
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup.urllib.request, "urlopen", self.urlopen(seen)), \
                mock.patch.dict(setup.os.environ, {"STRATA_SOURCE": "huggingface", "STRATA_ALLOW_UNPINNED": "1"}), \
                mock.patch.object(setup, "_unpinned_said", []):
            dst = Path(d) / "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
            _, out = quiet(setup.download, pinned_url("Q2_0/" + dst.name), dst)
            self.assertEqual(dst.read_bytes(), b"current file")     # not checked against the pin: it is not the pinned file
        self.assertIn("STRATA_ALLOW_UNPINNED=1", out)

    def test_the_opt_out_is_off_by_default(self):
        with mock.patch.dict(setup.os.environ, {}, clear=False):
            setup.os.environ.pop("STRATA_ALLOW_UNPINNED", None)
            self.assertFalse(setup.allow_unpinned())
            for value in ("", "0", "no"):
                setup.os.environ["STRATA_ALLOW_UNPINNED"] = value
                self.assertFalse(setup.allow_unpinned())


class InstalledModelsNeedNoNetwork(unittest.TestCase):
    """Once a model is downloaded, setup and the start do not contact Hugging Face (they run offline)."""

    def test_an_installed_file_is_not_asked_for(self):
        def urlopen(req, timeout=None):
            raise AssertionError("network used for an installed file: " + req.full_url)

        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup.urllib.request, "urlopen", urlopen), \
                mock.patch.dict(setup.os.environ, {"HF_ENDPOINT": "http://127.0.0.1:9", "STRATA_SOURCE": "auto"}):
            dst = Path(d) / "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
            dst.write_bytes(b"installed by an earlier setup")       # no hash in its mark (installed before the pins)
            setup.mark(dst)
            _, out = quiet(setup.download, pinned_url("Q2_0/" + dst.name), dst)
            self.assertIn("already downloaded", out)
            self.assertEqual(dst.read_bytes(), b"installed by an earlier setup")   # kept, not hashed, not deleted

    def test_a_dead_host_is_reported_for_a_download_only(self):
        def urlopen(req, timeout=None):
            raise urllib.error.URLError("no route to host")

        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup.urllib.request, "urlopen", urlopen), \
                mock.patch.object(setup.time, "sleep", lambda s: None), \
                mock.patch.dict(setup.os.environ, {"STRATA_SOURCE": "huggingface"}):
            with self.assertRaises(SystemExit):
                quiet(setup.download, pinned_url("Q2_0/x.gguf"), Path(d) / "x.gguf")

    def test_the_runtime_has_no_model_host_in_it(self):
        """strata launch, serve, the monitor and the API reach no model host: the names are in setup.py and the install
        tools only (and strata_control.acquisition refuses a download under STRATA_NO_ACQUIRE=1)."""
        hosts = re.compile(r"huggingface\.co|hf-mirror|HF_ENDPOINT|modelscope|hf_hub_download", re.I)
        # The harmless mentions: the model-card link a benchmark result file names (nothing opens it), and the pinned
        # DFlash2 drafter address, which only `strata drafting fetch` downloads from, behind acquisition.require (refused
        # under STRATA_NO_ACQUIRE, so a start or a server never reaches it). Dropping just those lines from the scan keeps
        # the rule exact for everything else.
        harmless = ('MODEL_CARD_URL = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next"',
                    'BASE_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"')
        found = []
        for folder in ("strata_cli", "serve", "strata_control"):
            for p in (ROOT / folder).rglob("*.py"):
                if "test" in p.name:
                    continue
                text = p.read_text(encoding="utf-8", errors="replace")
                for line in harmless:
                    text = text.replace(line, "")
                if hosts.search(text):
                    found.append(str(p.relative_to(ROOT)))
        self.assertEqual(found, [])

    def test_a_download_is_refused_while_serving(self):
        from strata_control import acquisition
        with acquisition.forbidden():
            with self.assertRaises(acquisition.AcquisitionRefused):
                setup.download(pinned_url("Q2_0/x.gguf"), Path(tempfile.gettempdir()) / "strata-never.gguf")


if __name__ == "__main__":
    unittest.main()
