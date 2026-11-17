"""Signed update feeds and their relative downloads use the URL path, not its query."""

from __future__ import annotations

import hashlib
import http.server
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.parse
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strata_control import feed, signing, transfer, updater  # noqa: E402


class FeedURLs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = signing.generate_key()
        cls.trusted = dict([signing.parse_public(signing.public_part(cls.key))])
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("VERSION", "9.9.9\n")
        cls.archive = archive.getvalue()
        cls.payload = json.dumps({
            "schema_version": 1, "sequence": 1, "channel": "stable",
            "generated_at": "2026-01-01T00:00:00Z",
            "control": {"version": "9.9.9", "packages": [{
                "platform": "linux-x86_64", "format": "zip", "url": "control.zip",
                "size": len(cls.archive), "installed_bytes": 6,
                "sha256": hashlib.sha256(cls.archive).hexdigest(),
            }]},
        }).encode()
        cls.signature = json.dumps(signing.sign_bytes(cls.key, cls.payload)).encode()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.requests = []
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                owner.requests.append(self.path)
                parts = urllib.parse.urlsplit(self.path)
                name = parts.path.rsplit("/", 1)[-1]
                if parts.path.startswith("/release/") and name in (
                    "strata-feed.json", "strata-feed-linux-aarch64.json", "named.json"
                ) and parts.query in ("", "token=a/b"):
                    body = owner.payload
                elif parts.path.startswith("/release/") and name in (
                    "strata-feed.json.sig", "strata-feed-linux-aarch64.json.sig", "named.json.sig"
                ) and parts.query in ("", "token=a/b"):
                    body = owner.signature
                elif parts.path == "/release/control.zip" and not parts.query:
                    body = owner.archive
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(worker.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.origin = f"http://127.0.0.1:{server.server_port}"

    def test_signed_feed_paths_preserve_query_and_ignore_fragment(self):
        cases = (
            ("/release/named.json?token=a/b", "named.json"),
            ("/release/named.json#notes/path", "named.json"),
            ("/release?token=a/b", feed.feed_name()),
            ("/release/?token=a/b#notes/path", feed.feed_name()),
            ("/release/named.json", "named.json"),
            ("/release/", feed.feed_name()),
        )
        for suffix, name in cases:
            with self.subTest(source=suffix):
                self.requests.clear()
                loaded = feed.fetch(self.origin + suffix, self.trusted)
                self.assertEqual(loaded.key_id, self.key["key_id"])
                query = "?token=a/b" if "?token=" in suffix else ""
                self.assertEqual(self.requests, [f"/release/{name}{query}", f"/release/{name}.sig{query}"])

    def test_relative_artifact_uses_feed_directory(self):
        for suffix in ("?token=a/b", "#notes/path", "?token=a/b#notes/path", ""):
            with self.subTest(suffix=suffix):
                self.requests.clear()
                loaded = feed.verify_and_parse(self.payload, self.signature,
                    source=self.origin + "/release/named.json" + suffix, trusted=self.trusted)
                package = loaded.control[0]
                self.assertEqual(loaded.url(package.url), self.origin + "/release/control.zip")
                ctx = updater.Context(components=self.root / str(len(suffix)), source=loaded.source,
                    trusted=self.trusted, hardware={"platform": "linux-x86_64", "backend": "cpu"})
                path = updater._download(ctx, "control", package.url, package.size,
                    package.sha256, package.label, loaded)
                self.assertEqual(path.read_bytes(), self.archive)
                self.assertEqual(self.requests, ["/release/control.zip"])

    def test_real_cli_checks_query_feed(self):
        keys = self.root / "keys.json"
        keys.write_text(json.dumps({"keys": [signing.public_part(self.key)]}))
        components = self.root / "components"
        env = {**os.environ, "STRATA_UPDATE_TRUSTED_KEYS": str(keys),
               "STRATA_COMPONENTS_DIR": str(components)}
        for suffix in ("/release/named.json?token=a/b", "/release/named.json"):
            with self.subTest(source=suffix):
                result = subprocess.run([sys.executable, str(ROOT / "strata.py"), "update", "--check",
                    "--feed", self.origin + suffix, "--components-dir", str(components), "--output", "json"],
                    capture_output=True, text=True, env=env, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads(result.stdout)
                self.assertTrue(report["ok"])
                self.assertEqual(report["feed"]["key_id"], self.key["key_id"])

    def test_local_feed_signature_and_download_are_unchanged(self):
        folder = self.root / "offline"
        folder.mkdir()
        (folder / feed.FEED_NAME).write_bytes(self.payload)
        (folder / (feed.FEED_NAME + feed.SIGNATURE_SUFFIX)).write_bytes(self.signature)
        (folder / "control.zip").write_bytes(self.archive)
        for source in (str(folder), str(folder / feed.FEED_NAME), (folder / feed.FEED_NAME).as_uri()):
            with self.subTest(source=source):
                loaded = feed.fetch(source, self.trusted)
                self.assertEqual(Path(loaded.url("control.zip")).read_bytes(), self.archive)

    def test_signature_hash_and_reference_guards_remain_enforced(self):
        source = self.origin + "/release/named.json?token=a/b"
        with self.assertRaises(feed.FeedError):
            feed.verify_and_parse(self.payload + b" ", self.signature, source=source, trusted=self.trusted)
        with self.assertRaises(transfer.ChecksumMismatch):
            transfer.download(self.origin + "/release/control.zip", self.root / "bad.part",
                size=len(self.archive), sha256="0" * 64)
        for ref in ("../outside.zip", "http://example.com/package.zip"):
            with self.subTest(ref=ref), self.assertRaises(transfer.TransferError):
                transfer.resolve_ref(transfer.base_of(source), ref, environ={})


if __name__ == "__main__":
    unittest.main()
