"""Shared pieces for the update tests: stub engine packages, signed archives, a feed folder and a local HTTP server that
supports Range requests and can cut a transfer short. Nothing here touches the network beyond 127.0.0.1."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))

import feed_tool  # noqa: E402
import package_engine  # noqa: E402
from strata_control import components, signing  # noqa: E402

WINDOWS = os.name == "nt"

GOOD_ENGINE = (
    "import sys, os\nfrom pathlib import Path\n"
    "Path(os.environ['FAKE_ENGINE_MARK']).write_text(os.path.abspath(__file__) + '\\n' + ' '.join(sys.argv[1:]))\n"
    "if '--version' in sys.argv:\n    print('strata-engine stub'); sys.exit(0)\n"
    "print('READY 4096 stop', flush=True)\n"
    "for line in sys.stdin:\n"
    "    if line.strip() == 'QUIT': break\n"
    "    if line.startswith(('GEN ', 'BGEN ')):\n"
    "        slot = int(line.split()[1]) if line.startswith('BGEN ') else None\n"
    "        print('T 97', flush=True)\n"
    "        print('DONE 1 1 1.0 1.0 length 0 0 0', flush=True)\n"
    "        if slot is not None: print(f'BADM {slot} 0', flush=True)\n")
BAD_ENGINE = "import sys\nif '--version' in sys.argv:\n    sys.exit(0)\nsys.exit(3)\n"      # starts, then dies before READY


def make_key() -> dict:
    return signing.generate_key()


def trusted_of(key: dict) -> dict:
    return {key["key_id"]: base64.b64decode(key["public"])}


def build_package(parent: Path, version: str, *, engine_source: str = GOOD_ENGINE, key: dict | None = None,
                  backend: str = "cuda", variant: str = "cuda13", build_id: str | None = None, archs=("sm_86", "compute_89"),
                  min_driver=None, padding: int = 0) -> Path:
    """A schema-2 package made by the real packager (a stub engine behind a manifest-declared entrypoint). `padding` bytes of
    an incompressible 'library' make the archive big enough for a transfer to be cut part-way."""
    src = parent / f"src-{version}-{variant}"
    src.mkdir(parents=True)
    (src / "engine_main.py").write_text(engine_source, encoding="utf-8")
    if WINDOWS:
        entry = "strata.cmd"
        (src / entry).write_bytes(f'@echo off\r\n"{sys.executable}" "%~dp0engine_main.py" %*\r\n'.encode())
        library = "cudart64_13.dll"
    else:
        entry = "strata"
        (src / entry).write_text(f'#!/bin/sh\nexec "{sys.executable}" "$(dirname "$0")/engine_main.py" "$@"\n')
        (src / entry).chmod(0o755)
        library = "libcudart.so.13"
    (src / library).write_bytes(os.urandom(padding) if padding else b"a runtime library the stub never loads " * 50)
    out = parent / "built"
    out.mkdir(exist_ok=True)
    folder = package_engine.build(src, backend, version, components.host_platform(), out, entry=entry, archs=tuple(archs),
                                  min_driver=min_driver or {"nvidia": "580.65"}, build_id=build_id or f"stub-{version}",
                                  variant=variant, include=["engine_main.py"])
    if key is not None:
        feed_tool.sign_package(folder, key)
    return folder


def minisign_pair() -> tuple[str, bytes]:
    """(Tauri-style public key value, 32-byte seed): a TEST minisign key, ED25519 with the BLAKE2b-prehash 'ED' mode."""
    seed = os.urandom(32)
    public = signing.public_from_seed(seed)
    key_id = os.urandom(8)
    text = "untrusted comment: minisign public key: TEST\n" + base64.b64encode(b"Ed" + key_id + public).decode() + "\n"
    return base64.b64encode(text.encode()).decode(), key_id + seed


def minisign_sign(blob: bytes, key_id_and_seed: bytes, comment: str = "timestamp:1") -> str:
    key_id, seed = key_id_and_seed[:8], key_id_and_seed[8:]
    digest = hashlib.blake2b(blob, digest_size=64).digest()
    sig = signing.sign_raw(seed, digest)
    trusted = comment.encode()
    global_sig = signing.sign_raw(seed, sig + trusted)
    text = ("untrusted comment: signature from minisign secret key\n" + base64.b64encode(b"ED" + key_id + sig).decode() + "\n"
            f"trusted comment: {comment}\n" + base64.b64encode(global_sig).decode() + "\n")
    return base64.b64encode(text.encode()).decode()


class FeedServer:
    """Serves `root` on 127.0.0.1 with Range support. `cut` = {path-suffix: bytes}: the first response for that path stops after
    that many bytes (once), closing the connection. `requests` records (method, path, range)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.requests: list[tuple[str, str, str | None]] = []
        self.cut: dict[str, int] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                rng = self.headers.get("Range")
                outer.requests.append(("GET", self.path, rng))
                target = (outer.root / self.path.lstrip("/")).resolve()
                if outer.root.resolve() not in target.parents or not target.is_file():
                    self.send_error(404)
                    return
                data = target.read_bytes()
                start, status = 0, 200
                if rng:
                    match = re.fullmatch(r"bytes=(\d+)-", rng)
                    if match:
                        start, status = int(match.group(1)), 206
                body = data[start:]
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                self.end_headers()
                limit = next((n for suffix, n in list(outer.cut.items()) if self.path.endswith(suffix)), None)
                if limit is not None:
                    del outer.cut[next(s for s in outer.cut if self.path.endswith(s))]
                    self.wfile.write(body[:limit])
                    self.wfile.flush()
                    self.connection.close()
                    return
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        return False
