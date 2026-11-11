"""Tests for setup.py's parallel ranged downloads (0.2.0): a local threaded HTTP server that supports Range stands in
for Hugging Face's CDN.  Covers the parallel path, a resume after a kill and after a failed round, a server without Range,
a piece that fails once, a signed address that runs out, the SHA-256 pin, STRATA_DL_CONNECTIONS=1 and the progress line.

    python -m unittest tools.test_setup_download
"""
from __future__ import annotations

import contextlib
import hashlib
import http.server
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import strata_mcp  # noqa: E402

CHUNK = 64 << 10                                       # the tests' piece size (setup uses 32 MiB)
DATA = bytes((i * 7 + (i >> 8)) & 0xFF for i in range(CHUNK * 40 + 12345))       # 40 whole pieces and a short last one
NAME = "t.gguf"


class Server:
    """A threaded HTTP server for one file.  `faults(path, start, end, nth)` may answer a request with an error."""

    def __init__(self, data=DATA, ranges=True, length=True, redirect=False, delay=0.002, faults=None):
        self.data, self.ranges, self.length, self.redirect, self.delay = data, ranges, length, redirect, delay
        self.faults = faults
        self.lock = threading.Lock()
        self.gets, self.heads, self.redirects = [], 0, 0           # gets: (start, end) of every ranged GET
        self.active = self.max_active = 0
        self.nth = {}
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_HEAD(self):
                outer.handle(self, head=True)

            def do_GET(self):
                outer.handle(self, head=False)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/{NAME}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    def handle(self, h, head):
        path = h.path.split("?")[0]
        if self.redirect and path == "/" + NAME:
            with self.lock:
                self.redirects += 1
            h.send_response(302)
            h.send_header("Location", f"http://127.0.0.1:{self.port}/cdn/{NAME}?Expires={int(time.time()) + 3600}&Signature=x")
            h.send_header("Content-Length", "0")
            h.end_headers()
            return
        size = len(self.data)
        rng = h.headers.get("Range")
        start, end = 0, size - 1
        if rng and self.ranges:
            m = re.match(r"bytes=(\d+)-(\d*)$", rng)
            start = int(m.group(1))
            end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
        if head:
            with self.lock:
                self.heads += 1
            h.send_response(200)
            if self.length:
                h.send_header("Content-Length", str(size))
            h.send_header("Content-Type", "application/octet-stream")
            h.end_headers()
            return
        with self.lock:
            self.gets.append((start, end) if rng and self.ranges else (0, size - 1))
            key = (start, end)
            self.nth[key] = n = self.nth.get(key, 0) + 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            fault = self.faults(path, start, end, n) if self.faults else None
            if fault and fault[0] == "status":
                h.send_response(fault[1])
                h.send_header("Content-Length", "0")
                h.end_headers()
                return
            body = self.data[start:end + 1]
            if rng and self.ranges:
                h.send_response(206)
                h.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                h.send_response(200)
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            cut = fault[1] if fault and fault[0] == "cut" else None
            try:
                for i in range(0, len(body), 16384):
                    if cut is not None and i >= cut:
                        h.connection.shutdown(2)           # the connection drops in the middle of the body
                        return
                    h.wfile.write(body[i:i + 16384])
                    if self.delay:
                        time.sleep(self.delay)
            except OSError:
                pass
        finally:
            with self.lock:
                self.active -= 1


def pin_of(data):
    return len(data), hashlib.sha256(data).hexdigest()


@contextlib.contextmanager
def small_pieces(**extra):
    """Setup's piece size, size floor and waits scaled down to the test file."""
    consts = dict(DL_CHUNK=CHUNK, DL_RUN=1, DL_MIN_PARALLEL=0, DL_BACKOFF=0.0, DL_SAVE_EVERY=0.05, DL_CONNECTIONS=4)
    consts.update(extra)
    with contextlib.ExitStack() as st:
        for k, v in consts.items():
            st.enter_context(mock.patch.object(setup, k, v))
        st.enter_context(mock.patch.object(setup.time, "sleep", lambda s: None))
        env = {k: v for k, v in os.environ.items() if k not in ("STRATA_DL_CONNECTIONS", "STRATA_ALLOW_UNPINNED")}
        env["STRATA_SOURCE"] = "huggingface"
        st.enter_context(mock.patch.dict(os.environ, env, clear=True))
        yield


def run(server, d, pin="auto", name=NAME):
    """setup.download() of the server's file into folder d; returns (exit code or None, stdout)."""
    dst = Path(d) / name
    want = pin_of(DATA) if pin == "auto" else pin
    out = io.StringIO()
    code = None
    with mock.patch.object(setup, "file_pin", lambda u: want), contextlib.redirect_stdout(out):
        try:
            setup.download(server.url, dst, "the model")
        except SystemExit as e:
            code = e.code if e.code is not None else 0
    return code, out.getvalue()


def asked(srv):
    """The pieces (indexes) the server was asked for, one entry per request; the one-byte probe is left out."""
    return [s // CHUNK for s, e in srv.gets if (s, e) != (0, 0)]


def files(d):
    return sorted(p.name for p in Path(d).iterdir())


class ParallelDownload(unittest.TestCase):
    def test_a_big_file_comes_over_several_connections(self):
        with small_pieces(), Server(redirect=True) as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(files(d), [NAME, NAME + ".done"])             # no .part, no piece list left behind
            self.assertIn("sha256 " + hashlib.sha256(DATA).hexdigest(), (Path(d) / (NAME + ".done")).read_text())
            self.assertGreater(srv.max_active, 1)
            self.assertLessEqual(srv.max_active, 4)
            self.assertEqual(sorted(asked(srv)), list(range(41)))           # every piece once, none twice
            self.assertEqual(srv.redirects, 2)                              # HEAD and the one probe: the pieces use the CDN address

    def test_the_redirect_is_resolved_once(self):
        with small_pieces(), Server(redirect=True) as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertLessEqual(srv.redirects, 2)

    def test_a_small_file_is_one_stream(self):
        with small_pieces(DL_MIN_PARALLEL=len(DATA) + 1), Server() as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(srv.max_active, 1)
            self.assertEqual(len(srv.gets), 1)

    def test_one_connection_variable_restores_one_stream(self):
        with small_pieces(), Server() as srv, tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"STRATA_DL_CONNECTIONS": "1"}):
                code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(srv.max_active, 1)
            self.assertEqual(srv.gets, [(0, len(DATA) - 1)])                # one request for the whole file

    def test_connection_count_variable(self):
        with small_pieces():
            self.assertEqual(setup.dl_connections(), 4)
            for raw, want in (("16", 16), ("0", 1), ("-3", 1), ("99", 32), ("x", 4), ("", 4)):
                with mock.patch.dict(os.environ, {"STRATA_DL_CONNECTIONS": raw}):
                    self.assertEqual(setup.dl_connections(), want, raw)


class NoRangeSupport(unittest.TestCase):
    def test_a_server_that_ignores_range_is_read_in_one_stream(self):
        with small_pieces(), Server(ranges=False) as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(srv.max_active, 1)
            self.assertEqual(files(d), [NAME, NAME + ".done"])

    def test_a_server_without_any_length_is_read_in_one_stream(self):
        with small_pieces(), Server(length=False, ranges=False) as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d, pin=None)                                # no length, so nothing to check a pin against first
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(srv.max_active, 1)

    def test_a_head_without_a_length_still_gets_the_connections(self):
        """ModelScope's HEAD has no Content-Length (#1915): the length comes from a one-byte ranged GET, not 'unknown'."""
        with small_pieces(), Server(length=False) as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertGreater(srv.max_active, 1)
            self.assertEqual(sorted(asked(srv)), list(range(41)))
            self.assertEqual(files(d), [NAME, NAME + ".done"])

    def test_range_stopping_in_the_middle_falls_back_to_the_unbroken_start(self):
        def faults(path, start, end, nth):
            return None

        with small_pieces(), Server(faults=faults) as srv, tempfile.TemporaryDirectory() as d:
            real = setup._fetch_run
            calls = []

            def fetch(src, st, f, run):
                calls.append(run)
                if len(calls) == 6:
                    raise setup._NoRange("HTTP 200")
                return real(src, st, f, run)

            with mock.patch.object(setup, "_fetch_run", fetch):
                code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)           # the pin check passed: no gaps were kept
            self.assertEqual(files(d), [NAME, NAME + ".done"])


class Resume(unittest.TestCase):
    def test_a_failed_round_keeps_the_finished_pieces(self):
        broken = {"on": True}

        def faults(path, start, end, nth):
            if broken["on"] and start >= 20 * CHUNK:
                return ("status", 500)
            return None

        with small_pieces(DL_ROUNDS=1, DL_CHUNK_TRIES=1), Server(faults=faults) as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertEqual(code, 1)                                       # could not finish: setup stops with a message
            self.assertIn("could not finish downloading", out)
            part = Path(d) / (NAME + ".part")
            side = Path(d) / (NAME + ".part.chunks")
            self.assertTrue(part.exists() and side.exists())
            saved = json.loads(side.read_text())
            done = {i for a, b in saved["done"] for i in range(a, b)}
            self.assertTrue(done and all(i < 20 for i in done), done)
            srv.gets.clear()
            broken["on"] = False
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            again = set(asked(srv))
            self.assertFalse(again & done, "pieces that were finished are fetched again")
            self.assertEqual(again | done, set(range(41)))
            self.assertEqual(files(d), [NAME, NAME + ".done"])

    def test_a_killed_download_continues(self):
        runner = (
            "import sys; sys.path[:0] = [%r, %r]\n"
            "import setup\n"
            "setup.DL_CHUNK = %d; setup.DL_MIN_PARALLEL = 0; setup.DL_SAVE_EVERY = 0.05; setup.DL_CONNECTIONS = 4\n"
            "pin = (%d, %r)\n"
            "setup.file_pin = lambda u: pin\n"
            "setup.download(sys.argv[1], __import__('pathlib').Path(sys.argv[2]) / %r, 'the model')\n"
            % (str(ROOT), str(ROOT / "tools"), CHUNK, len(DATA), hashlib.sha256(DATA).hexdigest(), NAME))
        with small_pieces(), Server(delay=0.05) as srv, tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, STRATA_SOURCE="huggingface")
            proc = subprocess.Popen([sys.executable, "-c", runner, srv.url, d], env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                deadline = time.time() + 60
                side = Path(d) / (NAME + ".part.chunks")
                while time.time() < deadline and proc.poll() is None:
                    try:
                        if sum(b - a for a, b in json.loads(side.read_text())["done"]) >= 12:
                            break
                    except (OSError, ValueError, KeyError):
                        pass
                    time.sleep(0.05)
            finally:
                proc.kill()
                proc.wait()
            self.assertTrue((Path(d) / (NAME + ".part")).exists(), "the download finished before it was killed")
            saved = json.loads(side.read_text())
            done = {i for a, b in saved["done"] for i in range(a, b)}
            self.assertGreaterEqual(len(done), 12)
            srv.gets.clear()
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertFalse(set(asked(srv)) & done, "pieces that were finished before the kill are fetched again")

    def test_a_part_from_one_stream_continues_after_its_last_whole_piece(self):
        with small_pieces(), Server() as srv, tempfile.TemporaryDirectory() as d:
            have = CHUNK * 10 + 777                                         # an older setup's .part: bytes from the start
            (Path(d) / (NAME + ".part")).write_bytes(DATA[:have])
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(sorted(asked(srv)), list(range(10, 41)))       # pieces 0-9 were not asked for again

    def test_one_stream_continues_a_parallel_part_from_its_unbroken_start(self):
        with small_pieces(), Server() as srv, tempfile.TemporaryDirectory() as d:
            part = Path(d) / (NAME + ".part")
            junk = bytearray(DATA)
            junk[8 * CHUNK:9 * CHUNK] = b"\0" * CHUNK                       # a gap behind the later pieces
            part.write_bytes(bytes(junk[:20 * CHUNK]))
            (Path(d) / (NAME + ".part.chunks")).write_text(json.dumps(
                {"v": 1, "total": len(DATA), "chunk": CHUNK, "done": [[0, 8], [9, 20]]}))
            with mock.patch.dict(os.environ, {"STRATA_DL_CONNECTIONS": "1"}):
                code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(srv.gets, [(8 * CHUNK, len(DATA) - 1)])        # resumed after piece 7, the gap refetched
            self.assertEqual(files(d), [NAME, NAME + ".done"])

    def test_a_piece_list_of_another_file_starts_over(self):
        with small_pieces(), Server() as srv, tempfile.TemporaryDirectory() as d:
            (Path(d) / (NAME + ".part")).write_bytes(b"x" * (CHUNK * 3))
            (Path(d) / (NAME + ".part.chunks")).write_text(json.dumps(
                {"v": 1, "total": 999, "chunk": CHUNK, "done": [[0, 3]]}))
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)


class Failures(unittest.TestCase):
    def test_a_piece_that_fails_once_is_tried_again(self):
        def faults(path, start, end, nth):
            if start == 5 * CHUNK and nth == 1:
                return ("status", 500)
            if start == 9 * CHUNK and nth == 1:
                return ("cut", CHUNK // 2)                                  # drops half way: continues from where it was
            return None

        with small_pieces(), Server(faults=faults) as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(sum(1 for s, _ in srv.gets if s == 5 * CHUNK), 2)
            self.assertEqual(sum(1 for s, _ in srv.gets if s == 9 * CHUNK), 1)
            resumed = [s for s, _ in srv.gets if 9 * CHUNK < s < 10 * CHUNK]
            self.assertEqual(len(resumed), 1)                               # the rest of the piece, not all of it again
            self.assertGreater(resumed[0], 9 * CHUNK)

    def test_a_signed_address_that_ran_out_is_asked_for_again(self):
        def faults(path, start, end, nth):
            if path.startswith("/cdn/") and nth == 1 and start in (4 * CHUNK, 11 * CHUNK):
                return ("status", 403)
            return None

        with small_pieces(), Server(redirect=True, faults=faults) as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertGreater(srv.redirects, 2)                            # HEAD + first probe + a new address after the 403

    def test_a_pinned_hash_mismatch_deletes_the_file_and_stops(self):
        wrong = bytes(reversed(DATA))
        with small_pieces(), Server() as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d, pin=pin_of(wrong))
            self.assertEqual(code, 1)
            self.assertEqual(files(d), [])                                  # no file, no .done, no .part, no piece list

    def test_a_pinned_hash_mismatch_with_the_same_size(self):
        other = bytes((b ^ 0x55) for b in DATA)
        with small_pieces(), Server(data=other) as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertEqual(code, 1)
            self.assertIn("wrong SHA-256", out)
            self.assertEqual(files(d), [])

    def test_a_full_disk_stops_with_a_message(self):
        with small_pieces(), Server() as srv, tempfile.TemporaryDirectory() as d:
            real = setup._piece_worker

            def worker(src, st, todo, part):
                st.fatal = setup._DiskError("No space left on device")
                st.stop.set()

            with mock.patch.object(setup, "_piece_worker", worker):
                code, out = run(srv, d)
            self.assertEqual(code, 1)
            self.assertIn("cannot write", out)

    def test_the_file_is_not_preallocated(self):
        """The app counts a .part's size as the bytes done, so it must not jump to the full size at the start."""
        seen = []

        def faults(path, start, end, nth):
            if start == 30 * CHUNK:
                p = Path(self.d) / (NAME + ".part")
                seen.append(p.stat().st_size if p.exists() else None)
            return None

        with small_pieces(DL_CONNECTIONS=2), Server(faults=faults, delay=0.01) as srv, tempfile.TemporaryDirectory() as d:
            self.d = d
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertTrue(seen and seen[0] is not None and seen[0] <= 32 * CHUNK, seen)


class Progress(unittest.TestCase):
    def test_the_progress_line_is_the_one_the_app_reads(self):
        with small_pieces(), Server() as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            lines = [x for x in out.replace("\r", "\n").split("\n") if x.strip()]
            bar = [x for x in lines if "GB (" in x]
            self.assertTrue(bar)
            for x in bar:
                self.assertRegex(x, r"^  the model: +[\d.]+ / [\d.]+ GB \(\d+%\) +$")
            parsed = strata_mcp.install_progress(bar)["download"]
            self.assertEqual(parsed["file"], "the model")
            self.assertEqual(parsed["percent"], 100)
            self.assertEqual(parsed["total_gb"], round(len(DATA) / 1e9, 2))

    def test_the_format_matches_the_one_stream_path(self):
        with small_pieces(), Server() as srv, tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            _, par = run(srv, d1)
            with mock.patch.dict(os.environ, {"STRATA_DL_CONNECTIONS": "1"}):
                _, one = run(srv, d2)
            def shape(out):
                return {re.sub(r"[\d.]+", "N", x) for x in out.replace("\r", "\n").split("\n") if "GB (" in x}
            self.assertEqual(shape(par), shape(one))


class SignedAddress(unittest.TestCase):
    def test_expiry_of_a_cdn_address(self):
        now = time.time()
        self.assertAlmostEqual(setup._url_ttl(f"https://cdn.example/f?Expires={int(now) + 600}&Signature=x"), 600, delta=3)
        self.assertIsNone(setup._url_ttl("https://cdn.example/f"))
        self.assertIsNone(setup._url_ttl(f"https://cdn.example/f?Expires={int(now) - 5}"))      # already out, or a wrong clock
        self.assertIsNone(setup._url_ttl("https://cdn.example/f?Expires=soon"))
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
        self.assertAlmostEqual(setup._url_ttl(f"https://cdn.example/f?X-Amz-Date={stamp}&X-Amz-Expires=900"), 900, delta=3)


if __name__ == "__main__":
    unittest.main()


class Runs(unittest.TestCase):
    """One request asks for several neighbouring pieces (DL_RUN): fewer connections to open, the same piece list."""

    def test_a_request_covers_several_pieces(self):
        with small_pieces(DL_RUN=4), Server() as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(files(d), [NAME, NAME + ".done"])
            spans = [(s, e) for s, e in srv.gets if (s, e) != (0, 0)]
            self.assertLess(len(spans), 41)                                 # fewer requests than pieces
            self.assertTrue(any(e - s + 1 > CHUNK for s, e in spans), spans)
            self.assertTrue(all(e - s + 1 <= 4 * CHUNK for s, e in spans), spans)
            covered = sorted(i for s, e in spans for i in range(s // CHUNK, e // CHUNK + 1))
            self.assertEqual(covered, list(range(41)))                       # every piece once, none twice

    def test_the_last_pieces_are_asked_for_one_by_one(self):
        st = setup._Pieces(CHUNK * 100, set(), conns=4)
        todo = list(range(100))
        self.assertEqual(setup._take_run(st, todo), [0, 1, 2, 3])
        with mock.patch.object(setup, "DL_RUN", 4):
            self.assertEqual(setup._take_run(st, list(range(60, 100))), [60, 61, 62, 63])
            self.assertEqual(setup._take_run(st, list(range(93, 100))), [93])      # fewer than two rounds left
            self.assertEqual(setup._take_run(st, [5, 9, 10]), [5])                  # only neighbours go together

    def test_a_broken_run_continues_from_the_byte_it_reached(self):
        cuts = []

        def faults(path, start, end, nth):
            if end - start + 1 > 2 * CHUNK and not cuts:
                cuts.append(start)
                return ("cut", int(2.5 * CHUNK))                           # drops inside the third piece of the run
            return None

        with small_pieces(DL_RUN=4, DL_CONNECTIONS=2), Server(faults=faults) as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(len(cuts), 1)
            first = cuts[0]
            again = [s for s, e in srv.gets if first < s < first + 4 * CHUNK]
            self.assertEqual(len(again), 1)                                  # one more request: for the rest
            self.assertGreaterEqual(again[0], first + 2 * CHUNK)              # the two whole pieces are not fetched again

    def test_a_connection_that_keeps_moving_is_not_given_up_on(self):
        """Seven breaks in a row (more than DL_CHUNK_TRIES) that each carried bytes forward do not end the request."""
        count = [0]

        def faults(path, start, end, nth):
            if start < 4 * CHUNK and end - start + 1 > CHUNK // 2 and count[0] < 7:
                count[0] += 1
                return ("cut", CHUNK // 3)
            return None

        with small_pieces(DL_RUN=4, DL_CONNECTIONS=2, DL_ROUNDS=1), Server(faults=faults) as srv,                 tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(count[0], 7)

    def test_a_request_without_progress_still_gives_up(self):
        def faults(path, start, end, nth):
            if start >= 4 * CHUNK:
                return ("status", 500)
            return None

        with small_pieces(DL_RUN=4, DL_ROUNDS=1, DL_CHUNK_TRIES=2), Server(faults=faults) as srv,                 tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertEqual(code, 1)
            self.assertIn("could not finish downloading", out)

    def test_the_default_is_many_connections_and_runs(self):
        self.assertGreaterEqual(setup.DL_CONNECTIONS, 8)
        self.assertGreaterEqual(setup.DL_RUN, 2)
        self.assertEqual(setup.DL_CHUNK, 32 << 20)                           # the piece list on disk names this size: resumes need it
