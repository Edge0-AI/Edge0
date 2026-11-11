"""0.2.2: a model download never leaves wrong bytes behind, and a file with a wrong SHA-256 is repaired, not thrown away.

A user's 39.8 GB shard arrived with the wrong SHA-256 on 0.2.1 (16 connections, several pieces per request).  These tests run
setup.download() against a local HTTP server that misbehaves in each way a server, a CDN or a disk can: a 200 for a ranged
request, a 206 for another range or another total, a body that ends early, another version of the file part-way, a changed
ETag, a .part written from another mirror, no piece list, a full disk, a raw write that takes fewer bytes than given, and
bytes that are wrong in the first download (the repair reads the file again and writes only the pieces that differ).
Pieces are 64 KiB here instead of 32 MiB, so a few MiB stand for the 40 GB."""

from __future__ import annotations

import contextlib
import hashlib
import http.server
import io
import json
import random
import re
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import setup
from strata_control import transfer

PIECE = 64 << 10
SIZE = 40 * PIECE + 12345            # 41 pieces, the last one short
DATA = random.Random(7).randbytes(SIZE)
SHA = hashlib.sha256(DATA).hexdigest()


class Behavior:
    """What the server does with request number n (1 = the first).  `plan` returns overrides for that answer."""

    def __init__(self, plan=None, data=DATA, etag='"v1"'):
        self.plan = plan or (lambda n, start, end: {})
        self.data = data
        self.etag = etag
        self.lock = threading.Lock()
        self.count = 0
        self.served = []                               # (start, end) of every ranged answer
        self.once = set()

    def next(self):
        with self.lock:
            self.count += 1
            return self.count


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"                      # one request per connection: ending a body early is a reset

    def log_message(self, *a):
        pass

    def do_HEAD(self):
        beh = self.server.beh
        self.send_response(200)
        self.send_header("Content-Length", str(len(beh.data)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        beh = self.server.beh
        n = beh.next()
        total = len(beh.data)
        m = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range") or "")
        start = int(m.group(1)) if m else 0
        end = min(int(m.group(2)), total - 1) if m and m.group(2) else total - 1
        if start >= total:
            self.send_response(416)
            self.end_headers()
            return
        o = beh.plan(n, start, end) or {}
        if o.get("status") == 404 or o.get("status") == 503:
            self.send_response(o["status"])
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        status = o.get("status", 206 if m else 200)
        b0, b1 = o.get("body", (start, end))
        body = bytearray(beh.data if status == 200 else beh.data[b0:b1 + 1])
        for off in o.get("corrupt", ()):
            lo = 0 if status == 200 else b0
            if lo <= off < lo + len(body):
                body[off - lo] ^= 0xFF
        self.send_response(status)
        self.send_header("Content-Length", str(o.get("cl", len(body))))
        if status == 206:
            self.send_header("Content-Range", o.get("cr", f"bytes {start}-{end}/{total}"))
        self.send_header("ETag", o.get("etag", beh.etag))
        self.end_headers()
        with beh.lock:
            beh.served.append((start, end))
        cut = o.get("cut")
        try:
            self.wfile.write(bytes(body[:cut] if cut is not None else body))
        except OSError:
            pass


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass


class DownloadCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dlint"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        patches = [mock.patch.object(setup, "DL_CHUNK", PIECE), mock.patch.object(setup, "DL_MIN_PARALLEL", PIECE * 4),
                   mock.patch.object(setup, "DL_BACKOFF", 0.0), mock.patch.object(setup, "DL_SAVE_EVERY", 0.0),
                   mock.patch.object(setup, "DL_ROUNDS", 3), mock.patch.object(setup.time, "sleep", lambda s: None),
                   mock.patch.object(setup, "file_pin", lambda url: (SIZE, SHA)),
                   mock.patch.dict("os.environ", {"STRATA_DL_CONNECTIONS": "4", "STRATA_SOURCE": "huggingface"})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.dst = self.tmp / "model.gguf"
        self.part = self.tmp / "model.gguf.part"

    def serve(self, beh):
        srv = Server(("127.0.0.1", 0), Handler)
        srv.beh = beh
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        self.beh = beh
        return f"http://127.0.0.1:{srv.server_address[1]}/m/model.gguf"

    def run_download(self, url, expect_exit=False):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if expect_exit:
                with self.assertRaises(SystemExit):
                    setup.download(url, self.dst)
            else:
                setup.download(url, self.dst)
        self.out = buf.getvalue()
        return self.out

    def assertGood(self):
        self.assertTrue(self.dst.is_file(), self.out)
        self.assertEqual(hashlib.sha256(self.dst.read_bytes()).hexdigest(), SHA)
        self.assertFalse(self.part.exists())
        self.assertFalse(self.tmp.joinpath("model.gguf.part.chunks").exists())


class Misbehaving(DownloadCase):
    def test_clean_download(self):
        self.run_download(self.serve(Behavior()))
        self.assertGood()

    def test_ranged_requests_answered_with_200_use_one_stream(self):
        self.run_download(self.serve(Behavior(lambda n, s, e: {"status": 200})))
        self.assertGood()

    def test_ranges_stop_being_honoured_part_way(self):
        # the probe and the first pieces are right; later requests get the whole file with a 200
        self.run_download(self.serve(Behavior(lambda n, s, e: {"status": 200} if n > 6 else {})))
        self.assertGood()

    def test_206_for_another_range_is_not_written(self):
        # Content-Range names a range that starts later than the one asked for, from the 5th request on
        self.run_download(self.serve(Behavior(lambda n, s, e: {"cr": f"bytes {s + 7}-{e}/{SIZE}", "body": (s + 7, e)}
                                              if n > 4 and s > 0 else {})))
        self.assertGood()

    def test_206_with_another_total_is_not_written(self):
        self.run_download(self.serve(Behavior(lambda n, s, e: {"cr": f"bytes {s}-{e}/{SIZE + 1}"} if n > 4 else {})))
        self.assertGood()

    def test_206_with_a_longer_body_than_asked(self):
        # Content-Length does not fit the range: tried again, then right
        self.run_download(self.serve(Behavior(lambda n, s, e: {"cl": e - s + 5} if n in (4, 5, 9) else {})))
        self.assertGood()

    def test_connection_reset_in_the_middle_of_pieces(self):
        # every 3rd answer ends after a random part of its body; the piece it broke in is written again from its start
        rnd = random.Random(3)
        self.run_download(self.serve(Behavior(lambda n, s, e: {"cut": rnd.randrange(1, max(2, e - s))} if n % 3 == 0 else {})))
        self.assertGood()

    def test_an_answer_cut_after_a_whole_number_of_blocks_still_ends_in_the_right_bytes(self):
        self.run_download(self.serve(Behavior(lambda n, s, e: {"cut": PIECE + 17} if n in (3, 4, 5, 6) else {})))
        self.assertGood()

    def test_one_answer_from_another_version_is_asked_again(self):
        self.run_download(self.serve(Behavior(lambda n, s, e: {"etag": '"v2"'} if n in (4, 7) else {})))
        self.assertGood()

    def test_another_version_for_good_stops_and_leaves_no_file(self):
        out = self.run_download(self.serve(Behavior(lambda n, s, e: {"etag": '"v2"'} if n > 3 else {})), expect_exit=True)
        self.assertFalse(self.dst.exists())

    def test_the_file_changing_between_resolves_is_noticed(self):
        src = setup._Source("http://x/y", SIZE)
        src.etag = "aaa"

        class R:
            status = 206
            headers = {"Content-Range": f"bytes 0-0/{SIZE}", "ETag": '"bbb"'}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, n):
                return b"x"

            def geturl(self):
                return "http://x/y"

        with mock.patch.object(setup, "_urlopen", lambda *a, **k: R()):
            with self.assertRaises(setup._Changed):
                src.resolve()

    def test_server_error_part_way_keeps_pieces_and_a_second_run_finishes(self):
        url = self.serve(Behavior(lambda n, s, e: {"status": 503} if n > 12 else {}))
        self.run_download(url, expect_exit=True)
        self.assertTrue(self.part.exists())
        self.assertTrue(self.tmp.joinpath("model.gguf.part.chunks").exists())
        first_run = len(self.beh.served)
        self.beh.plan = lambda n, s, e: {}                     # the same address, now answering
        self.beh.served.clear()
        self.run_download(url)
        self.assertGood()
        fetched = sum(e - s + 1 for s, e in self.beh.served if e > s)       # without the one-byte probes
        self.assertLess(fetched, SIZE, "the second run took the whole file again instead of resuming")
        self.assertGreater(first_run, 1)

    def test_two_interrupted_runs_then_a_good_one(self):
        url = self.serve(Behavior())
        for limit in (6, 6):
            self.beh.count = 0
            self.beh.plan = lambda n, s, e, limit=limit: {"status": 503} if n > limit else {}
            self.run_download(url, expect_exit=True)
        self.beh.plan = lambda n, s, e: {}
        self.run_download(url)
        self.assertGood()

    def test_a_part_written_from_another_mirror_is_dropped(self):
        url = self.serve(Behavior())
        self.part.write_bytes(b"\xaa" * (PIECE * 10))
        self.tmp.joinpath("model.gguf.part.chunks").write_text(
            json.dumps({"v": 1, "total": SIZE, "chunk": PIECE, "done": [[0, 10]]}), encoding="utf-8")
        self.tmp.joinpath("model.gguf.part.src").write_text(
            json.dumps({"url": "https://www.modelscope.cn/models/x/resolve/master/model.gguf", "etag": None}), encoding="utf-8")
        self.run_download(url)
        self.assertGood()                                      # the 0xaa pieces were not trusted

    def test_a_part_of_another_version_is_dropped(self):
        url = self.serve(Behavior())
        self.part.write_bytes(b"\xaa" * (PIECE * 10))
        self.tmp.joinpath("model.gguf.part.chunks").write_text(
            json.dumps({"v": 1, "total": SIZE, "chunk": PIECE, "done": [[0, 10]]}), encoding="utf-8")
        self.tmp.joinpath("model.gguf.part.src").write_text(json.dumps({"url": url, "etag": '"old"'}), encoding="utf-8")
        self.run_download(url)
        self.assertGood()

    def test_without_a_writable_piece_list_the_file_is_not_written_in_parts(self):
        url = self.serve(Behavior())
        with mock.patch.object(setup, "_save_pieces", return_value=False):
            self.run_download(url)
        self.assertGood()

    def test_a_stale_piece_list_never_names_bytes_the_part_lacks(self):
        self.part.write_bytes(b"\x00" * (PIECE * 3))
        self.tmp.joinpath("model.gguf.part.chunks").write_text(
            json.dumps({"v": 1, "total": SIZE, "chunk": PIECE, "done": [[0, 30]]}), encoding="utf-8")
        self.assertEqual(setup._load_pieces(self.part, SIZE), {0, 1, 2})

    def test_full_disk_stops_with_a_plain_message_and_keeps_the_part(self):
        url = self.serve(Behavior())

        def full(f, pos, data):
            raise OSError(28, "No space left on device")

        with mock.patch.object(setup, "_write_all", full):
            out = self.run_download(url, expect_exit=True)
        self.assertIn("cannot write", out)
        self.assertIn("free some disk space", out)
        self.assertFalse(self.dst.exists())

    def test_a_raw_write_that_takes_fewer_bytes_is_finished(self):
        class Slow:
            def __init__(self):
                self.buf = bytearray(100)
                self.pos = 0

            def seek(self, p):
                self.pos = p

            def write(self, v):
                n = min(7, len(v))
                self.buf[self.pos:self.pos + n] = bytes(v[:n])
                self.pos += n
                return n

        f = Slow()
        setup._write_all(f, 10, bytes(range(60)))
        self.assertEqual(bytes(f.buf[10:70]), bytes(range(60)))

    def test_a_raw_write_that_takes_nothing_is_an_error(self):
        class Zero:
            def seek(self, p):
                pass

            def write(self, v):
                return 0

        with self.assertRaises(OSError):
            setup._write_all(Zero(), 0, b"abc")

    def test_single_stream_resume_ignores_a_206_from_the_wrong_offset(self):
        url = self.serve(Behavior(lambda n, s, e: {"cr": f"bytes {s + 5}-{e}/{SIZE}", "body": (s + 5, e)} if s > 0 else {}))
        self.part.write_bytes(DATA[:PIECE * 3])
        with mock.patch.dict("os.environ", {"STRATA_DL_CONNECTIONS": "1"}):
            self.run_download(url)
        self.assertGood()

    def test_single_stream_resume_ignores_a_200(self):
        url = self.serve(Behavior(lambda n, s, e: {"status": 200}))
        self.part.write_bytes(DATA[:PIECE * 3])
        with mock.patch.dict("os.environ", {"STRATA_DL_CONNECTIONS": "1"}):
            self.run_download(url)
        self.assertGood()

    def test_finished_file_without_a_mark_is_checked_not_marked(self):
        self.dst.write_bytes(DATA[:-1] + b"\x00")
        url = self.serve(Behavior())
        self.run_download(url)
        self.assertIn("Some parts of model.gguf were damaged", self.out)
        self.assertGood()
        self.assertTrue(self.tmp.joinpath("model.gguf.done").exists())


class Repair(DownloadCase):
    def test_wrong_bytes_in_two_pieces_are_found_and_only_those_rewritten(self):
        bad = [5 * PIECE + 100, 30 * PIECE + 7]
        seen = set()

        def plan(n, s, e):
            hit = [o for o in bad if s <= o <= e and o not in seen]
            seen.update(hit)
            return {"corrupt": hit} if hit else {}

        out = self.run_download(self.serve(Behavior(plan)))
        self.assertGood()
        self.assertIn("Some parts of model.gguf were damaged; downloading only those parts again", out)
        self.assertIn("2 of 41 parts were different", out)
        self.assertTrue(self.tmp.joinpath("model.gguf.done").exists())

    def test_one_flipped_byte_in_the_last_short_piece(self):
        bad = [SIZE - 3]
        seen = set()

        def plan(n, s, e):
            hit = [o for o in bad if s <= o <= e and o not in seen]
            seen.update(hit)
            return {"corrupt": hit} if hit else {}

        out = self.run_download(self.serve(Behavior(plan)))
        self.assertGood()
        self.assertIn("1 of 41 parts were different", out)

    def test_a_repair_that_loses_the_connection_part_way_still_ends_right(self):
        bad = [3 * PIECE + 1]
        seen = set()

        def plan(n, s, e):
            hit = [o for o in bad if s <= o <= e and o not in seen]
            seen.update(hit)
            o = {"corrupt": hit} if hit else {}
            if seen and n % 4 == 0:
                o["cut"] = 5000
            return o

        self.run_download(self.serve(Behavior(plan)))
        self.assertGood()

    def test_a_server_that_keeps_serving_the_wrong_file_ends_with_the_file_deleted(self):
        wrong = bytearray(DATA)
        wrong[1000] ^= 0xFF
        url = self.serve(Behavior(data=bytes(wrong)))
        out = self.run_download(url, expect_exit=True)
        self.assertIn("wrong SHA-256", out)
        self.assertFalse(self.dst.exists())
        self.assertFalse(self.tmp.joinpath("model.gguf.done").exists())

    def test_the_repair_is_off_for_a_file_the_person_owns(self):
        self.dst.write_bytes(b"x" * SIZE)
        called = []
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                setup.verify_sha256(self.dst, SIZE, SHA, own=True, repair=lambda: called.append(1))
        self.assertEqual(called, [])
        self.assertTrue(self.dst.exists())


class UpdateTransfer(unittest.TestCase):
    """strata_control/transfer.py (update archives): a resumed answer is appended only when it starts at the byte asked for."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dlint"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_resume_check(self):
        class R:
            def __init__(self, status, cr):
                self.status = status
                self.headers = {"Content-Range": cr} if cr else {}

        self.assertTrue(transfer._resumes_at(R(206, "bytes 100-999/1000"), 100, 1000))
        self.assertFalse(transfer._resumes_at(R(206, "bytes 0-999/1000"), 100, 1000))
        self.assertFalse(transfer._resumes_at(R(206, "bytes 100-999/2000"), 100, 1000))
        self.assertFalse(transfer._resumes_at(R(206, ""), 100, 1000))
        self.assertFalse(transfer._resumes_at(R(200, "bytes 100-999/1000"), 100, 1000))

    def test_a_206_from_the_wrong_offset_restarts_the_file(self):
        data = DATA[:200000]
        beh = Behavior(lambda n, s, e: {"cr": f"bytes {s + 9}-{e}/{len(data)}", "body": (s + 9, e)} if s > 0 else {}, data=data)
        srv = Server(("127.0.0.1", 0), Handler)
        srv.beh = beh
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        part = self.tmp / "a.part"
        part.write_bytes(data[:50000])
        got = transfer.download(f"http://127.0.0.1:{srv.server_address[1]}/a", part, size=len(data),
                                sha256=hashlib.sha256(data).hexdigest())
        self.assertEqual(got.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
