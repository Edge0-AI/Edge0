"""Tests for tools/mtp_fetch.py (#327): a range read needs a 206 with the asked Content-Range, the pinned revision's
tensors are checked against their sha256 (a wrong one is fetched again; a corrupt install is found by `verify` and
setup fetches it again), and an inventory from another revision is read again.  A fake checkpoint behind a mocked
urlopen - nothing is downloaded.

    python -m unittest tools.test_mtp_fetch
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import mtp_fetch as M  # noqa: E402


def shard(tensors):
    """A safetensors file: [u64 header length][JSON header][data]."""
    header, data = {}, b""
    for name, blob in tensors:
        header[name] = {"dtype": "BF16", "shape": [len(blob) // 2], "data_offsets": [len(data), len(data) + len(blob)]}
        data += blob
    h = json.dumps(header).encode()
    return struct.pack("<Q", len(h)) + h + data


A, B, X = bytes(range(64)), bytes(range(100, 164)) * 2, b"\x07" * 48
FILES = {"model-1.safetensors": shard([("model.x", X), ("mtp.a", A)]), "model-2.safetensors": shard([("mtp.b", B)])}
INDEX = json.dumps({"weight_map": {"model.x": "model-1.safetensors", "mtp.a": "model-1.safetensors",
                                   "mtp.b": "model-2.safetensors"}}).encode()
HASHES = {"mtp.a": hashlib.sha256(A).hexdigest(), "mtp.b": hashlib.sha256(B).hexdigest()}


class Response(io.BytesIO):
    def __init__(self, body, status, headers):
        super().__init__(body)
        self.status, self.headers = status, headers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Mirror:
    """urlopen for the fake checkpoint.  `ignore_range`: 200 and the whole file (`cut`: cut to the asked length, as
    the proxy in #327 did); `wrong`: the bytes of these tensors' ranges are flipped (a source that sends bad data)."""

    def __init__(self, ignore_range=False, cut=False, wrong=()):
        self.ignore_range, self.cut, self.wrong, self.ranges = ignore_range, cut, set(wrong), []

    def __call__(self, req, timeout=None):
        name = req.full_url.rsplit("/", 1)[1]
        body = INDEX if name == "model.safetensors.index.json" else FILES[name]
        rng = req.get_header("Range")
        if rng is None:
            return Response(body, 200, {"Content-Length": str(len(body))})
        a, b = map(int, rng.split("=")[1].split("-"))
        self.ranges.append((name, a, b))
        if self.ignore_range:
            return Response(body[:b - a + 1] if self.cut else body, 200, {})
        part = body[a:b + 1]
        if any(t in self.wrong and self.span(name, t) == (a, b) for t in ("mtp.a", "mtp.b")):
            part = bytes(x ^ 0xFF for x in part)
        return Response(part, 206, {"Content-Range": "bytes %d-%d/%d" % (a, b, len(body))})

    @staticmethod
    def span(name, tensor):
        raw = FILES[name]
        n = struct.unpack("<Q", raw[:8])[0]
        meta = json.loads(raw[8:8 + n]).get(tensor)
        return (8 + n + meta["data_offsets"][0], 8 + n + meta["data_offsets"][1] - 1) if meta else None


class FetchCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.patches = [mock.patch.object(M, "SHA256", HASHES), mock.patch.object(M.time, "sleep", lambda s: None),
                        mock.patch.object(M, "REPO", M.PINNED)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def fetch(self, mirror):
        with mock.patch.object(M.urllib.request, "urlopen", mirror), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            M.fetch(str(self.out), None)
        return err.getvalue()

    def tensor(self, name):
        return (self.out / "tensors" / (name + ".bin")).read_bytes()


class Ranges(FetchCase):
    def test_check_range(self):
        M.check_range(206, "bytes 10-19/100", 10, 19)
        for status, cr in ((200, None), (200, "bytes 10-19/100"), (206, "bytes 0-9/100"), (206, None)):
            with self.subTest(status=status, cr=cr), self.assertRaises(IOError):
                M.check_range(status, cr, 10, 19)

    def test_a_mirror_that_ignores_range_is_refused(self):
        for cut in (False, True):                 # the whole file, or cut to the asked length (#327's proxy)
            with self.subTest(cut=cut), mock.patch.object(M.urllib.request, "urlopen", Mirror(True, cut)), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(IOError, "not honoured"):
                M.get(M.REPO + "model-1.safetensors", 8, 15)

    def test_fetch_from_an_honest_mirror(self):
        self.fetch(Mirror())
        self.assertEqual((self.tensor("mtp.a"), self.tensor("mtp.b")), (A, B))
        manifest = json.loads((self.out / "mtp-manifest.json").read_text())
        self.assertEqual({r["name"]: r["sha256"] for r in manifest}, HASHES)
        self.assertEqual(M.verify(str(self.out)), [])

    def test_ranges_are_read_in_parallel_and_come_back_in_order(self):
        data = bytes(range(256)) * 40
        seen = []

        def get(url, a, b):
            seen.append((a, b))
            return data[a:b + 1]

        for workers in (1, 4, 32):
            seen.clear()
            with self.subTest(workers=workers), mock.patch.object(M, "get", get):
                got = b"".join(M.ranged("u", 100, len(data) - 1, 1000, workers))
            self.assertEqual(got, data[100:])
            self.assertEqual(sorted(seen)[0], (100, 1099))
        with mock.patch.dict(M.os.environ, {"STRATA_DL_CONNECTIONS": "1"}):
            self.assertEqual(M.connections(), 1)
        with mock.patch.dict(M.os.environ, {"STRATA_DL_CONNECTIONS": ""}):
            self.assertEqual(M.connections(), 16)

    def test_the_cdn_address_is_reused_and_forgotten_after_an_error(self):
        """HF redirects /resolve/ to a signed CDN address: only the first range goes through the redirect (the rest of a
        tensor's ranges use the CDN address), and any error makes the next try ask the original address again."""
        asked = []

        def opener(req, timeout=None):
            asked.append(req.full_url)
            a, b = (int(x) for x in req.headers["Range"].split("=")[1].split("-"))
            r = Response(bytes(b - a + 1), 206, {"Content-Range": "bytes %d-%d/1000" % (a, b)})
            r.geturl = lambda: "https://cdn.example/x?sig=1"
            return r

        M._cdn.clear()
        with mock.patch.object(M.urllib.request, "urlopen", opener):
            for a in (0, 100, 200):
                M.get("https://hf.example/f", a, a + 99)
            self.assertEqual(asked, ["https://hf.example/f"] + ["https://cdn.example/x?sig=1"] * 2)
            asked.clear()
            fails = [True]

            def flaky(req, timeout=None):
                if fails.pop() if fails else False:
                    raise IOError("expired")
                return opener(req, timeout)

            with mock.patch.object(M.urllib.request, "urlopen", flaky), mock.patch.object(M.time, "sleep", lambda s: None),                     contextlib.redirect_stderr(io.StringIO()):
                M.get("https://hf.example/f", 300, 399)
            self.assertEqual(asked, ["https://hf.example/f"])               # the cached address failed: the original is asked again
        M._cdn.clear()

    def test_a_slow_first_range_does_not_stop_the_ones_behind_it(self):
        import threading
        gate = threading.Event()
        started = []

        def get(url, a, b):
            started.append(a)
            if a == 0:
                gate.wait(5)
            return bytes(b - a + 1)

        def release():                                   # lets the first range go once the later ones have been started
            for _ in range(200):
                if len(started) >= 9:
                    break
                M.time.sleep(0.01)
            gate.set()

        threading.Thread(target=release, daemon=True).start()
        with mock.patch.object(M, "get", get):
            list(M.ranged("u", 0, 40 * 1000 - 1, 1000, 4))
        self.assertGreaterEqual(len(started), 9)                           # four connections, nine ranges started before range 0 ended

    def test_a_failing_range_stops_the_parallel_read(self):
        def get(url, a, b):
            if a >= 2000:
                raise IOError("boom")
            return bytes(b - a + 1)

        with mock.patch.object(M, "get", get), self.assertRaises(IOError):
            list(M.ranged("u", 0, 9999, 1000, 4))

    def test_wrong_bytes_are_fetched_again_once_then_refused(self):
        self.fetch(Mirror())
        bad = bytes(len(A))                       # an install the #327 mirror corrupted: right size, wrong bytes
        (self.out / "tensors" / "mtp.a.bin").write_bytes(bad)
        self.assertEqual(M.verify(str(self.out)), ["mtp.a"])
        err = self.fetch(Mirror())                # setup runs fetch again: the bad one is fetched again
        self.assertIn("mtp.a: wrong bytes", err)
        self.assertEqual(self.tensor("mtp.a"), A)
        self.assertEqual(M.verify(str(self.out)), [])
        os.remove(self.out / "tensors" / "mtp.b.bin")
        with self.assertRaises(SystemExit) as cm:
            self.fetch(Mirror(wrong={"mtp.b"}))     # a source that keeps sending wrong data
        self.assertIn("is not the checkpoint's", str(cm.exception))
        self.assertFalse((self.out / "tensors" / "mtp.b.bin").exists())

    def test_an_inventory_from_another_revision_is_read_again(self):
        self.fetch(Mirror())
        inv = json.loads((self.out / "mtp-inventory.json").read_text())
        inv["repo"] = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/"
        for r in inv["tensors"]:
            r["start"] += 1                        # that revision's ranges: not this one's
            r["end"] += 1
        (self.out / "mtp-inventory.json").write_text(json.dumps(inv))
        (self.out / "tensors" / "mtp.b.bin").write_bytes(B[:10])     # half a tensor from that revision
        mirror = Mirror()
        self.fetch(mirror)
        self.assertEqual(json.loads((self.out / "mtp-inventory.json").read_text())["repo"], M.REPO)
        self.assertEqual((self.tensor("mtp.a"), self.tensor("mtp.b")), (A, B))
        fetched = [r for r in mirror.ranges if r[2] - r[1] + 1 in (len(A), len(B))]
        self.assertEqual(fetched, [("model-2.safetensors",) + Mirror.span("model-2.safetensors", "mtp.b")],
                         "mtp.a was right and is kept; mtp.b is fetched whole")

    def test_verify_keeps_its_verdict_and_skips_other_revisions(self):
        self.fetch(Mirror())
        self.assertEqual(M.verify(str(self.out)), [])
        stamps = json.loads((self.out / "tensors" / "verified.json").read_text())
        self.assertEqual(sorted(stamps), ["mtp.a", "mtp.b"])
        with mock.patch.object(M, "sha256_of", side_effect=AssertionError("hashed an unchanged file")):
            self.assertEqual(M.verify(str(self.out)), [])
        with mock.patch.object(M, "REPO", "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/"):
            (self.out / "tensors" / "mtp.a.bin").write_bytes(b"x")
            self.assertEqual(M.verify(str(self.out)), [])          # STRATA_MTP_REVISION: other hashes
        self.assertEqual(M.verify(str(self.out)), ["mtp.a"])
        self.assertEqual(M.verify(str(self.out / "nothing")), [])


class Pinned(unittest.TestCase):
    def test_every_tensor_has_a_hash(self):
        self.assertEqual(len(M.SHA256), 31)
        for name, h in M.SHA256.items():
            self.assertTrue(name.startswith("mtp."))
            self.assertRegex(h, "^[0-9a-f]{64}$")
        self.assertIn(M.PINNED_REVISION, M.PINNED)


class Setup(unittest.TestCase):
    def test_setup_rebuilds_a_corrupt_install(self):
        import setup
        with tempfile.TemporaryDirectory() as d:
            mtp = Path(d)
            self.assertFalse(setup.mtp_corrupt(mtp))              # no tensors kept: nothing to check
            (mtp / "tensors").mkdir()
            for code, want in ((0, False), (3, True), (1, False)):
                with self.subTest(code=code), mock.patch.object(
                        setup.subprocess, "run", return_value=mock.Mock(returncode=code)) as run:
                    self.assertEqual(setup.mtp_corrupt(mtp), want)
                    self.assertIn("verify", run.call_args[0][0])




class Progress(FetchCase):
    """#3257: the fetch says how far it is in the form of setup's own download line, so the app can show it."""

    def test_progress_lines_name_the_whole_download_and_end_at_100(self):
        err = self.fetch(Mirror())
        lines = [x for x in err.splitlines() if "MTP draft layer:" in x]
        total = len(A) + len(B)
        self.assertTrue(lines)
        self.assertIn("/ %.2f GB" % (total / 1e9), lines[-1])
        self.assertIn("(100%)", lines[-1])
        import re
        for x in lines:                                  # the shape tools/strata_mcp.py install_progress reads
            self.assertTrue(re.match(r"\s*(\S.*?): +([\d.]+) / ([\d.]+) GB \((\d+)%\)", x), x)

    def test_a_resumed_fetch_starts_from_what_is_already_there(self):
        self.fetch(Mirror())
        err = self.fetch(Mirror())
        self.assertIn("(100%)", [x for x in err.splitlines() if "MTP draft layer:" in x][0])

    def test_the_progress_figure_follows_the_job_in_strata_mcp(self):
        import strata_mcp as MCP
        got = MCP.install_progress(["=== Step 6: preparing the model for Strata ===",
                                    "  MTP draft layer:   1.23 / 5.04 GB (24%)"])
        self.assertEqual(got["download"], {"file": "MTP draft layer", "done_gb": 1.23, "total_gb": 5.04, "percent": 24})


class Flaky(FetchCase):
    def test_a_connection_that_drops_mid_tensor_does_not_end_the_run(self):
        """#3214: every range of the tensor fails past get()'s own retries once: the tensor is tried again (fewer connections)."""
        honest, fails = Mirror(), [0]

        def mirror(req, timeout=None):
            if req.get_header("Range") and req.full_url.endswith("model-2.safetensors") and fails[0] < 12:
                a, b = map(int, req.get_header("Range").split("=")[1].split("-"))
                if (a, b) == Mirror.span("model-2.safetensors", "mtp.b"):
                    fails[0] += 1
                    raise IOError("timed out")
            return honest(req, timeout)

        err = self.fetch(mirror)
        self.assertEqual(self.tensor("mtp.b"), B)
        self.assertIn("continuing with", err)

    def test_a_link_that_never_works_ends_with_a_plain_sentence(self):
        def mirror(req, timeout=None):
            if req.get_header("Range") and req.full_url.endswith("model-2.safetensors"):
                a, b = map(int, req.get_header("Range").split("=")[1].split("-"))
                if (a, b) == Mirror.span("model-2.safetensors", "mtp.b"):
                    raise IOError("timed out")
            return Mirror()(req, timeout)

        with self.assertRaisesRegex(SystemExit, "keeps failing"):
            self.fetch(mirror)


class OneAtATime(unittest.TestCase):
    def test_a_second_fetch_in_the_same_folder_waits_for_the_first(self):
        with tempfile.TemporaryDirectory() as d:
            first = M.lock_folder(d)
            told = io.StringIO()
            with mock.patch.object(M.time, "sleep", lambda s: None), contextlib.redirect_stderr(told), \
                    self.assertRaisesRegex(SystemExit, "another MTP download"):
                M.lock_folder(d, wait=0.0)
            first.close()
            M.lock_folder(d).close()                     # free again once the first ends

    def test_a_download_ends_with_the_process_that_started_it(self):
        """#3257: setup killed, its child must not go on: a grandchild whose parent is gone."""
        import subprocess
        import time
        child_code = ("import sys,time; sys.path.insert(0, %r); import mtp_fetch as M; M.exit_with_parent(); "
                      "print('up', flush=True); time.sleep(60)" % str(ROOT / "tools"))
        parent_code = ("import subprocess,sys,os; env=dict(os.environ, STRATA_EXIT_WITH_PARENT='1'); "
                       "p=subprocess.Popen([sys.executable,'-c',%r], env=env, stdout=subprocess.PIPE, text=True); "
                       "print(p.pid, flush=True); p.stdout.readline(); print('ready', flush=True); sys.stdout.flush(); os._exit(0)" % child_code)
        r = subprocess.Popen([sys.executable, "-c", parent_code], stdout=subprocess.PIPE, text=True)
        pid = int(r.stdout.readline())
        r.stdout.readline()
        r.wait(30)
        end = time.time() + 15
        alive = True
        while time.time() < end and alive:
            time.sleep(0.5)
            if os.name == "nt":
                out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid], capture_output=True, text=True).stdout
                alive = str(pid) in out
            else:
                try:
                    os.kill(pid, 0)
                    alive = True
                except OSError:
                    alive = False
        if alive:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
            else:
                os.kill(pid, 9)
        self.assertFalse(alive, "the download kept running after its parent ended")


if __name__ == "__main__":
    unittest.main()
