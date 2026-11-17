"""#533: POST /v1/vram (opt-in hot VRAM resize) - the engine command, the route, the config key, a reserve kept for
an unloaded engine and applied when it loads again."""
import io
import json
import queue
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, Service, StrataEngine, engine_args, serve
from serve.test_lifecycle import ResidentEngine


class VramEngine(ResidentEngine):
    """The resident test engine with an elastic cache: VRAM n frees what the reserve asks for, None takes the
    start's (700 MiB)."""
    def __init__(self, tok, elastic=True):
        super().__init__(tok)
        self.elastic, self.calls = elastic, []

    def vram(self, reserve_mib, timeout=120.0):
        self.calls.append(reserve_mib)
        if not self.elastic:
            raise ValueError("VRAM needs an engine started with --vram-elastic (one NVIDIA GPU, --serve)")
        r = 700 if reserve_mib is None else reserve_mib
        out = {"reserve_mib": r, "expert_slots": max(100, 4000 - r), "expert_cache_mib": max(500, 6000 - r),
               "expert_cache_full_mib": 6000, "vram_free_mib": r, "prompt_chunk": 4096}
        self.info["vram"] = out
        return out


class EngineCommand(unittest.TestCase):
    def engine(self, replies):
        e = StrataEngine("missing-executable", [], lazy=True)
        e.proc = mock.Mock()
        e.proc.poll.return_value = None
        e.proc.stdin = io.StringIO()
        e.ended = False
        e.lines = queue.Queue()
        for r in replies:
            e.lines.put(r)
        return e

    def test_the_line_and_the_answer(self):
        e = self.engine(["strata serve: something else\n",
                         "VRAM reserve_mib=6000 expert_slots=1200 expert_slots_full=3700 expert_cache_mib=1536 "
                         "expert_cache_full_mib=4900 vram_free_mib=6100 prompt_chunk=2048\n"])
        out = e.vram(6000)
        self.assertEqual(e.proc.stdin.getvalue(), "VRAM 6000\n")
        self.assertEqual(out["expert_slots"], 1200)
        self.assertEqual(out["prompt_chunk"], 2048)
        self.assertEqual(e.info["expert_slots"], 1200)
        e = self.engine(["VRAM reserve_mib=700 expert_slots=3700\n"])
        e.vram(None)
        self.assertEqual(e.proc.stdin.getvalue(), "VRAM\n")

    def test_refused_and_died(self):
        e = self.engine(["ERR VRAM needs an engine started with --vram-elastic\n"])
        with self.assertRaisesRegex(ValueError, "--vram-elastic"):
            e.vram(100)
        e = self.engine([None])
        with self.assertRaises(EngineDied):
            e.vram(100)

    def test_config_key(self):
        cfg = {"args": ["--max-context", "8192"]}
        self.assertEqual(engine_args(cfg), ["--max-context", "8192"])           # off unless asked
        self.assertEqual(engine_args({**cfg, "vram_elastic": True, "vram_segment_mib": 256}),
                         ["--max-context", "8192", "--vram-elastic", "--vram-segment-mib", "256"])
        self.assertEqual(engine_args({**cfg, "vram_elastic": "yes"}), ["--max-context", "8192"])


class Route(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = VramEngine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read().decode())

    def test_resize_and_status(self):
        self.engine.loaded = True
        code, body = self.request("/v1/vram", {"reserve_mib": 4000})
        self.assertEqual(code, 200, body)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["expert_cache_mib"], 2000)
        self.assertEqual(self.request("/v1/status")[1]["vram"]["expert_cache_mib"], 2000)
        code, body = self.request("/v1/vram", {"reserve_mib": None})   # back to the start's reserve
        self.assertEqual(code, 200)
        self.assertEqual(self.engine.calls, [4000, None])

    def test_bad_input_busy_and_foreign_pages(self):
        self.engine.loaded = True
        for bad in ({"reserve_mib": -1}, {"reserve_mib": "8 GiB"}, {"reserve_mib": True}, {"reserve_mib": 1.5}):
            self.assertEqual(self.request("/v1/vram", bad)[0], 400, bad)
        self.assertEqual(self.request("/v1/vram", {"reserve_mib": 1}, {"Origin": "https://other.example"})[0], 403)
        self.assertEqual(self.request("/v1/vram", {"reserve_mib": 1}, {"Content-Type": "text/plain"})[0], 415)
        self.svc.vram_wait_s = 0.2
        with self.svc.fifo:                                   # a request is running
            self.assertEqual(self.request("/v1/vram", {"reserve_mib": 1})[0], 409)
        self.assertEqual(self.engine.calls, [])

    def test_an_engine_without_the_flag_says_so(self):
        self.engine.loaded, self.engine.elastic = True, False
        code, body = self.request("/v1/vram", {"reserve_mib": 3000})
        self.assertEqual(code, 400)
        self.assertIn("--vram-elastic", body["error"]["message"])

    def test_unloaded_keeps_the_reserve_for_the_next_load(self):
        code, body = self.request("/v1/vram", {"reserve_mib": 5000})
        self.assertEqual(code, 200)
        self.assertEqual(body["status"], "not loaded")
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(self.request("/v1/load", {})[0], 200)
        self.assertEqual(self.engine.calls, [5000])


if __name__ == "__main__":
    unittest.main()
