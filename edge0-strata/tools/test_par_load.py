"""Self-test of tools/par_load.py and par_analyze.py against an in-process fake SSE server."""
import json, os, sys, tempfile, threading, time, unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.dirname(__file__))
import par_load, par_analyze


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"ok":1}')
    def do_POST(self):
        n = int(self.headers["Content-Length"]); b = json.loads(self.rfile.read(n))
        text = b["messages"][0]["content"]
        pt = len(text.split()) + 10
        if not b["stream"]:
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({"usage": {"prompt_tokens": pt, "completion_tokens": 1}}).encode()); return
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        k = min(b["max_tokens"], 12)
        for i in range(k):
            time.sleep(0.01)
            self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {"content": "w "}}]}) + "\n\n").encode()); self.wfile.flush()
        # two tokens per chunk at the end would make SSE counts wrong: usage is the truth
        self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "length"}], "usage": {"prompt_tokens": pt, "completion_tokens": k + 3}}) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=cls.s.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.s.server_address[1]}"
    @classmethod
    def tearDownClass(cls): cls.s.shutdown()

    def run_(self, *args):
        out = tempfile.mktemp(suffix=".json")
        rc = par_load.main([self.url, "--out", out, *args])
        return rc, json.load(open(out))

    def test_w2_usage_tokens(self):
        rc, L = self.run_("--workload", "w2", "--conc", "1,2", "--rounds", "2", "--max-tokens", "12", "--prompt-tokens", "200")
        self.assertEqual(rc, 0); self.assertTrue(L["valid"])
        b = L["blocks"][1]
        self.assertEqual(b["tokens"], 4 * 15)             # usage says 15, SSE chunks were 12
        self.assertEqual(b["requests"], 4)
        self.assertIsNotNone(b["ttft_p50"]); self.assertIsNotNone(b["tpot_p50"])

    def test_w4_w6_open(self):
        rc, L = self.run_("--workload", "w4", "--conc", "2", "--rounds", "1", "--joiner-at", "0.05", "--joiner-tokens", "400", "--prompt-tokens", "100")
        self.assertEqual(rc, 0); self.assertIn("joiner_ttft", L["blocks"][0])
        rc, L = self.run_("--workload", "w6", "--conc", "2", "--rounds", "1", "--share", "0,0.9", "--prompt-tokens", "300")
        self.assertEqual(len(L["blocks"]), 2)
        rc, L = self.run_("--workload", "w2", "--open-rates", "20", "--duration", "1", "--drain-s", "5", "--max-tokens", "8", "--prompt-tokens", "100")
        self.assertEqual(rc, 0); self.assertEqual(L["blocks"][0]["unfinished"], 0)

    def test_analyze(self):
        rows = []
        for i, (a, b) in enumerate([(10, 21), (10, 20), (11, 22)]):
            rows += [("x", {"workload": "w2", "level": 4, "arm": "A", "pair": str(i), "agg_tps": a, "errors": 0}),
                     ("x", {"workload": "w2", "level": 4, "arm": "B", "pair": str(i), "agg_tps": b, "errors": 0})]
        r = par_analyze.analyze(rows, "agg_tps")
        self.assertAlmostEqual(r[0]["geomean"], 2.0, delta=0.06); self.assertEqual(r[0]["pairs_better"], 3)


if __name__ == "__main__":
    unittest.main()
