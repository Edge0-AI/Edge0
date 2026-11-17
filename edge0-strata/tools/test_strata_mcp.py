"""tools/test_strata_mcp.py - tests for tools/strata_mcp.py (Strata's MCP server), standard library only.

    python -m unittest tools/test_strata_mcp.py -v

Setup and the model server are fakes (a temporary Strata folder with a stand-in setup.py and serve/server.py), so
nothing is downloaded and no GPU is used: the JSON-RPC handshake, argument checks, install / start / stop with real
background processes, status against a fake HTTP server, and that the built-in model table still matches setup.py.
"""
from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import strata_mcp as M  # noqa: E402

FAKE_SETUP = r'''
import json, sys, time
from pathlib import Path
def main():
    root = Path(__file__).resolve().parent
    (root / "setup-args.json").write_text(json.dumps(sys.argv[1:]))
    print("Strata - fake setup", flush=True)
    print("\n=== Step 1: checking your PC ===", flush=True)
    print("  [ok] GPU: fake", flush=True)
    print("\n=== Step 5: downloading Qwen3.8-Flash-Next Q2_0 ===", flush=True)
    for pct in (10, 20, 30):
        print(f"\r  model.gguf:  {pct * 0.5:.2f} / 50.00 GB ({pct}%)   ", end="", flush=True)
    print(flush=True)
    if (root / "slow").exists():
        end = time.time() + 30
        while not (root / "go").exists() and time.time() < end:
            time.sleep(0.1)
    if (root / "fail").exists():
        print("\n  [X]  not enough free disk space", flush=True)
        print("       use --models-dir on a bigger drive", flush=True)
        sys.exit(1)
    a = sys.argv
    model = a[a.index("--model") + 1].lower()
    cfg = {"exe": sys.executable, "args": ["--max-context", a[a.index("--context") + 1]], "port": 8080,
           "model_name": "qwen3.8-flash-next-" + model, "log": str(root / f"strata-{model}.log")}
    (root / f"strata-{model}.json").write_text(json.dumps(cfg))
    print("All set.", flush=True)


if __name__ == "__main__":
    main()
'''

FAKE_SERVER = r'''
import json, os, signal, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
a = sys.argv
port = int(a[a.index("--port") + 1])
cfg = json.loads(Path(a[a.index("--config") + 1]).read_text())
root = Path(__file__).resolve().parent.parent
print("loading the model (fake) ...", flush=True)
if cfg.get("crash"):
    print("engine failed: out of memory (fake)", flush=True)
    sys.exit(3)
time.sleep(cfg.get("delay", 0))
state = {"loaded": True}
class H(BaseHTTPRequestHandler):
    def log_message(self, *x):
        pass
    def js(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/health":
            self.js(200, {"status": "ok", "service": "strata", "model": cfg["model_name"], "max_context": 32768,
                          "images": False, "api_key": False, "loaded": state["loaded"]})
        elif p == "/status":
            self.js(200, {"busy": bool(cfg.get("busy")), "queued": 0})
        elif p == "/v1/status":
            self.js(200, {"service": "strata", "engine": "0.1.32", "uptime_s": 5, "activity": {"requests": 0},
                          "machine": {"gpu": {"name": "Fake GPU", "used_mib": 11000, "total_mib": 12227},
                                      "ram": {"used_gib": 40.0, "total_gib": 63.1}}})
        else:
            self.js(404, {"error": {"message": "not found"}})
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        p = self.path.split("?")[0]
        if p == "/unload":
            state["loaded"] = False
            with open(root / "unloaded.txt", "a") as f:
                f.write(str(port) + "\n")
            self.js(200, {"status": "unloaded"})
        elif p == "/v1/chat/completions":
            with open(root / "bench-request.json", "w") as f:
                json.dump(body, f)
            self.js(200, {"model": cfg["model_name"], "choices": [{"message": {"content": "x"}, "finish_reason": "length"}],
                          "usage": {"prompt_tokens": 30, "completion_tokens": body.get("max_tokens", 1)},
                          "timings": {"predicted_per_second": 61.5, "prompt_per_second": 400.0}})
        else:
            self.js(404, {})
httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
def term(*_):
    print("[strata] stopped (fake)", flush=True)
    os._exit(0)
signal.signal(signal.SIGTERM, term)
print(f"ready: http://127.0.0.1:{port}/v1", flush=True)
httpd.serve_forever()
'''


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_for(cond, seconds=30.0):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return cond()


FAKE_HW = {"os": "TestOS 1", "ram_gb": 64.0, "cpu": {"name": "Fake CPU", "cores": 12, "avx2": True, "avx512": True},
           "gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 5070", "vram_gb": 11.9, "arch": "120", "vendor": "nvidia",
                     "problem": None, "usable": True}]}


class FakeRoot(unittest.TestCase):
    """A temporary Strata folder: fake setup.py and serve/server.py, its own settings file (data folder)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = base / "Strata"
        (self.root / "serve").mkdir(parents=True)
        (self.root / "setup.py").write_text(FAKE_SETUP)
        (self.root / "serve" / "server.py").write_text(FAKE_SERVER)
        self.env_backup = {k: os.environ.get(k) for k in ("APPDATA", "XDG_CONFIG_HOME")}
        os.environ["APPDATA"] = os.environ["XDG_CONFIG_HOME"] = str(base / "config")
        self.strata = M.Strata(self.root)
        self.strata.python_override = sys.executable
        self.strata.health_timeout = 1.0
        self.strata._hardware = lambda: dict(FAKE_HW)
        self.strata.disk_free = lambda p: 1000.0
        self.tools = M.Tools(self.strata)
        self.extra = []
        # the tests must never see a Strata server that happens to run on the real default port on this PC
        self._default_port = M.DEFAULT_PORT
        M.DEFAULT_PORT = free_port()

    def tearDown(self):
        M.DEFAULT_PORT = self._default_port
        for p in self.extra:
            if p.poll() is None:
                p.kill()
                p.wait(10)
        srv = self.strata.state("server")
        if srv and M.proc_alive(srv.get("pid"), srv.get("ident")):
            M.proc_kill_tree(srv["pid"], srv["ident"])
            M.wait_gone(srv["pid"], srv["ident"], 10)
        job = self.strata.state("install")
        if job and M.proc_alive(job.get("pid"), job.get("ident")):
            M.proc_kill_tree(job["pid"], job["ident"])
            M.wait_gone(job["pid"], job["ident"], 10)
        for p in self.strata.children:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                pass
        for k, v in self.env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for _ in range(20):                             # Windows: a log may still be closing
            try:
                self.tmp.cleanup()
                break
            except OSError:
                time.sleep(0.25)

    def call(self, name, args=None):
        """A tools/call through the JSON-RPC layer: (result dict, isError)."""
        srv = M.McpServer(self.strata, io.BytesIO())
        r = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": name, "arguments": args or {}}})
        res = r["result"]
        return json.loads(res["content"][0]["text"]), res["isError"]

    def write_config(self, tag="q2_0", port=8080, **extra):
        cfg = {"exe": sys.executable, "args": ["--max-context", "32768"], "port": port,
               "model_name": f"qwen3.8-flash-next-{tag}", "log": str(self.root / f"strata-{tag}.log"), **extra}
        (self.root / f"strata-{tag}.json").write_text(json.dumps(cfg))
        (self.root / f"strata-{tag}.log").write_text("engine line 1\nengine line 2\n")
        return cfg


# ------------------------------------------------------------------------------------------------ the protocol
class Protocol(FakeRoot):
    def run_lines(self, msgs):
        inp = io.BytesIO(("\n".join(m if isinstance(m, str) else json.dumps(m) for m in msgs) + "\n").encode())
        out = io.BytesIO()
        M.McpServer(self.strata, out).serve(inp)
        return {m.get("id"): m for m in (json.loads(x) for x in out.getvalue().decode().splitlines())}

    def test_handshake_list_call(self):
        r = self.run_lines([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "strata_models", "arguments": {}}},
            {"jsonrpc": "2.0", "id": 4, "method": "ping"},
        ])
        init = r[1]["result"]
        self.assertEqual(init["protocolVersion"], "2025-06-18")
        self.assertIn("tools", init["capabilities"])
        self.assertEqual(init["serverInfo"]["name"], "strata")
        names = [t["name"] for t in r[2]["result"]["tools"]]
        self.assertEqual(names, ["strata_status", "strata_models", "strata_remove", "strata_install", "strata_start", "strata_stop",
                                 "strata_logs", "strata_benchmark", "strata_connect_info"])
        for t in r[2]["result"]["tools"]:
            self.assertEqual(t["inputSchema"]["type"], "object")
            self.assertFalse(t["inputSchema"]["additionalProperties"])
            self.assertGreater(len(t["description"]), 60)
        call = r[3]["result"]
        self.assertFalse(call["isError"])
        self.assertIn("families", call["structuredContent"])
        self.assertEqual(json.loads(call["content"][0]["text"])["recommendation"]["model"], "IQ3_S")
        self.assertEqual(r[4]["result"], {})
        self.assertNotIn(None, r)                       # the notification got no answer

    def test_older_protocol_and_errors(self):
        r = self.run_lines([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
            "{not json",
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "rm_rf", "arguments": {}}},
            {"jsonrpc": "2.0", "id": 3, "method": "no/such"},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "strata_logs", "arguments": []}},
        ])
        self.assertEqual(r[1]["result"]["protocolVersion"], "2024-11-05")
        self.assertEqual(r[None]["error"]["code"], -32700)
        self.assertEqual(r[2]["error"]["code"], -32602)
        self.assertEqual(r[3]["error"]["code"], -32601)
        self.assertTrue(r[4]["result"]["isError"])
        self.assertNotIn("structuredContent", r[4]["result"])   # 2024-11-05 has no structured results

    def test_real_stdio_process(self):
        """The script itself over a pipe: stdout carries only JSON-RPC lines."""
        msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                 "params": {"name": "strata_logs", "arguments": {"source": "setup"}}}]
        p = subprocess.run([sys.executable, str(HERE / "strata_mcp.py"), "--root", str(self.root)],
                           input="\n".join(json.dumps(m) for m in msgs) + "\n", capture_output=True, text=True,
                           timeout=60, env=dict(os.environ))
        lines = [json.loads(x) for x in p.stdout.splitlines()]
        self.assertEqual(sorted(m["id"] for m in lines), [1, 2, 3])
        self.assertEqual(len(next(m for m in lines if m["id"] == 2)["result"]["tools"]), 9)
        self.assertIn("ready", p.stderr)

    def test_not_a_strata_folder(self):
        p = subprocess.run([sys.executable, str(HERE / "strata_mcp.py"), "--root", self.tmp.name], input="",
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 2)


# ------------------------------------------------------------------------------------------------ validation
class Validation(FakeRoot):
    def assertRejected(self, tool, args, text):
        res, err = self.call(tool, args)
        self.assertTrue(err, f"{tool} {args} was accepted: {res}")
        self.assertIn(text, res["error"])

    def test_install_arguments(self):
        self.assertRejected("strata_install", {"model": "Q9_X"}, "model must be one of")
        self.assertRejected("strata_install", {"family": "llama"}, "family must be one of")
        self.assertRejected("strata_install", {"context": 100}, "context must be between")
        self.assertRejected("strata_install", {"gpus": "0;rm -rf /"}, "gpus=")
        self.assertRejected("strata_install", {"gpus": "0 && calc"}, "gpus=")
        self.assertRejected("strata_install", {"gpu": -1}, "between")
        self.assertRejected("strata_install", {"gpu": "1; calc"}, "whole number")
        self.assertRejected("strata_install", {"gpu": 0, "gpus": "0,1"}, "not both")
        self.assertRejected("strata_install", {"port": 80}, "between")
        self.assertRejected("strata_install", {"backend": "rocm"}, "backend must be one of")
        self.assertRejected("strata_install", {"command": "setup.py --build"}, "unknown argument")
        res, err = self.call("strata_install", {"family": "swift", "model": "IQ3_S"})
        self.assertFalse(err, res)                      # Swift 1.5 got its IQ3_S (same shard layout as its IQ2_XS)
        self.assertIn("--family swift --model IQ3_S", res["plan"]["setup_command"])
        res, err = self.call("strata_install", {"family": "unsloth", "model": "UD-Q4_K_XL", "vision": "yes"})
        self.assertFalse(err, res)                      # #967: images are allowed with UD-Q4_K_XL (setup warns)
        self.assertEqual(res["plan"]["images"], "yes")
        self.assertRejected("strata_install", {"vision": "yes\nrm"}, "control characters")

    def test_amd_vision_follows_the_picture_rule(self):
        """Owner rule 2026-10-18: pictures are on for AMD too, on Linux and Windows (setup picks the reader: CPU under 24 GB, GPU from 24 GB)."""
        for win in (False, True):
            with mock.patch.object(M, "WIN", win):
                for asked in ("yes", "gpu", "cpu"):
                    res, err = self.call("strata_install", {"backend": "hip", "vision": asked})
                    self.assertFalse(err, res)
                    self.assertEqual(res["plan"]["images"], asked)
                    self.assertIn("--vision " + asked, res["plan"]["setup_command"])
                res, err = self.call("strata_install", {"backend": "hip", "vision": "no"})
                self.assertFalse(err, res)
                self.assertEqual(res["plan"]["images"], "no")

    def test_pictures_default_on(self):
        """Nothing asked: pictures are on where the model can read them (any backend), off for a model that cannot."""
        for backend in (None, "cuda", "hip"):
            args = {"backend": backend} if backend else {}
            res, err = self.call("strata_install", args)
            self.assertFalse(err, res)
            self.assertEqual(res["plan"]["images"], "yes")
            self.assertIn("--vision yes", res["plan"]["setup_command"])
        res, err = self.call("strata_install", {"vision": "no"})
        self.assertFalse(err, res)
        self.assertEqual(res["plan"]["images"], "no")

    def test_data_dir_paths(self):
        self.assertRejected("strata_install", {"data_dir": "models"}, "absolute")
        self.assertRejected("strata_install", {"data_dir": str(self.root.parent / "x" / ".." / "Strata-x")}, "'..'")
        self.assertRejected("strata_install", {"data_dir": str(self.root.parent / "Documents")}, "named Strata")
        self.assertRejected("strata_install", {"data_dir": str(self.root.parent / "nope" / "Strata-data")},
                            "does not exist")
        self.assertRejected("strata_install", {"data_dir": "\\\\server\\share\\Strata-data"}, "network")
        sysdir = (os.environ.get("SystemRoot", "C:\\Windows") + "\\Strata-data") if M.WIN else "/etc/Strata-data"
        self.assertRejected("strata_install", {"data_dir": sysdir}, "system folder")
        ok = [self.root.parent / "Strata-data", self.root / "models-here", self.root.parent / "Strata-big"]
        for d in ok:
            self.assertEqual(M.norm(self.tools.check_data_dir(str(d))), M.norm(d))

    def test_the_persons_own_data_folder_is_accepted_whatever_it_is_called(self):
        """The app writes the folder the person chose into the settings file (apps/host model-folder.mjs); an AI tool naming a new
        folder still needs a Strata... name."""
        chosen = self.root.parent / "my-big-disk"
        chosen.mkdir()
        self.assertRejected("strata_install", {"data_dir": str(chosen)}, "named Strata")
        settings = self.tools.s.settings_path()
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text(json.dumps({"data_dir": str(chosen)}))
        self.assertEqual(M.norm(self.tools.check_data_dir(str(chosen))), M.norm(chosen))

    def test_other_tools_arguments(self):
        self.assertRejected("strata_start", {"model": "../../etc/passwd"}, "not valid")
        self.assertRejected("strata_start", {"model": "q2_0"}, "no model is installed")
        self.write_config("q2_0")
        self.assertRejected("strata_start", {"model": "iq3_s"}, "no installed model 'iq3_s'")
        self.assertRejected("strata_start", {"wait_seconds": 100000}, "between")
        self.assertRejected("strata_logs", {"lines": 0}, "between")
        self.assertRejected("strata_logs", {"source": "/etc/passwd"}, "source must be one of")
        self.assertRejected("strata_benchmark", {"max_tokens": 100000}, "between")
        self.assertRejected("strata_stop", {"force": "yes"}, "true or false")

    def test_engine_log_outside_is_not_read(self):
        self.write_config("q2_0", log=str(Path(self.tmp.name).parent / "elsewhere.log"))
        res, err = self.call("strata_logs", {"source": "engine", "model": "q2_0"})
        self.assertTrue(err)
        self.assertIn("outside the Strata folder", res["error"])
        self.write_config("q2_0")
        res, err = self.call("strata_logs", {"source": "engine"})
        self.assertFalse(err)
        self.assertEqual(res["lines"], ["engine line 1", "engine line 2"])


# ------------------------------------------------------------------------------------------------ install
class Install(FakeRoot):
    def wait_job(self):
        self.assertTrue(wait_for(lambda: not self.strata.install_job()["running"], 60))

    def test_plan_then_install(self):
        res, err = self.call("strata_install", {})
        self.assertFalse(err)
        self.assertIn("Nothing was done yet", res["summary"])
        self.assertEqual((res["plan"]["family"], res["plan"]["model"], res["plan"]["context"]), ("qwen", "IQ3_S", 32768))
        self.assertFalse((self.root / "setup-args.json").exists())
        self.assertIsNone(self.strata.install_job())

        res, err = self.call("strata_install", {"model": "Q2_0", "context": 65536, "vision": "no", "confirm": True})
        self.assertFalse(err, res)
        self.assertIn("started in the background", res["summary"])
        self.wait_job()
        args = json.loads((self.root / "setup-args.json").read_text())
        self.assertEqual(args[:2], ["--yes", "--no-start"])
        for pair in (["--family", "qwen"], ["--model", "Q2_0"], ["--context", "65536"], ["--vision", "no"]):
            i = args.index(pair[0])
            self.assertEqual(args[i:i + 2], pair)
        res, err = self.call("strata_install", {})
        self.assertEqual(res["install"]["result"], "installed")
        self.assertEqual(res["install"]["exit_code"], 0)
        self.assertEqual(res["next"], "strata_start")
        st, _ = self.call("strata_status", {"hardware": False})
        self.assertEqual([m["model"] for m in st["installed"]["models"]], ["q2_0"])

    def test_progress_while_running_and_cancel(self):
        (self.root / "slow").write_text("")
        res, err = self.call("strata_install", {"family": "coder", "confirm": True})
        self.assertFalse(err, res)
        self.assertTrue(wait_for(lambda: (self.strata.install_job() or {}).get("download") is not None, 30))
        res, err = self.call("strata_install", {"model": "IQ3_S", "confirm": True})   # a second call: progress only
        self.assertFalse(err)
        self.assertIn("an install is running", res["summary"])
        self.assertEqual(res["install"]["step"], "5/7: downloading Qwen3.8-Flash-Next Q2_0")
        self.assertEqual(res["install"]["download"]["percent"], 30)
        res, err = self.call("strata_start", {})
        self.assertTrue(err)
        self.assertIn("install is still running", res["error"])
        res, err = self.call("strata_install", {"cancel": True})
        self.assertIn("stopped", res["summary"])
        self.assertFalse(self.strata.install_job()["running"])

    def test_failure_is_reported(self):
        (self.root / "fail").write_text("")
        self.call("strata_install", {"model": "Q2_0", "confirm": True})
        self.wait_job()
        res, _ = self.call("strata_install", {})
        self.assertEqual(res["install"]["result"], "failed")
        self.assertIn("not enough free disk space - use --models-dir on a bigger drive", res["install"]["error"])

    def test_disk_space_is_checked(self):
        self.strata.disk_free = lambda p: 10.0
        res, err = self.call("strata_install", {"model": "Q2_0"})
        self.assertIn("not enough free disk space", res["plan"]["disk_short"])
        res, err = self.call("strata_install", {"model": "Q2_0", "confirm": True})
        self.assertTrue(err)
        self.assertIsNone(self.strata.install_job())


# ------------------------------------------------------------------------------------------------ start / stop / status
class StartStop(FakeRoot):
    def test_start_status_bench_stop(self):
        port = free_port()
        self.write_config("q2_0", port=port, delay=1.5)
        res, err = self.call("strata_start", {"wait_seconds": 30})
        self.assertFalse(err, res)
        self.assertIn("Strata is running", res["summary"])
        pid = res["server"]["pid"]

        st, _ = self.call("strata_status", {"hardware": False})
        self.assertIn(f"running on port {port}", st["summary"])
        srv = st["running"][0]
        self.assertEqual(srv["engine_version"], "0.1.32")
        self.assertEqual(srv["gpu"]["total_mib"], 12227)
        self.assertIn("this MCP server", srv["started_by"])

        res, err = self.call("strata_start", {})            # already running: nothing new
        self.assertIn("already running", res["summary"])

        b, err = self.call("strata_benchmark", {"max_tokens": 64})
        self.assertFalse(err, b)
        self.assertEqual(b["output_tok_s"], 61.5)
        sent = json.loads((self.root / "bench-request.json").read_text())
        self.assertEqual((sent["temperature"], sent["max_tokens"], sent["reasoning_effort"]), (0, 64, "none"))

        c, _ = self.call("strata_connect_info", {})
        self.assertEqual(c["openai_base_url"], f"http://127.0.0.1:{port}/v1")

        logs, _ = self.call("strata_logs", {"source": "server"})
        self.assertTrue(any(line.startswith("ready:") for line in logs["lines"]))

        res, err = self.call("strata_stop", {})
        self.assertFalse(err, res)
        self.assertIn("Strata stopped", res["summary"])
        self.assertEqual(res["engine_unloaded"], "unloaded")
        self.assertEqual((self.root / "unloaded.txt").read_text().split(), [str(port)])
        ident = M.proc_identity(pid)
        self.assertIsNone(ident)
        st, _ = self.call("strata_status", {"hardware": False})
        self.assertEqual(st["running"], [])
        res, _ = self.call("strata_stop", {})
        self.assertEqual(res["summary"], "Strata is not running")

    def test_start_reports_a_crash(self):
        self.write_config("q2_0", port=free_port(), crash=True)
        res, err = self.call("strata_start", {"wait_seconds": 30})
        self.assertTrue(err)
        self.assertIn("out of memory (fake)", res["error"])
        self.assertEqual(self.strata.state("server"), {})

    def test_loading_then_ready(self):
        port = free_port()
        self.write_config("q2_0", port=port, delay=3)
        res, err = self.call("strata_start", {"wait_seconds": 0})
        self.assertIn("loading", res["summary"])
        st, _ = self.call("strata_status", {"hardware": False})
        self.assertIn("loading", st["summary"])
        self.assertTrue(wait_for(lambda: self.strata.probe(port) is not None, 30))

    def test_stop_while_busy_needs_force(self):
        port = free_port()
        self.write_config("q2_0", port=port, busy=True)
        self.call("strata_start", {"wait_seconds": 30})
        res, err = self.call("strata_stop", {})
        self.assertTrue(err)
        self.assertIn("force=true", res["error"])
        res, err = self.call("strata_stop", {"force": True})
        self.assertFalse(err, res)

    def test_external_server_is_only_unloaded(self):
        port = free_port()
        self.write_config("q2_0", port=port)
        p = subprocess.Popen([sys.executable, str(self.root / "serve" / "server.py"), "--config",
                              str(self.root / "strata-q2_0.json"), "--port", str(port)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.extra.append(p)
        self.assertTrue(wait_for(lambda: self.strata.probe(port) is not None, 30))
        st, _ = self.call("strata_status", {"hardware": False})
        self.assertIn("outside this MCP server", st["running"][0]["started_by"])
        res, err = self.call("strata_start", {})
        self.assertIn("started outside this MCP server", res["summary"])
        res, err = self.call("strata_stop", {})
        self.assertFalse(err, res)
        self.assertIn("only its model was unloaded", res["summary"])
        time.sleep(0.5)
        self.assertIsNone(p.poll())                      # the server itself was not touched

    def test_never_kills_an_unrelated_process(self):
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.extra.append(other)
        # a stale record: this process id, but another start time (a reused id)
        self.strata.save_state("server", {"pid": other.pid, "ident": "win:1" if M.WIN else "linux:1",
                                          "port": free_port(), "model": "q2_0"})
        res, _ = self.call("strata_stop", {})
        self.assertEqual(res["summary"], "Strata is not running")
        self.assertFalse(M.proc_terminate(other.pid, "linux:1"))
        time.sleep(0.3)
        self.assertIsNone(other.poll())

    def test_status_sees_other_programs_on_the_port(self):
        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(404)
                self.end_headers()

            def log_message(self, *a):
                pass
        httpd = HTTPServer(("127.0.0.1", 0), H)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            self.write_config("q2_0", port=port)
            st, _ = self.call("strata_status", {"hardware": False})
            self.assertEqual(st["running"], [])
            self.assertEqual(st["ports_used_by_other_programs"][0]["port"], port)
            res, err = self.call("strata_start", {})
            self.assertTrue(err)
            self.assertIn("used by another program", res["error"])
        finally:
            httpd.shutdown()
            httpd.server_close()


# ------------------------------------------------------------------------------------------------ helpers
class Helpers(unittest.TestCase):
    def test_progress_from_setup_output(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "setup.log"
            f.write_text("=== Step 4: the Strata engine ===\n  [ok] engine\n=== Step 5: downloading X ===\n"
                         "\r  a.gguf:   1.00 / 50.00 GB (2%)   \r  a.gguf:  25.00 / 50.00 GB (50%)   \n")
            p = M.install_progress(M.read_tail(f, 50))
        self.assertEqual(p["step"], "5/7: downloading X")
        self.assertEqual(p["download"], {"file": "a.gguf", "done_gb": 25.0, "total_gb": 50.0, "percent": 50})
        self.assertNotIn("error", p)

    def test_builtin_table_matches_setup(self):
        s = M.Strata(HERE.parent)
        S = s.setup_module()
        self.assertIsNotNone(S, "the repository's setup.py did not import")
        self.assertEqual(list(S.MODELS), list(M.FALLBACK_MODELS))
        for m, d in S.MODELS.items():
            for k in ("download_gb", "ram_gb", "arena_gb"):
                self.assertEqual(d[k], M.FALLBACK_MODELS[m][k], f"{m} {k}")
            for k in ("experimental", "vision"):
                self.assertEqual(d.get(k), M.FALLBACK_MODELS[m].get(k), f"{m} {k}")
            self.assertEqual(d.get("families", ("qwen", "swift")), M.FALLBACK_MODELS[m].get("families", ("qwen", "swift")))
        self.assertEqual(list(S.FAMILIES), list(M.FALLBACK_FAMILIES))
        for f, d in S.FAMILIES.items():
            self.assertEqual(d["tag"], M.FALLBACK_FAMILIES[f]["tag"])
        self.assertEqual(list(S.CONTEXTS), M.FALLBACK_CONTEXTS)

    def test_unsloth_sizes(self):
        """0.1.39: UD-IQ4_XS is a regular size, listed first; MODEL-UPDATE Phase 2: UD-Q4_K_XL is no longer experimental,
        and UD-Q5_K_XL, UD-Q6_K_XL and Q8_0 follow by size."""
        s = M.Strata(HERE.parent)
        for models in (s.tables()[0], M.FALLBACK_MODELS):
            self.assertEqual(s.sizes_of(models, "unsloth"), ["UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"])
            self.assertFalse(models["UD-IQ4_XS"].get("experimental"))
            self.assertFalse(models["UD-Q4_K_XL"].get("experimental"))
        self.assertEqual(s.sizes_of(M.FALLBACK_MODELS, "qwen"), ["Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S"])
        fam = next(f for f in s.model_table({"ram_gb": 63.7, "gpus": []}) if f["family"] == "unsloth")
        self.assertFalse(fam["experimental"])
        self.assertTrue(fam["images"])
        by = {x["model"]: x for x in fam["sizes"]}
        self.assertEqual((by["UD-IQ4_XS"]["experimental"], by["UD-IQ4_XS"]["images"]), (False, True))
        self.assertEqual((by["UD-Q4_K_XL"]["experimental"], by["UD-Q4_K_XL"]["images"]), (False, True))   # #967
        self.assertFalse(by["UD-IQ4_XS"]["on_this_pc"].startswith("experimental"))
        self.assertIn("read from the SSD", by["UD-Q4_K_XL"]["on_this_pc"])      # 64 GB: the RAM budget
        fam = next(f for f in s.model_table({"ram_gb": 127.8, "gpus": []}) if f["family"] == "unsloth")
        by = {x["model"]: x for x in fam["sizes"]}
        self.assertEqual(by["UD-Q4_K_XL"]["on_this_pc"], "fits (all of its experts in RAM)")
        self.assertIn("read from the SSD", by["Q8_0"]["on_this_pc"])

    def test_qwen36_is_listed_with_its_three_sizes(self):
        """QA 2026-10-18: the Qwen3.6-35B-A3B page of the first-run setup showed no versions, because setup.py keeps the family
        in OWN_PATH_FAMILIES (not FAMILIES) and the model list only walked FAMILIES."""
        s = M.Strata(HERE.parent)
        self.assertIn("qwen36", s.tables()[1])
        fam = next(f for f in s.model_table({"ram_gb": 63.7, "gpus": [{"index": 0, "name": "x", "vram_gb": 12.0, "usable": True}]})
                   if f["family"] == "qwen36")
        by = {x["model"]: x for x in fam["sizes"]}
        self.assertEqual(list(by), ["UD-Q4_K_XL", "UD-Q6_K_XL", "Q8_0"])
        self.assertEqual(by["UD-Q4_K_XL"]["id"], "qwen36-ud-q4_k_xl")          # the tag the install writes its config under
        self.assertEqual(by["UD-Q4_K_XL"]["download_gb"], 22.9)
        self.assertTrue(all(x["images"] and x["engine_min"] == "0.1.45" and x["gpu_options"] is None for x in by.values()))
        self.assertEqual({x["on_this_pc"] for x in by.values()}, {"fits"})
        small = next(f for f in s.model_table({"ram_gb": 16, "gpus": []}) if f["family"] == "qwen36")
        self.assertEqual({x["on_this_pc"] for x in small["sizes"]}, {"does not fit"})
        self.assertEqual(s.sizes_of(s.tables()[0], "unsloth")[-1], "Q8_0")      # the Unsloth Flash-Next sizes are untouched

    def test_qwen36_install_plan(self):
        s = M.Strata(HERE.parent)
        s.hardware = lambda fresh=False: {"ram_gb": 64.0, "gpus": [{"index": 0, "name": "x", "vram_gb": 12.0, "usable": True, "vendor": "nvidia"}],
                                          "cpu": {"avx2": True}}
        s.disk_free = lambda p: 1000.0
        plan, args = M.Tools(s).install_plan("qwen36", "UD-Q4_K_XL", 32768, "yes", "int8", None, None, None, "auto", None, None)
        self.assertEqual((plan["title"], plan["download_gb"], plan["family"]), ("Qwen3.6-35B-A3B UD-Q4_K_XL", 22.9, "qwen36"))
        self.assertEqual(args[:10], ["--yes", "--no-start", "--family", "qwen36", "--model", "UD-Q4_K_XL", "--context", "32768",
                                     "--vision", "yes"])
        for flag in ("--kv", "--low-ram", "--experimental-speed-projection"):      # Flash-Next settings: setup would only warn
            self.assertNotIn(flag, args)
        plan, _a = M.Tools(s).install_plan("qwen36", None, 32768, "no", None, None, None, None, None, None, None)
        self.assertEqual(plan["model"], "UD-Q4_K_XL")
        with self.assertRaises(M.ToolError):
            M.Tools(s).install_plan("qwen36", "IQ3_S", 32768, "no", None, None, None, None, None, None, None)

    def test_recommendation_follows_ram(self):
        s = M.Strata(HERE.parent)
        gpu = [{"index": 0, "name": "x", "vram_gb": 24.0, "usable": True, "vendor": "nvidia"}]
        cases = {96: ("qwen", "IQ3_S", 131072), 48: ("qwen", "IQ3_S", 131072), 32: ("qwen", "IQ3_XXS", 131072), 12: ("coder", "IQ1_M", 131072)}   # the owner's tiers count RAM + VRAM (24 GB card): 120, 72, 56 GB; the Coder where the older rules stand
        for ram, want in cases.items():
            r = s.recommend({"ram_gb": ram, "gpus": gpu, "cpu": {"avx2": True}})
            self.assertEqual((r["family"], r["model"], r["context"]), want, ram)
        r = s.recommend({"ram_gb": 16, "gpus": [{**gpu[0], "vram_gb": 8.0}], "cpu": {"avx2": True}})
        self.assertIsNone(r["model"])
        r = s.recommend({"ram_gb": 64, "gpus": [], "cpu": {"avx2": True}})
        self.assertIsNone(r["model"])
        r = s.recommend({"ram_gb": 64, "gpus": [{**gpu[0], "vram_gb": 12.0}], "cpu": {"avx2": True}})
        self.assertEqual(r["context"], 32768)

    def test_a_pick_that_reads_from_the_ssd_says_so(self):
        # #3425: 36 GB RAM + 16 GB VRAM - 5 GB is the 48 GB tier; about 8 GB of IQ3_XXS would come from the SSD
        s = M.Strata(HERE.parent)
        gpu = [{"index": 0, "name": "x", "vram_gb": 15.9, "usable": True, "vendor": "nvidia"}]
        r = s.recommend({"ram_gb": 36.0, "gpus": gpu, "cpu": {"avx2": True}})
        self.assertEqual((r["family"], r["model"]), ("qwen", "IQ3_XXS"))
        self.assertTrue(any("read from the SSD" in n for n in r["notes"]), r["notes"])
        roomy = s.recommend({"ram_gb": 64, "gpus": gpu, "cpu": {"avx2": True}})
        self.assertFalse(any("read from the SSD" in n for n in roomy["notes"]), roomy["notes"])

    def test_3425_the_tiers_leave_5_gb_for_windows(self):
        # the reporter: RTX 5080 (16 GB) + 32 GB of RAM used to sum to 48 GB and get IQ3_XXS; now the 40 GB tier
        s = M.Strata(HERE.parent)
        gpu = [{"index": 0, "name": "x", "vram_gb": 15.9, "usable": True, "vendor": "nvidia"}]
        self.assertEqual(M.TIER_SYSTEM_RESERVE_GB, 5.0)
        r = s.recommend({"ram_gb": 31.9, "gpus": gpu, "cpu": {"avx2": True}})
        self.assertEqual((r["family"], r["model"]), ("swift", "IQ2_XS"))
        # boundary: 43 GB of RAM + VRAM is the lowest 40 GB tier PC, 51 GB the lowest 48 GB tier PC
        def pick(ram, vram):
            g = [{**gpu[0], "vram_gb": vram}]
            r = s.recommend({"ram_gb": ram, "gpus": g, "cpu": {"avx2": True}})
            return r["family"], r["model"]
        self.assertEqual(pick(30.0, 13.0), ("swift", "IQ2_XS"))
        self.assertNotEqual(pick(30.0, 12.9), ("swift", "IQ2_XS"))
        self.assertEqual(pick(38.0, 13.0), ("qwen", "IQ3_XXS"))
        self.assertEqual(pick(38.0, 12.9), ("swift", "IQ2_XS"))
        self.assertEqual(pick(54.0, 13.0), ("qwen", "IQ3_S"))
        self.assertEqual(pick(54.0, 12.9), ("qwen", "IQ3_XXS"))


class PhysicalCores(unittest.TestCase):
    """#3417: the hardware report's cpu.cores is the physical cores; the Work profile takes 80% of it for --pool-workers."""

    def test_physical_not_logical(self):
        from strata_control import tune_machine
        with mock.patch.object(tune_machine, "cpu", return_value={"model": "x", "logical": 16, "physical": 8, "hybrid": False}),                 mock.patch.object(M.os, "cpu_count", return_value=16):
            self.assertEqual(M.physical_cores(), 8)

    def test_unknown_falls_back_to_threads(self):
        from strata_control import tune_machine
        with mock.patch.object(tune_machine, "cpu", return_value={"physical": None}), mock.patch.object(M.os, "cpu_count", return_value=16):
            self.assertEqual(M.physical_cores(), 16)
        with mock.patch.object(tune_machine, "cpu", side_effect=OSError("no")), mock.patch.object(M.os, "cpu_count", return_value=4):
            self.assertEqual(M.physical_cores(), 4)


class MacCatalog(unittest.TestCase):
    """A Mac (Apple Silicon): one pool of unified memory, judged by strata_control/mac_fit.py (the rules metal/setup_mac.py uses)."""

    def hw(self, ram, macos="27.2", chip="Apple M4"):
        from strata_control import mac_fit
        s = M.Strata(HERE.parent)
        info = {"chip": chip, "ram_gb": float(ram), "macos": macos, "arm64": True, "clt": "", "perf_cores": 4}
        with mock.patch.object(M.sys, "platform", "darwin"), mock.patch.object(mac_fit, "mac_info", return_value=info):
            hw = s._hardware()
        return s, hw

    def test_the_hardware_is_one_apple_gpu_with_unified_memory(self):
        s, hw = self.hw(32)
        self.assertEqual(hw["ram_gb"], 32.0)
        self.assertEqual(len(hw["gpus"]), 1)
        g = hw["gpus"][0]
        self.assertEqual((g["vendor"], g["name"], g["usable"], g["vram_gb"], g["unified_gb"]), ("apple", "Apple M4", True, 0.0, 32.0))
        self.assertEqual(hw["mac"]["unified_gb"], 32.0)
        self.assertFalse(hw["nvidia_smi"])

    def test_recommendation_by_unified_memory(self):
        cases = {32: ("Q2_0", 32768), 48: ("Q2_0", 32768), 80: ("IQ3_XXS", 32768), 128: ("IQ3_S", 65536)}
        for ram, (model, ctx) in cases.items():
            s, hw = self.hw(ram)
            r = s.recommend(hw)
            self.assertEqual((r["family"], r["model"], r["context"], r["backend"]), ("qwen", model, ctx, "metal"), ram)
            self.assertNotIn("NVIDIA", r["why"])
            self.assertFalse([n for n in r["notes"] if "UD-" in n], "the Unsloth sizes are not offered in the app yet")
        s, hw = self.hw(32)
        self.assertIn("read from the SSD", s.recommend(hw)["why"])
        s, hw = self.hw(128)
        self.assertIn("stays in memory", s.recommend(hw)["why"])

    def test_a_mac_too_small_or_too_old_has_no_recommendation_and_says_why(self):
        s, hw = self.hw(8)
        r = s.recommend(hw)
        self.assertIsNone(r["model"])
        self.assertIn("8 GB of unified memory", r["why"])
        s, hw = self.hw(32, macos="14.6")
        r = s.recommend(hw)
        self.assertIsNone(r["model"])
        self.assertIn("macOS 15 or newer", r["why"])

    def test_the_model_list_is_the_macs_sizes_with_unified_memory_verdicts(self):
        s, hw = self.hw(32)
        fams = s.model_table(hw)
        self.assertEqual([f["family"] for f in fams], ["qwen", "gyro", "deepseek41"],
                         "no Unsloth, Swift or NVFP4 on a Mac in the app; Gyro after the regular sizes, DeepSeek (0.2.7.1) last")
        by = {x["model"]: x for x in fams[0]["sizes"]}
        self.assertEqual(list(by), ["Q2_0", "IQ3_XXS", "IQ3_S"])
        gy = {x["model"]: x for x in fams[1]["sizes"]}
        self.assertEqual(list(gy), ["Gyro-S", "Gyro-M"])
        self.assertEqual(gy["Gyro-S"]["download_gb"], 58.5)
        self.assertTrue(all(v["on_this_pc"] and "SSD" in v["on_this_pc"] for v in gy.values()), "32 GB: both stream (said, not refused)")
        self.assertIn("read from the SSD", by["Q2_0"]["on_this_pc"])
        self.assertIn("slower", by["Q2_0"]["on_this_pc"])
        self.assertIn("~14 tok/s", by["Q2_0"]["on_this_pc"])
        self.assertTrue(by["IQ3_S"]["on_this_pc"].startswith("slow here"))
        self.assertTrue(all(x["images"] for x in by.values()), "the Mac engine carries the Metal picture reader: pictures are on")
        s, hw = self.hw(128)
        self.assertEqual({x["on_this_pc"] for f in s.model_table(hw) if f["family"] != "deepseek41" for x in f["sizes"]}, {"fits"})
        s, hw = self.hw(8)
        self.assertEqual({x["on_this_pc"] for f in s.model_table(hw) if f["family"] != "deepseek41" for x in f["sizes"]}, {"does not fit"})
        self.assertTrue(all(x["on_this_pc"].startswith("does not fit") for f in s.model_table(hw) if f["family"] == "deepseek41" for x in f["sizes"]))

    def test_deepseek_on_a_mac_offers_the_two_sets_that_were_run_with_the_mac_words(self):
        s, hw = self.hw(128)
        ds = next(f for f in s.model_table(hw) if f["family"] == "deepseek41")
        by = {x["model"]: x for x in ds["sizes"]}
        self.assertEqual(list(by), ["IST-DAS-2.00bpw", "IST-DAS-2.25bpw"], "the 1.75 bpw set was never run on a Mac")
        for x in by.values():
            self.assertTrue(x["multi_file"] and not x["images"])
            self.assertTrue(x["on_this_pc"].startswith("runs on this Mac's memory"), x["on_this_pc"])
            self.assertIn("read from the SSD", x["on_this_pc"])
            self.assertGreater(x["disk_gb"], x["download_gb"])
        s, hw = self.hw(32)
        ds = next(f for f in s.model_table(hw) if f["family"] == "deepseek41")
        for x in ds["sizes"]:                           # a small Mac: said plainly, never refused
            self.assertTrue(x["on_this_pc"].startswith("slow here"), x["on_this_pc"])
            self.assertIn("can still be installed", x["on_this_pc"])
        from strata_cli.launch import catalog as C
        self.assertEqual(C.verdict(by["IST-DAS-2.00bpw"]["on_this_pc"])[0], C.FITS)
        self.assertEqual(C.verdict(ds["sizes"][0]["on_this_pc"])[0], C.TIGHT)

    def test_the_install_plan_on_a_mac_has_no_card_kv_or_ram_options(self):
        s, hw = self.hw(32)
        s.hardware = lambda fresh=False: hw
        s.disk_free = lambda p: 1000.0
        with tempfile.TemporaryDirectory() as own:
            plan, args = M.Tools(s).install_plan("qwen", "Q2_0", None, "no", "int8", 0, None, None, "auto", None, None, own)
        self.assertEqual(plan["context"], 32768)
        self.assertEqual(args[:9], ["--yes", "--no-start", "--family", "qwen", "--model", "Q2_0", "--context", "32768", "--vision"])
        for flag in ("--kv", "--gpu", "--gpus", "--low-ram", "--backend"):
            self.assertNotIn(flag, args)
        self.assertIn("--gguf-dir", args)
        with tempfile.TemporaryDirectory() as own:      # nothing said about pictures: on, as everywhere (the Mac's reader is the Metal one)
            _plan, args = M.Tools(s).install_plan("qwen", "Q2_0", None, None, "int8", 0, None, None, "auto", None, None, own)
        self.assertEqual(args[args.index("--vision") + 1], "yes")
        with self.assertRaises(M.ToolError):
            s2, hw2 = self.hw(8)
            s2.hardware = lambda fresh=False: hw2
            M.Tools(s2).install_plan(None, None, None, None, None, None, None, None, None, None, None)

    def test_the_install_plan_for_deepseek_on_a_mac_is_text_only_at_8k(self):
        s, hw = self.hw(128)
        s.hardware = lambda fresh=False: hw
        s.disk_free = lambda p: 1000.0
        with tempfile.TemporaryDirectory() as own:
            plan, args = M.Tools(s).install_plan("deepseek41", "IST-DAS-2.00bpw", None, None, None, None, None, None, None, None, None, None)
        self.assertEqual(args[:9], ["--yes", "--no-start", "--family", "deepseek41", "--model", "IST-DAS-2.00bpw", "--context", "8192", "--vision"])
        self.assertEqual(args[args.index("--vision") + 1], "no", "the Metal engine has not run DeepSeek's picture tower")
        for flag in ("--kv", "--gpu", "--gpus", "--low-ram", "--backend"):
            self.assertNotIn(flag, args)
        self.assertTrue(plan["ram"].startswith("runs on this Mac's memory"))
        with self.assertRaises(M.ToolError):            # UD-Q8_K_XL (508 GB) and the 1.75 set were never run on a Mac
            M.Tools(s).install_plan("deepseek41", "UD-Q8_K_XL", None, None, None, None, None, None, None, None, None, None)
        with self.assertRaises(M.ToolError):
            M.Tools(s).install_plan("deepseek41", "IST-DAS-2.25bpw", None, "yes", None, None, None, None, None, None, None, None)

    def test_a_pc_is_unchanged(self):
        s = M.Strata(HERE.parent)
        r = s.recommend({"ram_gb": 96, "gpus": [{"index": 0, "name": "x", "vram_gb": 24.0, "usable": True, "vendor": "nvidia"}],
                         "cpu": {"avx2": True}})
        self.assertEqual((r["family"], r["model"]), ("qwen", "IQ3_S"))
        self.assertIn("unsloth", [f["family"] for f in s.model_table({"ram_gb": 96, "gpus": []})])


class MixedGraphics(unittest.TestCase):
    def test_an_integrated_amd_gpu_beside_an_nvidia_card_is_not_offered(self):
        s = M.Strata(HERE.parent)
        S = s.setup_module()
        nv = [{"index": 0, "name": "NVIDIA GeForce RTX 5090", "vram_gb": 31.8, "arch": "120", "driver": "580.95"}]
        igpu = [{"index": 0, "name": "AMD Radeon Graphics", "vram_gb": 0.5, "arch": "gfx1036", "uma": True}]
        with mock.patch.object(M.sys, "platform", "linux"), mock.patch.object(M, "WIN", False),                 mock.patch.object(S, "gpus", lambda: [dict(g) for g in nv]), mock.patch.object(S, "gpu_problem", lambda g: None),                 mock.patch.object(S, "amd_gpus", lambda: [dict(g) for g in igpu]), mock.patch.object(S, "amd_problem", lambda g: None):
            hw = s._hardware()
        by = {g["vendor"]: g for g in hw["gpus"]}
        self.assertTrue(by["nvidia"]["usable"])
        self.assertFalse(by["amd"]["usable"])
        self.assertIn("NVIDIA card", by["amd"]["problem"])
        # the RTX and the Ryzen iGPU both say index 0 (each make counts from 0), so the report gives each a stable id of its own
        self.assertEqual((by["nvidia"]["index"], by["amd"]["index"]), (0, 0))
        self.assertEqual(sorted(g["id"] for g in hw["gpus"]), ["amd:0", "nvidia:0"])
        rec = s.recommend({**hw, "ram_gb": 92.0})
        self.assertEqual((rec["backend"], rec["gpu"]["name"]), ("cuda", "NVIDIA GeForce RTX 5090"))
        # with no NVIDIA card the integrated GPU stays what it was
        with mock.patch.object(M.sys, "platform", "linux"), mock.patch.object(M, "WIN", False),                 mock.patch.object(S, "gpus", lambda: []), mock.patch.object(S, "amd_gpus", lambda: [dict(g) for g in igpu]),                 mock.patch.object(S, "amd_problem", lambda g: None):
            self.assertTrue(s._hardware()["gpus"][0]["usable"])

    def test_two_gtx_1080_are_a_working_pc_on_the_cuda_12_engine(self):
        """#3655: Pascal cards were flagged 'not supported' and both started Off, so the app said no card was usable. When Pascal is all the PC has
        (and no AMD card), setup runs them with the CUDA 12 engine, and the hardware report now says so; a newer card keeps the old behaviour."""
        s = M.Strata(HERE.parent)
        S = s.setup_module()
        pascal = [{"index": 0, "name": "NVIDIA GeForce GTX 1080", "vram_gb": 8.0, "arch": "61", "driver": "581.15"},
                  {"index": 1, "name": "NVIDIA GeForce GTX 1080", "vram_gb": 8.0, "arch": "61", "driver": "581.15"}]
        igpu = [{"index": 0, "name": "AMD Radeon(TM) Graphics", "vram_gb": 0.5, "arch": "gfx90c", "uma": True}]
        patches = (mock.patch.object(M.sys, "platform", "linux"), mock.patch.object(M, "WIN", False),
                   mock.patch.object(S, "amd_gpus", lambda: [dict(g) for g in igpu]), mock.patch.object(S, "amd_problem", lambda g: "integrated graphics"),
                   mock.patch.dict(M.os.environ, {"STRATA_EXPERIMENTAL_SM60": ""}), mock.patch.object(S, "OLD_GPUS", None))
        with mock.patch.object(S, "gpus", lambda: [dict(g) for g in pascal]):
            for p in patches:
                p.start()
            try:
                hw = s._hardware()
            finally:
                for p in reversed(patches):
                    p.stop()
        nv = [g for g in hw["gpus"] if g["vendor"] == "nvidia"]
        self.assertEqual([(g["usable"], g["problem"], g.get("engine_variant")) for g in nv], [(True, None, "cuda12")] * 2)
        # a newer card in the PC: Pascal stays an opt-in, the ready-made engine is recommended
        mixed = [{"index": 0, "name": "NVIDIA GeForce GTX 1080", "vram_gb": 8.0, "arch": "61", "driver": "581.15"},
                 {"index": 1, "name": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0, "arch": "86", "driver": "581.15"}]
        with mock.patch.object(S, "gpus", lambda: [dict(g) for g in mixed]):
            for p in patches:
                p.start()
            try:
                hw = s._hardware()
            finally:
                for p in reversed(patches):
                    p.stop()
        by = {g["name"]: g for g in hw["gpus"] if g["vendor"] == "nvidia"}
        self.assertFalse(by["NVIDIA GeForce GTX 1080"]["usable"])
        self.assertTrue(by["NVIDIA GeForce RTX 3060"]["usable"])

    def test_an_amd_card_the_installed_engine_has_no_code_for_is_not_usable(self):
        """#3203: the engine package lists its GPU code; a card outside it is flagged unusable with the plain reason, so no list offers it."""
        status = {"usable": True, "engine": {"backend": "hip", "archs": ["gfx1200", "gfx1201"]}}
        hw = {"gpus": [{"vendor": "amd", "index": 0, "name": "AMD Radeon 760M", "arch": "gfx1103", "usable": False, "problem": "integrated"},
                       {"vendor": "amd", "index": 1, "name": "AMD Radeon RX 6800 XT", "arch": "gfx1030", "usable": True, "problem": None},
                       {"vendor": "amd", "index": 2, "name": "AMD Radeon RX 9070 XT", "arch": "gfx1201", "usable": True, "problem": None},
                       {"vendor": "nvidia", "index": 0, "name": "NVIDIA GeForce RTX 3060", "arch": "86", "usable": True, "problem": None}]}
        with mock.patch("strata_control.engine_component.status", lambda: status):
            M.Strata._engine_code(hw)
        by = {g["name"]: g for g in hw["gpus"]}
        self.assertFalse(by["AMD Radeon RX 6800 XT"]["usable"])
        self.assertIn("no code for gfx1030", by["AMD Radeon RX 6800 XT"]["problem"])
        self.assertEqual(by["AMD Radeon 760M"]["problem"], "integrated", "an earlier reason stays")
        self.assertTrue(by["AMD Radeon RX 9070 XT"]["usable"])
        self.assertTrue(by["NVIDIA GeForce RTX 3060"]["usable"])
        # no engine package that lists archs (a source build, no engine yet): nothing is judged
        for other in ({"usable": False}, {"usable": True, "engine": {"backend": "cuda", "archs": ["sm_86"]}}):
            again = {"gpus": [{"vendor": "amd", "index": 1, "name": "x", "arch": "gfx1030", "usable": True, "problem": None}]}
            with mock.patch("strata_control.engine_component.status", lambda other=other: other):
                M.Strata._engine_code(again)
            self.assertTrue(again["gpus"][0]["usable"])

    @mock.patch("strata_control.engine_component.status", lambda: {"usable": False})   # not this PC's installed (CUDA) engine
    def test_windows_lists_an_amd_card_and_not_the_integrated_radeon(self):
        s = M.Strata(HERE.parent)
        S = s.setup_module()
        cards = [{"index": 0, "name": "AMD Radeon RX 9070 XT", "vram_gb": 15.9, "arch": "gfx1201", "driver": "32.0.1"},
                 {"index": 1, "name": "AMD Radeon(TM) Graphics", "vram_gb": 0.5, "arch": "unknown (PCI 164E)", "driver": "32.0.1"}]
        with mock.patch.object(M, "WIN", True), mock.patch.object(S, "gpus", lambda: []),                 mock.patch.object(S, "amd_gpus", lambda: [dict(g) for g in cards]),                 mock.patch.object(M, "windows_video_controllers", lambda: [{"name": "AMD Radeon RX 9070 XT", "driver": "32"}]),                 mock.patch.object(S, "ram_gb", lambda: 64.0), mock.patch.object(S, "page_file_gb", lambda: 10.0):
            hw = s._hardware()
            by = {g["name"]: g for g in hw["gpus"]}
            self.assertTrue(by["AMD Radeon RX 9070 XT"]["usable"])
            self.assertFalse(by["AMD Radeon(TM) Graphics"]["usable"])
            rec = s.recommend({**hw, "ram_gb": 64.0})
            self.assertEqual((rec["backend"], rec["gpu"]["name"]), ("hip", "AMD Radeon RX 9070 XT"))


if __name__ == "__main__":
    unittest.main()


class RemoveMissingFilesTest(unittest.TestCase):
    """Deleting a model whose plan lists files that are already gone (or inside a folder removed earlier) is a success, not 'Some files could not be removed'."""

    def test_missing_paths_are_already_removed(self):
        import importlib.util, tempfile, threading
        from pathlib import Path
        spec = importlib.util.spec_from_file_location("strata_mcp_rm", Path(__file__).with_name("strata_mcp.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = root / "Gyro-M"
            folder.mkdir()
            (folder / "a.gguf").write_bytes(b"x")
            (folder / "b.gguf").write_bytes(b"y")
            gone = root / "already-gone.gguf"
            cfg = root / "model.json"
            cfg.write_text("{}")
            plan = {"model": "gyro-m", "gb": 0.1, "remove": [
                {"path": str(folder)}, {"path": str(folder / "a.gguf")}, {"path": str(gone)}, {"path": str(cfg)}]}

            class S:
                lock = threading.RLock()
                def find_config(self, model): return cfg
                def removal_plan(self, config): return plan
                def install_job(self): return None
                def tracked_server(self): return None
                def model_paths(self, c): return ([], [])
                def read_config(self, c): return {}

            result = mod.Tools(S()).strata_remove(model="gyro-m", confirm=True)
            self.assertEqual(result["failed"], [], result)
            self.assertFalse(folder.exists()); self.assertFalse(cfg.exists())
            again = mod.Tools(S()).strata_remove(model="gyro-m", confirm=True)
            self.assertEqual(again["failed"], [], "a second delete of the same model finds nothing left and still succeeds")


class DiskUsageTest(unittest.TestCase):
    """QA 2026-10-18 (F02): Storage showed IQ3_S as 0.0 GB while it uses 84 GB, and Gyro-S as 119 GB while it is 59 GB.  The sizes come from the real
    files: every part of a model in parts, the picture reader, the packs; a file one config names through several flags counts once; a
    file two models share counts once in the total; a model's own figure leaves out another model's files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = Path(self.tmp.name)
        self.root, self.data, self.own = d / "install", d / "data", d / "owner-folder"
        for p in (self.root, self.data / "models" / "gyro-Gyro-S", self.data / "packs" / "swift-iq2_xs", self.data / "mtp" / "rt", self.own):
            p.mkdir(parents=True)
        (d / "settings.json").write_text(json.dumps({"data_dir": str(self.data)}), encoding="utf-8")
        env = mock.patch.dict(os.environ, {"STRATA_SETTINGS_FILE": str(d / "settings.json")})
        env.start()
        self.addCleanup(env.stop)

    def blob(self, path: Path, size: int) -> str:
        path.write_bytes(b"x" * size)
        return str(path)

    def write_config(self, name: str, args: list, vision=None):
        cfg = {"exe": sys.executable, "args": args, "port": 8080}
        if vision:
            cfg["vision"] = {"mmproj": vision}
        (self.root / f"strata-{name}.json").write_text(json.dumps(cfg), encoding="utf-8")

    def test_sizes_come_from_the_files_not_from_what_may_be_deleted(self):
        gyro = self.blob(self.data / "models" / "gyro-Gyro-S" / "Gyro-S.gguf", 59_000)
        reader = self.blob(self.data / "models" / "mmproj.gguf", 900)
        part1 = self.blob(self.own / "Model-IQ3_S-00001-of-00002.gguf", 30_000)
        self.blob(self.own / "Model-IQ3_S-00002-of-00002.gguf", 54_000)
        # Gyro: the one file is --native, --ple-gguf, --embd-gguf and --native-dense-gguf at once
        self.write_config("gyro-gyro-s", ["--native", gyro, "--ple-gguf", gyro, "--embd-gguf", gyro, "--native-dense-gguf", gyro], reader)
        # IQ3_S: adopted from the person's own folder, the config names the first part only
        self.write_config("iq3_s", ["--native", part1], reader)
        s = M.Strata(self.root)
        use = s.disk_usage()
        self.assertEqual(use["models"]["gyro-gyro-s"]["bytes"], 59_000, "the Gyro file is counted once, not four times")
        self.assertEqual(use["models"]["iq3_s"]["bytes"], 84_000, "both parts of IQ3_S, although the config names the first")
        self.assertEqual(use["shared_bytes"], 900, "the picture reader both use is one file")
        self.assertEqual(use["total_bytes"], 59_000 + 84_000 + 900, "rows plus the shared files add up to the total")
        self.assertEqual(use["models"]["iq3_s"]["shared_bytes"], 900)
        # deleting IQ3_S would remove nothing: its files are the person's own
        by = {d["model"]: d for d in (s.describe_config(c, use) for c in s.configs())}
        self.assertEqual(by["iq3_s"]["disk_bytes"], 84_000)
        self.assertEqual(by["iq3_s"]["removable_gb"], 0.0)
        self.assertEqual(by["gyro-gyro-s"]["disk_bytes"], 59_000)

    def test_a_pack_folder_and_the_shared_draft_layer_count_once(self):
        gguf = self.blob(self.data / "models" / "gyro-Gyro-S" / "m.gguf", 10_000)
        pack = self.data / "packs" / "swift-iq2_xs"
        self.blob(pack / "experts.bin", 40_000)
        self.blob(pack / "index.txt", 100)
        mtp = self.data / "mtp" / "rt"
        self.blob(mtp / "experts.bin", 5_000)
        self.write_config("swift-iq2_xs", ["--native", gguf, "--pack", str(pack), "--mtp", str(mtp)])
        link_target = self.data / "models" / "gyro-Gyro-S" / "m2.gguf"
        try:
            os.link(gguf, link_target)                      # a hard link: the same bytes under another name
        except OSError:
            self.skipTest("no hard links here")
        self.write_config("other", ["--native", str(link_target), "--mtp", str(mtp)])
        use = M.Strata(self.root).disk_usage()
        self.assertEqual(use["models"]["swift-iq2_xs"]["bytes"], 40_100)
        self.assertEqual(use["models"]["other"]["bytes"], 0)
        self.assertEqual(use["shared_bytes"], 10_000 + 5_000, "the hard-linked GGUF and the draft layer are shared")
        self.assertEqual(use["total_bytes"], 55_100)

    def test_a_model_with_a_part_missing_is_not_ready(self):
        part1 = self.blob(self.own / "Model-00001-of-00003.gguf", 10)
        self.blob(self.own / "Model-00002-of-00003.gguf", 10)
        self.write_config("three-parts", ["--native", part1])
        s = M.Strata(self.root)
        desc = s.describe_config(s.configs()[0])
        self.assertFalse(desc["ready"], "one part of three is on disk: not downloaded")
        self.assertTrue(any(m.endswith("Model-00003-of-00003.gguf") for m in desc["missing_files"]), desc["missing_files"])
        self.blob(self.own / "Model-00003-of-00003.gguf", 10)
        desc = s.describe_config(s.configs()[0])
        self.assertEqual([m for m in desc["missing_files"] if m.endswith(".gguf")], [])


class DraftLayerAndCancel(unittest.TestCase):
    """#3257: the draft layer's download shows real progress, and a cancel ends the whole tree."""

    def test_step_6_shows_the_draft_layer_download_added_to_the_finished_model_files(self):
        lines = ["=== Step 6: preparing the model for Strata ===", "  > python mtp_fetch.py fetch",
                 "  MTP draft layer:   2.00 / 5.00 GB (40%)"]
        with tempfile.TemporaryDirectory() as d:
            shard = Path(d) / "m-00001-of-00001.gguf"
            shard.write_bytes(b"x" * 20_000_000)
            job = {"files": [{"path": str(shard), "bytes": None, "role": "model"}], "total_gb": 60}
            prog = M.combined_progress(M.install_progress(lines), job)
        self.assertEqual(prog["download"]["file"], "Draft layer")
        self.assertEqual([p["kind"] for p in prog["parts"]], ["model", "draft"])
        self.assertEqual(prog["parts"][0]["state"], "done")
        self.assertAlmostEqual(prog["parts"][1]["done_gb"], 2.0)
        self.assertAlmostEqual(prog["parts"][1]["total_gb"], 5.0)
        self.assertGreater(prog["download"]["total_gb"], 5.0)
        self.assertEqual(prog["download"]["percent"], min(99, int(100 * prog["download"]["done_gb"] / prog["download"]["total_gb"])))

    def test_without_a_file_list_the_draft_layer_alone_is_shown(self):
        prog = M.combined_progress(M.install_progress(["=== Step 6: x ===", "  MTP draft layer:   5.00 / 5.00 GB (100%)"]), {})
        self.assertEqual([(p["kind"], p["state"]) for p in prog["parts"]], [("draft", "done")])

    def test_other_step_6_output_still_shows_no_download(self):
        prog = M.combined_progress(M.install_progress(["=== Step 6: x ===", "  IQ2_XS:   0.00 / 0.00 GB (0%)"]), {})
        self.assertNotIn("download", prog)

    @unittest.skipUnless(M.WIN, "Windows process trees")
    def test_cancel_ends_a_grandchild_even_when_taskkill_does_not_reach_it(self):
        """setup.py -> tools/*.py: taskkill /T walks parent ids; a snapshot taken first still finds the grandchild."""
        code = ("import subprocess,sys,time; c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']);"
                "print(c.pid, flush=True); time.sleep(120)")
        top = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        grand = int(top.stdout.readline())
        try:
            ident, gident = M.proc_identity(top.pid), M.proc_identity(grand)
            self.assertTrue(ident and gident)
            self.assertIn((grand, gident), M.proc_tree_snapshot(top.pid))
            real_run = subprocess.run

            def no_taskkill(cmd, *a, **kw):                  # the case of #3257: taskkill ends the parent only
                if cmd and cmd[0] == "taskkill":
                    return real_run(["taskkill", "/PID", str(top.pid), "/F"], capture_output=True)
                return real_run(cmd, *a, **kw)
            with mock.patch.object(M.subprocess, "run", no_taskkill):
                M.proc_kill_tree(top.pid, ident)
            M.wait_gone(grand, gident, 10)
            self.assertFalse(M.proc_alive(grand, gident), "the grandchild kept running")
            self.assertFalse(M.proc_alive(top.pid, ident))
        finally:
            for pid in (grand, top.pid):
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            top.wait(10)
