"""Bounded subprocess and local HTTP regressions for MCP transport ownership."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from serve.mcp import DEFAULTS, HttpTransport, McpCancelled, McpError, McpServer, McpTimeout, StdioTransport


def captured(fn):
    result = {}

    def run():
        try:
            result["value"] = fn()
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, result


@unittest.skipIf(os.name == "nt", "POSIX pipe/process-group regression")
class StdioOwnership(unittest.TestCase):
    def transport(self, source, *args):
        transport = StdioTransport("lifecycle", {"command": sys.executable, "args": ["-u", "-c", source, *args]})
        transport.start()
        self.addCleanup(self.stop, transport)
        return transport

    @staticmethod
    def stop(transport):
        try:
            os.killpg(transport.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        transport.proc.wait(timeout=3)
        transport.close()

    def test_full_stdin_pipe_obeys_request_deadline(self):
        transport = self.transport("import time; time.sleep(30)")
        worker, result = captured(lambda: transport.request("tools/call", {"text": "x" * 2**20}, 0.1))
        worker.join(0.8)
        done = not worker.is_alive()
        if not done:
            self.stop(transport)
            worker.join(3)
        self.assertTrue(done, "the request stayed blocked in the child's full stdin pipe")
        self.assertIsInstance(result.get("error"), McpTimeout, result)

    def test_full_stdin_pipe_obeys_cancellation(self):
        transport = self.transport("import time; time.sleep(30)")
        cancel = threading.Event()
        worker, result = captured(lambda: transport.request("tools/call", {"text": "x" * 2**20}, 30, cancel))
        time.sleep(0.1)
        cancel.set()
        worker.join(0.8)
        done = not worker.is_alive()
        if not done:
            self.stop(transport)
            worker.join(3)
        self.assertTrue(done, "cancellation stayed blocked in the child's full stdin pipe")
        self.assertIsInstance(result.get("error"), McpCancelled, result)

    def test_concurrent_large_requests_preserve_message_boundaries(self):
        source = """import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':{'length':len(msg['params']['text'])}}), flush=True)
"""
        transport = self.transport(source)
        calls = [captured(lambda n=n: transport.request("echo", {"text": "x" * n}, 3))
                 for n in (100000, 100001, 100002, 100003)]
        for n, (worker, result) in zip((100000, 100001, 100002, 100003), calls):
            worker.join(4)
            self.assertFalse(worker.is_alive())
            self.assertEqual(result, {"value": {"length": n}})

    def test_close_ends_descendant_after_parent_exits_on_eof(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "child-pid"
            source = """import pathlib, subprocess, sys
child = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)'])
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
sys.stdin.buffer.read()
"""
            transport = self.transport(source, str(marker))
            deadline = time.monotonic() + 3
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(marker.exists())
            worker, result = captured(transport.close)
            worker.join(3.5)
            done = not worker.is_alive()
            root_exit = transport.proc.poll()
            if not done:
                self.stop(transport)
                worker.join(3)
            self.assertEqual(root_exit, 0, "the root process should exit normally on stdin EOF")
            self.assertTrue(done, "a descendant kept stdout/stderr and close() blocked")
            self.assertNotIn("error", result)

    def test_invalid_response_id_does_not_disable_later_responses(self):
        source = """import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if 'id' in msg:
        print(json.dumps({'jsonrpc':'2.0','id':[],'result':{}}), flush=True)
        print(json.dumps({'jsonrpc':'2.0','id':{},'result':{}}), flush=True)
        print(json.dumps({'jsonrpc':'2.0','id':True,'result':{'wrong':True}}), flush=True)
        print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':{'ok':True}}), flush=True)
"""
        transport = self.transport(source)
        self.assertEqual(transport.request("ping", {}, 0.5), {"ok": True})
        self.assertEqual(transport.request("ping", {}, 0.5), {"ok": True})

    def test_malformed_initialize_result_fails_and_reaps_server(self):
        source = """import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if 'id' in msg:
        print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':{'serverInfo':'invalid'}}), flush=True)
"""
        server = McpServer("malformed", {"command": sys.executable, "args": ["-u", "-c", source]}, DEFAULTS)
        try:
            self.assertFalse(server.start())
            self.assertEqual(server.status, "failed")
            self.assertIsNotNone(server.transport.proc.poll())
        finally:
            if server.transport is not None:
                self.stop(server.transport)

    def test_malformed_results_and_tool_names_fail_initialization(self):
        for bad_result in ([], "not an object", {"tools": "not an array"}, {"tools": [{"name": 7}]}):
            with self.subTest(result=bad_result):
                source = """import json, sys
bad = json.loads(sys.argv[1])
for line in sys.stdin:
    msg = json.loads(line)
    if 'id' in msg:
        result = {'serverInfo':{}} if msg['method']=='initialize' else bad
        print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':result}), flush=True)
"""
                server = McpServer("malformed", {"command": sys.executable,
                                   "args": ["-u", "-c", source, json.dumps(bad_result)]}, DEFAULTS)
                try:
                    self.assertFalse(server.start())
                    self.assertEqual(server.status, "failed")
                    self.assertEqual(server.tools, [])
                    self.assertIsNotNone(server.transport.proc.poll())
                finally:
                    if server.transport is not None:
                        self.stop(server.transport)


    def test_close_does_not_signal_a_retired_process_group_twice(self):
        transport = self.transport("import sys; sys.stdin.buffer.read()")
        transport.close()
        with patch("serve.mcp.os.killpg", side_effect=AssertionError("a retired process group was signalled")):
            transport.close()

    def test_close_does_not_signal_a_process_group_after_both_pipes_ended(self):
        transport = self.transport("pass")
        transport.proc.wait(timeout=3)
        transport._reader.join(2)
        transport._err_reader.join(2)
        self.assertFalse(transport._reader.is_alive())
        self.assertFalse(transport._err_reader.is_alive())
        with patch("serve.mcp.os.killpg", side_effect=AssertionError("an expired process group was signalled")):
            transport.close()


class HttpOwnership(unittest.TestCase):
    def setUp(self):
        self.stop_stream = threading.Event()
        self.stream_ended = threading.Event()
        self.server_response = threading.Event()
        self.notification = threading.Event()
        self.messages = []
        case = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                message = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                case.messages.append(message)
                if "method" not in message:
                    case.server_response.set()
                    self.send_response(202)
                    self.end_headers()
                    return
                if "id" not in message:
                    case.notification.set()
                    self.send_response(202)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    if message["method"] == "ping-before-result":
                        self.wfile.write(b'data: {"jsonrpc":"2.0","id":"server-ping","method":"ping"}\n\n')
                        self.wfile.flush()
                        if case.server_response.wait(0.7):
                            response = {"jsonrpc": "2.0", "id": message["id"], "result": {"ok": True}}
                            self.wfile.write(("data: " + json.dumps(response) + "\n\n").encode())
                            self.wfile.flush()
                    else:
                        while not case.stop_stream.wait(0.02):
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                except (ConnectionError, OSError):
                    pass
                finally:
                    case.stream_ended.set()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.http_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.http_thread.start()
        self.transport = HttpTransport("lifecycle", {"url": f"http://127.0.0.1:{self.httpd.server_port}/mcp"})

    def tearDown(self):
        self.stop_stream.set()
        self.transport.close()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.http_thread.join(3)





if __name__ == "__main__":
    unittest.main()
