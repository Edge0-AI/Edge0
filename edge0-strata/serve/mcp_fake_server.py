"""serve/mcp_fake_server.py - a small MCP server for serve/test_mcp.py (no MCP SDK, no network beyond localhost).

    python serve/mcp_fake_server.py                  stdio: JSON-RPC lines on stdin/stdout
    python serve/mcp_fake_server.py --http 8765      Streamable HTTP on 127.0.0.1:8765/mcp

Tools: echo (returns its text), fail (a tool error, isError), sleep (answers after `seconds`), big (`n` characters),
die (the process exits mid-call), add (two integers).  tools/list pages two tools at a time (`--page N`), so the
client's pagination is exercised.  `--crash-at-start` exits before answering anything.  With FAKE_MCP_LOG set, the
server appends what it saw (calls, cancellations) to that file, one JSON line each.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOOLS = [
    {"name": "echo", "description": "Returns the text it is given.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "fail", "description": "Always fails.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "sleep", "description": "Waits, then answers.",
     "inputSchema": {"type": "object", "properties": {"seconds": {"type": "number"}}}},
    {"name": "big", "description": "Returns n characters.",
     "inputSchema": {"type": "object", "properties": {"n": {"type": "integer"}}}},
    {"name": "die", "description": "Ends the server.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "add", "description": "Adds two integers.",
     "inputSchema": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}},
]
PAGE = 2
cancelled: set = set()


def log(event: dict):
    path = os.environ.get("FAKE_MCP_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")


def text(t, error=False):
    return {"content": [{"type": "text", "text": t}], "isError": error}


def handle(msg: dict):
    """-> the response for a request, None for a notification."""
    method, rid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if "id" not in msg:
        if method == "notifications/cancelled":
            cancelled.add(params.get("requestId"))
            log({"cancelled": params.get("requestId")})
        return None
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": params.get("protocolVersion", "2025-06-18"), "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake", "version": "0.1"}}}
    if method == "tools/list":
        start = int(params.get("cursor") or 0)
        page = {"tools": TOOLS[start:start + PAGE]}
        if start + PAGE < len(TOOLS):
            page["nextCursor"] = str(start + PAGE)
        return {"jsonrpc": "2.0", "id": rid, "result": page}
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        log({"call": name, "arguments": args})
        if name == "echo":
            result = text(str(args.get("text", "")))
        elif name == "fail":
            result = text("it failed on purpose", error=True)
        elif name == "sleep":
            time.sleep(float(args.get("seconds", 1)))
            result = text("slept")
        elif name == "big":
            result = text("y" * int(args.get("n", 100000)))
        elif name == "die":
            sys.stderr.write("dying on purpose\n")
            sys.stderr.flush()
            os._exit(3)
        elif name == "add":
            result = text(str(int(args.get("a", 0)) + int(args.get("b", 0))))
        else:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": f"unknown tool {name}"}}
        return {"jsonrpc": "2.0", "id": rid, "result": result}
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"no method {method}"}}


def stdio():
    for f in (sys.stdin, sys.stdout, sys.stderr):      # MCP's stdio is UTF-8, whatever the Windows code page is
        f.reconfigure(encoding="utf-8")
    out = threading.Lock()

    def answer(msg):
        r = handle(msg)
        if r is not None and msg.get("id") not in cancelled:
            with out:
                sys.stdout.write(json.dumps(r) + "\n")
                sys.stdout.flush()

    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        if msg.get("method") == "tools/call":           # calls on threads: a slow one must not block cancellation
            threading.Thread(target=answer, args=(msg,), daemon=True).start()
        else:
            answer(msg)


def http_server(port: int = 0) -> ThreadingHTTPServer:
    """Streamable HTTP at /mcp: JSON answers, except tools/call, which answers as an event stream (a progress
    notification first); a session id from initialize is required afterwards."""
    sessions = set()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):
            pass

        def do_POST(self):
            msg = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if msg.get("method") != "initialize" and self.headers.get("Mcp-Session-Id") not in sessions:
                self.send_response(400)
                self.end_headers()
                return
            r = handle(msg)
            if r is None:
                self.send_response(202)
                self.end_headers()
                return
            if msg.get("method") == "tools/call":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                note = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}}
                self.wfile.write(b"event: message\ndata: " + json.dumps(note).encode() + b"\n\n")
                self.wfile.write(b"event: message\ndata: " + json.dumps(r).encode() + b"\n\n")
                return
            body = json.dumps(r).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if msg.get("method") == "initialize":
                sid = f"s-{len(sessions) + 1}"
                sessions.add(sid)
                self.send_header("Mcp-Session-Id", sid)
            self.end_headers()
            self.wfile.write(body)

        def do_DELETE(self):
            sessions.discard(self.headers.get("Mcp-Session-Id"))
            self.send_response(200)
            self.end_headers()

    httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


if __name__ == "__main__":
    if "--page" in sys.argv:
        PAGE = int(sys.argv[sys.argv.index("--page") + 1])
    if "--crash-at-start" in sys.argv:
        sys.stderr.write("cannot start: missing configuration\n")
        sys.exit(2)
    if "--http" in sys.argv:
        server = http_server(int(sys.argv[sys.argv.index("--http") + 1]))
        print(f"listening on http://127.0.0.1:{server.server_address[1]}/mcp", flush=True)
        threading.Event().wait()
    stdio()
