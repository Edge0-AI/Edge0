"""HTTP app layer (Flask optional; stdlib fallback).

Preference order: if ``flask`` is importable it is used (the full
OpenAI-compatible surface with SSE streaming); otherwise a minimal
stdlib ``http.server`` handler serves the same JSON contract
(non-streaming).  Both share ``build_app_handlers`` so the route
semantics stay identical.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from edge0.server.chat import (QueueServer, parse_chat_request, sse_format,
                               decode_tokens)

try:  # pragma: no cover - environment dependent
    from flask import Flask, Response, jsonify, request  # type: ignore

    _HAS_FLASK = True
except ImportError:  # pragma: no cover
    _HAS_FLASK = False


def _error(status: int, msg: str) -> tuple:
    if _HAS_FLASK:
        return jsonify({"error": {"message": msg, "type": "invalid_request"}}), status
    return (json.dumps({"error": {"message": msg}}), status)


def _split_think(text: str, think: bool):
    """ling-mlx-server server.py parity: with thinking enabled the model
    streams the reasoning block first and closes it with ``</think>``
    (token 156904); everything before is ``reasoning_content``, after
    is ``content``.  A generation cut off inside the reasoning block
    surfaces what it has so the UI isn't a blank wait."""
    if think:
        if "</think>" in text:
            r, c = text.split("</think>", 1)
            return r.strip(), c.strip()
        return text.strip(), ""
    return "", text


def _chat_once(server: QueueServer, payload: dict):
    req = parse_chat_request(payload)
    tokens, meta = server.chat(req)
    text = decode_tokens(server.engine, tokens)
    think = bool(req.enable_thinking if req.enable_thinking is not None
                 else getattr(server.engine, "think", False))
    reasoning, content = _split_think(text, think)
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": server.model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
                "reasoning_content": reasoning,
            },
            "finish_reason": "stop",
        }],
        "usage": meta["usage"],
    }


def _chat_stream(server: QueueServer, payload: dict):
    req = parse_chat_request(payload)
    events: list[bytes] = []

    def on_token(tid: int):
        text = decode_tokens(server.engine, [tid])
        events.append(sse_format({
            "id": "chatcmpl-1", "object": "chat.completion.chunk",
            "created": int(time.time()), "model": server.model_name,
            "choices": [{"index": 0,
                         "delta": {"content": text},
                         "finish_reason": None}],
        }).encode("utf-8"))

    tokens, meta = server.chat(req, on_token=on_token)
    events.append(sse_format({
        "id": "chatcmpl-1", "object": "chat.completion.chunk",
        "created": int(time.time()), "model": server.model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": meta["usage"],
    }).encode("utf-8"))
    events.append(b"data: [DONE]\n\n")
    return b"".join(events)


def build_app_handlers(server: QueueServer):
    """Return a handler dispatch dict shared by both transports."""

    def handle_chat(payload: dict):
        if payload.get("stream"):
            if _HAS_FLASK:
                return Response(_chat_stream(server, payload),
                                mimetype="text/event-stream")
            return (_chat_stream(server, payload), 200,
                    {"Content-Type": "text/event-stream"})
        return _chat_once(server, payload)

    handlers = {
        "GET /healthz": lambda: {"status": "ok", "model": server.model_name},
        "GET /v1/models": lambda: {
            "object": "list",
            "data": [{
                "id": server.model_name,
                "object": "model",
                "owned_by": "edge0",
            }],
        },
        "POST /v1/chat/completions": handle_chat,
        "POST /v1/completions": lambda p: _error(
            400, "text completions not supported; use /v1/chat/completions"),
    }
    return handlers


def create_app(server: QueueServer):
    """Flask app (raises ImportError when flask is missing)."""
    if not _HAS_FLASK:
        raise ImportError("flask is required for create_app(); "
                          "use run_stdlib instead")
    app = Flask("edge0")
    handlers = build_app_handlers(server)

    @app.get("/healthz")
    def healthz():
        return handlers["GET /healthz"]()

    @app.get("/v1/models")
    def models():
        return handlers["GET /v1/models"]()

    @app.post("/v1/chat/completions")
    def chat():
        payload = request.get_json(force=True, silent=True) or {}
        out = handlers["POST /v1/chat/completions"](payload)
        if isinstance(out, Response):
            return out
        if isinstance(out, tuple) and out and isinstance(out[0], Response):
            return out
        return jsonify(out)

    @app.post("/v1/completions")
    def completions():
        payload = request.get_json(force=True, silent=True) or {}
        out = handlers["POST /v1/completions"](payload)
        if isinstance(out, tuple):
            return out
        return jsonify(out)

    return app


class _StdlibHandler(BaseHTTPRequestHandler):
    server_q = None  # type: QueueServer

    def _json(self, status: int, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        handlers = build_app_handlers(self.server_q)
        if path == "/healthz":
            self._json(200, handlers["GET /healthz"]())
        elif path == "/v1/models":
            self._json(200, handlers["GET /v1/models"]())
        else:
            self._json(404, {"error": {"message": f"no route {path}"}})

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        handlers = build_app_handlers(self.server_q)
        if path not in ("/v1/chat/completions", "/v1/completions"):
            self._json(404, {"error": {"message": f"no route {path}"}})
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        out = handlers[("POST " + path)](payload)
        if isinstance(out, tuple):
            body, status, headers = out
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json(200, out)

    def log_message(self, fmt, *args):  # quiet by default
        pass


def run_stdlib(server: QueueServer, host: str, port: int):
    handler = type("Edge0Handler", (_StdlibHandler,),
                   {"server_q": server})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.serve_forever()


def run_server(server: QueueServer, host: str = "127.0.0.1",
               port: int = 8000, use_flask: bool | None = None):
    """Run the HTTP server in the foreground (blocking)."""
    if use_flask is None:
        use_flask = _HAS_FLASK
    if use_flask:
        app = create_app(server)
        app.run(host=host, port=port, threaded=True)
    else:
        run_stdlib(server, host, port)
