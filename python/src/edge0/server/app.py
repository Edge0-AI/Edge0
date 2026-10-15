"""HTTP app layer (Flask optional; stdlib fallback).

Preference order: if ``flask`` is importable it is used (the full
OpenAI-compatible surface with SSE streaming); otherwise a minimal
stdlib ``http.server`` handler serves the same JSON contract and SSE
streaming.  Both share ``build_app_handlers`` so the route
semantics stay identical.
"""

from __future__ import annotations

import json
import queue
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
    return {"error": {"message": msg, "type": "invalid_request"}}, status


def _split_think(text: str, think: bool):
    """With thinking enabled the model
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


def _extract_tool_calls(server: QueueServer, req, content: str):
    """(content, tool_calls) after stripping any ``<tool_call>`` block the
    engine's chat template asked the model to emit. ``tool_calls`` is None
    when the engine has no parser (unsupported family) or the request
    didn't ask for tools, or when no call was found in ``content``."""
    if req.tool_choice == "none" or (not req.tools
                                     and req.tool_choice != "required"):
        return content, None
    parse = getattr(server.engine, "parse_tool_calls", None)
    if not callable(parse):
        return content, None
    content, calls = parse(content)
    return content, (calls or None)


def _chat_once(server: QueueServer, payload: dict):
    req = parse_chat_request(payload)
    tokens, meta = server.chat(req)
    text = decode_tokens(server.engine, tokens)
    think = bool(req.enable_thinking if req.enable_thinking is not None
                 else getattr(server.engine, "think", False))
    reasoning, content = _split_think(text, think)
    content, tool_calls = _extract_tool_calls(server, req, content)
    message = {
        "role": "assistant",
        "content": content,
        "reasoning_content": reasoning,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": server.model_name,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }],
        "usage": meta["usage"],
    }


def _chat_stream(server: QueueServer, payload: dict):
    req = parse_chat_request(payload)
    events = queue.Queue()
    finished = object()
    request_id = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())
    # Tool-call XML (<tool_call>...</tool_call>) must not leak into content
    # deltas verbatim -- the whole point of #115 is that a client asking
    # for tools gets a parsed message.tool_calls, not raw template markup.
    # Detecting a block incrementally, token by token, needs a lookback
    # buffer for a tag that can straddle token boundaries; simpler and
    # just as correct: buffer the full response and parse once, same as
    # the non-streaming path. Requests with no tools (the common case)
    # keep the existing immediate per-token streaming, unchanged.
    buffering = bool(req.tools) and req.tool_choice != "none" and callable(
        getattr(server.engine, "parse_tool_calls", None))

    def on_token(tid: int):
        if buffering:
            return
        text = decode_tokens(server.engine, [tid])
        events.put(sse_format({
            "id": request_id, "object": "chat.completion.chunk",
            "created": created, "model": server.model_name,
            "choices": [{"index": 0,
                         "delta": {"content": text},
                         "finish_reason": None}],
        }).encode("utf-8"))

    def produce():
        try:
            tokens, meta = server.chat(req, on_token=on_token)
            finish_reason = "stop"
            delta = {}
            if buffering:
                text = decode_tokens(server.engine, tokens)
                think = bool(
                    req.enable_thinking if req.enable_thinking is not None
                    else getattr(server.engine, "think", False))
                _, content = _split_think(text, think)
                content, tool_calls = _extract_tool_calls(
                    server, req, content)
                if tool_calls:
                    delta = {"content": content, "tool_calls": [
                        {"index": i, **call} for i, call in enumerate(tool_calls)
                    ]}
                    finish_reason = "tool_calls"
                else:
                    delta = {"content": content}
            events.put(sse_format({
                "id": request_id, "object": "chat.completion.chunk",
                "created": created, "model": server.model_name,
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": finish_reason}],
                "usage": meta["usage"],
            }).encode("utf-8"))
            events.put(b"data: [DONE]\n\n")
        except Exception as exc:  # pragma: no cover - transport-dependent
            events.put(sse_format({
                "error": {"message": str(exc), "type": "server_error"},
            }).encode("utf-8"))
        finally:
            events.put(finished)

    threading.Thread(target=produce, daemon=True).start()
    while True:
        event = events.get()
        if event is finished:
            return
        yield event


def build_app_handlers(server: QueueServer):
    """Return a handler dispatch dict shared by both transports."""

    def handle_chat(payload: dict):
        # Validate before sending streaming response headers.
        try:
            parse_chat_request(payload)
        except ValueError as exc:
            return _error(400, str(exc))
        if payload.get("stream"):
            if _HAS_FLASK:
                return Response(
                    _chat_stream(server, payload),
                    mimetype="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )
            return _chat_stream(server, payload)
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
        if isinstance(out, tuple):
            body, status = out
            return jsonify(body), status
        return jsonify(out)

    @app.post("/v1/completions")
    def completions():
        payload = request.get_json(force=True, silent=True) or {}
        out = handlers["POST /v1/completions"](payload)
        if isinstance(out, tuple):
            body, status = out
            return jsonify(body), status
        return jsonify(out)

    return app


class _StdlibHandler(BaseHTTPRequestHandler):
    server_q = None  # type: QueueServer
    protocol_version = "HTTP/1.1"

    def _json(self, status: int, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, events):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            for body in events:
                if not body:
                    continue
                size = f"{len(body):X}".encode("ascii")
                self.wfile.write(size + b"\r\n" + body + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

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
        if path == "/v1/chat/completions":
            try:
                parse_chat_request(payload)
            except ValueError as exc:
                self._json(400, {"error": {"message": str(exc),
                                          "type": "invalid_request"}})
                return
        if path == "/v1/chat/completions" and payload.get("stream"):
            self._sse(_chat_stream(self.server_q, payload))
            return
        out = handlers[("POST " + path)](payload)
        if isinstance(out, tuple):
            body, status = out
            self._json(status, body)
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
