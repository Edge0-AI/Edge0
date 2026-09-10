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
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from edge0.server.chat import (QueueServer, parse_chat_request, sse_format,
                               decode_tokens)
from edge0.server.limits import (
    MAX_REQUEST_BYTES,
    ChatRequestError,
    insecure_bind_warning,
)

try:  # pragma: no cover - environment dependent
    from flask import Flask, Response, jsonify, request  # type: ignore

    _HAS_FLASK = True
except ImportError:  # pragma: no cover
    _HAS_FLASK = False


def _error(status: int, msg: str) -> tuple:
    """Transport-agnostic error: ``(payload_dict, status)``."""
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
    # Parse eagerly so invalid requests 400 *before* SSE headers are sent.
    # The inner generator is what the transports iterate.
    req = parse_chat_request(payload)
    events = queue.Queue()
    finished = object()
    request_id = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())

    def on_token(tid: int):
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
            _, meta = server.chat(req, on_token=on_token)
            events.put(sse_format({
                "id": request_id, "object": "chat.completion.chunk",
                "created": created, "model": server.model_name,
                "choices": [{"index": 0, "delta": {},
                             "finish_reason": "stop"}],
                "usage": meta["usage"],
            }).encode("utf-8"))
            events.put(b"data: [DONE]\n\n")
        except Exception as exc:  # pragma: no cover - transport-dependent
            events.put(sse_format({
                "error": {"message": str(exc), "type": "server_error"},
            }).encode("utf-8"))
        finally:
            events.put(finished)

    def generate():
        threading.Thread(target=produce, daemon=True).start()
        while True:
            event = events.get()
            if event is finished:
                return
            yield event

    return generate()


def build_app_handlers(server: QueueServer):
    """Return a handler dispatch dict shared by both transports."""

    def handle_chat(payload: dict):
        try:
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
        except ChatRequestError as exc:
            return _error(exc.status, exc.message)

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
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
    handlers = build_app_handlers(server)

    @app.errorhandler(413)
    def _too_large(_err):
        return jsonify({
            "error": {
                "message": "request body too large",
                "type": "invalid_request",
            },
        }), 413

    @app.get("/healthz")
    def healthz():
        return handlers["GET /healthz"]()

    @app.get("/v1/models")
    def models():
        return handlers["GET /v1/models"]()

    @app.post("/v1/chat/completions")
    def chat():
        payload = request.get_json(force=True, silent=True) or {}
        if not isinstance(payload, dict):
            body, status = _error(400, "JSON object required")
            return jsonify(body), status
        out = handlers["POST /v1/chat/completions"](payload)
        if isinstance(out, Response):
            return out
        if isinstance(out, tuple):
            body, status = out[0], out[1]
            if isinstance(body, Response):
                return body, status
            return jsonify(body), status
        return jsonify(out)

    @app.post("/v1/completions")
    def completions():
        payload = request.get_json(force=True, silent=True) or {}
        out = handlers["POST /v1/completions"](payload)
        if isinstance(out, tuple):
            body, status = out[0], out[1]
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

    def _write_handler_result(self, out):
        """Serialize a handler return value (dict, 2-tuple, or 3-tuple)."""
        if not isinstance(out, tuple):
            self._json(200, out)
            return
        body, status = out[0], out[1]
        extra = dict(out[2]) if len(out) > 2 else {}
        if isinstance(body, dict):
            self._json(status, body)
            return
        get_data = getattr(body, "get_data", None)
        if callable(get_data):
            raw = get_data()
            ctype = getattr(body, "mimetype", None) or "application/json"
        else:
            raw = body.encode("utf-8") if isinstance(body, str) else body
            ctype = "application/json"
        self.send_response(status)
        sent_ct = False
        for k, v in extra.items():
            self.send_header(k, v)
            if k.lower() == "content-type":
                sent_ct = True
        if not sent_ct:
            self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_body(self):
        """Read a POST body capped at ``MAX_REQUEST_BYTES``.

        Returns the parsed object, or None if an error response was already
        written.
        """
        raw_len = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_len)
        except (TypeError, ValueError):
            self._json(400, {"error": {"message": "invalid Content-Length",
                                       "type": "invalid_request"}})
            return None
        if length < 0:
            self._json(400, {"error": {"message": "invalid Content-Length",
                                       "type": "invalid_request"}})
            return None
        if length > MAX_REQUEST_BYTES:
            self.close_connection = True
            self._json(413, {"error": {"message": "request body too large",
                                       "type": "invalid_request"}})
            return None
        raw = self.rfile.read(length) or b"{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            self._json(400, {"error": {"message": "invalid JSON",
                                       "type": "invalid_request"}})
            return None

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        handlers = build_app_handlers(self.server_q)
        if path not in ("/v1/chat/completions", "/v1/completions"):
            self._json(404, {"error": {"message": f"no route {path}"}})
            return
        payload = self._read_json_body()
        if payload is None:
            return
        if not isinstance(payload, dict):
            self._json(400, {"error": {"message": "JSON object required",
                                       "type": "invalid_request"}})
            return
        try:
            if path == "/v1/chat/completions" and payload.get("stream"):
                self._sse(_chat_stream(self.server_q, payload))
                return
            out = handlers[("POST " + path)](payload)
        except ChatRequestError as exc:
            self._json(exc.status, {"error": {"message": exc.message,
                                              "type": "invalid_request"}})
            return
        self._write_handler_result(out)

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
    warning = insecure_bind_warning(host, port)
    if warning:
        print(warning, file=sys.stderr)
    if use_flask is None:
        use_flask = _HAS_FLASK
    if use_flask:
        app = create_app(server)
        app.run(host=host, port=port, threaded=True)
    else:
        run_stdlib(server, host, port)
