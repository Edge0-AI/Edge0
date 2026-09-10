"""Server-layer tests: request parsing, chat session, SSE framing and the
stdlib HTTP transport, all against a fake engine (no real checkpoints).

Covers the OpenAI-compatible surface the deployment servers expose:
/healthz, /v1/models, POST /v1/chat/completions (non-streaming and SSE
streaming), and the /v1/completions rejection path.
"""

from __future__ import annotations

import json
import threading
import time
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib import request as urlrequest
from urllib.error import HTTPError

import pytest

from edge0.config import GenerationConfig
from edge0.server.app import (
    _HAS_FLASK,
    _chat_once,
    _chat_stream,
    _StdlibHandler,
    build_app_handlers,
    create_app,
    run_stdlib,
)
from edge0.server.chat import (
    ChatMessage,
    ChatRequest,
    ChatSession,
    QueueServer,
    decode_tokens,
    parse_chat_request,
    sse_format,
)
from edge0.server.limits import (
    MAX_MAX_TOKENS,
    MAX_MESSAGES,
    MAX_PROMPT_CHARS,
    MAX_REQUEST_BYTES,
    ChatRequestError,
    insecure_bind_warning,
    is_loopback_host,
)


# ---- fake engine ----------------------------------------------------------


class FakeTok:
    """Minimal tokenizer stub (transformers-like surface)."""

    def __init__(self, ids=(7, 8, 9)):
        self.ids = list(ids)
        self.bos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False):
        text = "<tpl>" + json.dumps(messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def encode(self, text):
        return self.ids

    def decode(self, tokens):
        return "".join(f"T{t}" for t in tokens)


class PlainTok(FakeTok):
    """Tokenizer without apply_chat_template (stdlib fallback path).

    Deleting the method (vs raising) keeps hasattr() False so the
    server's plain-text fallback renders instead."""

    def __init__(self):
        super().__init__()
        try:
            del self.apply_chat_template
        except AttributeError:
            pass


class FakeEngine:
    """Duck-typed engine: enough of Edge0Engine for the server layer."""

    name = "fake"

    def __init__(self, tok=None):
        self._tok = tok or FakeTok()
        self.pos = 3
        self.cfg = SimpleNamespace(
            gen=GenerationConfig(temperature=0.7, top_p=0.95))
        self.generated = []

    def generate(self, ids, gen_config=None, on_token=None, **kw):
        toks = [11, 12, 13]
        self.generated.append((list(ids), gen_config))
        if on_token is not None:
            for t in toks:
                on_token(t)
        return toks

    def stats(self):
        return {"ok": 1}

    def reset(self):
        """Per-request state clear (server parity with the real engine)."""
        self.pos = 0


def _req(**kw) -> ChatRequest:
    payload = {
        "model": "fake",
        "messages": [{"role": "user", "content": "hi"}],
        **kw,
    }
    return parse_chat_request(payload)


# ---- parse_chat_request ---------------------------------------------------


def test_parse_basic_defaults():
    req = _req()
    assert req.model == "fake"
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"
    assert req.messages[0].content == "hi"
    assert req.temperature is None
    assert req.stream is False


def test_parse_multipart_content_joins_text():
    req = _req(messages=[{
        "role": "user",
        "content": [{"type": "text", "text": "a"},
                    {"type": "image_url", "image_url": {"url": "x"}},
                    {"type": "text", "text": "b"}],
    }])
    assert req.messages[0].content == "ab"


def test_parse_stream_flag_and_sampling():
    req = _req(temperature=0.3, top_p=0.9, top_k=32, max_tokens=16,
               stream=True)
    assert req.temperature == 0.3
    assert req.top_p == 0.9
    assert req.top_k == 32
    assert req.max_tokens == 16
    assert req.stream is True


def test_parse_clamps_max_tokens():
    req = _req(max_tokens=MAX_MAX_TOKENS + 1000)
    assert req.max_tokens == MAX_MAX_TOKENS


def test_parse_rejects_non_positive_max_tokens():
    with pytest.raises(ChatRequestError, match="max_tokens"):
        _req(max_tokens=0)
    with pytest.raises(ChatRequestError, match="max_tokens"):
        _req(max_tokens=-8)
    with pytest.raises(ChatRequestError, match="integer"):
        _req(max_tokens="nope")


def test_parse_rejects_too_many_messages():
    with pytest.raises(ChatRequestError, match="too many messages"):
        parse_chat_request({
            "messages": [{"role": "user", "content": "x"}]
            * (MAX_MESSAGES + 1),
        })


def test_parse_rejects_oversized_prompt():
    with pytest.raises(ChatRequestError, match="prompt too large"):
        parse_chat_request({
            "messages": [{"role": "user", "content": "a" * (MAX_PROMPT_CHARS + 1)}],
        })


def test_parse_rejects_non_list_messages():
    with pytest.raises(ChatRequestError, match="array"):
        parse_chat_request({"messages": "not-a-list"})


def test_gen_config_uses_clamped_max_tokens():
    eng = FakeEngine()
    sess = ChatSession(eng, _req(max_tokens=10_000))
    gen = sess.gen_config()
    assert gen.max_new_tokens == MAX_MAX_TOKENS


# ---- sse / decode ---------------------------------------------------------


def test_sse_format():
    line = sse_format({"id": "1", "x": "中文"})
    assert line.startswith("data: ")
    assert line.endswith("\n\n")
    assert json.loads(line[6:-2]) == {"id": "1", "x": "中文"}


def test_decode_tokens():
    eng = FakeEngine()
    assert decode_tokens(eng, [1, 2]) == "T1T2"
    eng._tok = None
    assert decode_tokens(eng, [1]) == ""


# ---- ChatSession ----------------------------------------------------------


def test_chat_session_prompt_ids_uses_template():
    eng = FakeEngine()
    sess = ChatSession(eng, _req())
    ids = sess.prompt_ids()
    assert ids == [7, 8, 9]
    # The template was called with enable_thinking=False (serve.py parity).
    text = eng._tok.apply_chat_template([{"role": "user", "content": "hi"}])
    assert "<|im_start|>assistant" in text


def test_chat_session_prompt_ids_fallback_text():
    eng = FakeEngine(tok=PlainTok())
    sess = ChatSession(eng, _req())
    ids = sess.prompt_ids()
    assert ids == [7, 8, 9]


def test_chat_session_fallback_text_format():
    eng = FakeEngine(tok=PlainTok())
    req = parse_chat_request({"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]})
    sess = ChatSession(eng, req)
    text = sess._chat_text()
    assert "<|im_start|>system\nsys<|im_end|>\n" in text
    assert "<|im_start|>user\nq<|im_end|>\n" in text
    assert "<|im_start|>assistant\na<|im_end|>\n" in text
    assert text.endswith("<|im_start|>assistant\n")


def test_gen_config_overrides_request_fields():
    eng = FakeEngine()
    sess = ChatSession(eng, _req(temperature=0.1, max_tokens=5))
    gen = sess.gen_config()
    assert gen.temperature == 0.1
    assert gen.max_new_tokens == 5
    assert gen.top_p == 0.95  # engine default preserved


def test_chat_session_run_usage_and_callback():
    eng = FakeEngine()
    seen = []
    sess = ChatSession(eng, _req())
    tokens, meta = sess.run(on_token=seen.append)
    assert tokens == [11, 12, 13]
    assert seen == [11, 12, 13]
    assert meta["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 3,
        "total_tokens": 6,
    }
    assert "wall_s" in meta


# ---- QueueServer ----------------------------------------------------------


def test_queue_server_health():
    eng = FakeEngine()
    srv = QueueServer(eng)
    h = srv.health()
    assert h["status"] == "ok"
    assert h["model"] == "fake"
    assert h["pos"] == 3


def test_queue_server_chat_serializes():
    eng = FakeEngine()
    srv = QueueServer(eng)
    tokens, meta = srv.chat(_req())
    assert meta["usage"]["completion_tokens"] == 3
    assert len(eng.generated) == 1
    gen = eng.generated[0][1]
    assert gen.temperature == 0.7


# ---- handlers -------------------------------------------------------------


def test_handlers_healthz_and_models():
    srv = QueueServer(FakeEngine())
    handlers = build_app_handlers(srv)
    assert handlers["GET /healthz"]()["status"] == "ok"
    models = handlers["GET /v1/models"]()
    assert models["object"] == "list"
    assert models["data"][0]["id"] == "fake"


def test_handlers_completions_rejected():
    srv = QueueServer(FakeEngine())
    handlers = build_app_handlers(srv)
    body, status = handlers["POST /v1/completions"]({})
    assert status == 400
    assert body["error"]["type"] == "invalid_request"


def test_handlers_reject_too_many_messages():
    srv = QueueServer(FakeEngine())
    handlers = build_app_handlers(srv)
    payload = {
        "messages": [{"role": "user", "content": "x"}] * (MAX_MESSAGES + 1),
    }
    body, status = handlers["POST /v1/chat/completions"](payload)
    assert status == 400
    assert "too many messages" in body["error"]["message"]


def test_chat_stream_rejects_invalid_before_yielding():
    srv = QueueServer(FakeEngine())
    with pytest.raises(ChatRequestError, match="too many messages"):
        _chat_stream(srv, {
            "messages": [{"role": "user", "content": "x"}] * (MAX_MESSAGES + 1),
            "stream": True,
        })


def test_chat_once_response_shape():
    srv = QueueServer(FakeEngine())
    out = _chat_once(srv, {"messages": [{"role": "user", "content": "hi"}]})
    assert out["object"] == "chat.completion"
    assert out["model"] == "fake"
    ch = out["choices"][0]
    assert ch["message"]["role"] == "assistant"
    assert ch["message"]["content"] == "T11T12T13"
    assert ch["finish_reason"] == "stop"
    assert out["usage"]["completion_tokens"] == 3


def test_chat_stream_events_sequence():
    srv = QueueServer(FakeEngine())
    events = _chat_stream(srv, {
        "messages": [{"role": "user", "content": "hi"}], "stream": True,
    })
    body = b"".join(events)
    events = [e for e in body.decode("utf-8").split("\n\n") if e]
    assert events[-1] == "data: [DONE]"
    chunks = [json.loads(e[6:]) for e in events[:-1]]  # drop [DONE]
    deltas = [c["choices"][0]["delta"]["content"]
              for c in chunks if c["choices"][0]["delta"]]
    assert deltas == ["T11", "T12", "T13"]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"][0]["delta"] == {}


class PausingFakeEngine(FakeEngine):
    """Pause after the first callback so transport buffering is observable."""

    def __init__(self):
        super().__init__()
        self.first_token = threading.Event()
        self.resume = threading.Event()
        self.finished = threading.Event()

    def generate(self, ids, gen_config=None, on_token=None, **kw):
        toks = [11, 12, 13]
        self.generated.append((list(ids), gen_config))
        if on_token is not None:
            on_token(toks[0])
            self.first_token.set()
            self.resume.wait(timeout=5)
            for tid in toks[1:]:
                on_token(tid)
        self.finished.set()
        return toks


def test_chat_stream_yields_before_generation_finishes():
    eng = PausingFakeEngine()
    srv = QueueServer(eng)
    stream = iter(_chat_stream(srv, {
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }))
    try:
        first = next(stream)
        assert first.startswith(b"data: ")
        assert eng.first_token.is_set()
        assert not eng.finished.is_set()
        eng.resume.set()
        body = first + b"".join(stream)
    finally:
        eng.resume.set()
    assert b"data: [DONE]\n\n" in body


# ---- stdlib HTTP transport ------------------------------------------------


def test_stdlib_http_end_to_end():
    eng = FakeEngine()
    srv = QueueServer(eng, model_name="fake")
    handler = type("Edge0Handler", (_StdlibHandler,), {"server_q": srv})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    try:
        with urlrequest.urlopen(f"{base}/healthz", timeout=10) as r:
            assert r.status == 200
            assert json.loads(r.read())["status"] == "ok"
        with urlrequest.urlopen(f"{base}/v1/models", timeout=10) as r:
            assert json.loads(r.read())["data"][0]["id"] == "fake"
        payload = json.dumps({
            "model": "fake",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        }).encode("utf-8")
        req = urlrequest.Request(f"{base}/v1/chat/completions",
                                 data=payload,
                                 headers={"Content-Type": "application/json"})
        with urlrequest.urlopen(req, timeout=10) as r:
            assert r.status == 200
            out = json.loads(r.read())
            assert out["choices"][0]["message"]["content"] == "T11T12T13"
        # unknown route -> 404
        with pytest.raises(Exception) as exc:
            urlrequest.urlopen(f"{base}/nope", timeout=10)
        assert getattr(exc.value, "code", None) == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)


def test_stdlib_http_streams_before_generation_finishes():
    eng = PausingFakeEngine()
    srv = QueueServer(eng, model_name="fake")
    handler = type("Edge0Handler", (_StdlibHandler,), {"server_q": srv})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    payload = json.dumps({
        "model": "fake",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }).encode("utf-8")
    req = urlrequest.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlrequest.urlopen(req, timeout=10) as response:
            assert response.status == 200
            assert response.headers["Transfer-Encoding"] == "chunked"
            first = response.readline()
            assert first.startswith(b"data: ")
            assert eng.first_token.is_set()
            assert not eng.finished.is_set()
            eng.resume.set()
            rest = response.read()
        assert b"data: [DONE]\n\n" in first + rest
    finally:
        eng.resume.set()
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)


@pytest.mark.skipif(not _HAS_FLASK, reason="flask is not installed")
def test_flask_http_streams_before_generation_finishes():
    eng = PausingFakeEngine()
    app = create_app(QueueServer(eng, model_name="fake"))
    response = app.test_client().post(
        "/v1/chat/completions",
        json={
            "model": "fake",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        buffered=False,
    )
    try:
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-cache"
        stream = iter(response.response)
        first = next(stream)
        assert first.startswith(b"data: ")
        assert eng.first_token.is_set()
        assert not eng.finished.is_set()
        eng.resume.set()
        body = first + b"".join(stream)
    finally:
        eng.resume.set()
        response.close()
    assert b"data: [DONE]\n\n" in body


def _stdlib_server(eng=None):
    eng = eng or FakeEngine()
    srv = QueueServer(eng, model_name="fake")
    handler = type("Edge0Handler", (_StdlibHandler,), {"server_q": srv})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, t, port, f"http://127.0.0.1:{port}"


def test_stdlib_http_completions_rejected():
    httpd, t, _port, base = _stdlib_server()
    try:
        req = urlrequest.Request(
            f"{base}/v1/completions", data=b"{}",
            headers={"Content-Type": "application/json"})
        with pytest.raises(HTTPError) as exc:
            urlrequest.urlopen(req, timeout=10)
        assert exc.value.code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)


def test_stdlib_http_rejects_too_many_messages():
    httpd, t, _port, base = _stdlib_server()
    payload = json.dumps({
        "messages": [{"role": "user", "content": "x"}] * (MAX_MESSAGES + 1),
    }).encode("utf-8")
    try:
        req = urlrequest.Request(
            f"{base}/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"})
        with pytest.raises(HTTPError) as exc:
            urlrequest.urlopen(req, timeout=10)
        assert exc.value.code == 400
        err = json.loads(exc.value.read())
        assert "too many messages" in err["error"]["message"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)


def test_stdlib_http_stream_rejects_too_many_messages():
    httpd, t, _port, base = _stdlib_server()
    payload = json.dumps({
        "messages": [{"role": "user", "content": "x"}] * (MAX_MESSAGES + 1),
        "stream": True,
    }).encode("utf-8")
    try:
        req = urlrequest.Request(
            f"{base}/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"})
        with pytest.raises(HTTPError) as exc:
            urlrequest.urlopen(req, timeout=10)
        assert exc.value.code == 400
        err = json.loads(exc.value.read())
        assert "too many messages" in err["error"]["message"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)


def test_stdlib_http_rejects_oversize_content_length():
    httpd, t, port, _base = _stdlib_server()
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.putrequest("POST", "/v1/chat/completions")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(MAX_REQUEST_BYTES + 1))
        conn.endheaders()
        conn.send(b"{}")
        resp = conn.getresponse()
        assert resp.status == 413
        body = json.loads(resp.read())
        assert "too large" in body["error"]["message"]
    finally:
        conn.close()
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)


def test_stdlib_http_clamps_max_tokens():
    eng = FakeEngine()
    httpd, t, _port, base = _stdlib_server(eng)
    payload = json.dumps({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": MAX_MAX_TOKENS + 50,
    }).encode("utf-8")
    try:
        req = urlrequest.Request(
            f"{base}/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"})
        with urlrequest.urlopen(req, timeout=10) as r:
            assert r.status == 200
        gen = eng.generated[0][1]
        assert gen.max_new_tokens == MAX_MAX_TOKENS
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)


@pytest.mark.skipif(not _HAS_FLASK, reason="flask is not installed")
def test_flask_rejects_oversize_body():
    app = create_app(QueueServer(FakeEngine(), model_name="fake"))
    resp = app.test_client().post(
        "/v1/chat/completions",
        data="x" * (MAX_REQUEST_BYTES + 1),
        content_type="application/json",
    )
    assert resp.status_code == 413


def test_loopback_bind_helpers():
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("127.0.0.2")
    assert is_loopback_host("localhost")
    assert is_loopback_host("::1")
    assert is_loopback_host("[::1]")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("::")
    assert not is_loopback_host("192.168.1.10")
    assert not is_loopback_host("example.com")
    warning = insecure_bind_warning("0.0.0.0", 8000)
    assert warning and "unauthenticated" in warning
    assert insecure_bind_warning("127.0.0.1", 8000) is None
