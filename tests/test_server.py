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
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib import request as urlrequest

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


# ---- fake engine ----------------------------------------------------------


class FakeTok:
    """Minimal tokenizer stub (transformers-like surface)."""

    def __init__(self, ids=(7, 8, 9)):
        self.ids = list(ids)
        self.bos_token_id = 0
        self.encoded_text = None

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False, tools=None):
        text = "<tpl>" + json.dumps(messages)
        if tools:
            text += "<tools>" + json.dumps(tools)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def encode(self, text):
        self.encoded_text = text
        return self.ids

    def decode(self, tokens):
        return "".join(f"T{t}" for t in tokens)


class PlainTok:
    """Tokenizer without apply_chat_template (stdlib fallback path).

    A standalone class rather than a FakeTok subclass: deleting the
    inherited method off the instance does not work, so ``hasattr`` would
    stay True and the fallback would never be exercised."""

    def __init__(self, ids=(7, 8, 9)):
        self.ids = list(ids)
        self.bos_token_id = 0
        self.encoded_text = None

    def encode(self, text):
        self.encoded_text = text
        return self.ids

    def decode(self, tokens):
        return "".join(f"T{t}" for t in tokens)


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


class ToolCallTok(FakeTok):
    """Decodes generated tokens to a Ling-style ``<tool_call>`` block, the
    same shape the real edge0-8b template asks the model to emit."""

    def decode(self, tokens):
        return ("<tool_call>calculator\n"
                "<arg_key>expression</arg_key>\n"
                "<arg_value>2 + 2</arg_value>\n"
                "</tool_call>")


class ToolCallEngine(FakeEngine):
    """FakeEngine + the real Ling tool-call parser, so tests exercise the
    actual edge0.server.tool_calls integration, not a re-implementation
    of it."""

    def __init__(self, tok=None):
        super().__init__(tok=tok or ToolCallTok())

    def parse_tool_calls(self, text: str):
        from edge0.server.tool_calls import parse_ling_tool_calls
        return parse_ling_tool_calls(text)


_TOOLS = [{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Calculate an arithmetic expression.",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]},
    },
}]


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


def test_parse_preserves_tools_and_tool_choice():
    req = _req(tools=_TOOLS, tool_choice="auto")
    assert req.tools == _TOOLS
    assert req.tool_choice == "auto"


def test_parse_message_tool_calls_and_tool_role_roundtrip():
    req = parse_chat_request({"messages": [
        {"role": "user", "content": "compute 2+2"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "calculator",
                          "arguments": '{"expression": "2 + 2"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "4"},
    ]})
    assert req.messages[1].tool_calls[0]["function"]["name"] == "calculator"
    assert req.messages[1].content == ""  # null, not the string "None"
    assert req.messages[2].tool_call_id == "call_1"


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
    # What actually gets encoded is the RENDERED template, not the
    # hardcoded ChatML (issue #11: the latter loses the template's empty
    # <think></think> closer and derails the turn).
    assert eng._tok.encoded_text.startswith("<tpl>")
    assert eng._tok.encoded_text.endswith("<|im_start|>assistant\n")
    assert eng._tok.encoded_text != sess._chat_text()
    # The template was called with enable_thinking=False (serve.py parity).
    text = eng._tok.apply_chat_template([{"role": "user", "content": "hi"}])
    assert "<|im_start|>assistant" in text


def test_chat_session_prompt_ids_fallback_text():
    eng = FakeEngine(tok=PlainTok())
    sess = ChatSession(eng, _req())
    ids = sess.prompt_ids()
    assert ids == [7, 8, 9]
    # No template on the tokenizer -> hardcoded ChatML (this path used to
    # raise NameError on the unbound ``text``).
    assert eng._tok.encoded_text == sess._chat_text()


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


# ---- tool calls (#115) -----------------------------------------------------


def test_chat_once_ignores_tool_call_text_without_tools_field():
    """No ``tools`` in the request -> the raw <tool_call> XML the fake
    generation returns is left as plain content, exactly today's
    (buggy) behavior -- this only changes when the client opts in."""
    srv = QueueServer(ToolCallEngine())
    out = _chat_once(srv, {"messages": [{"role": "user", "content": "hi"}]})
    ch = out["choices"][0]
    assert "tool_calls" not in ch["message"]
    assert "<tool_call>" in ch["message"]["content"]
    assert ch["finish_reason"] == "stop"


def test_chat_once_parses_tool_calls_when_requested():
    srv = QueueServer(ToolCallEngine())
    out = _chat_once(srv, {
        "messages": [{"role": "user", "content": "compute 2 + 2"}],
        "tools": _TOOLS, "tool_choice": "auto",
    })
    ch = out["choices"][0]
    assert ch["message"]["content"] is None
    assert ch["finish_reason"] == "tool_calls"
    calls = ch["message"]["tool_calls"]
    assert len(calls) == 1
    fn = calls[0]["function"]
    assert fn["name"] == "calculator"
    assert json.loads(fn["arguments"]) == {"expression": "2 + 2"}
    assert calls[0]["type"] == "function"
    assert calls[0]["id"].startswith("call_")


def test_chat_once_tool_choice_none_suppresses_parsing():
    srv = QueueServer(ToolCallEngine())
    out = _chat_once(srv, {
        "messages": [{"role": "user", "content": "compute 2 + 2"}],
        "tools": _TOOLS, "tool_choice": "none",
    })
    ch = out["choices"][0]
    assert "tool_calls" not in ch["message"]
    assert "<tool_call>" in ch["message"]["content"]


def test_chat_once_tools_requested_but_engine_cannot_parse():
    """An engine with no parse_tool_calls (unsupported family) must not
    crash a tools-enabled request; it just can't extract structured
    calls, same as today."""
    srv = QueueServer(FakeEngine())
    out = _chat_once(srv, {
        "messages": [{"role": "user", "content": "hi"}], "tools": _TOOLS,
    })
    ch = out["choices"][0]
    assert "tool_calls" not in ch["message"]
    assert ch["finish_reason"] == "stop"


def test_chat_stream_tools_requested_buffers_and_emits_final_tool_calls():
    srv = QueueServer(ToolCallEngine())
    events = _chat_stream(srv, {
        "messages": [{"role": "user", "content": "compute 2 + 2"}],
        "stream": True, "tools": _TOOLS, "tool_choice": "auto",
    })
    body = b"".join(events)
    frames = [e for e in body.decode("utf-8").split("\n\n") if e]
    assert frames[-1] == "data: [DONE]"
    chunks = [json.loads(e[6:]) for e in frames[:-1]]
    # No raw <tool_call> XML in any per-token delta -- buffered, not
    # streamed token-by-token, unlike the no-tools path.
    for c in chunks[:-1]:
        assert "<tool_call>" not in json.dumps(c)
    final = chunks[-1]["choices"][0]
    assert final["finish_reason"] == "tool_calls"
    assert final["delta"]["content"] is None
    calls = final["delta"]["tool_calls"]
    assert calls[0]["function"]["name"] == "calculator"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "expression": "2 + 2"}


def test_chat_stream_without_tools_keeps_immediate_per_token_deltas():
    """Regression guard: requests with no tools must keep the existing
    immediate per-token streaming -- buffering is scoped to tool-enabled
    requests only."""
    srv = QueueServer(ToolCallEngine())
    events = _chat_stream(srv, {
        "messages": [{"role": "user", "content": "hi"}], "stream": True,
    })
    body = b"".join(events)
    frames = [e for e in body.decode("utf-8").split("\n\n") if e]
    chunks = [json.loads(e[6:]) for e in frames[:-1]]
    deltas = [c["choices"][0]["delta"].get("content")
              for c in chunks if c["choices"][0]["delta"]]
    # unchanged from the no-tools path: whatever decode_tokens([tid])
    # returns per call, not the whole buffered response
    assert len(deltas) == 3
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


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
