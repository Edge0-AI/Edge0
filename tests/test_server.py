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
                            enable_thinking=False):
        text = "<tpl>" + json.dumps(messages)
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


@pytest.mark.parametrize("stats", [False, True])
def test_chat_stream_events_sequence(stats):
    srv = QueueServer(FakeEngine(), stats=stats)
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


@pytest.mark.parametrize("verbose", [False, True])
def test_server_diagnostics_per_request(verbose, capsys):
    class DiagnosticTok(FakeTok):
        def convert_ids_to_tokens(self, token):
            return f"T{token}"

        def decode(self, tokens, **kwargs):
            return super().decode(tokens)

    class DiagnosticEngine(FakeEngine):
        def stats(self):
            return {"cache_hits": len(self.generated) * 3}

    engine = DiagnosticEngine(tok=DiagnosticTok())
    server = QueueServer(engine, stats=not verbose, verbose_tokens=verbose)
    for _ in range(2):
        received = []
        tokens, meta = server.chat(_req(), on_token=received.append)
        assert received == tokens == [11, 12, 13]
        output = capsys.readouterr()
        assert output.out == ""
        records = [json.loads(line.removeprefix("[edge0 stats] "))
                   for line in output.err.splitlines() if line.startswith("[edge0 stats] ")]
        assert [r["phase"] for r in records] == ["start", "progress", "complete"]
        assert records[0]["cache_hits"] == 0
        assert records[-1]["cache_hits"] == 3
        assert records[-1]["generated_tokens"] == 3
        assert records[-1]["usage"] == meta["usage"]
        assert ("[edge0 token]" in output.err) == verbose
        if verbose:
            assert '"phase": "prompt"' in output.err
            assert '"phase": "generated"' in output.err


def test_server_diagnostics_disabled(capsys):
    QueueServer(FakeEngine()).chat(_req())
    assert capsys.readouterr().err == ""


def test_server_diagnostics_exception(capsys):
    server = QueueServer(FakeEngine(), stats=True)

    def fail(token):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        server.chat(_req(), on_token=fail)
    assert '"phase": "interrupted"' in capsys.readouterr().err
    server.chat(_req())
    assert '"phase": "complete"' in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--stats", "--verbose-tokens"])
def test_serve_cli_diagnostics(flag, monkeypatch):
    import edge0
    import edge0.server
    from edge0.cli import main

    engine = FakeEngine()
    engine.close = lambda: None
    monkeypatch.setattr(edge0.AutoEngine, "from_pretrained", lambda *a, **kw: engine)
    servers = []
    monkeypatch.setattr(edge0.server, "run_server", lambda server, **kw: servers.append(server))
    assert main(["serve", "--model-path", "fake.gguf", flag]) == 0
    assert servers[0].stats is True
    assert servers[0].verbose_tokens == (flag == "--verbose-tokens")
