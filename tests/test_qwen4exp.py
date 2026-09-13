import numpy as np
import pytest

from gguf_fixture import tiny_qwen4exp


@pytest.fixture
def engine(tmp_path):
    from edge0 import AutoEngine
    engine = AutoEngine.from_pretrained(model_path=tiny_qwen4exp(tmp_path / "tiny.gguf"),
        decoded_cache_bytes=128 * 1024, decode_chunk_bytes=4096)
    yield engine
    engine.close()


def test_chunked_prefill_reset_and_eos(engine):
    ids = [1, 3, 7, 4, 2, 256, 12, 10, 16]
    engine.prefill(ids, chunk_size=len(ids))
    expected = np.array(engine.next_logits())
    engine.reset()
    engine.prefill(ids, chunk_size=1)
    np.testing.assert_allclose(np.array(engine.next_logits()), expected, rtol=1e-4, atol=2e-5)
    assert len(engine.model.history) == 2
    engine.reset()
    assert engine.model.history == []
    engine.prefill(ids, chunk_size=3)
    np.testing.assert_allclose(np.array(engine.next_logits()), expected, rtol=1e-4, atol=2e-5)
    assert engine.stats()["peak_cached_pending_staging_bytes"] <= 128 * 1024


def test_history_hash_wrap_and_eos():
    from edge0.backends.mlx._impl.qwen4exp import ple_rows
    args = (3, 2, 99, [2**63 + 1, 2**62 + 1, 2**61 + 1], [0, 7, 18, 31], [7, 11, 13, 17])
    rows, history = ple_rows([4, 99, 3, 5], [], *args)
    assert len(history) == 2
    assert rows[2] == ple_rows([3], [], *args)[0][0]
    chunks, hist = [], []
    for tid in [4, 99, 3, 5]:
        r, hist = ple_rows([tid], hist, *args)
        chunks += r
    assert chunks == rows


def test_tokenizer_roundtrip_and_template(engine):
    text = "Café e\u0301 中文 123!\n"
    ids = engine._tok.encode(text)
    assert engine._tok.decode(ids) == text
    prompt = engine.encode_chat([{"role": "user", "content": "test"}])
    assert 257 in prompt and 256 in prompt


def test_persistent_conversation_cache_is_not_part_of_gguf(tmp_path):
    from edge0 import AutoConfig
    path = tiny_qwen4exp(tmp_path / "tiny.gguf")
    with pytest.raises(TypeError, match="unknown CustomConfig override 'conversation_cache'"):
        AutoConfig.from_pretrained(model_path=path, conversation_cache=object())


def test_upstream_tiny_logits(engine):
    from pathlib import Path
    fixture = Path(__file__).parent / "fixtures" / "qwen4exp_logits.npy"
    assert fixture.exists(), "generate tiny logits with pinned llama.cpp reference first"
    engine.prefill([1, 3, 7, 4, 2, 256, 12, 10, 16])
    np.testing.assert_allclose(np.array(engine.next_logits()), np.load(fixture), rtol=2e-4, atol=2e-5)


def test_http_streaming_and_nonstreaming(engine):
    import json
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import Request, urlopen
    from edge0.server.chat import QueueServer
    from edge0.server.app import _StdlibHandler
    handler = type("GGUFHandler", (_StdlibHandler,), {"server_q": QueueServer(engine, model_name=engine.name)})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        for stream in (False, True):
            payload = json.dumps({"model": engine.name, "messages": [{"role": "user", "content": "hi"}],
                "stream": stream, "max_tokens": 2, "temperature": 0}).encode()
            req = Request(f"http://127.0.0.1:{server.server_port}/v1/chat/completions", data=payload,
                headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=30) as response:
                text = response.read().decode()
                assert response.status == 200
            if stream:
                assert "data: [DONE]" in text
            else:
                assert "choices" in json.loads(text)
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_cli_custom_gguf(tmp_path, capsys):
    from edge0.cli import main
    path = tiny_qwen4exp(tmp_path / "tiny.gguf")
    for command in ("demo", "chat"):
        assert main([command, "--model-path", str(path), "--prompt", "hi", "--max-new", "1"]) == 0
        stderr = capsys.readouterr().err
        assert "1 tokens" in stderr
        assert "[edge0 stats]" not in stderr
    assert main(["chat", "--model-path", str(path), "--prompt", "hi",
                 "--max-new", "1", "--stats"]) == 0
    output = capsys.readouterr()
    assert '"phase": "complete"' in output.err
    assert '"weight_cache_budget_bytes": 3221225472' in output.err
    assert "[edge0 stats]" not in output.out
    assert "[edge0 token]" not in output.err
    assert main(["chat", "--model-path", str(path), "--prompt", "hi",
                 "--max-new", "1", "--verbose-tokens", "--cache-mib", "16"]) == 0
    output = capsys.readouterr()
    assert "[edge0 tokenize]" in output.err
    assert '"weight_cache_budget_bytes": 16777216' in output.err
    assert '"phase": "prompt"' in output.err
    assert '"phase": "generated"' in output.err
    assert '"since_previous_token_s": null' in output.err
    assert "[edge0 token]" not in output.out
