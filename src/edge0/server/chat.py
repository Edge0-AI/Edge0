"""edge0 HTTP server: a single-process, one-model OpenAI-style endpoint.

The engine is exclusive to one request at a time (a single MLX process
can stream one generation well; concurrent requests would thrash the
expert cache).  Requests queue on a lock: each generation streams
token-by-token over SSE, and the queue drains in FIFO order.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from edge0.engine.base import Edge0Engine


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatRequest:
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    seed: int | None = None
    stream: bool = False
    enable_thinking: bool | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def parse_chat_request(payload: dict) -> ChatRequest:
    msgs = []
    for m in payload.get("messages", []):
        role = str(m.get("role", "user"))
        content = m.get("content", "")
        if isinstance(content, list):  # multi-part content: join text parts
            content = "".join(
                p.get("text", "") for p in content if isinstance(p, dict))
        msgs.append(ChatMessage(role=role, content=str(content)))
    return ChatRequest(
        model=str(payload.get("model", "")),
        messages=msgs,
        temperature=payload.get("temperature"),
        top_p=payload.get("top_p"),
        top_k=payload.get("top_k"),
        max_tokens=payload.get("max_tokens"),
        seed=payload.get("seed"),
        stream=bool(payload.get("stream", False)),
        enable_thinking=payload.get("enable_thinking"),
        raw=payload,
    )


class ChatSession:
    """Tokenize + chat-template + generate for one request."""

    def __init__(self, engine: Edge0Engine, req: ChatRequest):
        self.engine = engine
        self.req = req
        self._tok = engine._tok

    def prompt_ids(self) -> list[int]:
        tok = self._tok
        # Families with a vendored chat template
        # (``engine.encode_chat`` renders a chat_template.jinja with
        # enable_thinking) must go
        # through it — the tokenizer's own apply_chat_template renders a
        # DIFFERENT prompt (no think-mode system line, no ``<think>``
        # opener), which is exactly the wrong-template bug.
        encode = getattr(self.engine, "encode_chat", None)
        if callable(encode):
            think = self.req.enable_thinking
            if think is None:
                think = getattr(self.engine, "think", False)
            return encode(
                [m.__dict__ for m in self.req.messages], think=bool(think))
        if hasattr(tok, "apply_chat_template"):
            # The qwen35 template supports ``enable_thinking``: False
            # renders the canonical no-think prompt — an EMPTY think
            # block closer, the model's direct-answer form.  Default OFF
            # (CLI demo parity).
            #
            # The successful render MUST be kept: substituting the
            # hardcoded ChatML drops that think-block closer, the model
            # then opens its own <think> and the turn derails (issue #11:
            # <think> leaking into content, then empty/garbled replies).
            # ``_chat_text()`` is a last resort, never the success path.
            msgs = [m.__dict__ for m in self.req.messages]
            think = self.req.enable_thinking
            if think is None:
                think = False
            try:
                text = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=bool(think))
            except TypeError:
                # tokenizer template without the kwarg: plain render
                try:
                    text = tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                except Exception:  # noqa: BLE001
                    text = self._chat_text()
            except Exception:  # noqa: BLE001 — render failed: last resort
                text = self._chat_text()
        else:
            # Tokenizer without a chat template: plain ChatML.  (This also
            # used to leave ``text`` unbound -> NameError.)
            text = self._chat_text()
        ids = tok.encode(text)
        if not ids:
            ids = [tok.bos_token_id or 0]
        return ids

    def _chat_text(self) -> str:
        parts = []
        for m in self.req.messages:
            if m.role == "system":
                parts.append(f"<|im_start|>system\n{m.content}<|im_end|>\n")
            elif m.role == "user":
                parts.append(f"<|im_start|>user\n{m.content}<|im_end|>\n")
            elif m.role == "assistant":
                parts.append(f"<|im_start|>assistant\n{m.content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def gen_config(self):
        from edge0.config import GenerationConfig
        base = getattr(self.engine.cfg, "gen", GenerationConfig())
        kw = {}
        for k in ("temperature", "top_p", "top_k", "max_new_tokens", "seed"):
            v = getattr(self.req, {
                "max_new_tokens": "max_tokens",
            }.get(k, k), None)
            if v is not None:
                kw[k] = v
        return GenerationConfig(**{**base.__dict__, **kw})

    def run(self, on_token=None, on_prompt=None) -> tuple[list[int], dict]:
        t0 = time.perf_counter()
        # Per-request clean
        # per-request state.  A previous degenerate/truncated turn leaves
        # bad pre-routing cross-token state and KV behind, which makes every
        # later request collapse from its first token.
        self.engine.reset()
        tokenize_started = time.perf_counter()
        ids = self.prompt_ids()
        self.engine.last_tokenization_s = time.perf_counter() - tokenize_started
        if on_prompt is not None:
            on_prompt(ids)

        gen = self.gen_config()
        tokens = self.engine.generate(
            ids, gen_config=gen, on_token=on_token)
        usage = {
            "prompt_tokens": len(ids),
            "completion_tokens": len(tokens),
            "total_tokens": len(ids) + len(tokens),
        }
        meta = {"wall_s": round(time.perf_counter() - t0, 3)}
        return tokens, {"usage": usage, **meta}


class QueueServer:
    """Single-slot serving loop: serialize generations, stream via callback."""

    def __init__(self, engine: Edge0Engine, model_name: str | None = None):
        self.engine = engine
        self.model_name = model_name or engine.name
        self._lock = threading.Lock()

    def health(self) -> dict:
        st = self.engine.stats()
        return {"status": "ok", "model": self.model_name,
                "pos": self.engine.pos}

    def chat(self, req: ChatRequest, on_token=None) -> dict:
        with self._lock:
            sess = ChatSession(self.engine, req)
            return sess.run(on_token=on_token)


def sse_format(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def decode_tokens(engine: Edge0Engine, tokens: list[int]) -> str:
    tok = engine._tok
    if tok is None:
        return ""
    return tok.decode(tokens)
