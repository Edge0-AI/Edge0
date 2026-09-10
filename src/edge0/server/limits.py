"""Request and bind-address limits for the unauthenticated serve API.

The OpenAI-compatible HTTP surface has no auth.  These caps exist to
keep a local ``edge0 serve`` from being an easy resource-exhaustion
target, and to make non-loopback binds noisy rather than silent.
"""

from __future__ import annotations

import ipaddress

# HTTP body cap for POST /v1/* (stdlib Content-Length check + Flask
# MAX_CONTENT_LENGTH).  413 when exceeded.
MAX_REQUEST_BYTES = 1 * 1024 * 1024  # 1 MiB

# Generation cap applied to request ``max_tokens`` (clamped, not rejected,
# so localhost clients that send OpenAI-style large caps still work).
MAX_MAX_TOKENS = 2048

# Chat-compleitions prompt shape.  400 when exceeded.
MAX_MESSAGES = 128
MAX_PROMPT_CHARS = 100_000


class ChatRequestError(ValueError):
    """Client-facing request validation failure (HTTP 400 unless noted)."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status
        self.message = message


def is_loopback_host(host: str | None) -> bool:
    """True for loopback literals (127.0.0.1/8, ::1, localhost)."""
    if not host:
        return False
    h = host.strip().lower()
    if h in {"localhost", "localhost."}:
        return True
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    try:
        addr = ipaddress.ip_address(h)
    except ValueError:
        return False
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        addr = mapped
    return bool(addr.is_loopback)


def insecure_bind_warning(host: str, port: int) -> str | None:
    """Return a warning if ``host`` is not a loopback bind, else None."""
    if is_loopback_host(host):
        return None
    return (
        f"[edge0] WARNING: binding {host}:{port} exposes an unauthenticated "
        "OpenAI-compatible API on a non-loopback interface. "
        "Do not expose /v1/* to untrusted networks. "
        "Prefer --host 127.0.0.1 (the default)."
    )


def clamp_max_tokens(value) -> int | None:
    """Validate and clamp ``max_tokens`` to ``[1, MAX_MAX_TOKENS]``.

    ``None`` (omitted) is left to the engine's generation default.
    """
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ChatRequestError("max_tokens must be an integer") from None
    if n < 1:
        raise ChatRequestError("max_tokens must be >= 1")
    if n > MAX_MAX_TOKENS:
        return MAX_MAX_TOKENS
    return n


def message_text(content) -> str:
    """Flatten OpenAI message ``content`` (string or multipart list) to text."""
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content if isinstance(p, dict))
    if content is None:
        return ""
    return str(content)
