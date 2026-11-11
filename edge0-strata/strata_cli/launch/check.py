"""Check my harness: send one simple prompt and one tool call to the running server the way an app configured by
`strata connect` would, and say in plain words whether each worked.

The check uses the harness's own wire format (chat completions, responses, or messages) with the address, key and model name its
config carries, so it proves the settings `strata connect` prints reach a server that answers correctly. It does not start the app
itself. The prompt's answer text is checked (it must contain the expected word) and the tool call's arguments are checked (the
city must be named), not just the HTTP status.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from strata_control import loopback

#: harness id -> the endpoint family it speaks (the others use chat completions)
WIRE = {"codex": "responses", "claude-code": "messages"}

PROMPT = "Reply with exactly one word: pong"
TOOL_PROMPT = "What is the weather in Paris? Use the get_weather tool."
TOOL_SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}


@dataclass
class Check:
    name: str            # "prompt" | "tool call"
    ok: bool
    detail: str          # plain words
    seconds: float = 0.0


def _post(url: str, key: str, body: dict, headers: dict | None = None, timeout: float = 240.0) -> dict:
    h = {"Content-Type": "application/json", "Accept": "application/json", **(headers or {})}
    if key:
        h["Authorization"] = "Bearer " + key
        h["x-api-key"] = key
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=h, method="POST")
    with loopback.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def wire_for(harness_id: str) -> str:
    return WIRE.get(harness_id, "chat")


def _prompt_request(wire: str, base: str, model: str) -> tuple[str, dict]:
    if wire == "responses":
        return base + "/v1/responses", {"model": model, "input": PROMPT, "max_output_tokens": 256}
    if wire == "messages":
        return base + "/v1/messages", {"model": model, "max_tokens": 256, "messages": [{"role": "user", "content": PROMPT}]}
    return base + "/v1/chat/completions", {"model": model, "max_tokens": 256, "messages": [{"role": "user", "content": PROMPT}]}


def _tool_request(wire: str, base: str, model: str) -> tuple[str, dict]:
    if wire == "responses":
        return base + "/v1/responses", {"model": model, "input": TOOL_PROMPT, "max_output_tokens": 1536, "tool_choice": "auto",
                                        "tools": [{"type": "function", "name": "get_weather", "description": "Weather of a city",
                                                   "parameters": TOOL_SCHEMA}]}
    if wire == "messages":
        return base + "/v1/messages", {"model": model, "max_tokens": 1536, "messages": [{"role": "user", "content": TOOL_PROMPT}],
                                       "tools": [{"name": "get_weather", "description": "Weather of a city", "input_schema": TOOL_SCHEMA}]}
    return base + "/v1/chat/completions", {"model": model, "max_tokens": 1536, "messages": [{"role": "user", "content": TOOL_PROMPT}],
                                           "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Weather of a city",
                                                                                      "parameters": TOOL_SCHEMA}}]}


def answer_text(wire: str, data: dict) -> str:
    if wire == "responses":
        parts = []
        for item in data.get("output") or []:
            if not isinstance(item, dict) or item.get("type") not in (None, "message"):
                continue                                   # reasoning items are not the answer
            for c in item.get("content") or []:
                if isinstance(c, dict) and c.get("type") in (None, "output_text", "text") and c.get("text"):
                    parts.append(c["text"])
        if parts:
            return "".join(parts)
        return data["output_text"] if isinstance(data.get("output_text"), str) else ""
    if wire == "messages":
        return "".join(b.get("text", "") for b in data.get("content") or [] if isinstance(b, dict) and b.get("type") == "text")
    try:
        return data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


def tool_calls(wire: str, data: dict) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    try:
        if wire == "responses":
            for item in data.get("output") or []:
                if item.get("type") == "function_call":
                    out.append((item.get("name", ""), json.loads(item.get("arguments") or "{}")))
        elif wire == "messages":
            for b in data.get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    out.append((b.get("name", ""), b.get("input") or {}))
        else:
            for tc in data["choices"][0]["message"].get("tool_calls") or []:
                fn = tc.get("function") or {}
                out.append((fn.get("name", ""), json.loads(fn.get("arguments") or "{}")))
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return out
    return out


def run_checks(harness_id: str, url: str, key: str, model: str, *, timeout: float = 240.0, poster=None) -> list[Check]:
    """The two checks for one harness. `poster` is replaceable for tests."""
    poster = poster or _post
    base = url.rstrip("/")
    wire = wire_for(harness_id)
    checks: list[Check] = []
    t0 = time.monotonic()
    try:
        u, body = _prompt_request(wire, base, model)
        data = poster(u, key, body, None, timeout)
        text = answer_text(wire, data).strip()
        ok = "pong" in text.lower()
        checks.append(Check("prompt", ok, f"It answered: {text[:60]!r}." if ok else
                            (f"It answered {text[:60]!r}, not the word asked for." if text else "The server replied but with no text."),
                            time.monotonic() - t0))
    except Exception as e:                                   # noqa: BLE001
        checks.append(Check("prompt", False, _why(e), time.monotonic() - t0))
    t1 = time.monotonic()
    try:
        u, body = _tool_request(wire, base, model)
        data = poster(u, key, body, None, timeout)
        calls = tool_calls(wire, data)
        hit = next(((n, a) for n, a in calls if n == "get_weather" and "paris" in json.dumps(a).lower()), None)
        if hit:
            checks.append(Check("tool call", True, f"It called {hit[0]} with {json.dumps(hit[1])}.", time.monotonic() - t1))
        elif calls:
            checks.append(Check("tool call", False, f"It called {calls[0][0]} with {json.dumps(calls[0][1])[:60]}, not the city asked for.",
                                time.monotonic() - t1))
        else:
            checks.append(Check("tool call", False, "The model answered in words and did not call the tool.", time.monotonic() - t1))
    except Exception as e:                                   # noqa: BLE001
        checks.append(Check("tool call", False, _why(e), time.monotonic() - t1))
    return checks


def _why(e: Exception) -> str:
    if isinstance(e, urllib.error.HTTPError):
        if e.code in (401, 403):
            return "The server refused the key. Check the API key in the config."
        if e.code == 404:
            return "The server does not have that address. Check the server address and the app's endpoint type."
        return f"The server answered with an error (HTTP {e.code})."
    if isinstance(e, urllib.error.URLError):
        return "The server could not be reached. Is it running (strata launch --status)?"
    return f"The check could not finish: {type(e).__name__}."
