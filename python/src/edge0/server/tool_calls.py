"""Parse the XML tool-call blocks each model family's chat template asks
it to emit, and convert them to the OpenAI ``message.tool_calls`` schema.

The two supported checkpoints use two different XML dialects (confirmed
against the real ``chat_template.jinja`` shipped with each checkpoint,
not assumed):

* **edge0-8b (Ling)** — ``<tool_call>{name}<arg_key>k</arg_key>
  <arg_value>v</arg_value>...</tool_call>``, one ``arg_key``/``arg_value``
  pair per argument.
* **edge0-35b (Qwen3.5)** — ``<tool_call>\\n<function={name}>\\n
  <parameter={key}>\\nvalue\\n</parameter>\\n...</function>\\n</tool_call>``.

Both formats can appear more than once per turn (multiple tool calls in
one response); each ``<tool_call>...</tool_call>`` block becomes one
OpenAI tool-call entry with a synthesized ``id``.
"""

from __future__ import annotations

import json
import re
import uuid

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

_LING_ARG = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)

_QWEN_FUNCTION = re.compile(
    r"\s*<function=([^>]+)>\s*(.*?)\s*</function>\s*", re.DOTALL)
_QWEN_PARAM = re.compile(
    r"<parameter=([^>]+)>\n(.*?)\n</parameter>", re.DOTALL)


def _coerce(raw: str):
    """A bare arg value is a string unless it parses as JSON (numbers,
    booleans, null, objects, arrays) -- mirrors the template's own
    ``v if v is string else v|tojson`` split, in reverse."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _to_openai(name: str, args: dict) -> dict:
    return {
        "id": _tool_call_id(),
        "type": "function",
        "function": {"name": name.strip(), "arguments": json.dumps(args)},
    }


def _split(text: str, parse_block) -> tuple[str | None, list[dict]]:
    """Shared block extraction: find every ``<tool_call>...</tool_call>``,
    remove it from the content, and parse its body with ``parse_block``.
    Blocks ``parse_block`` cannot make sense of are left in ``content``
    rather than silently dropped."""
    calls = []
    leftover_spans = []
    pos = 0
    for m in _TOOL_CALL_BLOCK.finditer(text):
        parsed = parse_block(m.group(1))
        if parsed is None:
            continue
        calls.append(_to_openai(*parsed))
        leftover_spans.append((pos, m.start()))
        pos = m.end()
    leftover_spans.append((pos, len(text)))
    content = "".join(text[a:b] for a, b in leftover_spans).strip()
    if calls and not content:
        return None, calls
    return content, calls


def _parse_arguments(text: str, pattern) -> dict | None:
    args = {}
    pos = 0
    for match in pattern.finditer(text):
        if text[pos:match.start()].strip():
            return None
        key = match.group(1).strip()
        if not key:
            return None
        args[key] = _coerce(match.group(2))
        pos = match.end()
    if text[pos:].strip():
        return None
    return args


def _parse_ling_block(body: str):
    name_match = re.match(r"^([^<]+)", body)
    if not name_match:
        return None
    name = name_match.group(1).strip()
    if not name:
        return None
    args = _parse_arguments(body[name_match.end():], _LING_ARG)
    if args is None:
        return None
    return name, args


def parse_ling_tool_calls(text: str) -> tuple[str | None, list[dict]]:
    """Split ``text`` into (remaining content, OpenAI tool_calls) for the
    edge0-8b (Ling) XML dialect. Returns ``(text.strip(), [])`` when no
    ``<tool_call>`` block is present -- same stripping convention as
    ``app._split_think``."""
    return _split(text, _parse_ling_block)


def _parse_qwen_block(body: str):
    fn_match = _QWEN_FUNCTION.fullmatch(body)
    if not fn_match:
        return None
    name = fn_match.group(1).strip()
    if not name:
        return None
    params = fn_match.group(2)
    args = _parse_arguments(params, _QWEN_PARAM)
    if args is None:
        return None
    return name, args


def parse_qwen_tool_calls(text: str) -> tuple[str | None, list[dict]]:
    """Split ``text`` into (remaining content, OpenAI tool_calls) for the
    edge0-35b (Qwen3.5) XML dialect. Returns ``(text.strip(), [])`` when
    no ``<tool_call>`` block is present -- same stripping convention as
    ``app._split_think``."""
    return _split(text, _parse_qwen_block)
