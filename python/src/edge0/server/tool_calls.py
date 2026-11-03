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
    ``v if v is string else v|tojson`` split, in reverse.

    ``NaN``, ``Infinity`` and ``-Infinity`` stay strings: ``json.loads``
    accepts them, but ``tojson`` never writes them and ``json.dumps``
    would turn them back into tokens that are not valid JSON in
    ``function.arguments``."""
    try:
        return json.loads(raw, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError):
        return raw


def _reject_constant(name: str):
    raise ValueError(f"{name} is not valid JSON")


def _string_parameters(tools, name: str) -> set[str]:
    """Names of ``name``'s parameters that the request's ``tools`` declare
    as plain strings (``"type": "string"``).

    Both chat templates write a string argument bare (``v if v is string
    else v|tojson``), so ``12345``, ``true`` or ``null`` in a string
    parameter look exactly like JSON. Without the schema, ``_coerce``
    turns a zip code or an id into a number; with it, the value stays the
    string the template rendered."""
    if not isinstance(tools, list):
        return set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict) or function.get("name") != name:
            continue
        parameters = function.get("parameters")
        properties = (parameters.get("properties")
                      if isinstance(parameters, dict) else None)
        if not isinstance(properties, dict):
            return set()
        return {key for key, schema in properties.items()
                if isinstance(schema, dict) and schema.get("type") == "string"}
    return set()


def _tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _to_openai(name: str, args: dict) -> dict:
    return {
        "id": _tool_call_id(),
        "type": "function",
        "function": {"name": name.strip(), "arguments": json.dumps(args)},
    }


def _split(text: str, parse_block,
           tools=None) -> tuple[str | None, list[dict]]:
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
        name, raw_args = parsed
        strings = _string_parameters(tools, name.strip())
        args = {key: raw if key in strings else _coerce(raw)
                for key, raw in raw_args.items()}
        calls.append(_to_openai(name, args))
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
        args[key] = match.group(2)
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


def parse_ling_tool_calls(text: str,
                          tools=None) -> tuple[str | None, list[dict]]:
    """Split ``text`` into (remaining content, OpenAI tool_calls) for the
    edge0-8b (Ling) XML dialect. Returns ``(text.strip(), [])`` when no
    ``<tool_call>`` block is present -- same stripping convention as
    ``app._split_think``. ``tools`` is the request's tool list; parameters
    it declares as strings are not coerced from JSON."""
    return _split(text, _parse_ling_block, tools)


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


def parse_qwen_tool_calls(text: str,
                          tools=None) -> tuple[str | None, list[dict]]:
    """Split ``text`` into (remaining content, OpenAI tool_calls) for the
    edge0-35b (Qwen3.5) XML dialect. Returns ``(text.strip(), [])`` when
    no ``<tool_call>`` block is present -- same stripping convention as
    ``app._split_think``. ``tools`` is the request's tool list; parameters
    it declares as strings are not coerced from JSON."""
    return _split(text, _parse_qwen_block, tools)
