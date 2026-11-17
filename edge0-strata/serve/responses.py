"""serve/responses.py - #451: OpenAI's Responses API (POST /v1/responses), stateless, on the chat path.

A Responses request becomes the same template messages, tools and kwargs a Chat Completions request does, runs through
the same Service.run, and its events (reasoning, text, tool calls) come back as Responses output items and typed SSE
events.  Nothing is stored: the client sends the whole conversation in `input` every time (`store: false`, as Codex CLI
does), including the reasoning, message and function_call items of earlier answers.

Codex CLI is the first client.  What it needs, and what this does:
  * `instructions` and `developer` messages -> the system message (leading ones merged, so the prompt start is stable
    and the conversation cache holds across turns);
  * function tools, also inside `namespace` tools (Codex's MCP and agent tools: the model sees `namespace.name`) and
    `custom` tools (a free-form `input` string); hosted tools (web search, ...) are left out, the model cannot run them;
  * reasoning items: the model's thinking is returned as `reasoning_text` content (no summaries: this model does not
    write them, see docs/DETAILS.md) and, when `include` asks for `reasoning.encrypted_content`, also as an opaque
    `encrypted_content` string (base64, not encrypted: the client already holds the text).  Either form sent back is
    put into the conversation again, so the prompt matches what the model wrote and the cache is reused;
  * the official event order: response.created, response.in_progress, response.output_item.added, deltas,
    ...done events, response.output_item.done, and response.completed / response.incomplete / response.failed.

Based on #630 by CC-David-CC (the item/event assembler, namespace tools, call_id ordering of tool results).
"""
from __future__ import annotations

import base64
import copy
import json
import time
import uuid

from serve.frontend import Event, _late_system_to_user, _parts_of, effort_kwargs

ENCRYPTED_PREFIX = "strata.r1:"                   # our own reasoning replay strings; others' are ignored
HOSTED_TOOLS = ("web_search", "web_search_preview", "file_search", "computer_use_preview", "computer_use",
                "code_interpreter", "image_generation", "local_shell", "mcp", "tool_search")


class ResponsesError(ValueError):
    """A request error in the Responses format: {"error": {"message", "type", "param", "code"}}."""

    def __init__(self, message, param=None, code=None, status=400, kind="invalid_request_error"):
        super().__init__(message)
        self.param, self.code, self.status, self.kind = param, code, status, kind

    def body(self):
        return error_body(str(self), self.kind, self.param, self.code)


def error_body(message, kind="invalid_request_error", param=None, code=None) -> dict:
    return {"error": {"message": message, "type": kind, "param": param, "code": code}}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


# ------------------------------------------------------------------------------------------------ request -> chat
def _content(content, param):
    """A message's (or a tool output's) content -> the template's form: a string, or text and image parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ResponsesError("expected a string or an array of content parts", param)
    parts = []
    for j, part in enumerate(content):
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            raise ResponsesError("expected a content part object", f"{param}[{j}]")
        kind = part.get("type")
        if kind in ("input_text", "output_text", "text", "summary_text", "reasoning_text"):
            parts.append({"type": "text", "text": part.get("text") or ""})
        elif kind == "refusal":
            parts.append({"type": "text", "text": part.get("refusal") or ""})
        elif kind in ("input_image", "image_url"):
            if not part.get("image_url"):
                raise ResponsesError("only images given as image_url (a data: or http(s) URL) are supported; "
                                     "this server keeps no files", f"{param}[{j}]", "unsupported_parameter")
            parts.append({"type": "input_image", "image_url": part["image_url"]})
        else:
            raise ResponsesError(f"content parts of type {kind!r} are not supported (text and images are)",
                                 f"{param}[{j}].type", "unsupported_parameter")
    return _parts_of(parts)


def _reasoning_text(item, param) -> str:
    """The thinking a reasoning item carries: its reasoning_text content, else our own encrypted_content.  A summary
    alone, or another server's encrypted_content, carries none that can go back into the prompt."""
    texts = [p.get("text") or "" for p in item.get("content") or [] if isinstance(p, dict)
             and p.get("type") in ("reasoning_text", "text")]
    if texts:
        return "".join(texts)
    enc = item.get("encrypted_content")
    if isinstance(enc, str) and enc.startswith(ENCRYPTED_PREFIX):
        try:
            return json.loads(base64.b64decode(enc[len(ENCRYPTED_PREFIX):], validate=True).decode("utf-8"))["text"]
        except (ValueError, KeyError, TypeError, UnicodeError):
            raise ResponsesError("this reasoning item's encrypted_content is damaged", param + ".encrypted_content",
                                 "invalid_encrypted_content") from None
    return ""


def _arguments(raw, param) -> dict:
    """A function_call's arguments (a JSON string) -> the mapping the template renders.  Arguments that are not a JSON
    object (a call the output ended inside) are kept as text rather than refusing the whole conversation."""
    if isinstance(raw, dict):
        return raw
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}
    if not isinstance(raw, str):
        raise ResponsesError("function_call arguments must be a JSON string", param)
    try:
        value = json.loads(raw)
    except ValueError:
        return {"arguments": raw}
    return value if isinstance(value, dict) else {"arguments": value}


def _tool_output(item, param):
    out = item.get("output")
    if isinstance(out, dict):                        # some clients send a JSON result as an object
        return json.dumps(out, ensure_ascii=False)
    return out if isinstance(out, str) else _content(out, param + ".output")


def input_messages(req: dict) -> list[dict]:
    """`instructions` + `input` -> the chat template's messages."""
    messages = []
    instructions = req.get("instructions")
    if instructions not in (None, ""):
        messages.append({"role": "system", "content": _content(instructions, "instructions")
                         if not isinstance(instructions, str) else instructions})
    items = req.get("input")
    if items is None:
        raise ResponsesError("input is required: a string or an array of input items", "input", "missing_required_parameter")
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    if not isinstance(items, list):
        raise ResponsesError("input must be a string or an array of input items", "input")
    thinking = []                                    # reasoning waiting for the assistant turn it belongs to
    open_turn = None                                 # the assistant message the next function_call joins

    def turn(content=""):
        nonlocal open_turn
        open_turn = {"role": "assistant", "content": content}
        if thinking:
            open_turn["reasoning_content"] = "".join(thinking)
            thinking.clear()
        messages.append(open_turn)
        return open_turn

    for i, item in enumerate(items):
        param = f"input[{i}]"
        if not isinstance(item, dict):
            raise ResponsesError("expected an input item object", param)
        kind = item.get("type") or ("message" if "role" in item else None)
        if kind == "message":
            role = item.get("role")
            if role not in ("user", "assistant", "system", "developer"):
                raise ResponsesError(f"unknown message role {role!r}", param + ".role")
            content = _content(item.get("content"), param + ".content")
            if role == "assistant":
                turn(content if isinstance(content, str) else "".join(
                    p.get("text", "") for p in content if p.get("type") == "text"))
            else:
                open_turn = None
                thinking.clear()
                messages.append({"role": "system" if role == "developer" else role, "content": content})
        elif kind == "reasoning":
            text = _reasoning_text(item, param)
            if text:
                if open_turn is not None and (open_turn.get("tool_calls") or open_turn["content"]):
                    open_turn = None                 # thinking starts a new model turn
                thinking.append(text)
        elif kind in ("function_call", "custom_tool_call"):
            name = item.get("name")
            if not isinstance(name, str) or not name:
                raise ResponsesError("a function_call needs a name", param + ".name")
            if item.get("namespace"):
                name = f"{item['namespace']}.{name}"
            if kind == "function_call":
                args = _arguments(item.get("arguments"), param + ".arguments")
            else:
                args = {"input": item.get("input") if isinstance(item.get("input"), str) else ""}
            target = open_turn if open_turn is not None and not thinking else turn()
            target.setdefault("tool_calls", []).append({"function": {"name": name, "arguments": args},
                                                        "_call_id": item.get("call_id")})
        elif kind in ("function_call_output", "custom_tool_call_output"):
            messages.append({"role": "tool", "content": _tool_output(item, param), "_call_id": item.get("call_id")})
            open_turn = None
            thinking.clear()
        elif kind == "additional_tools":
            continue                                 # #782 (Codex): tools the client adds as an input item; see request_tools
        elif kind == "item_reference":
            raise ResponsesError("item references need stored responses, and this server keeps none: send the items "
                                 "themselves", param, "unsupported_parameter")
        else:
            raise ResponsesError(f"input items of type {kind!r} are not supported", param + ".type",
                                 "unsupported_parameter")
    _order_tool_results(messages)
    # leading system/developer messages (Codex: instructions, then its developer message) become one system message
    # at the start; later ones become user messages in place, as on the chat path
    lead = 0
    while lead < len(messages) and messages[lead]["role"] == "system" and isinstance(messages[lead]["content"], str):
        lead += 1
    if lead > 1:
        messages[:lead] = [{"role": "system", "content": "\n\n".join(m["content"] for m in messages[:lead])}]
    return _late_system_to_user(messages)


def _order_tool_results(messages):
    """The template places tool results by position: each run of tool messages is put in the order of the calls of
    the assistant turn before it (a client may send the results in the order they finished), then the private call
    ids are dropped."""
    i = 0
    while i < len(messages):
        if messages[i]["role"] != "tool":
            i += 1
            continue
        j = i
        while j < len(messages) and messages[j]["role"] == "tool":
            j += 1
        prev = messages[i - 1] if i else {}
        order = [c.get("_call_id") for c in prev.get("tool_calls") or []]
        rank = {cid: k for k, cid in enumerate(order) if cid}
        messages[i:j] = sorted(messages[i:j], key=lambda m: rank.get(m.get("_call_id"), len(order)))
        i = j
    for m in messages:
        m.pop("_call_id", None)
        for c in m.get("tool_calls") or []:
            c.pop("_call_id", None)


def request_tools(req: dict):
    """-> (template tools or None, {template name: (namespace, name, kind)}, names of hosted tools left out)."""
    tools, names, skipped = [], {}, []
    given = req.get("tools") or []
    if not isinstance(given, list):
        raise ResponsesError("tools must be an array", "tools")
    # #782 (Codex v0.160): `additional_tools` input items carry more tool definitions (a `tools` array, as the request
    # has); they join the request's own, and an item without a usable array adds nothing
    extra = [t for item in (req.get("input") if isinstance(req.get("input"), list) else [])
             if isinstance(item, dict) and item.get("type") == "additional_tools" and isinstance(item.get("tools"), list)
             for t in item["tools"]]
    given = list(given) + extra

    def add(tool, param, namespace=None, ns_description=""):
        kind = tool.get("type")
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise ResponsesError("a tool needs a name", param + ".name")
        flat = f"{namespace}.{name}" if namespace else name
        description = tool.get("description") or ""
        if ns_description:
            description = ns_description + ("\n\n" + description if description else "")
        if kind == "function":
            params = tool.get("parameters") or {"type": "object", "properties": {}}
        else:                                        # custom: free-form text, maybe in a grammar the model reads
            fmt = tool.get("format") or {}
            if isinstance(fmt, dict) and fmt.get("type") == "grammar" and fmt.get("definition"):
                description += f"\n\nThe input must follow this {fmt.get('syntax', '')} grammar:\n{fmt['definition']}"
            params = {"type": "object", "properties": {"input": {"type": "string", "description": "the raw input"}},
                      "required": ["input"]}
        tools.append({"name": flat, "description": description, "parameters": params})
        names[flat] = (namespace, name, kind)

    for i, tool in enumerate(given):
        param = f"tools[{i}]"
        if not isinstance(tool, dict):
            raise ResponsesError("expected a tool object", param)
        kind = tool.get("type")
        if kind in ("function", "custom"):
            add(tool, param)
        elif kind == "namespace":
            for j, member in enumerate(tool.get("tools") or []):
                if not isinstance(member, dict) or member.get("type", "function") not in ("function", "custom"):
                    raise ResponsesError("a namespace holds function tools", f"{param}.tools[{j}]")
                add({"type": "function", **member}, f"{param}.tools[{j}]", tool.get("name"),
                    tool.get("description") or "")
        elif kind in HOSTED_TOOLS or isinstance(kind, str):
            skipped.append(kind)                     # the model cannot run OpenAI's hosted tools: left out
        else:
            raise ResponsesError("a tool needs a type", param + ".type")
    choice = req.get("tool_choice")
    if choice == "none":
        return None, names, skipped
    return tools or None, names, skipped


def _turn_body(req: dict):
    meta = req.get("client_metadata")
    raw = meta.get("x-codex-turn-metadata") if isinstance(meta, dict) else None
    try:
        turn = json.loads(raw) if isinstance(raw, str) else None
    except ValueError:
        return None
    return turn if isinstance(turn, dict) else None


def _title_text(req: dict) -> str:
    """The user line Codex asked to name, cut to the schema's 36 characters."""
    parts = []
    items = req.get("input")
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict) or item.get("role") != "user":
                continue
            content = item.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for piece in content:
                    if isinstance(piece, dict) and isinstance(piece.get("text"), str):
                        parts.append(piece["text"])
    text = "\n".join(parts)
    if "User prompt:" in text:
        text = text.split("User prompt:", 1)[1]
    line = next((x.strip() for x in text.splitlines() if x.strip()), "Untitled")
    line = line.strip(" \"'`")
    while line and line[-1] in ".。!！?？,，;；:：":
        line = line[:-1].rstrip()
    return line[:36].rstrip() or "Untitled"


def thread_title_events(req: dict, model: str):
    """Codex's thread-title turn, answered here. None when this request is not one.

    Codex 0.160 sends this beside the user turn: another session, no tools, the full instructions. Running it on the
    one prefix cache replaces the conversation.
    """
    if (_turn_body(req) or {}).get("thread_source") != "thread_title":
        return None
    check_request(req)
    text = json.dumps({"title": _title_text(req)}, ensure_ascii=False)
    asm = Assembler(req, model, 0, {}, False, json_mode=False)
    events = asm.start()
    events += asm._open({"id": new_id("msg"), "type": "message", "status": "in_progress", "role": "assistant",
                         "content": []})
    asm.item["content"][0]["text"] = text
    events.append(asm.event("response.output_text.delta", item_id=asm.item["id"], output_index=asm.index,
                            content_index=0, delta=text, logprobs=[]))
    events += asm.close()
    asm.response["usage"] = {
        "input_tokens": 0, "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 0}
    asm.response["status"] = "completed"
    asm.response["completed_at"] = int(time.time())
    events.append(asm.event("response.completed", response=asm.snapshot()))
    return events


# Codex's local compaction (the context is full, or /compact) sends its conversation again with `tools: []`
# (codex-rs/core/src/compact.rs: `Prompt { input, base_instructions, ..Default::default() }`).  The template writes
# the tools at the top of the prompt, so that prompt shares almost nothing with the one the engine holds and the whole
# conversation is read again, at its longest.  So the tools a Codex conversation's last prompt was rendered with are
# kept, and that conversation's compaction prompt is rendered with them.  Only the prompt: the parser and the response
# get the request's own tools, which are none.  Codex says what a request is and whose it is in
# client_metadata["x-codex-turn-metadata"] (a JSON string, Codex 0.140 and later): `request_kind` and the
# conversation's `session_id` and `thread_id` (a sub-agent shares its parent's session, not its thread; a fork gets
# its own).  `prompt_cache_key` is not used: it groups requests for caching, and Codex gives sub-agents and ephemeral
# forks their parent's.  A request without that metadata is rendered as sent and changes nothing.  One entry, replaced
# by each prompt of a Codex conversation: nothing grows, and a compaction of any other conversation finds a different
# (session_id, thread_id) and renders its request as sent.  Lost (a restart): the prompt is read again, as without it.
_kept_prompt_tools = (None, None)                    # ((session_id, thread_id), its prompt's tools as JSON or None)


def _codex_turn(req: dict):
    """-> (request_kind, (session_id, thread_id)) from Codex's turn metadata; the conversation is None without both."""
    meta = req.get("client_metadata")
    raw = meta.get("x-codex-turn-metadata") if isinstance(meta, dict) else None
    try:
        turn = json.loads(raw) if isinstance(raw, str) else None
    except ValueError:
        turn = None
    if not isinstance(turn, dict):
        return None, None
    conversation = (turn.get("session_id"), turn.get("thread_id"))
    return turn.get("request_kind"), conversation if all(isinstance(x, str) and x for x in conversation) else None


def prompt_tools(req: dict, tools):
    """-> the tools the prompt is rendered with: request_tools' `tools`, or for a Codex compaction request without
    tools the ones its own conversation's last prompt was rendered with (so its prompt starts as that one did)."""
    kind, conversation = _codex_turn(req)
    owner, kept = _kept_prompt_tools
    if kind != "compaction" or conversation is None or owner != conversation or kept is None or req.get("tools"):
        return tools
    return json.loads(kept)                          # a fresh copy: the kept one is nobody's to change


def prompt_made(req: dict, tools):
    """The prompt of `req` was rendered with `tools` and goes to the engine: a Codex request that is not a compaction
    leaves its conversation and those tools for that conversation's compaction. A thread-title turn is another
    session Codex sends beside the real one, and it does not replace this entry."""
    global _kept_prompt_tools
    if (_turn_body(req) or {}).get("thread_source") == "thread_title":
        return
    kind, conversation = _codex_turn(req)
    if conversation is not None and kind != "compaction":
        _kept_prompt_tools = (conversation, json.dumps(tools) if tools else None)


def text_format(req: dict):
    """`text.format` -> the chat path's response_format (None: plain text)."""
    text = req.get("text")
    if text is None:
        return None
    if not isinstance(text, dict):
        raise ResponsesError("text must be an object", "text")
    fmt = text.get("format")
    if fmt is None:
        return None
    if not isinstance(fmt, dict):
        raise ResponsesError("text.format must be an object", "text.format")
    kind = fmt.get("type")
    if kind in (None, "text"):
        return None
    extra = {"repair_latex_escapes": True} if fmt.get("repair_latex_escapes") is True else {}   # #2893, opt-in
    if kind == "json_object":
        return {"type": "json_object", **extra}
    if kind == "json_schema":
        spec = {k: fmt[k] for k in ("name", "schema", "strict", "description") if k in fmt}
        return {"type": "json_schema", "json_schema": spec, **extra}
    raise ResponsesError("text.format.type must be text, json_object or json_schema", "text.format.type")


def check_request(req: dict) -> None:
    """What this stateless server cannot do, refused before anything runs."""
    if req.get("previous_response_id"):
        raise ResponsesError("this server keeps no responses (stateless): send the whole conversation in input "
                             "instead of previous_response_id", "previous_response_id", "unsupported_parameter")
    if req.get("conversation"):
        raise ResponsesError("this server keeps no conversations (stateless): send the whole conversation in input",
                             "conversation", "unsupported_parameter")
    if req.get("background"):
        raise ResponsesError("background responses are not supported: use stream or wait for the answer",
                             "background", "unsupported_parameter")
    if req.get("prompt"):
        raise ResponsesError("stored prompt templates are not supported: send instructions and input",
                             "prompt", "unsupported_parameter")
    n = req.get("max_output_tokens")
    # -1 and 0 mean no cap, as max_tokens does on the chat path (docs/DETAILS.md); anything lower is a mistake
    if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n < -1):
        raise ResponsesError("max_output_tokens must be a whole number, 1 or more, or -1 or 0 for no cap",
                             "max_output_tokens")
    reasoning = req.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, dict):
        raise ResponsesError("reasoning must be an object", "reasoning")


def template_kwargs(req: dict, shared: dict) -> dict:
    effort = (req.get("reasoning") or {}).get("effort")
    if effort is None:
        effort = shared.get("reasoning_effort")      # the Chat settings shared with apps, as on the chat path
    try:
        return effort_kwargs(effort)
    except ValueError as e:
        raise ResponsesError(str(e), "reasoning.effort") from None


# ------------------------------------------------------------------------------------------------ output
class Assembler:
    """One response: its output items and its typed events, built from Service.run's events.  The non-streaming
    answer is the `response` of the last event, so both modes are the same items."""

    def __init__(self, req: dict, model: str, prompt_tokens: int, names: dict, encrypted: bool,
                 json_mode: bool = False):
        self.names, self.encrypted, self.json_mode = names, encrypted, json_mode
        self.prompt_tokens = prompt_tokens
        self.seq = 0
        self.item = None                             # the open output item
        self.json_text = []                          # structured output: the answer, held until it is checked
        self.called = False                          # a tool call was made: this turn is not the final answer
        reasoning = req.get("reasoning") if isinstance(req.get("reasoning"), dict) else {}
        self.response = {
            "id": new_id("resp"), "object": "response", "created_at": int(time.time()), "status": "in_progress",
            "background": False, "error": None, "incomplete_details": None,
            "instructions": req.get("instructions"), "max_output_tokens": req.get("max_output_tokens"),
            "model": model, "output": [], "parallel_tool_calls": req.get("parallel_tool_calls", True),
            "previous_response_id": None,
            "reasoning": {"effort": reasoning.get("effort"), "summary": reasoning.get("summary")},
            "store": False, "temperature": req.get("temperature"),
            "text": req.get("text") if isinstance(req.get("text"), dict) else {"format": {"type": "text"}},
            "tool_choice": req.get("tool_choice", "auto"), "tools": req.get("tools") or [],
            "top_p": req.get("top_p"), "truncation": "disabled", "usage": None, "user": None,
            "metadata": req.get("metadata") or {}}

    # --- events
    def event(self, kind, **data) -> dict:
        e = {"type": kind, "sequence_number": self.seq, **data}
        self.seq += 1
        return e

    def snapshot(self) -> dict:
        return copy.deepcopy(self.response)

    def start(self) -> list[dict]:
        return [self.event("response.created", response=self.snapshot()),
                self.event("response.in_progress", response=self.snapshot())]

    def in_progress(self) -> dict:
        """A keep-alive a client counts as activity (SSE comments are not events: Codex's idle timer ignores them)."""
        return self.event("response.in_progress", response=self.snapshot())

    def _open(self, item) -> list[dict]:
        self.response["output"].append(item)
        self.item = item
        out = [self.event("response.output_item.added", output_index=self.index, item=copy.deepcopy(item))]
        if item["type"] == "message":
            part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
            item["content"].append(part)
            out.append(self.event("response.content_part.added", item_id=item["id"], output_index=self.index,
                                  content_index=0, part=copy.deepcopy(part)))
        return out

    @property
    def index(self) -> int:
        return len(self.response["output"]) - 1

    def close(self, status=None) -> list[dict]:
        item, self.item = self.item, None
        if item is None:
            return []
        at, out = self.response["output"].index(item), []
        item["status"] = status or ("incomplete" if item["type"] in ("function_call", "custom_tool_call") else "completed")
        if item["type"] == "message":
            part = item["content"][0]
            out += [self.event("response.output_text.done", item_id=item["id"], output_index=at, content_index=0,
                               text=part["text"], logprobs=[]),
                    self.event("response.content_part.done", item_id=item["id"], output_index=at, content_index=0,
                               part=copy.deepcopy(part))]
        elif item["type"] == "reasoning":
            text = item["content"][0]["text"]
            out.append(self.event("response.reasoning_text.done", item_id=item["id"], output_index=at,
                                  content_index=0, text=text))
            if self.encrypted:
                item["encrypted_content"] = ENCRYPTED_PREFIX + base64.b64encode(
                    json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")).decode("ascii")
        elif item["type"] == "function_call":
            out.append(self.event("response.function_call_arguments.done", item_id=item["id"], output_index=at,
                                  name=item["name"], arguments=item["arguments"]))
        elif item["type"] == "custom_tool_call":
            out.append(self.event("response.custom_tool_call_input.done", item_id=item["id"], output_index=at,
                                  input=item["input"]))
        out.append(self.event("response.output_item.done", output_index=at, item=copy.deepcopy(item)))
        return out

    def feed(self, ev: Event) -> list[dict]:
        out = []
        kind = ev.kind
        if kind == "reasoning":
            if not ev.text:
                return out
            if self.item is None or self.item["type"] != "reasoning":
                out += self.close()
                out += self._open({"id": new_id("rs"), "type": "reasoning", "status": "in_progress", "summary": [],
                                   "content": [{"type": "reasoning_text", "text": ""}]})
            self.item["content"][0]["text"] += ev.text
            out.append(self.event("response.reasoning_text.delta", item_id=self.item["id"], output_index=self.index,
                                  content_index=0, delta=ev.text))
        elif kind == "content":
            if not ev.text:
                return out
            if self.json_mode:                       # sent once it is checked (finish)
                out += self.close()
                self.json_text.append(ev.text)
                return out
            if self.item is None or self.item["type"] != "message":
                out += self.close()
                out += self._open({"id": new_id("msg"), "type": "message", "status": "in_progress",
                                   "role": "assistant", "content": []})
            self.item["content"][0]["text"] += ev.text
            out.append(self.event("response.output_text.delta", item_id=self.item["id"], output_index=self.index,
                                  content_index=0, delta=ev.text, logprobs=[]))
        elif kind == "tool_start":
            out += self.close()
            out += self._call_started()
            out += self._open(self._call_item(ev))
        elif kind == "tool_args":
            item = self.item
            if item is None or item.get("call_id") != ev.call.id:
                return out
            if item["type"] == "function_call":         # a custom tool's input is sent whole, at its end
                item["arguments"] += ev.text
                out.append(self.event("response.function_call_arguments.delta", item_id=item["id"],
                                      output_index=self.index, delta=ev.text))
        elif kind == "tool_call":
            item = self.item
            if item is None or item.get("call_id") != ev.call.id:     # a call that was not announced while written
                out += self.close()
                out += self._call_started()
                out += self._open(self._call_item(ev))
                item = self.item
                if item["type"] == "function_call":
                    item["arguments"] = json.dumps(ev.call.arguments, ensure_ascii=False)
                    out.append(self.event("response.function_call_arguments.delta", item_id=item["id"],
                                          output_index=self.index, delta=item["arguments"]))
            if item["type"] == "custom_tool_call":
                value = ev.call.arguments.get("input") if isinstance(ev.call.arguments, dict) else None
                item["input"] = value if isinstance(value, str) else json.dumps(ev.call.arguments, ensure_ascii=False)
                out.append(self.event("response.custom_tool_call_input.delta", item_id=item["id"],
                                      output_index=self.index, delta=item["input"]))
            out += self.close("completed")
        return out

    def _release(self, text: str) -> list[dict]:
        """A held structured answer (or the words written beside a tool call) as one message item."""
        if not text.strip():
            return []
        out = self._open({"id": new_id("msg"), "type": "message", "status": "in_progress", "role": "assistant",
                          "content": []})
        self.item["content"][0]["text"] = text
        out.append(self.event("response.output_text.delta", item_id=self.item["id"], output_index=self.index,
                              content_index=0, delta=text, logprobs=[]))
        return out + self.close()

    def _call_started(self) -> list[dict]:
        """The first tool call of a JSON-format turn: this turn is not the final answer, so what the model wrote
        before the call goes out as it is, ahead of the call, and the schema is checked on the turn that ends in
        text (#782)."""
        out = []
        if self.json_mode and not self.called:
            out = self._release("".join(self.json_text))
            self.json_text = []
        self.called = True
        return out

    def _call_item(self, ev: Event) -> dict:
        namespace, name, kind = self.names.get(ev.call.name, (None, ev.call.name, "function"))
        if kind == "custom":
            item = {"id": new_id("ctc"), "type": "custom_tool_call", "status": "in_progress", "call_id": ev.call.id,
                    "name": name, "input": ""}
        else:
            item = {"id": new_id("fc"), "type": "function_call", "status": "in_progress", "call_id": ev.call.id,
                    "name": name, "arguments": ""}
        if namespace:
            item["namespace"] = namespace
        return item

    def finish(self, done: dict, validate=None) -> list[dict]:
        """The end: close what is open, then response.completed (or .incomplete when max_output_tokens ran out).
        `validate(text, finish)` checks a structured answer and returns its text (or raises)."""
        length = done.get("finish") == "length"
        # a call still open was cut off (its tool_call never came, #211): incomplete, whatever the finish
        unfinished = self.item is not None and self.item["type"] in ("function_call", "custom_tool_call")
        out = self.close("incomplete" if length or unfinished else "completed")
        if self.json_mode:
            text = "".join(self.json_text)
            self.json_text = []
            if self.called:
                out += self._release(text)           # beside a tool call: as written (#782)
            else:
                out += self._release(validate(text, done.get("finish")))
        n = done.get("completion_tokens", 0)
        prompt = done.get("prompt_tokens", self.prompt_tokens)
        self.response["usage"] = {
            "input_tokens": prompt, "input_tokens_details": {"cached_tokens": min(done.get("reused") or 0, prompt)},
            "output_tokens": n, "output_tokens_details": {"reasoning_tokens": done.get("reasoning_tokens", 0)},
            "total_tokens": prompt + n}
        if length:
            self.response["status"] = "incomplete"
            self.response["incomplete_details"] = {"reason": "max_output_tokens"}
            return out + [self._with_timings(self.event("response.incomplete", response=self.snapshot()), done)]
        self.response["status"] = "completed"
        self.response["completed_at"] = int(time.time())
        return out + [self._with_timings(self.event("response.completed", response=self.snapshot()), done)]

    @staticmethod
    def _with_timings(event: dict, done: dict) -> dict:
        """Top-level `timings` on the final event, as /v1/chat/completions has them (#107)."""
        if done.get("timings"):
            event["timings"] = done["timings"]
        return event

    def failed(self, message: str, code: str = "server_error") -> dict:
        """response.failed: an error after the stream started (the HTTP status is already sent)."""
        if self.item is not None:
            self.item["status"] = "incomplete"
            self.item = None
        self.response["status"] = "failed"
        self.response["error"] = {"code": code, "message": message}
        return self.event("response.failed", response=self.snapshot())


def collect(events) -> dict:
    """The non-streaming answer: the final response object."""
    last = None
    for e in events:
        if e is not None:
            last = e
    if last.get("timings"):                          # as on a whole /v1/chat/completions reply (#107)
        return {**last["response"], "timings": last["timings"]}
    return last["response"]
