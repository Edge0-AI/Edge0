"""edge0.server.tool_calls: XML tool-call parsing for both checkpoint
families.

The fixture strings below are not invented: they were captured by
rendering the *real* ``chat_template.jinja`` shipped with each checkpoint
(edge0-8b / Ling, edge0-35b / Qwen3.5) with jinja2 directly, for an
assistant message carrying one ``tool_calls`` entry, and copying the
generated span verbatim -- including the Ling template's own leading-
whitespace quirk before the first ``<arg_key>`` (a real jinja2 whitespace-
control artifact in that template, not a typo here). This guards against
inventing a plausible-looking XML dialect that the real templates don't
actually produce.
"""

from __future__ import annotations

import json

import pytest

from edge0.server.tool_calls import parse_ling_tool_calls, parse_qwen_tool_calls

# Captured from edge0-8b/chat_template.jinja rendered with
# tools=[calculator] and one assistant tool_calls entry.
LING_REAL_SPAN = (
    "<tool_call>calculator\n"
    "                <arg_key>expression</arg_key>\n"
    "<arg_value>2 + 2</arg_value>\n"
    "</tool_call>"
)

# Captured from edge0-35b/chat_template.jinja's own in-prompt format
# example (example_function_name / example_parameter_1/2, the second
# spanning multiple lines -- the format the "IMPORTANT" reminder block
# documents to the model).
QWEN_REAL_SPAN = (
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def test_ling_parses_real_template_span():
    content, calls = parse_ling_tool_calls(LING_REAL_SPAN)
    assert content is None
    assert len(calls) == 1
    fn = calls[0]["function"]
    assert fn["name"] == "calculator"
    assert json.loads(fn["arguments"]) == {"expression": "2 + 2"}
    assert calls[0]["type"] == "function"
    assert calls[0]["id"].startswith("call_")


def test_qwen_parses_real_template_span_with_multiline_value():
    content, calls = parse_qwen_tool_calls(QWEN_REAL_SPAN)
    assert content is None
    assert len(calls) == 1
    fn = calls[0]["function"]
    assert fn["name"] == "example_function_name"
    args = json.loads(fn["arguments"])
    assert args["example_parameter_1"] == "value_1"
    assert args["example_parameter_2"] == (
        "This is the value for the second parameter\n"
        "that can span\nmultiple lines")


def test_no_tool_call_leaves_content_unchanged_stripped():
    for parser in (parse_ling_tool_calls, parse_qwen_tool_calls):
        content, calls = parser("  just a plain answer  ")
        assert content == "just a plain answer"
        assert calls == []


def test_prose_before_tool_call_is_preserved_as_content():
    text = "Sure, let me compute that.\n" + LING_REAL_SPAN
    content, calls = parse_ling_tool_calls(text)
    assert content == "Sure, let me compute that."
    assert len(calls) == 1


def test_multiple_tool_calls_in_one_response():
    text = LING_REAL_SPAN + "\n" + (
        "<tool_call>calculator\n<arg_key>expression</arg_key>\n"
        "<arg_value>3 * 3</arg_value>\n</tool_call>")
    content, calls = parse_ling_tool_calls(text)
    assert content is None
    assert len(calls) == 2
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "expression": "2 + 2"}
    assert json.loads(calls[1]["function"]["arguments"]) == {
        "expression": "3 * 3"}
    # each call gets its own id
    assert calls[0]["id"] != calls[1]["id"]


def test_qwen_malformed_block_kept_as_content():
    # missing '<function=...>' -- not what the template ever emits, but a
    # truncated generation could still produce it; must not crash or
    # silently eat the block.
    text = "<tool_call>\nnot a function block\n</tool_call>"
    content, calls = parse_qwen_tool_calls(text)
    assert calls == []
    assert "<tool_call>" in content


@pytest.mark.parametrize("parser,body", [
    (parse_ling_tool_calls, "calculator<arg_key>x</arg_key>"),
    (parse_ling_tool_calls, "calculator<arg_key></arg_key><arg_value>1</arg_value>"),
    (parse_ling_tool_calls, "calculator<arg_key>x</arg_key><arg_value>1</arg_value>junk"),
    (parse_qwen_tool_calls, "<function=calculator><parameter=x>broken</function>"),
    (parse_qwen_tool_calls, "<function=calculator></function>junk"),
    (parse_qwen_tool_calls, "<function=calculator><parameter=>\n1\n</parameter></function>"),
])
def test_incomplete_or_unconsumed_arguments_are_preserved(parser, body):
    text = f"<tool_call>{body}</tool_call>"
    content, calls = parser(text)
    assert content == text
    assert calls == []


@pytest.mark.parametrize("parser,text", [
    (parse_ling_tool_calls, "<tool_call>get_time</tool_call>"),
    (parse_qwen_tool_calls, "<tool_call><function=get_time></function></tool_call>"),
])
def test_zero_argument_tool_call(parser, text):
    content, calls = parser(text)
    assert content is None
    assert json.loads(calls[0]["function"]["arguments"]) == {}


def test_qwen_function_allows_surrounding_whitespace():
    text = "<tool_call>\n  <function=get_time></function>\n  </tool_call>"
    content, calls = parse_qwen_tool_calls(text)
    assert content is None
    assert calls[0]["function"]["name"] == "get_time"


def test_malformed_blocks_survive_between_valid_calls():
    malformed = "<tool_call>calculator<arg_key>x</arg_key></tool_call>"
    content, calls = parse_ling_tool_calls(LING_REAL_SPAN + malformed + LING_REAL_SPAN)
    assert content == malformed
    assert len(calls) == 2


def test_ling_argument_coercion_numbers_and_json():
    text = (
        "<tool_call>set_temperature"
        "<arg_key>value</arg_key>\n<arg_value>21.5</arg_value>"
        "<arg_key>enabled</arg_key>\n<arg_value>true</arg_value>"
        "<arg_key>label</arg_key>\n<arg_value>living room</arg_value>"
        "\n</tool_call>"
    )
    _, calls = parse_ling_tool_calls(text)
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"value": 21.5, "enabled": True, "label": "living room"}
