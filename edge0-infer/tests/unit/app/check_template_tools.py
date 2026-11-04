"""Byte-exact rendering against the transformers 4.57.1 Jinja environment."""
import itertools
import json
from pathlib import Path
import subprocess
import sys
from jinja2.sandbox import ImmutableSandboxedEnvironment

root = Path(__file__).resolve().parents[3]
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                   extensions=["jinja2.ext.loopcontrols"])
env.filters["tojson"] = lambda x: json.dumps(x, ensure_ascii=False)
def raise_exception(message):
    raise ValueError(message)
env.globals["raise_exception"] = raise_exception
template = env.from_string((root / "tests/fixtures/models/edge0_35b/chat_template.jinja").read_text())
values = json.loads(r'{"city":"\u5317\u4eac\n\"\\/\u0001","n":-0,"float":1.0,"half":0.5,"small":1e-05,"large":1e16,"yes":true,"none":null,"obj":{"z":1,"a":2},"arr":[1,"x"]}')
tool = {"type": "function", "function": {"name": "get_weather",
        "description": values["city"], "parameters": {"type": "object",
        "properties": {"city": {"type": "string"}}}, "extra": values}}
call = {"function": {"name": "get_weather", "arguments": values}}
count = 0
for thinking, preserve, system, content, reasoning, multiple in itertools.product(
        [False, True], [False, True], [None, "", " \n ", " Be brief "],
        ["", " Answer "], [None, "", " reason "], [False, True]):
    messages = ([] if system is None else [{"role": "system", "content": system}])
    messages += [{"role": "user", "content": "Weather?"}]
    assistant = {"role": "assistant", "content": content,
                 "tool_calls": [call, call] if multiple else [call]}
    if reasoning is not None:
        assistant["reasoning_content"] = reasoning
    messages.append(assistant)
    messages += [{"role": "tool", "content": "  result  "}] * (2 if multiple else 1)
    messages += [{"role": "user", "content": "<tool_response>more</tool_response>"},
                 {"role": "assistant", "content": "<think>\nold\n</think>\n\nanswer"},
                 {"role": "user", "content": "Next"}]
    for tools in [[], [tool]]:
        case = dict(messages=messages, thinking=thinking, preserve=preserve, tools=tools)
        kwargs = dict(messages=messages, enable_thinking=thinking,
                      preserve_thinking=preserve, add_generation_prompt=True)
        if tools:
            kwargs["tools"] = tools
        expected = template.render(**kwargs).encode()
        result = subprocess.run([sys.argv[1]], input=json.dumps(case).encode(),
                                capture_output=True, check=True)
        assert result.stdout == expected, (case, result.stdout, expected)
        count += 1
subprocess.run([sys.executable, str(root / "tests/unit/model/check_json.py"), sys.argv[2]], check=True)
print(f"{count} exact template comparisons passed")
