"""One bounded coding smoke check, separate from throughput measurements."""
import argparse
import ast
import json
import re
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://127.0.0.1:8080")
args = parser.parse_args()

request = {
    "model": "strata",
    "messages": [{"role": "user", "content":
        "Write only Python code defining clamp_score(value: int) -> int. "
        "Return value clamped to the inclusive range 0 through 100. "
        "No imports, no markdown, no tests, no extra text."}],
    "max_tokens": 256, "temperature": 0, "reasoning_effort": "none",
}
wire = urllib.request.Request(
    args.url.rstrip("/") + "/v1/chat/completions",
    data=json.dumps(request).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(wire, timeout=300) as response:
    result = json.load(response)
code = result["choices"][0]["message"]["content"].strip()
code = re.sub(r"^```(?:python)?\s*\n|\n```$", "", code)
tree = ast.parse(code)
allowed_nodes = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
                 ast.If, ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq,
                 ast.Call, ast.Name, ast.Load, ast.Constant, ast.UnaryOp, ast.USub,
                 ast.Expr)
allowed_names = {"clamp_score", "value", "int", "min", "max"}
if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
    raise ValueError("Expected one function definition")
if tree.body[0].name != "clamp_score" or tree.body[0].decorator_list:
    raise ValueError("Unexpected function name or decorator")
for node in ast.walk(tree):
    if not isinstance(node, allowed_nodes):
        raise ValueError(f"Unsupported generated syntax: {type(node).__name__}")
    if isinstance(node, ast.Name) and node.id not in allowed_names:
        raise ValueError(f"Unexpected generated name: {node.id}")
namespace = {"__builtins__": {"int": int, "min": min, "max": max}}
exec(compile(tree, "generated-clamp-score", "exec"), namespace)
failures = []
for value in range(-1000, 1001):
    actual = namespace["clamp_score"](value)
    if actual != min(100, max(0, value)):
        failures.append({"input": value, "actual": actual})
report = {"url": args.url, "request": request, "response": result, "generated_code": code,
          "cases": 2001, "failures": failures, "passed": not failures,
          "scope": "One simple function only; not a general coding-quality benchmark."}
Path(__file__).with_name("coding-smoke-check.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"cases": 2001, "passed": not failures, "code": code}))
