"""Keep checkpoint-supplied Python opt-in across backend loaders."""

from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _function(path: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return next(node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name)


def _default_false(function: ast.FunctionDef, name: str) -> bool:
    positional = function.args.args
    positional_defaults = [None] * (
        len(positional) - len(function.args.defaults))
    positional_defaults.extend(function.args.defaults)
    keyword_only = function.args.kwonlyargs
    defaults = list(zip(positional, positional_defaults))
    defaults.extend(zip(keyword_only, function.args.kw_defaults))
    return any(arg.arg == name and isinstance(default, ast.Constant)
               and default.value is False for arg, default in defaults)


def _keyword_value(function: ast.FunctionDef, method: str,
                   keyword: str) -> ast.expr | None:
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        called_name = (node.func.attr if isinstance(node.func, ast.Attribute)
                       else node.func.id if isinstance(node.func, ast.Name)
                       else None)
        if called_name == method:
            for item in node.keywords:
                if item.arg == keyword:
                    return item.value
    return None


def test_tokenizer_remote_code_is_opt_in_for_both_backends():
    for path in ("src/edge0/backends/cuda/io.py",
                 "src/edge0/backends/mlx/io.py"):
        loader = _function(path, "load_tokenizer")
        assert _default_false(loader, "trust_remote_code")
        value = _keyword_value(loader, "from_pretrained", "trust_remote_code")
        assert isinstance(value, ast.Name)
        assert value.id == "trust_remote_code"


def test_cuda_model_remote_code_is_explicitly_opt_in():
    loader = _function("src/edge0/backends/cuda/io.py", "load_model")
    resolver = _function("src/edge0/backends/cuda/io.py",
                         "_resolve_model_class")
    assert _default_false(loader, "trust_remote_code")
    assert _default_false(resolver, "trust_remote_code")

    resolver_call = _keyword_value(
        loader, "_resolve_model_class", "trust_remote_code")
    assert isinstance(resolver_call, ast.Name)
    assert resolver_call.id == "trust_remote_code"

    refusal = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Name)
        and node.test.operand.id == "trust_remote_code"
        and any(isinstance(child, ast.Raise)
                and isinstance(child.exc, ast.Call)
                and isinstance(child.exc.func, ast.Name)
                and child.exc.func.id == "ValueError"
                for child in ast.walk(node))
        for node in ast.walk(resolver))
    assert refusal, "CUDA loader must refuse custom code unless opted in"

    for method in ("from_pretrained", "from_config"):
        value = _keyword_value(resolver if method == "from_pretrained"
                               else loader, method, "trust_remote_code")
        assert isinstance(value, ast.Name)
        assert value.id == "trust_remote_code"
