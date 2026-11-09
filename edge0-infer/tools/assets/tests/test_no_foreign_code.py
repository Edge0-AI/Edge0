"""Asset tooling must not reach code outside this repository (gguf-py, an Edge0 checkout, llama.cpp)."""
import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.l1

ROOT = Path(__file__).resolve().parents[1]
BUILD_CODE = ['edge0_35b/make_assets.py', 'edge0_35b/fetch_source.py', 'edge0_35b/edge0_convert']


def python_files(root: Path):
    return sorted(p for p in root.rglob('*.py') if '.venv' not in p.parts and '__pycache__' not in p.parts)


def docstrings(tree: ast.AST) -> set[int]:
    """Docstrings are documentation, like comments; they are not scanned for strings."""
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                ids.add(id(first.value))
    return ids


def foreign_references(root: Path) -> list[str]:
    found = []
    for path in python_files(root):
        rel = path.relative_to(root).as_posix()
        if rel == 'tests/test_no_foreign_code.py':
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        skip = docstrings(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(a.name.split('.')[0] == 'gguf' for a in node.names):
                found.append(f'{rel}:{node.lineno}: import gguf')
            if isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] == 'gguf':
                found.append(f'{rel}:{node.lineno}: from gguf import')
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
                # Test fixtures may spell gguf-py as data; tooling code may not.
                if 'gguf-py' in node.value and not rel.startswith('tests/'):
                    found.append(f'{rel}:{node.lineno}: gguf-py string')
                if 'MOE_EDGE0' in node.value:
                    found.append(f'{rel}:{node.lineno}: MOE_EDGE0')
                if rel.startswith(tuple(BUILD_CODE)) and any(k in node.value for k in ['--edge0', '--llama']):
                    found.append(f'{rel}:{node.lineno}: foreign code parameter in build code')
    return found


def test_asset_tooling_has_no_foreign_code_references():
    assert foreign_references(ROOT) == []


def test_scanner_catches_seeded_references(tmp_path):
    (tmp_path / 'edge0_35b').mkdir()
    (tmp_path / 'a.py').write_text('import gguf\nfrom gguf.reader import X\n')
    (tmp_path / 'b.py').write_text('import os\nPATH = os.path.join("llama", "gguf-py")\nE = "MOE_EDGE0"\n')
    (tmp_path / 'edge0_35b/make_assets.py').write_text('ARGS = ["--edge0", "--llama"]\n')
    (tmp_path / 'comment.py').write_text('"""Docs may mention gguf-py."""\n# gguf-py is mentioned in a comment only\n')
    found = foreign_references(tmp_path)
    assert len(found) == 6
    assert not any(f.startswith('comment.py') for f in found)
