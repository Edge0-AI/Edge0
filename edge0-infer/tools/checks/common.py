"""Shared source traversal and CMake command parsing."""
import os
from pathlib import Path
import re
import tomllib

INTERNAL = {"docs-internal", "openspec", ".agents", ".claude", ".git", ".codex"}
# Tool-generated, Git-ignored directories; never repository sources.
GENERATED = {".venv", "__pycache__", "node_modules", ".gradle", ".cxx", ".kotlin", ".build", "DerivedData"}
SOURCE_SUFFIXES = {".h", ".hh", ".hpp", ".hxx", ".c", ".cc", ".cpp", ".cxx", ".c++", ".m", ".mm"}


def third_party(root):
    manifest = root / "tools/checks/third_party.toml"
    if not manifest.exists():
        return {"ggml": {"checker": "check_ggml_origin.py"}}
    return tomllib.loads(manifest.read_text())["libraries"]


def dependency_allowed(path, library):
    return not path.startswith(("src/", "include/")) and any(
        path.startswith(prefix) for prefix in library.get("allowed_prefixes", []))


def portal_header(root, file, path, header):
    if not path.startswith(("tools/cli/", "tests/unit/cli/")):
        return False
    candidates = [root / header, file.parent / header]
    return any(candidate.is_file() and candidate.resolve().is_relative_to(
        (root / "tools/cli").resolve()) for candidate in candidates)


def files(root):
    found = []
    libraries = third_party(root)
    for folder, dirs, names in os.walk(root):
        rel_dir = Path(folder).relative_to(root)
        # Prune instead of filtering afterwards: a local virtualenv alone holds ~900 Python files.
        dirs[:] = [
            d for d in dirs
            if d not in INTERNAL and d not in GENERATED
            and not (rel_dir == Path(".") and d.startswith("build"))
            and not (rel_dir.parts and rel_dir.parts[0] == "examples" and d in {"build", "target", "dist"})
            and not (rel_dir == Path("third_party") and d in libraries)
        ]
        for name in names:
            path = Path(folder) / name
            if path.is_file() and not path.is_symlink():
                found.append(path)
    for path in sorted(found):
        yield path, path.relative_to(root).as_posix()


def cmake_commands(text):
    # Preserve offsets when removing comments for accurate line numbers.
    clean = re.sub(r"#[^\n]*", lambda m: " " * len(m[0]), text)
    for match in re.finditer(r"\b(\w+)\s*\(([^)]*)\)", clean, re.S):
        args = re.findall(r'"([^"]*)"|([^\s]+)', match[2])
        yield match[1].lower(), [a or b for a, b in args], clean[:match.start()].count("\n") + 1
