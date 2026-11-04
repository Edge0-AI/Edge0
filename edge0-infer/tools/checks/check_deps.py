"""Reject undeclared dependency sources and reference build inclusion."""
import hashlib
import ast
import importlib.util
from pathlib import Path
import re
import sys

_spec = importlib.util.spec_from_file_location("checks_common", Path(__file__).with_name("common.py"))
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)


def check(root):
    errors = []

    def error(path, line, rule):
        errors.append(f"deps: {path}:{line}: {rule}")

    if (root / ".gitmodules").exists():
        error(".gitmodules", 1, "submodule: forbidden dependency source")
    libraries = _common.third_party(root)
    third = root / "third_party"
    if third.exists():
        for directory in third.iterdir():
            if directory.is_dir() and directory.name not in libraries:
                error(directory.relative_to(root).as_posix(), 1, "third-party: unregistered directory")
    for name, library in libraries.items():
        if name == "ggml":
            continue
        directory = third / name
        registered = library.get("files", {})
        for file in directory.rglob("*"):
            if file.is_symlink() or (file.is_file() and file.relative_to(directory).as_posix() not in registered):
                error(file.relative_to(root).as_posix(), 1, "third-party: unregistered file or symlink")
        for relative, metadata in registered.items():
            file = directory / relative
            if not file.is_file():
                error(file.relative_to(root).as_posix(), 1, "third-party: missing registered file")
            elif hashlib.sha256(file.read_bytes()).hexdigest() != metadata["sha256"]:
                error(file.relative_to(root).as_posix(), 1, "third-party: SHA-256 mismatch: " + hashlib.sha256(file.read_bytes()).hexdigest())
    for file, path in _common.files(root):
        if file.suffix == ".md":
            continue
        if not (file.name == "CMakeLists.txt" or file.suffix in _common.SOURCE_SUFFIXES | {".cmake", ".py", ".toml", ".txt", ".json"}):
            continue
        text = file.read_text()
        if file.suffix in _common.SOURCE_SUFFIXES:
            for match in re.finditer(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', text, re.M):
                for library in libraries.values():
                    if match[1] in library.get("headers", []) and not _common.dependency_allowed(path, library):
                        error(path, text[:match.start()].count("\n") + 1, "third-party: forbidden include")
        for number, line in enumerate(text.splitlines(), 1):
            if "gguf-py" in line and not (file.suffix == ".py" and line.lstrip().startswith("#")):
                # Python literals in checker fixtures are not dependency imports.
                if file.suffix != ".py":
                    error(path, number, "gguf-py: forbidden Python dependency")
        if "gguf-py" in path:
            error(path, 1, "gguf-py: forbidden Python dependency path")
        if file.suffix == ".py":
            try:
                tree = ast.parse(text)
            except SyntaxError as exc:
                error(path, exc.lineno, "python-parse: invalid source")
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "gguf" for a in node.names):
                    error(path, node.lineno, "gguf-py: forbidden Python import")
                if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "gguf":
                    error(path, node.lineno, "gguf-py: forbidden Python import")
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.search(r"(?:^|[/\\])gguf-py[/\\]", node.value):
                    error(path, node.lineno, "gguf-py: forbidden dependency reference")
        if file.name == "pyproject.toml" or file.name.startswith("requirements"):
            for number, line in enumerate(text.splitlines(), 1):
                if re.search(r"(?:^|[\"'\s])gguf(?:[<>=!~\[\"'\s]|$)", line, re.I):
                    error(path, number, "gguf-py: forbidden dependency package")
        if file.name != "CMakeLists.txt" and file.suffix != ".cmake":
            continue
        for command, args, number in _common.cmake_commands(text):
            if command.startswith(("fetchcontent", "externalproject")) or (command == "include" and args and args[0].lower() in {"fetchcontent", "externalproject"}):
                error(path, number, "dependency-source: network dependency forbidden")
            if command == "find_package" and args and "llama" in args[0].lower():
                error(path, number, "llama-library: forbidden package")
            if command == "target_link_libraries":
                if any(re.search(r"(?:^|::|/)(?:lib)?llama(?:$|[.:-])", arg, re.I) for arg in args[1:]):
                    error(path, number, "llama-library: forbidden link")
            if command == "add_subdirectory" and args:
                dest = (file.parent / args[0]).resolve()
                if not dest.is_relative_to(root):
                    error(path, number, "dependency-source: external subdirectory")
                elif dest.is_relative_to(root / "third_party") and dest != root / "third_party/ggml":
                    error(path, number, "dependency-source: third-party subdirectory forbidden")
    print("\n".join(errors) if errors else "deps: passed")
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(check(Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()))
