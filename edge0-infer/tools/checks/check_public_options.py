"""Keep the reference CLI flags and the public configuration reference in step.

Every flag the CLI parses must be documented, and every flag the documents
name in code spans must still exist. Defaults and value ranges are not checked.
"""
from pathlib import Path
import re
import sys

CLI = "tools/cli"
DOCS = ("docs/configuration.md", "docs/local-server.md")
# The CLI compares every argument against a string literal such as "--threads".
CLI_FLAG = re.compile(r'"(--[a-z][a-z0-9-]*)')
CODE_SPAN = re.compile(r"`([^`\n]+)`")
DOC_FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")


def cli_flags(root):
    flags = {}
    for file in sorted((root / CLI).glob("*")):
        if file.suffix not in {".cpp", ".h"}:
            continue
        for line, text in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
            for match in CLI_FLAG.finditer(text):
                flags.setdefault(match[1], f"{CLI}/{file.name}:{line}")
    return flags


def doc_flags(root, path):
    flags = {}
    for line, text in enumerate((root / path).read_text(encoding="utf-8").splitlines(), 1):
        for span in CODE_SPAN.finditer(text):
            for match in DOC_FLAG.finditer(span[1]):
                flags.setdefault(match[1], f"{path}:{line}")
    return flags


def check(root):
    errors = []
    missing = [path for path in (CLI, *DOCS) if not (root / path).exists()]
    if missing:
        errors.extend(f"public-options: {path}: missing" for path in missing)
    else:
        code = cli_flags(root)
        documented = {}
        for path in DOCS:
            for flag, where in doc_flags(root, path).items():
                documented.setdefault(flag, where)
        errors.extend(f"public-options: {where}: {flag} is not documented in {DOCS[0]}"
                      for flag, where in sorted(code.items()) if flag not in documented)
        errors.extend(f"public-options: {where}: {flag} is not a CLI flag"
                      for flag, where in sorted(documented.items()) if flag not in code)
    print("\n".join(errors) if errors else "public-options: passed")
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(check(Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()))
