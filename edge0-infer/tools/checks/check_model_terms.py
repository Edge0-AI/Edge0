"""Reject concrete model terms in framework C/C++ sources."""
from pathlib import Path
import importlib.util
import re
import sys

_spec = importlib.util.spec_from_file_location("checks_common", Path(__file__).with_name("common.py"))
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)

TERMS = re.compile(r"edge0[_-]35b|edge0[_-]8b|\b35b\b|\b8b\b|qwen3next|deltanet|prerouter|\blora\b", re.I)
EXCEPTIONS = {"src/app/adapters.cpp"}


def check(root):
    errors = []
    for file, path in _common.files(root):
        if file.suffix not in _common.SOURCE_SUFFIXES or path in EXCEPTIONS:
            continue
        if not path.startswith(("src/", "include/edge0/", "tools/cli/")):
            continue
        if path.startswith("src/models/") and not path.startswith("src/models/api/"):
            continue
        for line, text in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
            for match in TERMS.finditer(text):
                errors.append(f"model-terms: {path}:{line}: {match[0]}")
    print("\n".join(errors) if errors else "model-terms: passed")
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(check(Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()))
