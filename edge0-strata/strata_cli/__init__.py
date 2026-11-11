"""Terminal clients for the shared Strata server."""

import re
from pathlib import Path

SCHEMA_VERSION = 1

#: The version people are told: the Strata app's own (apps/desktop/src-tauri/Cargo.toml `version`; tests/terminal/test_tuiqa_version.py keeps the
#: two equal, except for a Terminal-only release such as 0.2.3.1 or 0.2.4, whose app stays 0.2.3). `version()` below is the files' build number, which the engine-compatibility checks compare, so it stays numeric.
PRODUCT_VERSION = "0.2.9"


def version() -> str:
    """The terminal's version: the VERSION file of an archive, else the source tree's CMake project version."""
    root = Path(__file__).resolve().parents[1]
    try:
        text = (root / "VERSION").read_text(encoding="utf-8").strip()
        if text:
            return text
    except OSError:
        pass
    try:
        m = re.search(r"^\s*project\([^)]*?VERSION\s+([0-9]+(?:\.[0-9]+)+)",
                      (root / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
        if m:
            return m.group(1)
    except OSError:
        pass
    return "unknown"
