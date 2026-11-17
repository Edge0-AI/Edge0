"""Which files of serve/web the server hands out below /web/: the page's translations and the vendored KaTeX.

The flat files (app.js, app.css, ...) are served by serve/server.py itself. The two folders here hold more than one
level, so each has its own whitelist: a name pattern and the types it may have. Nothing outside these folders is ever
read, a name with a backslash, a colon or a percent sign is refused, and the resolved path must stay inside the folder.
"""
from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"

# /web/locales/<lang>.json   e.g. de.json, pt-BR.json
LOCALE = re.compile(r"^/web/locales/(?P<name>[a-z]{2,3}(?:-[A-Z]{2})?\.json)$")
# /web/vendor/katex/dist/... : the script, the style sheet and the fonts only
VENDOR_TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".woff2": "font/woff2"}
VENDOR_PREFIX = "/web/vendor/katex/"


def subtree_file(path: str, web: Path = WEB):
    """(file, content type) for a /web/locales or /web/vendor path, or None when it is not served."""
    if any(c in path for c in ("\\", ":", "%", "\0")) or ".." in path:
        return None
    m = LOCALE.match(path)
    if m:
        f = web / "locales" / m.group("name")
        return (f, "application/json; charset=utf-8") if f.is_file() else None
    if path.startswith(VENDOR_PREFIX):
        vendor = (web / "vendor" / "katex").resolve()
        f = (vendor / path[len(VENDOR_PREFIX):]).resolve()
        ctype = VENDOR_TYPES.get(f.suffix)
        if ctype and f.is_relative_to(vendor) and f.is_file():
            return f, ctype
    return None
