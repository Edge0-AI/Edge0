"""The web page's translations: read the English text out of serve/web, write locales/en.json, check the catalogs.

    python tools/web_i18n.py            # check every catalog against the text in the code (exit 1 on a problem)
    python tools/web_i18n.py --write    # also rewrite serve/web/locales/en.json from the code

The English text lives in the code (see serve/web/i18n.js):
    tr("key", "English text", vars)                          in serve/web/*.js
    tr("key", {one: "{n} chat", other: "{n} chats"}, vars)   a plural
    <span data-i18n="key">English text</span>                in serve/web/index.html (also data-i18n-title, -label, -placeholder)
Both must be plain string literals (no template strings, no joined strings), so that this reader and a translator see
the same text. serve/test_web_i18n.py runs the check.
"""
from __future__ import annotations

import html.parser
import json
import re
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "serve" / "web"
LOCALES = WEB / "locales"
PLACEHOLDER = re.compile(r"\{(\w+)\}")
CALL = re.compile(r"(?<![\w.$])tr\(\s*\"([\w.\-]+)\"\s*,\s*")
PLURAL_CATEGORIES = ("zero", "one", "two", "few", "many", "other")


class Problem(Exception):
    pass


def _string(src: str, i: int):
    """A double-quoted JS string literal at src[i]: (the text, the index after it)."""
    if src[i] != '"':
        raise Problem(f"expected a string literal at: {src[i:i + 40]!r}")
    j = i + 1
    while j < len(src):
        if src[j] == "\\":
            j += 2
            continue
        if src[j] == '"':
            try:
                return json.loads(src[i:j + 1]), j + 1
            except ValueError as e:
                raise Problem(f"not a plain string literal ({e}): {src[i:j + 1]!r}")
        j += 1
    raise Problem("unterminated string literal")


def _plural(src: str, i: int):
    """{one: "...", other: "..."} at src[i]."""
    j = i + 1
    out = {}
    while True:
        m = re.compile(r"\s*(\w+)\s*:\s*").match(src, j)
        if not m:
            raise Problem(f"expected a plural form at: {src[j:j + 40]!r}")
        text, j = _string(src, m.end())
        out[m.group(1)] = text
        m = re.compile(r"\s*([,}])").match(src, j)
        if not m:
            raise Problem(f"expected , or }} after a plural form at: {src[j:j + 40]!r}")
        j = m.end()
        if m.group(1) == "}":
            break
    if "other" not in out:
        raise Problem(f"a plural needs an 'other' form: {out}")
    return out, j


def js_strings(src: str, name: str = "?") -> dict:
    found = {}
    for m in CALL.finditer(src):
        key = m.group(1)
        i = m.end()
        try:
            value, _ = _plural(src, i) if src[i] == "{" else _string(src, i)
        except Problem as e:
            line = src.count("\n", 0, m.start()) + 1
            raise Problem(f"{name}:{line}: tr({key!r}): {e}")
        found[key] = value
    return found


class _Html(html.parser.HTMLParser):
    ATTRS = {"data-i18n-title": "title", "data-i18n-label": "aria-label", "data-i18n-placeholder": "placeholder"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.found: dict = {}
        self.open = []          # [(tag, key, [text])]

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        for mark, attr in self.ATTRS.items():
            if mark in a:
                self.found[a[mark]] = a.get(attr) or ""
        if "data-i18n" in a:
            self.open.append((tag, a["data-i18n"], []))
        elif self.open:
            self.open.append((tag, None, self.open[-1][2]))

    def handle_data(self, data):
        if self.open:
            self.open[-1][2].append(data)

    def handle_endtag(self, tag):
        if self.open and self.open[-1][0] == tag:
            _, key, text = self.open.pop()
            if key:
                self.found[key] = " ".join("".join(text).split())


def html_strings(src: str) -> dict:
    p = _Html()
    p.feed(src)
    return p.found


def english() -> dict:
    """Every key in serve/web with its English text; the same key twice must have the same text."""
    out: dict = {}
    sources = {p.name: js_strings(p.read_text(encoding="utf-8"), p.name) for p in sorted(WEB.glob("*.js")) if p.name != "i18n.js"}
    sources["index.html"] = html_strings((WEB / "index.html").read_text(encoding="utf-8"))
    for name, found in sources.items():
        for key, value in found.items():
            if key in out and out[key] != value:
                raise Problem(f"{key!r} has two different English texts (one is in {name})")
            out[key] = value
    return dict(sorted(out.items()))


def languages() -> list:
    src = (WEB / "i18n.js").read_text(encoding="utf-8")
    block = re.search(r"const LANGUAGES = (\[.*?\]);", src, re.S).group(1)
    return [m for m in re.findall(r'\["([\w-]+)",', block)]


def placeholders(value) -> set:
    forms = value.values() if isinstance(value, dict) else [value]
    return {name for form in forms for name in PLACEHOLDER.findall(form)}


def check_catalog(code: str, catalog: dict, en: dict) -> list:
    problems = []
    for key in sorted(set(en) - set(catalog)):
        problems.append(f"{code}: missing {key!r}")
    for key in sorted(set(catalog) - set(en)):
        problems.append(f"{code}: {key!r} is not in the page (renamed or removed?)")
    for key, value in catalog.items():
        if key not in en:
            continue
        forms = value if isinstance(value, dict) else {"": value}
        if isinstance(value, dict):
            bad = [c for c in value if c not in PLURAL_CATEGORIES]
            if bad or "other" not in value:
                problems.append(f"{code}: {key!r}: plural forms must be CLDR names with an 'other' form, got {sorted(value)}")
            if not isinstance(en[key], dict):
                problems.append(f"{code}: {key!r} is a plural here but not in the code")
        elif isinstance(en[key], dict):
            problems.append(f"{code}: {key!r} is a plain text here but a plural in the code")
        for form, text in forms.items():
            if not isinstance(text, str) or not text.strip():
                problems.append(f"{code}: {key!r} {form} is empty or not a string")
        if placeholders(value) != placeholders(en[key]) and not (
                isinstance(value, dict) and isinstance(en[key], dict)
                and all(placeholders(f) <= placeholders(en[key]) for f in value.values())):
            problems.append(f"{code}: {key!r} uses {sorted(placeholders(value))}, the code uses {sorted(placeholders(en[key]))}")
    return problems


def check(write: bool = False) -> list:
    en = english()
    if write:
        LOCALES.mkdir(exist_ok=True)
        (LOCALES / "en.json").write_text(json.dumps(en, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    problems = []
    on_disk = LOCALES / "en.json"
    if not on_disk.is_file() or json.loads(on_disk.read_text(encoding="utf-8")) != en:
        problems.append("en.json is not the text in the code (python tools/web_i18n.py --write)")
    for code in languages():
        if code == "en":
            continue
        f = LOCALES / f"{code}.json"
        if not f.is_file():
            problems.append(f"{code}: serve/web/locales/{code}.json is missing")
            continue
        problems += check_catalog(code, json.loads(f.read_text(encoding="utf-8")), en)
    for f in sorted(LOCALES.glob("*.json")):
        if f.stem not in languages():
            problems.append(f"{f.name}: not in LANGUAGES of serve/web/i18n.js")
    return problems


if __name__ == "__main__":
    try:
        problems = check(write="--write" in sys.argv)
    except Problem as e:
        print(f"web_i18n: {e}")
        sys.exit(1)
    for p in problems:
        print("web_i18n:", p)
    sys.exit(1 if problems else 0)
