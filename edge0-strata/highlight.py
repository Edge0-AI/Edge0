"""Syntax highlighting for chat.py's streamed output: fenced code blocks get colored (tags, strings, comments,
keywords, numbers), everything else passes through unchanged. Standard library only, no dependency.

    from highlight import StreamHighlighter
    hl = StreamHighlighter(color=True)
    out = hl.feed(streamed_chunk)   # complete lines, highlighted; a partial last line is held back
    out += hl.flush()               # the held-back tail (unclosed fence included)
"""
from __future__ import annotations

import re


# ANSI colors; a disabled colorer returns the text unchanged
class _Palette:
    def __init__(self, color: bool):
        self.on = color

    def paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.on else text


# one regex per language family; the first matching family paints the line
_PY_KEYWORDS = ("False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
                "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in",
                "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield")
_JS_KEYWORDS = ("async", "await", "break", "case", "catch", "class", "const", "continue", "default", "delete",
                "do", "else", "export", "extends", "false", "finally", "for", "function", "if", "import", "in",
                "instanceof", "let", "new", "null", "return", "static", "super", "switch", "this", "throw", "true",
                "try", "typeof", "undefined", "var", "void", "while", "yield")
_C_KEYWORDS = ("alignas", "auto", "bool", "break", "case", "catch", "char", "class", "const", "constexpr",
               "continue", "default", "delete", "do", "double", "else", "enum", "explicit", "extern", "false",
               "float", "for", "friend", "if", "inline", "int", "long", "namespace", "new", "noexcept", "nullptr",
               "private", "protected", "public", "return", "short", "signed", "sizeof", "static", "struct",
               "switch", "template", "this", "throw", "true", "try", "typedef", "typename", "union", "unsigned",
               "using", "virtual", "void", "while")
_SHELL_KEYWORDS = ("case", "do", "done", "elif", "else", "esac", "fi", "for", "function", "if", "in", "then",
                   "until", "while")
_JSON_KEYS = ("false", "null", "true")

_STR = r"(\"(?:[^\"\\]|\\.)*\")"        # "..." with escapes
_SQ_STR = r"('(?:[^'\\]|\\.)*')"
_COMMENT = r"(#.*$)"
_SLASH_COMMENT = r"(//.*$)"
_NUM = r"\b(\d[\w.]*)\b"

_FAMILIES = {
    "markup": ({"html", "htm", "xml", "svg", "vue", "jsx", "tsx"},
               re.compile(r"(<!--.*?-->)|(<[A-Za-z][\w:.-]*/?>?)|(</[A-Za-z][\w:.-]*>)|([A-Za-z-]+)(?==)"),
               {1: "36;1", 2: "96", 3: "96", 4: "94"}),
    "python": ({"py", "python"},
               re.compile(rf"{_COMMENT}|{_SQ_STR}|{_STR}|\b({'|'.join(_PY_KEYWORDS)})\b|{_NUM}"),
               {1: "90", 2: "33", 3: "33", 4: "35;1", 5: "34"}),
    "c-like": ({"c", "cc", "cpp", "c++", "h", "hpp", "cu", "hip", "java", "kt", "rs", "go", "cs"},
               re.compile(rf"{_SLASH_COMMENT}|{_STR}|\b({'|'.join(_C_KEYWORDS)})\b|\b(0x[0-9a-fA-F]+|\d[\w.]*)\b"),
               {1: "90", 2: "33", 3: "35;1", 4: "34"}),
    "js-like": ({"js", "javascript", "ts", "typescript", "mjs"},
                re.compile(rf"{_SLASH_COMMENT}|{_STR}|{_SQ_STR}|(`[^`]*`)|\b({'|'.join(_JS_KEYWORDS)})\b|{_NUM}"),
                {1: "90", 2: "33", 3: "33", 4: "33", 5: "35;1", 6: "34"}),
    "shell": ({"sh", "bash", "zsh", "shell", "console", "bat", "cmd", "powershell", "ps1"},
              re.compile(rf"{_COMMENT}|{_SQ_STR}|{_STR}|\b({'|'.join(_SHELL_KEYWORDS)})\b|(^\s*\$)"),
              {1: "90", 2: "33", 3: "33", 4: "35;1", 5: "90"}),
    "json": ({"json", "jsonc", "json5"},
             re.compile(rf"{_SLASH_COMMENT}|{_STR}(?=\s*:)|\b({'|'.join(_JSON_KEYS)})\b|\b(-?\d[\w.]*)\b"),
             {1: "90", 2: "94", 3: "35;1", 4: "34"}),
    "css": ({"css", "scss", "less"},
            re.compile(r"(/\*.*?\*/)|([.#][\w-]+)|([\w-]+)(?=\s*:)|\b(\d[\w.%]*)\b"),
            {1: "90", 2: "96", 3: "94", 4: "34"}),
    "sql": ({"sql"},
            re.compile(r"(--.*$)|('[^']*')|\b(select|from|where|insert|into|update|set|delete|join|left|right|inner|"
                       r"outer|on|group|by|order|having|limit|create|table|alter|drop|and|or|not|null|as)\b",
                       re.IGNORECASE),
            {1: "90", 2: "33", 3: "35;1"}),
    "diff": ({"diff", "patch"},
             re.compile(r"^(\+[^+]*)|^(-.*)|^(@@.*@@)"),
             {1: "32", 2: "31", 3: "36"}),
}

_FENCE_RE = re.compile(r"^\s*(```|~~~)\s*([\w+#-]*)")
_LANG_ALIAS = {"c++": "c-like", "javascript": "js-like", "typescript": "js-like", "python": "python",
               "py": "python", "shell": "shell", "bash": "shell", "console": "shell"}


def _family(lang: str):
    lang = _LANG_ALIAS.get(lang.lower(), lang.lower())
    for exts, rx, colors in _FAMILIES.values():
        if lang in exts:
            return rx, colors
    return None, None


class StreamHighlighter:
    """Line-buffered highlighter for streamed text. feed() returns complete lines (highlighted inside fenced code
    blocks); flush() returns the held-back tail. Outside code fences the text is returned unchanged (color on or
    off), so prose never gets painted. All characters other than the added ANSI escapes pass through unchanged."""

    def __init__(self, color: bool = True):
        self.color = color
        self._pal = _Palette(color)
        self._buf = ""          # held-back partial line
        self._in_fence = False
        self._rx = None
        self._colors = None

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out = []
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            out.append(self._line(line))
            out.append("\n")
        return "".join(out)

    def flush(self) -> str:
        tail, self._buf = self._buf, ""
        return self._line(tail) if tail else ""

    def _line(self, line: str) -> str:
        fence = _FENCE_RE.match(line)
        if fence:
            marker, lang = fence.group(1), fence.group(2)
            if self._in_fence:
                self._in_fence, self._rx, self._colors = False, None, None
                return self._pal.paint(line, "90")
            self._in_fence = True
            self._rx, self._colors = _family(lang)
            if not self._rx:
                return self._pal.paint(line, "90")
            return self._pal.paint(marker, "90") + self._pal.paint(lang, "94") + line[fence.end():]
        if not self._in_fence or not self._rx:
            return line
        return self._rx.sub(self._sub, line)

    def _sub(self, m: re.Match) -> str:
        for gi, code in self._colors.items():
            if m.group(gi) is not None:
                return self._pal.paint(m.group(gi), code)
        return m.group(0)
