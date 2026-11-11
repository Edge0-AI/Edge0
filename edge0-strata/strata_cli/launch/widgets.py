"""The look of the launch screens: the monitor's colours, rules and key hints, and the one list widget they share.

Colours are the monitor's own (strata_cli/monitor.py: background #0e1113, panel #161a1d, text #eef1f3, muted #8e979f, accent
#34d399, warning yellow); the launch mock-up adds a selection tint (#1f2a27), a warning amber and a soft red. Everything is plain
text in a column of lines, as the monitor's text views are: a heading in the accent colour, a dim rule, the content, a dim
explanation line and a dim line of key hints.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

from rich.text import Text
from textual.widgets import Static

BG, PANEL, FG, MUTE = "#0e1113", "#161a1d", "#eef1f3", "#8e979f"
ACCENT, WARN, BAD, SEL, RULE = "#34d399", "#f5c56b", "#f28b82", "#1f2a27", "#3a4146"


@dataclass(frozen=True)
class Palette:
    """The colours of one theme. The launch screens are dark; the monitor's Connect view and its pop-ups follow the monitor's light theme."""

    bg: str
    panel: str
    fg: str
    mute: str
    accent: str
    warn: str
    bad: str
    sel: str
    rule: str


DARK = Palette(BG, PANEL, FG, MUTE, ACCENT, WARN, BAD, SEL, RULE)
# Light: every text colour reads at 4.5:1 or better on the white panel (tests/terminal/test_tuiqa_monitor.py computes it).
LIGHT = Palette("#f9fafb", "#ffffff", "#221e1f", "#5b5f66", "#0a6b4a", "#8a5a00", "#b01a2e", "#cfeadf", "#c9ced3")


def unicode_ok() -> bool:
    if os.environ.get("TERM") == "dumb":
        return False
    from strata_cli import consoleinfo
    if consoleinfo.legacy_console():          # the classic Windows console's font has no arrows, check marks or block glyphs ...
        if os.environ.get("STRATA_GLYPHS", "").strip().lower() in ("ascii", "legacy"):
            return False
        if consoleinfo.classic_style()["mode"] != "full":         # ... unless the window's font could be switched to Cascadia Mono
            return False
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        "›─↑↓•█░".encode(enc)
        return True
    except (LookupError, UnicodeEncodeError):
        return False


class Glyphs:
    def __init__(self, unicode: bool | None = None):
        u = unicode_ok() if unicode is None else unicode
        self.unicode = u
        self.cursor = "›" if u else ">"      # (U+203A is in Consolas too, but one rule for the whole screen is simpler)
        self.rule = "─" if u else "-"
        self.updown = "↑↓" if u else "Up/Down"
        self.dot = "·" if u else "-"
        self.full = "█" if u else "#"
        self.empty = "░" if u else "."
        self.tick = "✓" if u else "ok"
        self.mask = "•" if u else "*"
        self.arrow = "→" if u else "->"


CSS = f"""
Screen {{ background: {BG}; color: {FG}; }}
#shell {{ width: 100%; height: 100%; background: {PANEL}; padding: 0 1; }}
#head {{ height: 1; }}
#rule1, #rule2 {{ height: 1; color: {RULE}; }}
#body {{ height: 1fr; overflow-y: auto; }}
#explain {{ height: auto; max-height: 4; color: {MUTE}; }}
#hints {{ height: auto; max-height: 3; color: {MUTE}; }}
#message {{ height: auto; max-height: 3; color: {MUTE}; }}
Menu {{ height: auto; max-height: 100%; }}
.field {{ height: 3; border: solid {ACCENT}; display: none; }}
.field.slim {{ height: 1; border: none; padding: 0 1; background: {SEL}; }}
#cmdbox {{ height: 6; border-top: solid {RULE}; scrollbar-size-vertical: 1; }}
.field.open {{ display: block; }}
Input {{ background: {PANEL}; color: {FG}; }}
"""


@dataclass
class Cell:
    text: str
    style: str = ""


@dataclass
class Item:
    cells: list[Cell | str]
    key: Any = None
    enabled: bool = True
    style: str = ""          # whole-row style when not selected


def cell(c: Cell | str) -> Cell:
    return c if isinstance(c, Cell) else Cell(str(c))


class Menu(Static):
    """A column list with a cursor: `items` is a list of `Item`, `columns` a list of (title, width) (width 0 = the rest)."""

    DEFAULT_CSS = "Menu { height: auto; text-wrap: nowrap; text-overflow: clip; }"

    def __init__(self, columns: list[tuple[str, int]] | None = None, *, glyphs: Glyphs | None = None, id: str | None = None,
                 window: int = 0, palette: Palette | None = None) -> None:
        super().__init__("", id=id, markup=False)
        self.palette = palette or DARK
        self.columns = columns or []
        self.items: list[Item] = []
        self.index = 0
        self.top = 0
        self.window = window            # 0 = show all rows
        self.g = glyphs or Glyphs()
        self.show_header = bool(self.columns and any(t for t, _ in self.columns))

    def set_items(self, items: list[Item], index: int | None = None) -> None:
        self.items = items
        if index is not None:
            self.index = index
        self.index = max(0, min(self.index, len(items) - 1)) if items else 0
        self._refresh()

    @property
    def current(self) -> Item | None:
        return self.items[self.index] if self.items else None

    def move(self, step: int) -> None:
        if not self.items:
            return
        n = len(self.items)
        i = self.index
        for _ in range(n):
            i = (i + step) % n
            if self.items[i].enabled or not any(it.enabled for it in self.items):
                break
        self.index = i
        self._refresh()

    def select(self, key: Any) -> None:
        for i, it in enumerate(self.items):
            if it.key == key:
                self.index = i
        self._refresh()

    def page(self, step: int) -> None:
        self.index = max(0, min(len(self.items) - 1, self.index + step * max(1, (self.window or 8) - 1)))
        self._refresh()

    def _refresh(self) -> None:
        win = self.window or len(self.items)
        if self.index < self.top:
            self.top = self.index
        if self.index >= self.top + win:
            self.top = self.index - win + 1
        self.top = max(0, min(self.top, max(0, len(self.items) - win)))
        self.update(self.build())

    def build(self) -> Text:
        out = Text(no_wrap=True, overflow="crop")
        pad = "  "
        widths = [w for _t, w in self.columns]
        if self.show_header:
            out.append(pad)
            for (title, w) in self.columns:
                out.append(title.ljust(w) if w else title, style="bold")
            out.append("\n")
        win = self.window or len(self.items)
        rows = self.items[self.top:self.top + win]
        for n, it in enumerate(rows):
            selected = (self.top + n) == self.index
            line = Text(no_wrap=True, overflow="crop")
            line.append((self.g.cursor + " ") if selected else pad, style=f"bold {self.palette.accent}" if selected else "")
            for k, c in enumerate(it.cells):
                c = cell(c)
                w = widths[k] if k < len(widths) else 0
                txt = c.text.ljust(w) if w else c.text
                style = c.style if not selected else (c.style if c.style and not selected else "")
                if selected:
                    style = f"bold {self.palette.accent}" if c.style in ("", None) else c.style
                elif not c.style and it.style:
                    style = it.style
                if not it.enabled and not selected:
                    style = self.palette.mute
                line.append(txt, style=style)
            if selected:
                line.stylize(f"on {self.palette.sel}")
                line.pad_right(max(0, (self.size.width or 100) - len(line.plain)))
                line.stylize(f"on {self.palette.sel}")
            out.append_text(line)
            if n < len(rows) - 1:
                out.append("\n")
        if self.window and len(self.items) > win:
            out.append(f"\n  {self.top + 1}-{min(len(self.items), self.top + win)} of {len(self.items)}", style=self.palette.mute)
        return out

    def on_resize(self, _event) -> None:
        self._refresh()


def heading(text: str) -> Text:
    return Text(text, style=f"bold {ACCENT}")


def muted(text: str) -> Text:
    return Text(text, style=MUTE)


def warn(text: str) -> Text:
    return Text(text, style=WARN)


def bad(text: str) -> Text:
    return Text(text, style=BAD)


def bar(fraction: float, width: int, g: Glyphs) -> str:
    fraction = max(0.0, min(1.0, fraction))
    full = int(round(fraction * width))
    return g.full * full + g.empty * (width - full)
