"""The monitor's view tabs: ONE model for every view and every width.

There are six views (Overview, Serving, Requests, Engine, Hardware, Logs) and, when `strata launch` opened the monitor, a seventh, Connect.
Every screen draws the same tab line from here, so the selected tab is always marked (in square brackets, and in colour when there is
colour), the hints say 1-6 or 1-7 to match, and a narrow terminal keeps the tab names: shorter names first, then only the tabs around the
selected one with < and > for the ones out of sight.
"""

from __future__ import annotations

from dataclasses import dataclass

NAMES = ("Overview", "Serving", "Requests", "Engine", "Hardware", "Logs", "Connect")
SHORT = ("Over", "Serve", "Reqs", "Engine", "GPU", "Logs", "Conn")


def count(connect: bool) -> int:
    return 7 if connect else 6


def keys(connect: bool) -> str:
    """The number keys that switch views: "1-6", or "1-7" with the Connect view."""
    return "1-7" if connect else "1-6"


def title(index: int) -> str:
    return NAMES[max(0, min(index, len(NAMES) - 1))]


@dataclass
class Tabs:
    pieces: list[tuple[str, bool]]      # (text, selected)
    sep: str
    left: bool = False                  # tabs hidden on the left
    right: bool = False                 # tabs hidden on the right

    def width(self) -> int:
        return (2 if self.left else 0) + (2 if self.right else 0) + sum(len(t) for t, _ in self.pieces) + len(self.sep) * max(0, len(self.pieces) - 1)


def _items(labels: tuple[str, ...], current: int, n: int) -> list[str]:
    return [f"[{i + 1} {labels[i]}]" if i == current else f"{i + 1} {labels[i]}" for i in range(n)]


def layout(current: int, width: int, connect: bool = False) -> Tabs:
    """The tabs for a line `width` columns wide: all names, else short names, else only the tabs around the selected one."""
    n = count(connect)
    current = max(0, min(current, n - 1))
    for labels, sep in ((NAMES, "  "), (SHORT, " ")):
        tabs = Tabs([(t, i == current) for i, t in enumerate(_items(labels, current, n))], sep)
        if tabs.width() <= width:
            return tabs
    items = _items(SHORT, current, n)
    lo = hi = current
    while True:
        grew = False
        for nxt, is_hi in ((hi + 1, True), (lo - 1, False)):
            if not 0 <= nxt < n:
                continue
            new_lo, new_hi = (lo, nxt) if is_hi else (nxt, hi)
            trial = Tabs([(items[i], i == current) for i in range(new_lo, new_hi + 1)], " ", new_lo > 0, new_hi < n - 1)
            if trial.width() <= width:
                lo, hi, grew = new_lo, new_hi, True
        if not grew:
            break
    return Tabs([(items[i], i == current) for i in range(lo, hi + 1)], " ", lo > 0, hi < n - 1)


def plain(current: int, width: int, connect: bool = False) -> str:
    """The tab line as plain text, at most `width` columns."""
    tabs = layout(current, width, connect)
    line = ("< " if tabs.left else "") + tabs.sep.join(t for t, _ in tabs.pieces) + (" >" if tabs.right else "")
    return line[:max(1, width)]


def styled(current: int, width: int, connect: bool = False, *, accent: str = "", color: bool = True):
    """The same line as rich Text: the selected tab in bold and the accent colour (just bold, in brackets, without colour).
    (rich is imported here: the plain-text parts of this module also run in the thin stdlib-only monitor bundle.)"""
    from rich.text import Text
    tabs = layout(current, width, connect)
    out = Text(no_wrap=True, overflow="crop")
    if tabs.left:
        out.append("< ", style="dim")
    for i, (text, selected) in enumerate(tabs.pieces):
        if i:
            out.append(tabs.sep)
        out.append(text, style=(f"bold {accent}" if color and accent else "bold") if selected else "")
    if tabs.right:
        out.append(" >", style="dim")
    return out
