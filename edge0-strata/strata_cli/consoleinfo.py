"""What kind of terminal this is, for the few places that must change what they draw.

The classic Windows console window (conhost: the default window of Windows PowerShell and cmd on Windows 10) uses Consolas or Lucida Console,
which have no glyphs for the eighth-block sparkline characters or the check mark: they show as empty boxes (Consolas has the full, half and
shade blocks, the box-drawing set and the rounded corners; Lucida Console has no rounded corners and no filled circle).
Windows Terminal, VS Code's terminal, ConEmu and every Linux/macOS terminal with a modern font have them. `STRATA_GLYPHS=ascii|unicode`
overrides the guess (`solid` draws the classic console's solid look without touching the window's font).

In the classic console only, `classic_style()` does the best it can with no admin rights and no registry change:
* Cascadia Mono installed: this window's font is switched to it (the console font API) and the full Unicode look is drawn;
* otherwise the "solid" look: solid block bars with a thin track, ▄ / █ sparklines, square boxes, the 16 palette colours conhost really has,
  and `•` instead of `●` when the font is not Consolas (Lucida has no `●`).
Windows Terminal, Linux and macOS never get either: `legacy_console()` is False there.
"""

from __future__ import annotations

import os
import sys

TIP = "For the full look, use Windows Terminal (built into Windows 11)"


def legacy_console() -> bool:
    """True for the classic Windows console (no Windows Terminal, no VS Code, no ConEmu, no ssh session into it)."""
    forced = os.environ.get("STRATA_GLYPHS", "").strip().lower()
    if forced in ("ascii", "legacy", "solid"):
        return True
    if forced == "unicode":
        return False
    if os.name != "nt" and sys.platform != "win32":
        return False
    env = os.environ
    return not (env.get("WT_SESSION") or env.get("TERM_PROGRAM") or env.get("ConEmuANSI") or env.get("ANSICON")
                or env.get("TERM", "").startswith(("xterm", "screen", "tmux", "vt")))


def console_columns() -> int | None:
    """The visible width of the Windows console window in columns (not its buffer), or None when it cannot be read."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class COORD(ctypes.Structure):
            _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

        class RECT(ctypes.Structure):
            _fields_ = [("Left", ctypes.c_short), ("Top", ctypes.c_short), ("Right", ctypes.c_short), ("Bottom", ctypes.c_short)]

        class INFO(ctypes.Structure):
            _fields_ = [("size", COORD), ("cursor", COORD), ("attr", wintypes.WORD), ("window", RECT), ("maxsize", COORD)]

        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        info = INFO()
        if not ctypes.windll.kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(info)):
            return None
        return int(info.window.Right - info.window.Left + 1)
    except Exception:                                        # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------------ the font of the classic console

def _font_struct():
    import ctypes

    class COORD(ctypes.Structure):
        _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

    class CONSOLE_FONT_INFOEX(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_ulong), ("nFont", ctypes.c_ulong), ("dwFontSize", COORD), ("FontFamily", ctypes.c_uint),
                    ("FontWeight", ctypes.c_uint), ("FaceName", ctypes.c_wchar * 32)]
    return CONSOLE_FONT_INFOEX


def console_font() -> str | None:
    """The face name of this console window's font (Windows only), or None."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        S = _font_struct()
        info = S()
        info.cbSize = ctypes.sizeof(S)
        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        if not ctypes.windll.kernel32.GetCurrentConsoleFontEx(handle, False, ctypes.byref(info)):
            return None
        return info.FaceName or None
    except Exception:                                        # noqa: BLE001
        return None


def cascadia_installed() -> bool:
    """Is Cascadia Mono installed (it comes with Windows Terminal and Visual Studio, and can be installed alone)?"""
    if os.name != "nt":
        return False
    try:
        import winreg
        for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(root, "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Fonts") as key:
                    i = 0
                    while True:
                        name = winreg.EnumValue(key, i)[0]
                        if name.lower().startswith("cascadia mono"):
                            return True
                        i += 1
            except OSError:
                continue
    except Exception:                                        # noqa: BLE001
        return False
    fonts = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    return any(os.path.isfile(os.path.join(fonts, n)) for n in ("CascadiaMono.ttf", "CascadiaMonoPL.ttf"))


def switch_font(face: str = "Cascadia Mono") -> bool:
    """Set THIS console window's font through the console font API (no admin, nothing saved, gone when the window closes); True if it took."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        S = _font_struct()
        info = S()
        info.cbSize = ctypes.sizeof(S)
        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        if not ctypes.windll.kernel32.GetCurrentConsoleFontEx(handle, False, ctypes.byref(info)):
            return False
        height = info.dwFontSize.Y or 16
        info.FaceName = face
        info.dwFontSize.X = 0
        info.dwFontSize.Y = height
        info.FontFamily = 54                                 # TMPF_TRUETYPE | fixed pitch | modern
        info.FontWeight = 400
        if not ctypes.windll.kernel32.SetCurrentConsoleFontEx(handle, False, ctypes.byref(info)):
            return False
        return console_font() == face
    except Exception:                                        # noqa: BLE001
        return False


def decide_style(*, legacy: bool, cascadia_ok: bool, font: str | None) -> dict:
    """The look for this console (pure: the tests give it the facts).

    {"mode": "none"}                                      not the classic console: nothing here changes
    {"mode": "full", "font": "Cascadia Mono", ...}        the font was switched to Cascadia Mono: the full Unicode look
    {"mode": "solid", "font": face, "bullet": "●"|"•"}    solid blocks and palette colours; the bullet depends on the font
    `tip` is the one line every classic console shows."""
    if not legacy:
        return {"mode": "none", "bullet": "●", "tip": None, "font": font}
    if cascadia_ok:
        return {"mode": "full", "bullet": "●", "tip": TIP, "font": "Cascadia Mono"}
    bullet = "●" if (font or "").lower().startswith("consolas") else "•"       # Lucida Console (and unknown fonts) have no ●
    return {"mode": "solid", "bullet": bullet, "tip": TIP, "font": font}


_cached: dict | None = None


def classic_style() -> dict:
    """decide_style for this process, switching the font to Cascadia Mono once when that is possible. Cheap after the first call."""
    global _cached
    if _cached is not None:
        return _cached
    legacy = legacy_console()
    forced = os.environ.get("STRATA_GLYPHS", "").strip().lower() in ("ascii", "legacy", "solid")
    switched = False
    font = console_font() if (legacy and forced) else None
    if legacy and not forced and os.name == "nt":
        font = console_font()
        if font and font.lower().startswith("cascadia mono"):
            switched = True
        elif cascadia_installed():
            switched = switch_font("Cascadia Mono")
            font = console_font() if switched else font
    _cached = decide_style(legacy=legacy, cascadia_ok=switched, font=font)
    return _cached


def forget_style() -> None:
    """For the tests: decide again."""
    global _cached
    _cached = None
