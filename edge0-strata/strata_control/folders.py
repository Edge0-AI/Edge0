"""Is this a folder Strata can keep model files in? Said in plain words before anything is downloaded (the Folder step, --data-dir, setup).

`problem(path)` returns (what is wrong, what to do) or None. Nothing is created and nothing is left behind: a missing folder is judged by the
nearest folder above it that exists (can a file be made there, so the folder could be created), and "can be written" is tried by creating and
deleting a small file, never read from the permission bits.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

NEXT = "Pick another folder, for example one on a drive you own, like D:\\Strata-models. Nothing was downloaded or changed."
NEXT_POSIX = "Pick another folder you own, for example ~/strata-models. Nothing was downloaded or changed."

_WIN_BAD = re.compile(r'[<>"|?*\x00-\x1f]')
_POSIX_SYSTEM = ("/bin", "/boot", "/dev", "/etc", "/lib", "/lib32", "/lib64", "/proc", "/root", "/run", "/sbin", "/sys", "/usr", "/var/lib", "/System",
                 "/Library", "/private/etc")


def _next() -> str:
    return NEXT if os.name == "nt" else NEXT_POSIX


def _system_roots() -> list[str]:
    if os.name != "nt":
        return list(_POSIX_SYSTEM)
    roots = []
    for var in ("SystemRoot", "windir", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
        value = os.environ.get(var)
        if value:
            roots.append(value)
    drive = os.environ.get("SystemDrive", "C:")
    roots += [drive + "\\Windows", drive + "\\Program Files", drive + "\\Program Files (x86)"]
    return roots


def _norm(path: str) -> str:
    return os.path.normcase(os.path.normpath(path)).rstrip("\\/")


def _inside(path: str, root: str) -> bool:
    p, r = _norm(path), _norm(root)
    return p == r or p.startswith(r + os.sep)


def _can_write(folder: Path) -> bool:
    probe = folder / f".strata-write-test-{uuid.uuid4().hex[:8]}"
    try:
        with open(probe, "wb") as fh:
            fh.write(b"x")
    except OSError:
        return False
    try:
        probe.unlink()
    except OSError:
        pass
    return True


def problem(raw: str | os.PathLike | None) -> tuple[str, str] | None:
    """(what is wrong, what to do) for a folder meant for model files, None when it is usable (or empty: the default folder is used)."""
    text = os.fspath(raw).strip().strip('"') if raw else ""
    if not text:
        return None
    if _WIN_BAD.search(text.replace("\\\\?\\", "", 1)) or (os.name == "nt" and ":" in text[2:] and not text.startswith("\\\\?\\")):
        return (f"This does not look like one folder path: {text}.",
                "Type one full path, like D:\\Strata-models, with nothing else in front of or behind it. Nothing was downloaded or changed."
                if os.name == "nt" else "Type one full path, like ~/strata-models. Nothing was downloaded or changed.")
    path = Path(text).expanduser()
    if not path.is_absolute():
        return (f"This is not a full path: {text}.",
                "Type the whole path, starting with the drive, like D:\\Strata-models. Nothing was downloaded or changed."
                if os.name == "nt" else "Type the whole path, starting with /, like /data/strata-models. Nothing was downloaded or changed.")
    full = os.path.abspath(path)
    for root in _system_roots():
        if _inside(full, root) and not _inside(full, os.path.expanduser("~")):         # the running user's own home is theirs (root's /root included)
            return (f"{full} is a system folder, and Strata should not keep model files there.", _next())
    try:
        if path.exists():
            if not path.is_dir():
                return (f"{full} is a file, not a folder.", _next())
            if not _can_write(path):
                return (f"Strata cannot write to {full}: Windows or the folder's owner does not allow it." if os.name == "nt"
                        else f"Strata cannot write to {full}: you do not have permission.", _next())
            return None
        above = path
        while not above.exists():
            if above.parent == above:
                return (f"The drive or folder for {full} is not there. Check the spelling, and that the drive is connected.", _next())
            above = above.parent
        if not above.is_dir():
            return (f"{above} is a file, so {full} cannot be created inside it.", _next())
        if not _can_write(above):
            return (f"{full} does not exist yet and cannot be created: Strata cannot write to {above}.", _next())
    except OSError as e:
        return (f"Strata cannot look at {full} ({type(e).__name__}).", _next())
    return None


def words(problem_: tuple[str, str]) -> str:
    return f"{problem_[0]} {problem_[1]}".strip()
