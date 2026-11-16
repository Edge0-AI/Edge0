"""Windows path length (MAX_PATH).

A Windows PC with "long paths" off (the default on a fresh install) refuses any ordinary file path of 260 characters or more.
The AMD engine packages hold paths up to 149 characters (hipBLASLt tensile libraries), so under a long user name the unpacked
engine can go past the limit. Two defences, neither of which needs a setting from the user:

* every file Strata itself touches inside a component folder goes through `long_path`, which turns the path into the
  extended-length form (``\\\\?\\C:\\...``) that Windows accepts at any length (up to 32,767 characters);
* when a folder is still too deep for the engine's *own* file loading (the GPU math libraries open their files with ordinary
  paths), `warning_for` gives one plain sentence with the fix; the install continues (recommend, never refuse).

Nothing here reads or writes the registry. `plain_long_paths_work` finds out by trying.
"""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

#: the longest ordinary path Windows accepts with long paths off (MAX_PATH is 260 including the terminating null)
PLAIN_LIMIT = 259
_PREFIX = "\\\\?\\"
_UNC_PREFIX = "\\\\?\\UNC\\"

#: longest path inside an engine package, by backend, as shipped in 0.2.1 (hip: rocm/bin/hipblaslt/library/gfx1200/TensileLibrary_...).
#: The warning before a download can only estimate; after unpacking the real manifest is measured.
LONGEST_IN_PACKAGE = {"hip": 150}
LONGEST_IN_PACKAGE_DEFAULT = 60


def long_path(path) -> Path:
    """`path` in the form that works at any length: absolute, normalised, with the ``\\\\?\\`` prefix on Windows. Elsewhere unchanged."""
    p = Path(path)
    if os.name != "nt":
        return p
    text = str(p)
    if text.startswith(_PREFIX):
        return p
    text = os.path.normpath(os.path.abspath(text))
    if text.startswith("\\\\"):
        return Path(_UNC_PREFIX + text[2:])
    return Path(_PREFIX + text)


def plain(path) -> str:
    """The path as a person would read it, without the extended-length prefix."""
    text = str(path)
    if text.startswith(_UNC_PREFIX):
        return "\\\\" + text[len(_UNC_PREFIX):]
    return text[len(_PREFIX):] if text.startswith(_PREFIX) else text


def same_tree(path: Path, root: Path) -> bool:
    """Whether the (already resolved) `path` is `root` or below it; the prefix on either side does not matter."""
    a, b = os.path.normcase(plain(path)), os.path.normcase(plain(root))
    return a == b or a.startswith(b.rstrip("\\/") + os.sep)


def too_long(path, *, extra: int = 0) -> bool:
    """Would an ordinary (not extended) path be refused? Only ever true on Windows."""
    if os.name != "nt":
        return False
    return len(os.path.abspath(plain(path))) + extra > PLAIN_LIMIT


def plain_long_paths_work(directory: Path) -> bool:
    """Whether this process can open a path longer than 259 characters without the extended prefix (so Windows long paths are on).
    Tried, not read from the registry: a few folders are made through the extended form and reached through the ordinary one."""
    if os.name != "nt":
        return True
    base = long_path(Path(directory) / f".lp-{uuid.uuid4().hex[:6]}")
    try:
        deep = base / ("a" * 90) / ("b" * 90) / ("c" * 90)
        os.makedirs(deep)
        try:
            return os.path.isdir(plain(deep)) and len(plain(deep)) > PLAIN_LIMIT
        except OSError:
            return False
    except OSError:
        return True                                    # could not even test: say nothing rather than warn wrongly
    finally:
        shutil.rmtree(base, ignore_errors=True)


def warning_for(folder: Path, longest_inside: int) -> str | None:
    """One plain sentence when the files of a package placed in `folder` would pass the ordinary path limit, else None."""
    if os.name != "nt" or not too_long(Path(folder) / ("x" * (longest_inside + 1))):
        return None
    reach = len(plain(folder)) + 1 + longest_inside
    return (f"Strata's folder is deep: some engine files will have paths of about {reach} characters, and Windows stops at 259 unless "
            f"long paths are turned on. The install carries on, but the engine's graphics libraries may fail to load those files. "
            f"To fix it, turn on Windows long paths (Group Policy: Enable Win32 long paths) or put Strata in a shorter folder "
            f"(for example set STRATA_COMPONENTS_DIR to C:\\Strata) and update again.")


def describe_oserror(exc: BaseException) -> str:
    """Plain words for a failed file operation: which file, and when the cause is a path that is too long, that."""
    name = getattr(exc, "filename", None) or getattr(exc, "filename2", None)
    kind = type(exc).__name__
    if name is not None:
        shown = plain(os.fsdecode(name))
        if too_long(shown):
            return (f"a file path is longer than Windows allows ({len(os.path.abspath(shown))} characters, the limit is {PLAIN_LIMIT}): "
                    f"{shown}. Windows long paths are off on this PC. Put Strata in a shorter folder (set STRATA_COMPONENTS_DIR "
                    f"to something short like C:\\Strata) or turn on Windows long paths, then try again ({kind})")
        if isinstance(exc, FileNotFoundError):
            return f"Strata could not find or create {shown} ({kind})"
        if isinstance(exc, PermissionError):
            return f"Windows or another program would not let Strata use {shown} ({kind})"
        reason = getattr(exc, "strerror", None)
        return f"Strata could not use {shown}" + (f": {reason}" if reason else "") + f" ({kind})"
    return f"a file operation failed ({kind})"


def file_words(exc: BaseException) -> str:
    """The one-line message the command line prints for a failed local file operation."""
    return "local file operation failed: " + describe_oserror(exc)
