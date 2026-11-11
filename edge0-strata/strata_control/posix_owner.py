"""POSIX 'owner-only' checks that repair what is ours and explain what is not.

Strata keeps its private state in folders (0700) and files (0600) of the user. The checks used to refuse as soon as a mode had drifted
(a `chmod -R`, an rsync, a restored backup) or the owner was not the current user (one `sudo strata ...` leaves root-owned folders behind).
Now:

* the folder or file is ours and only the mode bits are wrong: the mode is put back (0700 / 0600) and the command goes on;
* the filesystem cannot hold modes at all (a Windows drive seen from WSL, vfat, some network shares): one note, then it goes on;
* another account owns it (root after sudo): the check still refuses, because that is the real boundary, but the message names the
  cause and the exact command that fixes it.

Windows has no mode bits; its own check lives in `private_files` and is not touched here.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

POSIX = os.name == "posix"
_NOTED: set[str] = set()


def owner_only_wrong(path: str | os.PathLike, details: os.stat_result) -> bool:
    """True when `path` (whose lstat or fstat is `details`) is not private to this user and could not be made so.

    Replaces `details.st_uid != os.getuid() or stat.S_IMODE(details.st_mode) & 0o077`. Never follows a link and never touches
    a file that another account owns.
    """
    if not POSIX:
        return False
    if details.st_uid != os.getuid():
        return True
    if not stat.S_IMODE(details.st_mode) & 0o077:
        return False
    if stat.S_ISLNK(details.st_mode):
        return True
    want = 0o700 if stat.S_ISDIR(details.st_mode) else 0o600
    try:
        os.chmod(path, want)                              # not a link (checked above), so this is the file itself
        after = os.lstat(path)
    except (OSError, NotImplementedError):
        after = None
    if after is not None and after.st_uid == os.getuid() and not stat.S_IMODE(after.st_mode) & 0o077:
        return False
    # The mode did not stick although the owner is right: the filesystem has no real modes. Say it once and carry on.
    folder = str(path if stat.S_ISDIR(details.st_mode) else Path(path).parent)
    if folder not in _NOTED:
        _NOTED.add(folder)
        print(f"strata: {folder} is on a drive that cannot keep files private to your account (a Windows drive seen from Linux, "
              "a USB stick or a network share). Strata goes on; keep your data on the Linux disk to make it private.", file=sys.stderr)
    return False


def _owner_name(uid: int) -> str:
    try:
        import pwd
        return pwd.getpwuid(uid).pw_name
    except (ImportError, KeyError):
        return f"user {uid}"


def fix_target(path: str | os.PathLike) -> Path:
    """The highest folder above `path` (below the home folder) that another account owns: one `chown` there fixes the whole tree."""
    target = Path(path)
    home = Path.home()
    uid = os.getuid()
    best = target
    current = target
    for _ in range(6):
        parent = current.parent
        if parent == current or parent == home or parent == Path(parent.anchor):
            break
        try:
            if parent.lstat().st_uid == uid:
                break
        except OSError:
            break
        best = parent
        current = parent
    return best


def owner_only_text(text: str, path: str | os.PathLike) -> str:
    """The message for a refused check. `text` is the old sentence ("recipe directory must be owner-only"); when another account owns
    `path` it becomes a sentence with the cause and the exact command, otherwise it is returned unchanged."""
    if not POSIX:
        return text
    try:
        owner = os.lstat(path).st_uid
    except OSError:
        return text
    if owner == os.getuid():
        return text
    subject = text.split(" must ")[0].split(" is not ")[0].strip()
    target = fix_target(path)
    return (f"{subject} belongs to {_owner_name(owner)}, not to you (usually from running Strata once with sudo). "
            f"Run this once, then try again: sudo chown -R \"$USER\" {target}")
