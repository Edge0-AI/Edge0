"""The model library when `--data-dir` names another folder: what is in the folder in use, and the one question about it.

Nothing here moves a file.  `strata launch` asks (or is told with --move-data); setup.py does the moving, and only when
it is given --move-data.  Without it the old folder stays where it is, stays usable, and the new folder is used for new
downloads."""
from __future__ import annotations

import os
import sys
from pathlib import Path

DATA_ITEMS = ("models", "packs", "mtp")


def library_gb(folder: Path) -> float:
    """GB of model files (models, packs, mtp) in a data folder; 0 when there are none."""
    total = 0
    for item in DATA_ITEMS:
        for root, _dirs, files in os.walk(folder / item):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return total / 1e9


def same_place(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def same_drive(a: Path, b: Path) -> bool:
    probe = lambda p: next((q for q in (p, *p.parents) if q.exists()), None)   # noqa: E731 - the new folder may not exist yet
    pa, pb = probe(a), probe(b)
    try:
        return pa is not None and pb is not None and os.stat(pa).st_dev == os.stat(pb).st_dev
    except OSError:
        return False


def movable_library(old: Path | None, new: Path) -> float:
    """GB of models in `old` that could be moved into `new` (same drive, a different folder); 0 when nothing is there to move."""
    if old is None or same_place(old, new) or not Path(old).is_dir() or not same_drive(Path(old), new):
        return 0.0
    return library_gb(Path(old))


def ask_move(gb: float, new: Path, *, stdin_tty: bool | None = None, out=None) -> bool:
    """Ask: keep the models where they are (and use the new folder for new downloads), or move them.  True = move.
    No keyboard (a pipe, a script) or Enter: keep, the answer that loses nothing."""
    out = out or sys.stdout
    tty = sys.stdin.isatty() if stdin_tty is None else stdin_tty
    if not tty:
        return False
    size = f"{gb:.0f} GB" if gb >= 10 else f"{gb:.1f} GB"
    out.write(f"\n[!] Move your {size} of models to {new}, or use the new folder only for new downloads?\n"
              "      k) keep my models where they are; the new folder is for new downloads (default)\n"
              "      m) move my models to the new folder\n")
    out.flush()
    try:
        answer = input("Keep or move? [K/m] ").strip().lower()
    except EOFError:
        return False
    return answer in ("m", "move")
