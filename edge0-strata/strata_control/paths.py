"""Per-user state locations, one rule for the terminal and the host.

POSIX keeps $XDG_STATE_HOME (default ~/.local/state) under ``strata/``. Windows has no such place: state goes to
``%LOCALAPPDATA%\\Strata``. The Node host (apps/host/server/platform-paths.mjs) resolves the same folders.
"""

from __future__ import annotations

import os
from pathlib import Path


def state_base(environ=None, *, windows: bool | None = None, home: Path | None = None) -> Path:
    """The folder that holds Strata's per-user state: runtime/, tuning/, time-skip/."""
    environ = os.environ if environ is None else environ
    windows = os.name == "nt" if windows is None else windows
    home = Path.home() if home is None else home
    if windows:
        local = environ.get("LOCALAPPDATA") or str(home / "AppData" / "Local")
        return Path(local) / "Strata"
    return Path(environ.get("XDG_STATE_HOME") or str(home / ".local" / "state")) / "strata"
