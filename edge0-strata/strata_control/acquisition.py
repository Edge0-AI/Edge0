"""The line between serving and acquiring.

Starting a model and serving an API never install, download or compile anything. Everything that
reaches the network or runs pip, git, cmake or a package manager lives behind an explicit command:
SETUP, UPDATE or `strata install`. This module is that line, in one place:

* `forbidden()` marks a stretch of code (a start, a serve) in which acquiring is not allowed;
* `require(what, command)` is called by every acquiring primitive (a download, a pip install, a
  network probe, a compile) and raises `AcquisitionRefused` inside such a stretch;
* `AcquisitionRefused` is the typed error: it says what is missing and names the explicit command.

The environment variable `STRATA_NO_ACQUIRE=1` forbids acquiring for a whole process. A process
that serves (`strata serve`, `strata runtime start`, the API server) never imports the installer, so
for it this is a second line of defence, not the first.

Nothing here imports anything outside the standard library.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading

ENV = "STRATA_NO_ACQUIRE"

_lock = threading.Lock()
_depth = 0


class AcquisitionRefused(RuntimeError):
    """Something would have been downloaded or installed during a start or a serve."""

    code = "acquisition_refused"
    exit_code = 5

    def __init__(self, what: str, command: str | None = None):
        self.what = what
        self.command = command or explicit_commands()
        super().__init__(
            f"{what} is missing or out of date. Starting and serving never install or download "
            f"anything. Run: {self.command}")


def explicit_commands() -> str:
    """The commands that acquire, named for this platform."""
    if os.name == "nt":
        return "SETUP.bat (first-time setup), UPDATE.bat (update) or `strata install`"
    return "./setup.sh (first-time setup), ./update.sh (update) or `strata install`"


def is_forbidden() -> bool:
    return _depth > 0 or os.environ.get(ENV, "").strip().lower() in ("1", "true", "yes", "on")


@contextlib.contextmanager
def forbidden():
    """No acquisition inside this block. Nests."""
    global _depth
    with _lock:
        _depth += 1
    try:
        yield
    finally:
        with _lock:
            _depth -= 1


def require(what: str, command: str | None = None) -> None:
    """Call before acquiring `what`; raises AcquisitionRefused where acquiring is forbidden."""
    if is_forbidden():
        raise AcquisitionRefused(what, command)


def main_guard(fn, *, fail=None):
    """Run `fn()`; a refused acquisition becomes `fail(message, hint)` (or an exit-5 message)."""
    try:
        return fn()
    except AcquisitionRefused as exc:
        if fail is not None:
            return fail(str(exc), None)
        print(f"strata: {exc}", file=sys.stderr)
        return exc.exit_code
