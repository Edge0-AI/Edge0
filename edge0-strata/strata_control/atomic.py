"""Replace a file that another program may be reading, on Windows as well as everywhere else.

A job record (a Time-skip `run.json`, a tuning `job.json`) is rewritten at every phase and READ by the app's host every couple of seconds.
On Windows, `os.replace` onto a file that is open in another process fails with PermissionError ("Access is denied") for as long as the
reader holds it. That is a split second, but a job saves many times, so a Time-skip or a tune died at random with that error. The fix is
to wait that split second and try again; any other error, and a PermissionError that does not go away, is raised as before.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Union

PathLike = Union[str, "os.PathLike[str]"]

#: how long a held file is waited for, in total (seconds) and between tries
PATIENCE_S = 5.0
STEP_S = 0.02


def replace(src: PathLike, dst: PathLike, *, patience: float = PATIENCE_S, sleep=time.sleep, clock=time.monotonic) -> None:
    deadline = clock() + patience
    step = STEP_S
    while True:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if os.name != "nt" or clock() >= deadline:
                raise
            sleep(step)
            step = min(step * 1.5, 0.25)


def write_text(path: PathLike, text: str, *, encoding: str = "utf-8") -> None:
    """Write `text` to `path` through a neighbouring temporary file, so a reader never sees half a record."""
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp") if target.suffix else target.with_name(target.name + ".tmp")
    tmp.write_text(text, encoding=encoding)
    replace(tmp, target)
