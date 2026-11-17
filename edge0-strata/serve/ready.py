"""Ready file for a foreground Strata server instance.

A consumer must still compare the instance ID with live telemetry before acting.
An abruptly killed process can leave this file behind; a stale file is never
accepted as proof of a running model by this module.
POSIX files use mode 0600; Windows access follows the parent directory ACL.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def preflight(path: str) -> Path:
    target = Path(path)
    if not target.is_absolute():
        raise ValueError("--ready-file must be an absolute path")
    if not target.parent.is_dir():
        raise ValueError("--ready-file parent directory must exist")
    if target.exists() or target.is_symlink():
        raise ValueError("--ready-file already exists; verify and remove a stale file before reuse")
    return target


def publish(path: Path, payload: dict) -> None:
    data = (json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def remove_if_owned(path: Path, instance_id: str) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("instance_id") == instance_id and data.get("pid") == os.getpid():
            path.unlink(missing_ok=True)
    except (OSError, ValueError):
        pass
