"""Saved tuning presets after the engine changed: mark them for the short re-check, never delete them.

A tuning preset is valid for one identity (GPU and driver, engine build, model file, context). When an engine update
changes the engine part of that identity the presets are not wrong, only unconfirmed. The plan (4b) says to offer a short
check of about two minutes, not a full re-tune, and to keep every preset.

This module is the one thing the updater writes about presets, and it is deliberately not a preset store: it writes a
small, separate file next to the tuning state, ``<state>/tuning/recheck.json``, and never opens, moves or rewrites a
preset. The tuner (S5) reads `pending()` when it loads a preset and, once the short check passed or the user re-tuned,
calls `clear(mark_id)`. The file is created only when an engine identity actually changed, so a PC without an earlier
engine, or an update to a build with the same identity, writes nothing.

    {"schema_version": 1,
     "pending": [{"id": "...", "reason": "engine_changed", "component": "strata-engine",
                  "from": {"version", "build_id", "manifest_sha256", "backend", "platform", "variant"},
                  "to":   {... same ...}, "marked_at": "<iso>"}]}
"""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import uuid
from pathlib import Path

from strata_control import paths
from strata_control import atomic as _atomic_mod

SCHEMA_VERSION = 1
IDENTITY_KEYS = ("version", "build_id", "manifest_sha256", "backend", "platform", "variant")


def marks_path(base: Path | None = None) -> Path:
    return (base or paths.state_base()) / "tuning" / "recheck.json"


def identity_of(component, manifest_sha256: str) -> dict:
    """The engine part of a preset's identity, from an installed `EngineComponent`."""
    return {"version": component.version, "build_id": component.build_id, "manifest_sha256": manifest_sha256,
            "backend": component.backend, "platform": component.platform, "variant": component.variant}


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {"schema_version": SCHEMA_VERSION, "pending": []}
    if not isinstance(data, dict) or not isinstance(data.get("pending"), list):
        return {"schema_version": SCHEMA_VERSION, "pending": []}
    return data


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp = tempfile.mkstemp(prefix=".recheck-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=1, sort_keys=True)
            stream.write("\n")
        _atomic_mod.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def pending(base: Path | None = None) -> list[dict]:
    return list(_read(marks_path(base))["pending"])


def mark_engine_changed(before: dict | None, after: dict, *, base: Path | None = None,
                        now: dt.datetime | None = None) -> dict | None:
    """Record that presets measured on `before` are unconfirmed on `after`. Returns the mark, or None when nothing is owed:
    there was no earlier engine, or the identity did not change. Presets are not touched."""
    if before is None:
        return None
    if all(before.get(key) == after.get(key) for key in IDENTITY_KEYS):
        return None
    path = marks_path(base)
    data = _read(path)
    when = (now or dt.datetime.now(dt.timezone.utc)).isoformat()
    mark = {"id": uuid.uuid4().hex, "reason": "engine_changed", "component": "strata-engine",
            "from": {k: before.get(k) for k in IDENTITY_KEYS}, "to": {k: after.get(k) for k in IDENTITY_KEYS},
            "marked_at": when}
    # a second change before the check supersedes the first: one pending re-check from the identity the presets were measured on
    older = [m for m in data["pending"] if m.get("reason") == "engine_changed"]
    if older:
        mark["from"] = older[0]["from"]
        data["pending"] = [m for m in data["pending"] if m.get("reason") != "engine_changed"]
        if all(mark["from"].get(k) == mark["to"].get(k) for k in IDENTITY_KEYS):
            _write(path, data)          # rolled back to the identity the presets were measured on: nothing is owed any more
            return None
    data["pending"].append(mark)
    _write(path, data)
    return mark


def clear(mark_id: str, base: Path | None = None) -> bool:
    path = marks_path(base)
    data = _read(path)
    kept = [m for m in data["pending"] if m.get("id") != mark_id]
    if len(kept) == len(data["pending"]):
        return False
    data["pending"] = kept
    _write(path, data)
    return True
