"""Saved launch presets: the whole answer set of the setup screens under a name.

One small JSON file per preset in `launch-presets/` beside the profiles folder (so it is found wherever the profiles are, and
`STRATA_LAUNCH_PRESET_DIR` moves it for tests). The files are private to the user (they can hold the API key). The tuned
settings themselves are NOT copied here: a preset points at the tuning preset the tuner saved (`strata presets`, the store the
desktop app also uses), so a re-tune in either place is seen in both. The profile the run uses (`strata profile`) is the shared
one too (named `launch-<slug>`).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from strata_control import atomic as _atomic_mod
from pathlib import Path
from typing import Any

from strata_cli.launch.settings import LaunchSettings, slug


class PresetError(ValueError):
    """A saved launch preset is missing or damaged; the message is for the user."""


def preset_dir() -> Path:
    explicit = os.environ.get("STRATA_LAUNCH_PRESET_DIR")
    if explicit:
        return Path(explicit).expanduser()
    from strata_control import profiles
    return profiles.profile_dir().parent / "launch-presets"


def _path(name: str) -> Path:
    return preset_dir() / f"{slug(name)}.json"


def _private_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".launch-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        _atomic_mod.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    if os.name == "nt":
        try:
            from strata_control import private_files
            private_files.restrict_to_current_user(path, directory=False)
        except Exception:                                    # noqa: BLE001 - a file the user can read is still theirs
            pass


def now_text() -> str:
    now = time.time()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1000000):06d}"


def check_name(name: str) -> None:
    """The error `save` would raise for this name, without writing anything."""
    if not (name or "").strip():
        raise PresetError("A preset needs a name.")
    _path(name.strip())


def save(settings: LaunchSettings, name: str | None = None) -> LaunchSettings:
    """Write (or replace) the preset called `name` (default: the settings' own name)."""
    name = (name or settings.name or "").strip()
    if not name:
        raise PresetError("A preset needs a name.")
    path = _path(name)
    previous = load(name, missing_ok=True)
    s = settings.copy(name=name)
    s.created = (previous.created if previous else None) or s.created or now_text()
    _private_write(path, (json.dumps(s.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    return s


def load(name: str, *, missing_ok: bool = False) -> LaunchSettings | None:
    path = _path(name)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        if missing_ok:
            return None
        raise PresetError(f"There is no saved preset called '{name}'." + _near(name)) from None
    except (OSError, ValueError):
        raise PresetError(f"The saved preset '{name}' cannot be read. Delete it (strata launch --delete-preset \"{name}\") "
                          "and save it again.") from None
    if not isinstance(data, dict):
        raise PresetError(f"The saved preset '{name}' is damaged.")
    s = LaunchSettings.from_dict(data)
    s.name = s.name or name
    return s


def _near(name: str) -> str:
    from strata_cli.launch.settings import closest
    names = [p.name for p in list_presets() if p.name]
    near = closest(name, names) if names else None
    if near:
        return f" Did you mean '{near}'?"
    return (" Saved presets: " + ", ".join(names) + ".") if names else " No presets are saved yet."


def list_presets() -> list[LaunchSettings]:
    folder = preset_dir()
    out: list[LaunchSettings] = []
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            s = LaunchSettings.from_dict(data)
            s.name = s.name or path.stem
            out.append(s)
        except (OSError, ValueError):
            continue
    out.sort(key=lambda s: s.last_used or s.created or "", reverse=True)
    return out


def delete(name: str) -> None:
    path = _path(name)
    if not path.is_file():
        raise PresetError(f"There is no saved preset called '{name}'." + _near(name))
    path.unlink()


def touch_used(name: str) -> None:
    s = load(name, missing_ok=True)
    if s is not None:
        s.last_used = now_text()
        save(s, name)


def summary(s: LaunchSettings) -> dict[str, Any]:
    """The columns of the start screen."""
    return {"name": s.name, "model": s.model or ("own files" if s.model_path else "-"),
            "context": s.context, "parallel": s.parallel, "tuned": s.is_tuned, "tuned_at": s.tuned_at,
            "last_used": s.last_used, "access": s.access}
