"""Versioned, owner-editable serving profiles separate from setup's run config.

The source JSON belongs to setup.py; this small layer keeps operator overrides in
the user's configuration directory so an installer update cannot silently erase
them. It deliberately does not start processes or contact a network service.
"""

from __future__ import annotations

from strata_control.private_files import private_mkdir as _private_mkdir
import hashlib
import json
import math
import os
import re
import shlex
import stat
import subprocess
import tempfile
from strata_control import atomic as _atomic_mod
from contextlib import contextmanager
from pathlib import Path


PROFILE_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
OVERRIDE_KEYS = frozenset({"host", "port", "parallel", "batch_groups", "max_queue", "queue_timeout", "offline",
                           "batch_telemetry"})


class ProfileError(ValueError):
    """A profile is missing, invalid or changed while it was being edited."""

    def __init__(self, message: str, exit_code: int = 2):
        super().__init__(message)
        self.exit_code = exit_code


def profile_dir() -> Path:
    explicit = os.environ.get("STRATA_PROFILE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    if os.name == "nt":
        base = os.environ.get("APPDATA")
        return (Path(base) if base else Path.home() / "AppData" / "Roaming") / "Strata" / "profiles"
    return Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "strata" / "profiles"


def _name(name: str) -> str:
    if not PROFILE_ID.fullmatch(name):
        raise ProfileError("profile name must start with a lowercase letter and use only a-z, 0-9, _ or -")
    return name


def _path(name: str) -> Path:
    return profile_dir() / f"{_name(name)}.json"


def _bytes(profile: dict) -> bytes:
    return (json.dumps(profile, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _revision(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def parse_assignment(text: str) -> tuple[str, object]:
    key, sep, raw = text.partition("=")
    if not sep or key not in OVERRIDE_KEYS:
        raise ProfileError(f"expected one of {', '.join(sorted(OVERRIDE_KEYS))} as KEY=VALUE")
    if key in ("port", "parallel", "max_queue"):
        try:
            value: object = int(raw)
        except ValueError:
            raise ProfileError(f"{key} must be a whole number") from None
    elif key == "queue_timeout":
        try:
            value = float(raw)
        except ValueError:
            raise ProfileError("queue_timeout must be a number of seconds") from None
    elif key == "batch_groups":
        if raw == "auto":
            value = raw
        else:
            try:
                value = int(raw)
            except ValueError:
                raise ProfileError("batch_groups must be auto or a positive whole number") from None
    elif key in ("offline", "batch_telemetry"):
        if raw not in ("true", "false"):
            raise ProfileError(f"{key} must be true or false")
        value = raw == "true"
    else:
        value = raw
    validate_overrides({key: value})
    return key, value


def validate_overrides(overrides: object) -> dict:
    if not isinstance(overrides, dict) or set(overrides) - OVERRIDE_KEYS:
        raise ProfileError("overrides must contain only supported serving settings")
    for key, value in overrides.items():
        if key == "host" and (not isinstance(value, str) or not value or any(c.isspace() for c in value)):
            raise ProfileError("host must be a nonempty address without whitespace")
        if key == "port" and (type(value) is not int or not 0 <= value <= 65535):
            raise ProfileError("port must be in 0..65535")
        if key == "parallel" and (type(value) is not int or not 1 <= value <= 1024):
            raise ProfileError("parallel must be in 1..1024")
        if key == "max_queue" and (type(value) is not int or not 0 <= value <= 4096):
            raise ProfileError("max_queue must be in 0..4096")
        if key == "queue_timeout" and (type(value) not in (int, float) or not math.isfinite(value)
                                        or not 0.1 <= value <= 600):
            raise ProfileError("queue_timeout must be finite seconds in 0.1..600")
        if key == "batch_groups" and value != "auto" and (type(value) is not int or not 1 <= value <= 1024):
            raise ProfileError("batch_groups must be auto or in 1..1024")
        if key in ("offline", "batch_telemetry") and type(value) is not bool:
            raise ProfileError(f"{key} must be true or false")
    return overrides


def validate_profile(value: object, *, expected_name: str | None = None) -> dict:
    if not isinstance(value, dict) or set(value) != {"schema_version", "id", "source_config", "overrides"}:
        raise ProfileError("profile needs schema_version, id, source_config and overrides")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ProfileError("unsupported profile schema version")
    name = _name(value["id"] if isinstance(value["id"], str) else "")
    if expected_name is not None and name != expected_name:
        raise ProfileError("profile ID does not match its filename")
    source = value["source_config"]
    if not isinstance(source, str) or not Path(source).is_absolute():
        raise ProfileError("source_config must be an absolute path")
    validate_overrides(value["overrides"])
    return value


def read_run_config(path: Path) -> dict:
    try:
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ProfileError(f"cannot read run config ({type(exc).__name__})", 5) from None
    if not isinstance(cfg, dict) or not isinstance(cfg.get("args"), list) or not all(
        isinstance(arg, str) for arg in cfg["args"]
    ):
        raise ProfileError("run config needs an args array of strings")
    for key in ("exe", "tokenizer"):
        if not isinstance(cfg.get(key), str) or not cfg[key]:
            raise ProfileError(f"run config needs a nonempty {key} path")
    return cfg


def load(name: str) -> tuple[dict, str]:
    path = _path(name)
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ProfileError("profile must be a regular file", 5)
        from strata_control import private_files
        data = private_files.read_bytes_repairing(path)      # a profile an elevated run locked is repaired once, in place
        if len(data) > 1024 * 1024:
            raise ProfileError("profile is larger than 1 MiB", 5)
        value = json.loads(data.decode("utf-8-sig"))
    except FileNotFoundError:
        raise ProfileError(f"profile {name!r} is not saved; use strata profile create", 5) from None
    except (OSError, ValueError, UnicodeError) as exc:
        if isinstance(exc, ProfileError):
            raise
        raise ProfileError(f"cannot read profile ({type(exc).__name__})", 5) from None
    return validate_profile(value, expected_name=name), _revision(data)


def list_profiles() -> list[dict]:
    directory = profile_dir()
    if not directory.exists():
        return []
    result = []
    for path in sorted(directory.glob("*.json")):
        if not PROFILE_ID.fullmatch(path.stem):
            continue
        try:
            profile, revision = load(path.stem)
            result.append({"id": path.stem, "revision": revision, "source_config": profile["source_config"]})
        except ProfileError as exc:
            result.append({"id": path.stem, "error": str(exc)})
    return result


@contextmanager
def _lock(name: str):
    directory = profile_dir()
    _private_mkdir(directory, parents=True, exist_ok=True)
    lock = directory / f".{_name(name)}.lock"
    with open(lock, "a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _atomic(path: Path, data: bytes) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_mod.replace(temp_name, path)
        if os.name != "nt":
            try:
                parent_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
            except OSError:
                # Replacement already succeeded; do not report a failed save after it changed the profile.
                pass
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def create(name: str, source_config: str) -> tuple[dict, str]:
    name = _name(name)
    try:
        source = Path(source_config).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ProfileError(f"run config is unavailable ({type(exc).__name__})", 5) from None
    read_run_config(source)
    profile = {"schema_version": 1, "id": name, "source_config": str(source), "overrides": {}}
    with _lock(name):
        path = _path(name)
        if path.exists() or path.is_symlink():
            raise ProfileError(f"profile {name!r} already exists")
        _atomic(path, _bytes(profile))
    return load(name)


def propose_source(name: str, source_config: str) -> tuple[dict, str]:
    """Select another installed run config without changing its contents."""
    current, revision = load(name)
    try:
        source = Path(source_config).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ProfileError(f"run config is unavailable ({type(exc).__name__})", 5) from None
    if not stat.S_ISREG(source.lstat().st_mode):
        raise ProfileError("run config must be a regular file", 5)
    read_run_config(source)
    proposed = {**current, "source_config": str(source)}
    return validate_profile(proposed, expected_name=name), revision


def save(name: str, proposed: dict, *, expected_revision: str,
         expected_source_sha256: str | None = None) -> tuple[dict, str]:
    name = _name(name)
    validate_profile(proposed, expected_name=name)
    read_run_config(Path(proposed["source_config"]))
    with _lock(name):
        current, revision = load(name)
        if revision != expected_revision:
            raise ProfileError("profile changed during edit; reload it before saving", 6)
        if expected_source_sha256 is not None:
            if not re.fullmatch(r"[0-9a-f]{64}", expected_source_sha256):
                raise ProfileError("source digest must be a lowercase SHA-256")
            try:
                source = Path(proposed["source_config"]).read_bytes()
            except OSError:
                raise ProfileError("run config changed after review; retry", 6) from None
            if len(source) > 1024 * 1024 or hashlib.sha256(source).hexdigest() != expected_source_sha256:
                raise ProfileError("run config changed after review; retry", 6)
        if current == proposed:
            return current, revision
        history = profile_dir() / ".history" / name
        _private_mkdir(history, parents=True, exist_ok=True)
        old = _path(name).read_bytes()
        if _revision(old) != revision:
            raise ProfileError("profile changed during edit; reload it before saving", 6)
        old_path = history / f"{revision}.json"
        if not old_path.exists():
            _atomic(old_path, old)
        _atomic(_path(name), _bytes(proposed))
    return load(name)


def set_values(name: str, assignments: list[str]) -> tuple[dict, str, str]:
    proposed, revision = propose_values(name, assignments)
    saved, new_revision = save(name, proposed, expected_revision=revision)
    return saved, revision, new_revision


def propose_values(name: str, assignments: list[str]) -> tuple[dict, str]:
    current, revision = load(name)
    changed = {key: value for key, value in (parse_assignment(item) for item in assignments)}
    proposed = {**current, "overrides": {**current["overrides"], **changed}}
    validate_profile(proposed, expected_name=name)
    return proposed, revision


def history(name: str) -> list[dict]:
    _name(name)
    directory = profile_dir() / ".history" / name
    entries = []
    if directory.exists():
        for path in directory.glob("*.json"):
            if re.fullmatch(r"[0-9a-f]{16}", path.stem) and path.is_file() and not path.is_symlink():
                entries.append({"revision": path.stem, "saved_at": path.stat().st_mtime})
    _, current = load(name)
    return [{"revision": current, "current": True},
            *sorted((item for item in entries if item["revision"] != current),
                    key=lambda item: item["saved_at"], reverse=True)]


def rollback(name: str, target_revision: str) -> tuple[dict, str, str]:
    _name(name)
    if not re.fullmatch(r"[0-9a-f]{16}", target_revision):
        raise ProfileError("rollback target must be a full 16-character revision")
    _, current_revision = load(name)
    path = profile_dir() / ".history" / name / f"{target_revision}.json"
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ProfileError("profile history must be a regular file", 5)
        data = path.read_bytes()
        if _revision(data) != target_revision:
            raise ProfileError("profile history content does not match its revision", 5)
        target = validate_profile(json.loads(data.decode("utf-8")), expected_name=name)
    except FileNotFoundError:
        raise ProfileError(f"profile revision {target_revision} was not found", 5) from None
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, ProfileError):
            raise
        raise ProfileError(f"cannot read profile history ({type(exc).__name__})", 5) from None
    saved, new_revision = save(name, target, expected_revision=current_revision)
    return saved, current_revision, new_revision


def edit(name: str) -> tuple[dict, str, str]:
    current, revision = load(name)
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if not editor:
        raise ProfileError("set VISUAL or EDITOR, or use strata profile set")
    command = shlex.split(editor, posix=os.name != "nt")
    if not command:
        raise ProfileError("VISUAL or EDITOR is empty")
    fd, temp_name = tempfile.mkstemp(prefix=f".{name}-edit-", suffix=".json", dir=profile_dir())
    keep_draft = False
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_bytes(current))
        result = subprocess.run([*command, temp_name], check=False)
        if result.returncode:
            keep_draft = True
            raise ProfileError(f"editor exited with status {result.returncode}; draft kept at {temp_name}", 6)
        try:
            proposed = json.loads(Path(temp_name).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            keep_draft = True
            raise ProfileError(f"edited profile is not valid JSON ({type(exc).__name__}); draft kept at {temp_name}") from None
        try:
            saved, new_revision = save(name, proposed, expected_revision=revision)
        except ProfileError as exc:
            keep_draft = True
            raise ProfileError(f"{exc}; draft kept at {temp_name}", exc.exit_code) from None
        return saved, revision, new_revision
    finally:
        if not keep_draft:
            Path(temp_name).unlink(missing_ok=True)


def resolve(ref: str) -> tuple[Path, dict | None, str | None]:
    """A saved name or an explicit JSON path; path-like values never fall back to names."""
    path = Path(ref).expanduser()
    if path.suffix.lower() == ".json" or path.is_absolute() or "/" in ref or "\\" in ref:
        return path.resolve(), None, None
    profile, revision = load(ref)
    return Path(profile["source_config"]), profile, revision
