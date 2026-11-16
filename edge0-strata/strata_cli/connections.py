"""Owner-local monitor connections; store references, never credential values.

This module is standard-library only so the portable monitor bundle and source
CLI resolve the same saved connection without importing server or installer code.
"""

from __future__ import annotations

import os
try:
    from strata_control.private_files import private_mkdir as _private_mkdir
except ImportError:                                     # a thin bundle (monitor only) ships without strata_control
    def _private_mkdir(path, *, parents=False, exist_ok=True):
        path.mkdir(mode=0o700, parents=parents, exist_ok=exist_ok)
        return path
try:
    from strata_control import posix_owner
except ImportError:                                     # the same thin bundle: the old plain check
    import stat as _stat

    class posix_owner:                                  # noqa: N801
        owner_only_wrong = staticmethod(lambda path, details: os.name == "posix" and (details.st_uid != os.getuid() or _stat.S_IMODE(details.st_mode) & 0o077))
        owner_only_text = staticmethod(lambda text, path: text)
try:
    from strata_control import atomic as _atomic_mod
except ImportError:                                     # the same thin bundle: a plain replace
    class _atomic_mod:                                  # noqa: N801
        replace = staticmethod(os.replace)
import hashlib
import json
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

from strata_cli.client import server_url, url_prefix


NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
MAX_BYTES = 4096
DEFAULT_URL = "http://127.0.0.1:8095"


class ConnectionError(ValueError):
    def __init__(self, message: str, exit_code: int = 2):
        super().__init__(message)
        self.exit_code = exit_code


def _name(value: str) -> str:
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ConnectionError("connection name must use lowercase a-z, 0-9, _ or - and start with a letter")
    return value


def prefix_path(value: str) -> str:
    try:
        return url_prefix(value)
    except ValueError as exc:
        raise ConnectionError(str(exc)) from None


def credential_name(value: str) -> str:
    if not isinstance(value, str) or not ENV_NAME.fullmatch(value):
        raise ConnectionError("credential reference must be an ASCII environment variable name")
    return value


def connection_dir(*, create: bool = False) -> Path:
    explicit = os.environ.get("STRATA_CONNECTION_DIR")
    if explicit:
        directory = Path(explicit).expanduser()
        if not directory.is_absolute():
            raise ConnectionError("STRATA_CONNECTION_DIR must be absolute")
    elif os.name == "nt":
        base = os.environ.get("APPDATA")
        directory = (Path(base) if base else Path.home() / "AppData" / "Roaming") / "Strata" / "connections"
    else:
        directory = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "strata" / "connections"
    if create:
        _private_mkdir(directory, parents=True, exist_ok=True)
    if directory.exists() or directory.is_symlink():
        details = directory.lstat()
        if not stat.S_ISDIR(details.st_mode) or (os.name == "posix" and
                (posix_owner.owner_only_wrong(directory, details))):
            raise ConnectionError(posix_owner.owner_only_text("connection directory must be owner-only", directory), 5)
    return directory


def _path(name: str, *, create_dir: bool = False) -> Path:
    return connection_dir(create=create_dir) / f"{_name(name)}.json"


def _validate(value: object, *, expected_name: str | None = None) -> dict:
    if not isinstance(value, dict) or set(value) != {"schema_version", "id", "url", "prefix", "credential_env"}:
        raise ConnectionError("connection has an unsupported schema", 5)
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ConnectionError("connection has an unsupported schema version", 5)
    name = _name(value["id"])
    if expected_name is not None and name != expected_name:
        raise ConnectionError("connection ID does not match its filename", 5)
    try:
        url = server_url(value["url"])
    except (ValueError, TypeError):
        raise ConnectionError("connection URL must be a safe server root", 5) from None
    prefix = prefix_path(value["prefix"])
    credential = credential_name(value["credential_env"])
    if url != value["url"] or prefix != value["prefix"]:
        raise ConnectionError("connection URL or prefix is not canonical", 5)
    return {"schema_version": 1, "id": name, "url": url, "prefix": prefix,
            "credential_env": credential}


def _bytes(value: dict) -> bytes:
    data = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if len(data) > MAX_BYTES:
        raise ConnectionError("connection record exceeds 4 KiB", 5)
    return data


def _revision(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def load(name: str) -> tuple[dict, str]:
    path = _path(name)
    try:
        details = path.lstat()
        if not stat.S_ISREG(details.st_mode) or details.st_size > MAX_BYTES or \
                (os.name == "posix" and (posix_owner.owner_only_wrong(path, details))):
            raise ConnectionError(posix_owner.owner_only_text("connection file must be a private regular file under 4 KiB", path), 5)
        data = path.read_bytes()
        value = json.loads(data.decode("utf-8"))
    except FileNotFoundError:
        raise ConnectionError(f"connection {name!r} is not saved; use strata connection create", 5) from None
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, ConnectionError):
            raise
        raise ConnectionError("connection file is unreadable", 5) from None
    try:
        return _validate(value, expected_name=name), _revision(data)
    except (ConnectionError, ValueError, TypeError) as exc:
        raise ConnectionError(f"connection file has invalid values ({type(exc).__name__})", 5) from None


def list_connections() -> list[dict]:
    directory = connection_dir()
    if not directory.exists():
        return []
    paths = [path for path in sorted(directory.glob("*.json")) if NAME.fullmatch(path.stem)]
    if len(paths) > 100:
        raise ConnectionError("more than 100 connection records need inspection", 5)
    result = []
    for path in paths:
        try:
            value, revision = load(path.stem)
            result.append({"id": value["id"], "url": value["url"], "prefix": value["prefix"],
                           "credential_env": value["credential_env"], "revision": revision})
        except ConnectionError as exc:
            result.append({"id": path.stem, "error": str(exc)})
    return result


def _replace(path: Path, data: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_mod.replace(name, path)
        if os.name == "posix":
            try:
                parent = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(parent)
                finally:
                    os.close(parent)
            except OSError:
                # The replacement succeeded; do not report a failed save afterward.
                pass
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def _lock(name: str):
    path = connection_dir(create=True) / f".{_name(name)}.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "r+b") as stream:
        details = os.fstat(stream.fileno())
        if not stat.S_ISREG(details.st_mode) or (os.name == "posix" and
                (posix_owner.owner_only_wrong(path, details))):
            raise ConnectionError(posix_owner.owner_only_text("connection lock must be owner-only", path), 5)
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


def create(name: str, url: str, *, prefix: str = "",
           credential_env: str = "STRATA_TELEMETRY_KEY") -> tuple[dict, str]:
    name = _name(name)
    value = _validate({"schema_version": 1, "id": name, "url": server_url(url),
                       "prefix": prefix_path(prefix), "credential_env": credential_name(credential_env)},
                      expected_name=name)
    data = _bytes(value)
    path = _path(name, create_dir=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        raise ConnectionError(f"connection {name!r} already exists") from None
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return value, _revision(data)


def propose(name: str, assignment: str) -> tuple[dict, str]:
    current, revision = load(name)
    key, sep, raw = assignment.partition("=")
    if not sep or key not in ("url", "prefix", "credential_env"):
        raise ConnectionError("expected url=ROOT, prefix=/PATH or credential_env=ENV_NAME")
    candidate = dict(current)
    candidate[key] = (server_url(raw) if key == "url" else
                      prefix_path(raw) if key == "prefix" else credential_name(raw))
    return _validate(candidate, expected_name=name), revision


def save(name: str, value: dict, *, expected_revision: str) -> tuple[dict, str]:
    name = _name(name)
    value = _validate(value, expected_name=name)
    if not isinstance(expected_revision, str) or not re.fullmatch(r"[0-9a-f]{16}", expected_revision):
        raise ConnectionError("connection revision must be 16 lowercase hex characters")
    path = _path(name)
    with _lock(name):
        current, revision = load(name)
        if revision != expected_revision:
            raise ConnectionError("connection changed since preview; inspect it again", 6)
        data = _bytes(value)
        if value == current:
            return current, revision
        _replace(path, data)
        return value, _revision(data)


def resolve(*, url: str | None = None, connection: str | None = None,
            credential_env: str | None = None, timeout: float | None = None,
            environment: dict | None = None) -> dict:
    env = os.environ if environment is None else environment
    selected = connection or (None if url is not None else env.get("STRATA_CONNECTION"))
    if url is not None and connection is not None:
        raise ConnectionError("choose --url or --connection, not both")
    if selected:
        saved, revision = load(selected)
        root, prefix = saved["url"], saved["prefix"]
        key_ref = saved["credential_env"]
        origins = {"url": "connection", "prefix": "connection", "credential_env": "connection"}
    else:
        revision = None
        root = url if url is not None else env.get("STRATA_URL", DEFAULT_URL)
        prefix = ""
        key_ref = "STRATA_API_KEY"
        origins = {"url": "flag" if url is not None else "environment" if "STRATA_URL" in env else "default",
                   "prefix": "default", "credential_env": "default"}
    if credential_env is not None:
        key_ref = credential_env
        origins["credential_env"] = "flag"
    if timeout is not None and (type(timeout) not in (int, float) or not 0 < timeout <= 30):
        raise ConnectionError("timeout must be greater than zero and at most 30 seconds")
    origins["timeout_s"] = "flag" if timeout is not None else "default"
    normalized_url, normalized_prefix = server_url(root), prefix_path(prefix)
    return {"url": normalized_url, "prefix": normalized_prefix,
            "base_url": normalized_url + normalized_prefix,
            "credential_env": credential_name(key_ref), "credential_present": bool(env.get(key_ref)),
            "timeout_s": timeout if timeout is not None else 3.0,
            "connection": selected, "connection_revision": revision, "origins": origins}
