"""Typed, owner-editable local operation recipes for saved serving profiles.

Recipes contain no executable command text. Each kind routes to an existing
Strata operation. This is a source-run store; Desktop must later use the same
schema and operation authority rather than add a separate implementation.
"""

from __future__ import annotations

from strata_control.private_files import private_mkdir as _private_mkdir
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from strata_control import profiles
from strata_control.profiles import ProfileError
from strata_control import posix_owner


KINDS = frozenset({"runtime.restart", "engine.calibrate.measure", "components.verify"})
MAX_BYTES = 64 * 1024


def _identity(profile: str, name: str) -> tuple[str, str]:
    if not profiles.PROFILE_ID.fullmatch(profile) or not profiles.PROFILE_ID.fullmatch(name):
        raise ProfileError("recipe profile and name must use lowercase letters, digits, _ or -")
    return profile, name


def _directory(profile: str, *, create: bool = False) -> Path:
    profiles.load(profile)
    base = profiles.profile_dir()
    directory = base / ".ops" / profile
    for path in (base, directory.parent):
        if not path.exists() and not path.is_symlink():
            continue
        details = path.lstat()
        if not stat.S_ISDIR(details.st_mode):
            raise ProfileError("recipe directory must not be a symlink", 5)
        if os.name == "posix" and (posix_owner.owner_only_wrong(path, details)):
            raise ProfileError(posix_owner.owner_only_text("recipe directory must be owner-only (mode 0700)", path), 5)
    if create:
        _private_mkdir(directory.parent, exist_ok=True)
        _private_mkdir(directory, exist_ok=True)
    if directory.exists() or directory.is_symlink():
        details = directory.lstat()
        if not stat.S_ISDIR(details.st_mode):
            raise ProfileError("recipe directory must not be a symlink", 5)
        if os.name == "posix" and (posix_owner.owner_only_wrong(directory, details)):
            raise ProfileError(posix_owner.owner_only_text("recipe directory must be owner-only (mode 0700)", directory), 5)
    return directory


def _path(profile: str, name: str, *, create: bool = False) -> Path:
    _identity(profile, name)
    return _directory(profile, create=create) / f"{name}.json"


def _lock_name(profile: str) -> str:
    return "ops" + hashlib.sha256(profile.encode("utf-8")).hexdigest()[:24]


def _history_dir(profile: str, name: str, *, create: bool = False) -> Path:
    root = _path(profile, name).parent / ".history"
    directory = root / name
    for path in (root, directory):
        if path.is_symlink():
            raise ProfileError("recipe history directory must not be a symlink", 5)
    if create:
        _private_mkdir(root, exist_ok=True)
        _private_mkdir(directory, exist_ok=True)
    for path in (root, directory):
        if path.exists():
            details = path.lstat()
            if not stat.S_ISDIR(details.st_mode) or (os.name == "posix" and
                    (posix_owner.owner_only_wrong(path, details))):
                raise ProfileError(posix_owner.owner_only_text("recipe history directory must be owner-only", path), 5)
    return directory


def _bytes(recipe: dict) -> bytes:
    return (json.dumps(recipe, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _validate(recipe: object, profile: str, name: str) -> dict:
    if not isinstance(recipe, dict) or set(recipe) != {"schema_version", "profile", "id", "kind", "params"}:
        raise ProfileError("recipe needs schema_version, profile, id, kind and params")
    if type(recipe["schema_version"]) is not int or recipe["schema_version"] != 1:
        raise ProfileError("unsupported recipe schema version")
    if recipe["profile"] != profile or recipe["id"] != name:
        raise ProfileError("recipe profile or ID does not match its file")
    kind, params = recipe["kind"], recipe["params"]
    if not isinstance(kind, str) or kind not in KINDS or not isinstance(params, dict):
        raise ProfileError("recipe kind or parameters are unsupported")
    if kind == "runtime.restart":
        if set(params) != {"startup_wait_s"} or type(params["startup_wait_s"]) is not int or \
                not 1 <= params["startup_wait_s"] <= 600:
            raise ProfileError("restart recipe needs startup_wait_s in 1..600")
    elif kind == "engine.calibrate.measure":
        directory = params.get("report_dir")
        if set(params) != {"report_dir"} or not isinstance(directory, str) or not directory or "\x00" in directory or \
                not Path(directory).is_absolute():
            raise ProfileError("calibration recipe needs an absolute report_dir")
    elif set(params) - {"live"} or ("live" in params and type(params["live"]) is not bool):
        raise ProfileError("component verification recipe accepts only live=true|false")
    return recipe


def _default(profile: str, name: str, kind: str) -> dict:
    if not isinstance(kind, str) or kind not in KINDS:
        raise ProfileError("recipe kind is unsupported")
    if kind == "runtime.restart":
        params = {"startup_wait_s": 300}
    elif kind == "engine.calibrate.measure":
        params = {"report_dir": str(profiles.profile_dir().resolve() / ".reports")}
    else:
        params = {}
    return _validate({"schema_version": 1, "profile": profile, "id": name,
                      "kind": kind, "params": params}, profile, name)


def create(profile: str, name: str, kind: str) -> tuple[dict, str]:
    profile, name = _identity(profile, name)
    value = _default(profile, name, kind)
    path = _path(profile, name, create=True)
    with profiles._lock(_lock_name(profile)):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            raise ProfileError(f"recipe {name!r} already exists") from None
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(_bytes(value))
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    return load(profile, name)


def initialize(profile: str) -> list[str]:
    profiles.load(profile)
    defaults = (("restart", "runtime.restart"),
                ("calibrate", "engine.calibrate.measure"),
                ("check", "components.verify"))
    for name, kind in defaults:
        path = _path(profile, name)
        if path.exists() or path.is_symlink():
            saved, _ = load(profile, name)
            if saved["kind"] != kind:
                raise ProfileError(f"recipe {name!r} exists with a different kind; choose another name")
    created = []
    for name, kind in defaults:
        path = _path(profile, name)
        if not path.exists() and not path.is_symlink():
            try:
                create(profile, name, kind)
                created.append(name)
            except ProfileError as exc:
                if "already exists" not in str(exc):
                    raise
        saved, _ = load(profile, name)
        if saved["kind"] != kind:
            raise ProfileError(f"recipe {name!r} exists with a different kind; choose another name")
    return created


def load(profile: str, name: str) -> tuple[dict, str]:
    path = _path(profile, name)
    try:
        details = path.lstat()
        if not stat.S_ISREG(details.st_mode) or details.st_size > MAX_BYTES:
            raise ProfileError("recipe must be a regular file under 64 KiB", 5)
        if os.name == "posix" and (posix_owner.owner_only_wrong(path, details)):
            raise ProfileError(posix_owner.owner_only_text("recipe file must be owner-only (mode 0600)", path), 5)
        data = path.read_bytes()
        value = json.loads(data.decode("utf-8-sig"))
    except FileNotFoundError:
        raise ProfileError(f"recipe {name!r} is missing; use strata ops init --profile {profile}", 5) from None
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, ProfileError):
            raise
        raise ProfileError(f"cannot read recipe ({type(exc).__name__})", 5) from None
    return _validate(value, profile, name), profiles._revision(data)


def list_recipes(profile: str) -> list[dict]:
    directory = _directory(profile)
    if not directory.exists():
        return []
    items = []
    for path in sorted(directory.glob("*.json")):
        if profiles.PROFILE_ID.fullmatch(path.stem):
            try:
                recipe, revision = load(profile, path.stem)
                items.append({"id": path.stem, "kind": recipe["kind"], "revision": revision})
            except ProfileError as exc:
                items.append({"id": path.stem, "error": str(exc)})
    return items


def save(profile: str, name: str, proposed: dict, expected_revision: str) -> tuple[dict, str]:
    _validate(proposed, profile, name)
    with profiles._lock(_lock_name(profile)):
        current, revision = load(profile, name)
        if revision != expected_revision:
            raise ProfileError("recipe changed during edit; reload it before saving", 6)
        if current == proposed:
            return current, revision
        path = _path(profile, name)
        old = path.read_bytes()
        if profiles._revision(old) != revision:
            raise ProfileError("recipe changed during edit; reload it before saving", 6)
        history_dir = _history_dir(profile, name, create=True)
        backup = history_dir / f"{revision}.json"
        if backup.exists() or backup.is_symlink():
            details = backup.lstat()
            if not stat.S_ISREG(details.st_mode) or (os.name == "posix" and
                    (posix_owner.owner_only_wrong(backup, details))) or \
                    backup.read_bytes() != old:
                raise ProfileError("recipe history conflicts with the saved revision", 5)
        else:
            profiles._atomic(backup, old)
        profiles._atomic(path, _bytes(proposed))
    return load(profile, name)


def history(profile: str, name: str) -> list[dict]:
    _, current = load(profile, name)
    directory = _history_dir(profile, name)
    entries = []
    if directory.exists():
        for path in directory.glob("*.json"):
            if re.fullmatch(r"[0-9a-f]{16}", path.stem) and path.is_file() and not path.is_symlink():
                entries.append({"revision": path.stem, "saved_at": path.stat().st_mtime})
    return [{"revision": current, "current": True},
            *sorted((item for item in entries if item["revision"] != current),
                    key=lambda item: item["saved_at"], reverse=True)]


def rollback(profile: str, name: str, target_revision: str) -> tuple[dict, str, str]:
    if not re.fullmatch(r"[0-9a-f]{16}", target_revision):
        raise ProfileError("rollback target must be a full 16-character revision")
    _, current_revision = load(profile, name)
    path = _history_dir(profile, name) / f"{target_revision}.json"
    try:
        details = path.lstat()
        if not stat.S_ISREG(details.st_mode) or details.st_size > MAX_BYTES:
            raise ProfileError("recipe history must be a regular file under 64 KiB", 5)
        if os.name == "posix" and (posix_owner.owner_only_wrong(path, details)):
            raise ProfileError(posix_owner.owner_only_text("recipe history file must be owner-only", path), 5)
        data = path.read_bytes()
        if profiles._revision(data) != target_revision:
            raise ProfileError("recipe history content does not match its revision", 5)
        proposed = _validate(json.loads(data.decode("utf-8")), profile, name)
    except FileNotFoundError:
        raise ProfileError(f"recipe revision {target_revision} was not found", 5) from None
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, ProfileError):
            raise
        raise ProfileError(f"cannot read recipe history ({type(exc).__name__})", 5) from None
    saved, new_revision = save(profile, name, proposed, current_revision)
    return saved, current_revision, new_revision


def propose_value(profile: str, name: str, assignment: str) -> tuple[dict, str]:
    current, revision = load(profile, name)
    key, separator, raw = assignment.partition("=")
    if not separator or not raw:
        raise ProfileError("expected a typed recipe parameter as KEY=VALUE")
    kind = current["kind"]
    if kind == "runtime.restart" and key == "startup_wait_s":
        try:
            value: object = int(raw)
        except ValueError:
            raise ProfileError("startup_wait_s must be a whole number") from None
    elif kind == "engine.calibrate.measure" and key == "report_dir":
        value = raw
    elif kind == "components.verify" and key == "live" and raw in ("true", "false"):
        value = raw == "true"
    else:
        raise ProfileError("parameter is not supported by this recipe kind")
    proposed = {**current, "params": {**current["params"], key: value}}
    return _validate(proposed, profile, name), revision


def edit(profile: str, name: str) -> tuple[dict, str, str]:
    current, revision = load(profile, name)
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if not editor:
        raise ProfileError("set VISUAL or EDITOR, or use strata ops set")
    command = shlex.split(editor, posix=os.name != "nt")
    if not command:
        raise ProfileError("VISUAL or EDITOR is empty")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{name}-edit-", suffix=".json", dir=_directory(profile))
    keep = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_bytes(current))
        result = subprocess.run([*command, temporary], check=False)
        if result.returncode:
            keep = True
            raise ProfileError(f"editor exited with status {result.returncode}; draft kept at {temporary}", 6)
        try:
            proposed = json.loads(Path(temporary).read_text(encoding="utf-8-sig"))
            saved, new_revision = save(profile, name, proposed, revision)
        except (OSError, ValueError) as exc:
            keep = True
            code = exc.exit_code if isinstance(exc, ProfileError) else 2
            raise ProfileError(f"invalid recipe edit ({exc}); draft kept at {temporary}", code) from None
        return saved, revision, new_revision
    finally:
        if not keep:
            Path(temporary).unlink(missing_ok=True)


def plan(profile: str, name: str) -> dict:
    recipe, revision = load(profile, name)
    kind = recipe["kind"]
    if kind == "runtime.restart":
        from strata_control import runtime
        preflight = runtime.plan_restart(profile)
        profile_revision = preflight.get("candidate_profile_revision")
        operation = {"ok": preflight["ok"], "exit_code": preflight.get("exit_code", 6),
                     "preflight": preflight, "startup_wait_s": recipe["params"]["startup_wait_s"],
                     "run_effects": {"drain_current_instance": True, "start_new_instance": True,
                                     "rollback_previous_launch_on_activation_failure": True}}
    elif kind == "engine.calibrate.measure":
        from strata_control import calibration
        preflight = calibration.preview(profile)
        profile_revision = preflight["profile_revision"]
        operation = {"ok": preflight["component_preflight"]["ok"], "exit_code": 5,
                     "preflight": preflight, "report_dir": recipe["params"]["report_dir"],
                     "run_effects": {"start_engine_for_decode_sweep": True, "write_new_report": True,
                                     "change_run_config": False, "requires_exclusive_acknowledgement": True}}
    else:
        from strata_control.doctor import inspect as inspect_components
        path, _, _ = profiles.resolve(profile)
        live = recipe["params"].get("live", False)
        report = inspect_components(path, profile_id=profile, live=live)
        profile_revision = profiles.load(profile)[1]
        operation = {"ok": report["ok"], "exit_code": 5, "preflight": report,
                     "run_effects": {"read_components": True, "query_host_gpus": live, "write_files": False}}
    if profile_revision is not None and profile_revision != profiles.load(profile)[1]:
        raise ProfileError("profile changed during recipe preview; retry", 6)
    return {"profile": profile, "profile_revision": profile_revision,
            "recipe": name, "recipe_revision": revision,
            "kind": kind, "operation": operation, "would_execute": False}


def run(profile: str, name: str, *, exclusive_ack: bool = False,
        expected_revision: str | None = None,
        expected_profile_revision: str | None = None,
        expected_owner_id: str | None = None,
        expected_instance_id: str | None = None,
        expected_source_sha256: str | None = None) -> dict:
    recipe, revision = load(profile, name)
    if expected_revision is not None and revision != expected_revision:
        raise ProfileError("recipe changed after preview; plan it again before running", 6)
    if expected_profile_revision is not None and profiles.load(profile)[1] != expected_profile_revision:
        raise ProfileError("profile changed after preview; plan it again before running", 6)
    kind = recipe["kind"]
    if kind == "runtime.restart":
        from strata_control import runtime
        if (expected_owner_id is None) != (expected_instance_id is None):
            raise ProfileError("restart recipe needs both reviewed owner and instance IDs")
        if any(value is not None for value in (expected_owner_id, expected_instance_id,
                                               expected_source_sha256)):
            preflight = runtime.plan_restart(profile)
            if not preflight["ok"]:
                raise ProfileError(preflight.get("error", "restart preflight failed"),
                                   preflight.get("exit_code", 6))
            if expected_owner_id is not None and preflight["current_owner_id"] != expected_owner_id:
                raise ProfileError("owner_changed_after_restart_plan", 6)
            if expected_instance_id is not None and preflight["current_instance_id"] != expected_instance_id:
                raise ProfileError("instance_changed_after_restart_plan", 6)
            if (expected_source_sha256 is not None and
                    preflight["candidate_source_sha256"] != expected_source_sha256):
                raise ProfileError("source_changed_after_preview", 6)
    elif any(value is not None for value in (expected_owner_id, expected_instance_id,
                                             expected_source_sha256)):
        raise ProfileError("owner, instance and source preconditions require a restart recipe")
    if kind == "engine.calibrate.measure" and not exclusive_ack:
        raise ProfileError("calibration recipe requires --exclusive for each run")
    from strata_control import journal
    profile_revision = profiles.load(profile)[1]
    record = journal.begin(profile, name, kind, revision, profile_revision,
                           intent=dict(recipe["params"]))
    try:
        if kind == "runtime.restart":
            from strata_control import runtime
            kwargs = {"startup_wait": recipe["params"]["startup_wait_s"]}
            if expected_profile_revision is not None:
                kwargs["expected_profile_revision"] = expected_profile_revision
            if expected_owner_id is not None:
                kwargs["expected_owner_id"] = expected_owner_id
                kwargs["expected_instance_id"] = expected_instance_id
            if expected_source_sha256 is not None:
                kwargs["expected_source_sha256"] = expected_source_sha256
            result = runtime.restart(profile, **kwargs)
        elif kind == "engine.calibrate.measure":
            from strata_control import calibration
            directory = Path(recipe["params"]["report_dir"])
            _private_mkdir(directory, exist_ok=True)
            if not stat.S_ISDIR(directory.lstat().st_mode):
                raise ProfileError("report_dir must be a directory, not a symlink", 5)
            report_path = directory / f"calibration-{uuid.uuid4().hex}.json"
            report = calibration.measure(profile, report_path, exclusive_ack=True,
                                         say=lambda message: print(message, file=sys.stderr))
            result = {"ok": True, "report_path": str(report_path), "report": report,
                      "requires_review_before_apply": True}
        else:
            from strata_control.doctor import inspect as inspect_components
            path, _, _ = profiles.resolve(profile)
            report = inspect_components(path, profile_id=profile, live=recipe["params"].get("live", False))
            result = {"ok": report["ok"], "report": report}
    except BaseException as exc:
        unknown = isinstance(exc, (KeyboardInterrupt, SystemExit)) or \
            (isinstance(exc, ProfileError) and exc.exit_code == 8)
        journal.finish(profile, record["operation_id"], "completion_unknown" if unknown else "failed",
                       {"error_type": type(exc).__name__,
                        "exit_code": exc.exit_code if isinstance(exc, ProfileError) else 130 if unknown else 6})
        raise
    journal.finish(profile, record["operation_id"], "succeeded" if result["ok"] else "failed",
                   journal.compact_result(kind, result))
    return {"profile": profile, "recipe": name, "recipe_revision": revision,
            "kind": kind, "operation_id": record["operation_id"], "result": result}
