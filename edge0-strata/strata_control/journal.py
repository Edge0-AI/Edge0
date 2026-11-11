"""Bounded local records for typed recipe and direct operator runs.

Records contain operation identity and compact outcomes, never credentials,
request bodies or full calibration reports. A running record after a caller
crash has unknown completion; it is not proof that a worker is still running.
"""

from __future__ import annotations

from strata_control.private_files import private_mkdir as _private_mkdir
import json
import math
import os
import re
import stat
import time
import uuid
from pathlib import Path

from strata_control import profiles, recipes
from strata_control.profiles import ProfileError
from strata_control import posix_owner


MAX_RECORDS = 500
MAX_RECORD_BYTES = 64 * 1024
ID = re.compile(r"[0-9a-f]{32}\Z")
TERMINAL = frozenset({"succeeded", "failed", "completion_unknown"})
RESULT_KEYS = frozenset({"ok", "state", "instance_id", "previous_instance_id", "restarted",
                         "source_switched", "source_migrated", "rollback_source_config",
                         "rollback_snapshot_cleanup_pending",
                         "rolled_back", "error", "exit_code", "report_path", "report_id",
                         "applicable", "issue_codes", "error_type", "changed",
                         "application_record", "setup_settings_removed", "generation_ready",
                         "api_identity_verified", "launch_attested", "readiness_state", "verified_now",
                         "run_id", "successful", "cancelled"})
RECORD_KEYS = frozenset({"schema_version", "operation_id", "profile", "recipe", "kind",
                         "recipe_revision", "profile_revision", "owner_pid", "intent",
                         "state", "started_at", "finished_at", "cancellation_policy",
                         "events", "result"})
DIRECT_RECORD_KEYS = (RECORD_KEYS - {"recipe", "recipe_revision"}) | {"origin"}
DIRECT_KINDS = frozenset({"runtime.start", "runtime.stop", "runtime.restart", "runtime.verify",
                          "bench.run",
                          "engine.calibrate.measure", "engine.calibrate.apply", "engine.calibrate.rollback"})


def _directory(profile: str, *, create: bool = False) -> Path:
    directory = recipes._directory(profile, create=create) / ".runs"
    if create:
        _private_mkdir(directory, exist_ok=True)
    if directory.exists() or directory.is_symlink():
        details = directory.lstat()
        if not stat.S_ISDIR(details.st_mode):
            raise ProfileError("operation journal directory must not be a symlink", 5)
        if os.name == "posix" and (posix_owner.owner_only_wrong(directory, details)):
            raise ProfileError(posix_owner.owner_only_text("operation journal directory must be owner-only", directory), 5)
    return directory


def _path(profile: str, operation_id: str, *, create: bool = False) -> Path:
    if not ID.fullmatch(operation_id):
        raise ProfileError("operation ID must be a 32-character hex value")
    return _directory(profile, create=create) / f"{operation_id}.json"


def _bytes(record: dict) -> bytes:
    data = (json.dumps(record, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if len(data) > MAX_RECORD_BYTES:
        raise ProfileError("operation journal record exceeds 64 KiB", 6)
    return data


def _lock_name(profile: str) -> str:
    import hashlib
    return "job" + hashlib.sha256(profile.encode("utf-8")).hexdigest()[:24]


def _validate_direct_intent(kind: str, intent: object) -> dict:
    if kind not in DIRECT_KINDS or not isinstance(intent, dict):
        raise ProfileError("direct operation kind or intent is unsupported", 5)
    expected = ({"bench.run": {"path", "suite", "concurrency", "repeat", "request_timeout",
                               "condition", "if_profile_revision", "if_source_sha256",
                               "if_owner_id", "if_instance_id"},
                 "runtime.start": {"wait_s", "shutdown_timeout_s",
                                    "expected_profile_revision", "expected_source_sha256"},
                 "runtime.stop": {"wait_s", "expected_owner_id", "expected_instance_id"},
                 "runtime.restart": {"wait_s", "startup_wait_s", "expected_profile_revision",
                                     "expected_owner_id", "expected_instance_id",
                                     "allow_source_switch", "allow_source_migration",
                                     "expected_source_sha256"},
                 "runtime.verify": {"wait_s", "load",
                                    "expected_owner_id", "expected_instance_id"}}
                .get(kind, {"report_path", "exclusive_ack"}))
    former_restart = expected - {"expected_owner_id", "expected_instance_id"}
    legacy_restart = (former_restart, former_restart - {"allow_source_migration"},
                      former_restart - {"allow_source_switch", "allow_source_migration",
                                         "expected_source_sha256"})
    legacy_start = kind == "runtime.start" and set(intent) == expected - {
        "expected_profile_revision", "expected_source_sha256"}
    legacy_stop = kind == "runtime.stop" and set(intent) == {"wait_s"}
    legacy_verify = kind == "runtime.verify" and set(intent) == {"wait_s", "load"}
    calibration_job = (kind.startswith("engine.calibrate.") and
                       set(intent) == expected | {"if_source_sha256", "if_report_sha256"})
    if (set(intent) != expected and not calibration_job and not legacy_start and not legacy_stop and not legacy_verify and
            not (kind == "runtime.restart" and set(intent) in legacy_restart)):
        raise ProfileError("direct operation intent has unsupported fields", 5)
    if kind == "bench.run":
        from strata_cli import bench
        path = intent["path"]
        suite = intent["suite"]
        maximum = 64 if suite == "parallel-overload-v1" else 16
        if (not isinstance(path, str) or len(path) > 1024 or not Path(path).is_absolute() or
                any(ord(char) < 32 or ord(char) == 127 or
                    0x202a <= ord(char) <= 0x202e or 0x2066 <= ord(char) <= 0x2069
                    for char in path) or
                not isinstance(suite, str) or suite not in bench.SUITES or
                type(intent["concurrency"]) is not int or
                not 1 <= intent["concurrency"] <= maximum or type(intent["repeat"]) is not int or
                not 1 <= intent["repeat"] <= 5 or type(intent["request_timeout"]) not in (int, float) or
                not math.isfinite(intent["request_timeout"]) or
                not 1 <= intent["request_timeout"] <= 600 or
                (intent["condition"] is not None and
                 (not isinstance(intent["condition"], str) or
                  not re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", intent["condition"]))) or
                not isinstance(intent["if_profile_revision"], str) or
                not re.fullmatch(r"[0-9a-f]{16}", intent["if_profile_revision"]) or
                not isinstance(intent["if_source_sha256"], str) or
                not re.fullmatch(r"[0-9a-f]{64}", intent["if_source_sha256"]) or
                any(not isinstance(intent[key], str) or not re.fullmatch(r"[0-9a-f]{32}", intent[key])
                    for key in ("if_owner_id", "if_instance_id"))):
            raise ProfileError("direct benchmark intent is invalid", 5)
    elif kind.startswith("runtime."):
        # The waits are bounded by the runtime that runs them, so the ceiling is read from it (#3949): a start or a
        # restart may wait for a big model up to MAX_START_WAIT. Repeating the number here is what broke a long start
        # twice over - the CLI validator and this record both stopped it at 600 s on their way to the runtime.
        from strata_control import runtime as _runtime
        start_wait_max = int(_runtime.MAX_START_WAIT)
        ranges = {"wait_s": (0, start_wait_max) if kind == "runtime.start" else
                    (1, 1800) if kind == "runtime.restart" else (1, 600),
                  "shutdown_timeout_s": (0, 300), "startup_wait_s": (1, start_wait_max)}
        for key in set(intent) - {"expected_profile_revision", "expected_source_sha256",
                               "expected_owner_id", "expected_instance_id", "load",
                               "allow_source_switch", "allow_source_migration"}:
            value = intent[key]
            low, high = ranges[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ProfileError("direct runtime intent has an invalid timeout", 5)
        revision = intent.get("expected_profile_revision")
        if kind == "runtime.verify" and type(intent["load"]) is not bool:
            raise ProfileError("direct verification intent has an invalid load flag", 5)
        if "allow_source_switch" in intent and type(intent["allow_source_switch"]) is not bool:
            raise ProfileError("direct restart intent has an invalid source switch flag", 5)
        if "allow_source_migration" in intent and type(intent["allow_source_migration"]) is not bool:
            raise ProfileError("direct restart intent has an invalid source migration flag", 5)
        source_digest = intent.get("expected_source_sha256")
        if kind in ("runtime.start", "runtime.restart") and ((source_digest is not None and
                                           (not isinstance(source_digest, str) or
                                            not re.fullmatch(r"[0-9a-f]{64}", source_digest))) or
                                          ((intent.get("allow_source_switch") is True or
                                            intent.get("allow_source_migration") is True) and
                                           source_digest is None)):
            raise ProfileError("direct runtime intent needs a valid candidate source digest", 5)
        if kind in ("runtime.start", "runtime.restart") and revision is not None and \
                (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{16}", revision)):
            raise ProfileError("direct runtime intent has an invalid profile revision", 5)
        if kind in ("runtime.stop", "runtime.restart", "runtime.verify"):
            for key in ("expected_owner_id", "expected_instance_id"):
                identity = intent.get(key)
                if identity is not None and (not isinstance(identity, str) or
                        not re.fullmatch(r"[0-9a-f]{32}", identity)):
                    raise ProfileError("direct runtime intent has an invalid identity", 5)
        if kind == "runtime.verify" and (intent.get("expected_owner_id") is None) != \
                (intent.get("expected_instance_id") is None):
            raise ProfileError("direct verification intent needs both identities", 5)
    else:
        path = intent["report_path"]
        if not isinstance(path, str) or len(path) > 4096 or not Path(path).is_absolute() or \
                intent["exclusive_ack"] is not True:
            raise ProfileError("direct calibration intent needs an absolute report path and exclusive access", 5)
        if calibration_job and (not isinstance(intent["if_source_sha256"], str) or
                not re.fullmatch(r"[0-9a-f]{64}", intent["if_source_sha256"]) or
                (kind == "engine.calibrate.measure" and intent["if_report_sha256"] is not None) or
                (kind != "engine.calibrate.measure" and
                 (not isinstance(intent["if_report_sha256"], str) or
                  not re.fullmatch(r"[0-9a-f]{64}", intent["if_report_sha256"])))):
            raise ProfileError("direct calibration job intent is invalid", 5)
    return intent


def _load_file(path: Path, profile: str) -> dict:
    try:
        details = path.lstat()
        if not stat.S_ISREG(details.st_mode) or details.st_size > MAX_RECORD_BYTES:
            raise ProfileError("operation record must be a regular file under 64 KiB", 5)
        if os.name == "posix" and (posix_owner.owner_only_wrong(path, details)):
            raise ProfileError(posix_owner.owner_only_text("operation record must be owner-only", path), 5)
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ProfileError("operation record was not found", 5) from None
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, ProfileError):
            raise
        raise ProfileError(f"cannot read operation record ({type(exc).__name__})", 5) from None
    schema = record.get("schema_version") if isinstance(record, dict) else None
    expected_keys = RECORD_KEYS if schema == 1 else DIRECT_RECORD_KEYS if schema == 2 else None
    if not isinstance(record, dict) or expected_keys is None or set(record) != expected_keys or \
            type(record.get("schema_version")) is not int or \
            record.get("operation_id") != path.stem or record.get("profile") != profile or \
            not isinstance(record.get("profile_revision"), str) or \
            not re.fullmatch(r"[0-9a-f]{16}", record["profile_revision"]) or \
            not isinstance(record.get("state"), str) or record["state"] not in {"running", *TERMINAL} or \
            type(record.get("owner_pid")) is not int or record["owner_pid"] < 1 or \
            not isinstance(record.get("started_at"), (int, float)) or \
            not math.isfinite(record["started_at"]) or \
            record.get("cancellation_policy") != "caller_interrupt_may_leave_remote_work_running" or \
            not isinstance(record.get("events"), list) or not 1 <= len(record["events"]) <= 16 or \
            (record["state"] == "running" and record.get("finished_at") is not None) or \
            (record["state"] == "running" and record.get("result") is not None) or \
            (record["state"] in TERMINAL and not isinstance(record.get("finished_at"), (int, float))) or \
            (record["state"] in TERMINAL and not math.isfinite(record["finished_at"])) or \
            (record["state"] in TERMINAL and not isinstance(record.get("result"), dict)):
        raise ProfileError("operation record has an unsupported shape", 5)
    try:
        if schema == 1:
            if not isinstance(record.get("recipe"), str) or \
                    not profiles.PROFILE_ID.fullmatch(record["recipe"]) or \
                    not isinstance(record.get("recipe_revision"), str) or \
                    not re.fullmatch(r"[0-9a-f]{16}", record["recipe_revision"]):
                raise ProfileError("operation record has an invalid recipe revision", 5)
            recipes._validate({"schema_version": 1, "profile": profile, "id": record["recipe"],
                               "kind": record["kind"], "params": record["intent"]},
                              profile, record["recipe"])
        elif record.get("origin") != "direct":
            raise ProfileError("operation record has an invalid origin", 5)
        else:
            _validate_direct_intent(record["kind"], record["intent"])
    except (ProfileError, TypeError):
        raise ProfileError("operation record has an unsupported intent", 5) from None
    if not all(isinstance(event, dict) and set(event) == {"seq", "at", "state"} and
               type(event["seq"]) is int and event["seq"] == index and
               isinstance(event["at"], (int, float)) and math.isfinite(event["at"]) and
               isinstance(event["state"], str) and event["state"] in {"running", *TERMINAL}
               for index, event in enumerate(record["events"], 1)):
        raise ProfileError("operation record has invalid events", 5)
    if record["events"][0]["state"] != "running" or record["events"][-1]["state"] != record["state"]:
        raise ProfileError("operation record has inconsistent events", 5)
    if record["result"] is not None and set(record["result"]) - RESULT_KEYS:
        raise ProfileError("operation record has unsupported result fields", 5)
    return record


def load(profile: str, operation_id: str) -> dict:
    record = _load_file(_path(profile, operation_id), profile)
    if record["state"] == "running":
        return {**record, "completion_unverified": True}
    return record


def list_runs(profile: str, *, limit: int = 20) -> list[dict]:
    if not 1 <= limit <= 100:
        raise ProfileError("operation list limit must be 1..100")
    directory = _directory(profile)
    if not directory.exists():
        return []
    paths = sorted((path for path in directory.glob("*.json") if ID.fullmatch(path.stem)),
                   key=lambda path: path.lstat().st_mtime, reverse=True)
    items = []
    for path in paths[:limit]:
        try:
            record = _load_file(path, profile)
            items.append({"operation_id": record["operation_id"], "recipe": record.get("recipe"),
                          "origin": "recipe" if record["schema_version"] == 1 else "direct",
                          "kind": record["kind"], "state": record["state"],
                          "started_at": record["started_at"], "finished_at": record.get("finished_at"),
                          "completion_unverified": record["state"] == "running"})
        except ProfileError as exc:
            items.append({"operation_id": path.stem, "error": str(exc)})
    return items


def _prune(directory: Path, profile: str) -> None:
    paths = sorted((path for path in directory.glob("*.json") if ID.fullmatch(path.stem)),
                   key=lambda path: path.lstat().st_mtime)
    count = len(paths)
    for path in paths:
        if count < MAX_RECORDS:
            break
        try:
            record = _load_file(path, profile)
        except ProfileError:
            continue
        if record["state"] in TERMINAL:
            path.unlink()
            (directory / f"{path.stem}.progress.json").unlink(missing_ok=True)
            (directory / f"{path.stem}.cancel").unlink(missing_ok=True)
            count -= 1
    if count >= MAX_RECORDS:
        raise ProfileError("operation journal is full; archive completed records before running another action", 5)


def _save_new(profile: str, record: dict) -> dict:
    directory = _directory(profile, create=True)
    path = _path(profile, record["operation_id"])
    with profiles._lock(_lock_name(profile)):
        _prune(directory, profile)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(_bytes(record))
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    return record


def _running_base(profile: str, kind: str, profile_revision: str, intent: dict,
                  operation_id: str | None = None) -> dict:
    now = time.time()
    return {"operation_id": operation_id or uuid.uuid4().hex, "profile": profile, "kind": kind,
            "profile_revision": profile_revision, "owner_pid": os.getpid(), "intent": intent,
            "state": "running", "started_at": now, "finished_at": None,
            "cancellation_policy": "caller_interrupt_may_leave_remote_work_running",
            "events": [{"seq": 1, "at": now, "state": "running"}], "result": None}


def begin(profile: str, recipe: str, kind: str, recipe_revision: str,
          profile_revision: str, *, intent: dict | None = None) -> dict:
    if intent is None:
        intent = {}
    if not isinstance(intent, dict):
        raise ProfileError("operation intent must be typed")
    recipes._validate({"schema_version": 1, "profile": profile, "id": recipe,
                       "kind": kind, "params": intent}, profile, recipe)
    if not isinstance(recipe_revision, str) or not isinstance(profile_revision, str) or \
            not re.fullmatch(r"[0-9a-f]{16}", recipe_revision) or \
            not re.fullmatch(r"[0-9a-f]{16}", profile_revision):
        raise ProfileError("operation revisions must be full 16-character values")
    record = {"schema_version": 1, **_running_base(profile, kind, profile_revision, intent),
              "recipe": recipe, "recipe_revision": recipe_revision}
    return _save_new(profile, record)


def begin_direct(profile: str, kind: str, profile_revision: str, intent: dict,
                 *, operation_id: str | None = None) -> dict:
    _, current_revision = profiles.load(profile)
    _validate_direct_intent(kind, intent)
    if not isinstance(profile_revision, str) or not re.fullmatch(r"[0-9a-f]{16}", profile_revision):
        raise ProfileError("operation profile revision must be a full 16-character value")
    if profile_revision != current_revision:
        raise ProfileError("profile changed before direct operation; preview again", 6)
    if operation_id is not None and (not isinstance(operation_id, str) or
                                     not ID.fullmatch(operation_id)):
        raise ProfileError("operation ID must be a 32-character hex value")
    record = {"schema_version": 2, **_running_base(profile, kind, profile_revision, dict(intent), operation_id),
              "origin": "direct"}
    return _save_new(profile, record)


def finish(profile: str, operation_id: str, state: str, result: dict) -> dict:
    if state not in TERMINAL:
        raise ProfileError("operation terminal state is invalid")
    if not isinstance(result, dict) or set(result) - RESULT_KEYS:
        raise ProfileError("operation result contains unsupported fields")
    path = _path(profile, operation_id)
    with profiles._lock(_lock_name(profile)):
        record = _load_file(path, profile)
        if record["state"] != "running":
            raise ProfileError("operation has already finished", 6)
        now = time.time()
        record.update({"state": state, "finished_at": now, "result": result})
        record["events"].append({"seq": len(record["events"]) + 1, "at": now, "state": state})
        profiles._atomic(path, _bytes(record))
    return record


def run_direct(profile: str, kind: str, profile_revision: str, intent: dict, action) -> tuple[str, dict]:
    """Record one already-authorized CLI action; uncertain interruption stays explicit."""
    record = begin_direct(profile, kind, profile_revision, intent)
    try:
        result = action()
        if not isinstance(result, dict):
            raise ProfileError("direct operation returned an invalid result", 6)
    except BaseException as exc:
        unknown = isinstance(exc, (KeyboardInterrupt, SystemExit)) or \
            (isinstance(exc, ProfileError) and exc.exit_code == 8)
        finish(profile, record["operation_id"], "completion_unknown" if unknown else "failed",
               {"error_type": type(exc).__name__,
                "exit_code": exc.exit_code if isinstance(exc, ProfileError) else 130 if unknown else 6})
        raise
    state = ("succeeded" if result.get("ok", True) else
             "completion_unknown" if result.get("exit_code") == 8 else "failed")
    finish(profile, record["operation_id"], state, compact_direct_result(kind, result))
    return record["operation_id"], result


def compact_direct_result(kind: str, result: dict) -> dict:
    if kind == "bench.run":
        summary = result.get("summary") or {}
        return {"ok": result.get("ok", True), "report_path": result.get("path"),
                "run_id": summary.get("run_id"), "instance_id": summary.get("instance_id"),
                "successful": result.get("successful"), "cancelled": summary.get("cancelled")}
    if kind == "runtime.verify":
        return {key: result.get(key) for key in ("ok", "state", "instance_id", "error", "exit_code",
                                                  "api_identity_verified", "launch_attested", "generation_ready",
                                                  "readiness_state", "verified_now")}
    if kind.startswith("runtime."):
        return {key: result.get(key) for key in ("ok", "state", "instance_id", "previous_instance_id",
                                                  "restarted", "rolled_back", "source_switched",
                                                  "source_migrated", "rollback_source_config",
                                                  "rollback_snapshot_cleanup_pending", "launch_attested",
                                                  "error", "exit_code")}
    if kind == "engine.calibrate.measure":
        report = result.get("report") or {}
        return {"ok": result.get("ok", True), "report_path": result.get("report_path"),
                "report_id": report.get("report_id"), "applicable": report.get("applicable")}
    if kind == "engine.calibrate.apply":
        application = result.get("application") or {}
        return {"ok": result.get("ok", True), "report_id": application.get("report_id"),
                "changed": application.get("changed"),
                "application_record": application.get("application_record")}
    rollback = result.get("rollback") or {}
    return {"ok": result.get("ok", True), "report_id": rollback.get("report_id"),
            "changed": rollback.get("changed"),
            "setup_settings_removed": rollback.get("setup_settings_removed")}


def compact_result(kind: str, result: dict) -> dict:
    """Only reviewed, bounded fields reach the persistent journal."""
    if kind == "runtime.restart":
        return {key: result.get(key) for key in ("ok", "state", "instance_id", "previous_instance_id",
                                                  "restarted", "rolled_back", "source_switched",
                                                  "source_migrated", "rollback_source_config",
                                                  "rollback_snapshot_cleanup_pending", "launch_attested",
                                                  "error", "exit_code")}
    if kind == "engine.calibrate.measure":
        report = result.get("report") or {}
        return {"ok": result.get("ok"), "report_path": result.get("report_path"),
                "report_id": report.get("report_id"), "applicable": report.get("applicable")}
    report = result.get("report") or {}
    return {"ok": result.get("ok"), "issue_codes": [item.get("code") for item in report.get("issues", [])][:50]}
