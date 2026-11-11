"""Owner-local receipts for reviewed Desktop calibration operations."""

from __future__ import annotations

import errno
import json
import math
import os
import stat
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from strata_control import calibration, journal, profiles
from strata_control.profiles import ProfileError
from strata_control import posix_owner


KINDS = {"measure": "engine.calibrate.measure", "apply": "engine.calibrate.apply",
         "rollback": "engine.calibrate.rollback"}
STAGES = frozenset({"preflight", "loading", "pcie_share", "draft_floor", "confirmation",
                    "cpu_workers", "expert_tier", "saving_report"})


#: Windows byte-range locks are mandatory: a locked byte cannot be read through any other handle. The lock
#: therefore sits far past the marker bytes, so another process can still read who holds the lease.
WINDOWS_LOCK_OFFSET = 1 << 16

def _directory(profile: str) -> Path:
    return journal._directory(profile, create=True)


def _lease_path(profile: str) -> Path:
    return _directory(profile) / ".calibration-worker.lock"


def _open_lease(profile: str, *, create: bool) -> int | None:
    flags = os.O_RDWR | (os.O_CREAT if create else 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(_lease_path(profile), flags, 0o600)
    except FileNotFoundError:
        return None
    details = os.fstat(descriptor)
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1 or details.st_size > 32 or \
            (os.name == "posix" and
             (posix_owner.owner_only_wrong(_lease_path(profile), details))):
        os.close(descriptor)
        raise ProfileError(posix_owner.owner_only_text("calibration worker lease must be owner-only", _lease_path(profile)), 5)
    return descriptor


def _try_lock(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt
        os.lseek(descriptor, WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt
        os.lseek(descriptor, WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(descriptor, fcntl.LOCK_UN)


@contextmanager
def _worker_lease(profile: str, job_id: str):
    descriptor = _open_lease(profile, create=True)
    if descriptor is None:
        raise ProfileError("calibration worker lease directory is unavailable", 5)
    try:
        try:
            _try_lock(descriptor)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EACCES) or getattr(exc, "winerror", None) == 33:
                raise ProfileError("another calibration worker is running for this profile", 6) from None
            raise
        try:
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.write(descriptor, job_id.encode("ascii")) != 32:
                raise ProfileError("cannot record calibration worker identity", 5)
            os.fsync(descriptor)
            yield
        finally:
            _unlock(descriptor)
    finally:
        os.close(descriptor)


def _activity(profile: str, job_id: str) -> str:
    descriptor = _open_lease(profile, create=False)
    if descriptor is None:
        return "untracked"
    try:
        try:
            _try_lock(descriptor)
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EACCES) and getattr(exc, "winerror", None) != 33:
                raise
            os.lseek(descriptor, 0, os.SEEK_SET)
            marker = os.read(descriptor, 32)
            return "active" if marker == job_id.encode("ascii") else "other_active"
        else:
            try:
                os.lseek(descriptor, 0, os.SEEK_SET)
                return "ended" if os.read(descriptor, 32) == job_id.encode("ascii") else "untracked"
            finally:
                _unlock(descriptor)
    finally:
        os.close(descriptor)


def _cancel_path(profile: str, job_id: str) -> Path:
    return _directory(profile) / f"{job_id}.cancel"


def _cancel_requested(profile: str, job_id: str) -> bool:
    path = _cancel_path(profile, job_id)
    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1 or details.st_size != 1 or \
            (os.name == "posix" and
             (posix_owner.owner_only_wrong(path, details))):
        raise ProfileError("calibration cancellation marker is invalid", 5)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        current = os.fstat(descriptor)
        if current.st_ino != details.st_ino or os.read(descriptor, 2) != b"1":
            raise ProfileError("calibration cancellation marker changed", 5)
    finally:
        os.close(descriptor)
    return True


@contextmanager
def _cancel_watcher(profile: str, job_id: str):
    requested = threading.Event()
    stop = threading.Event()

    def watch():
        while not stop.wait(0.2):
            try:
                if _cancel_requested(profile, job_id):
                    requested.set()
                    return
            except (OSError, ProfileError):
                requested.set()  # Invalid private state stops the measurement safely.
                return

    thread = threading.Thread(target=watch, name="strata-calibration-cancel", daemon=True)
    thread.start()
    try:
        yield requested
    finally:
        stop.set()
        thread.join(timeout=1)


def _progress_path(profile: str, job_id: str) -> Path:
    return _directory(profile) / f"{job_id}.progress.json"


def _save_progress(profile: str, job_id: str, stage: str, completed: int, total: int) -> None:
    if stage not in STAGES or type(completed) is not int or type(total) is not int or \
            not 0 <= completed <= total <= 1000:
        raise ProfileError("calibration progress is invalid", 5)
    value = {"schema_version": 1, "job_id": job_id, "stage": stage,
             "completed": completed, "total": total, "updated_at": time.time()}
    profiles._atomic(_progress_path(profile, job_id),
                     (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))


def _load_progress(profile: str, job_id: str) -> dict | None:
    path = _progress_path(profile, job_id)
    try:
        details = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1 or details.st_size > 1024 or \
            (os.name == "posix" and
             (posix_owner.owner_only_wrong(path, details))):
        raise ProfileError(posix_owner.owner_only_text("calibration progress file must be owner-only", path), 5)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise ProfileError("calibration progress file is invalid", 5) from None
    if not isinstance(value, dict) or set(value) != {"schema_version", "job_id", "stage",
                                                 "completed", "total", "updated_at"} or \
            value["schema_version"] != 1 or value["job_id"] != job_id or \
            value["stage"] not in STAGES or type(value["completed"]) is not int or \
            type(value["total"]) is not int or not 0 <= value["completed"] <= value["total"] <= 1000 or \
            type(value["updated_at"]) not in (int, float) or not math.isfinite(value["updated_at"]):
        raise ProfileError("calibration progress file is invalid", 5)
    return value


def _intent(params: dict) -> dict:
    return {"report_path": params["path"], "exclusive_ack": True,
            "if_source_sha256": params["if_source_sha256"],
            "if_report_sha256": params.get("if_report_sha256")}


def create(profile: str, action: str, params: dict) -> dict:
    """Create a receipt after management has checked every reviewed precondition."""
    if action not in KINDS or not isinstance(params.get("job_id"), str) or \
            not journal.ID.fullmatch(params["job_id"]):
        raise ProfileError("calibration job ID or action is invalid", 2)
    if _activity(profile, params["job_id"]) in ("active", "other_active"):
        raise ProfileError("another calibration worker is running for this profile", 6)
    record = journal.begin_direct(profile, KINDS[action], params["if_profile_revision"],
                                  _intent(params), operation_id=params["job_id"])
    return {"ok": True, "job_id": record["operation_id"], "action": action,
            "path": record["intent"]["report_path"]}


def _record(profile: str, job_id: str) -> dict:
    if not isinstance(job_id, str) or not journal.ID.fullmatch(job_id):
        raise ProfileError("calibration job ID must be a 32-character hex value", 2)
    record = journal.load(profile, job_id)
    if record.get("kind") not in KINDS.values() or record.get("origin") != "direct" or \
            "if_source_sha256" not in record["intent"]:
        raise ProfileError("calibration job ID does not identify a Desktop calibration", 5)
    return record


def run(profile: str, action: str, params: dict, work) -> tuple[str, dict]:
    job_id = params["job_id"]
    record = _record(profile, job_id)
    if record["state"] != "running" or record["kind"] != KINDS[action] or \
            record["profile_revision"] != params["if_profile_revision"] or \
            record["intent"] != _intent(params):
        raise ProfileError("calibration job no longer matches its reviewed request", 6)
    with _worker_lease(profile, job_id):
        return _run_owned(profile, action, job_id, work)


def _run_owned(profile: str, action: str, job_id: str, work) -> tuple[str, dict]:
    try:
        result = work()
        saved = journal.compact_direct_result(KINDS[action], result)
        try:
            journal.finish(profile, job_id, "succeeded", saved)
        except ProfileError:
            # A recovery read can reconcile a completed measurement first.
            current = _record(profile, job_id)
            if current["state"] != "succeeded" or current["result"] != saved:
                raise
        return job_id, result
    except BaseException as exc:
        if _record(profile, job_id)["state"] == "running":
            unknown = isinstance(exc, (KeyboardInterrupt, SystemExit)) or \
                (isinstance(exc, ProfileError) and exc.exit_code == 8)
            cancelled = (action == "measure" and isinstance(exc, ProfileError) and
                         exc.exit_code == 130)
            journal.finish(profile, job_id, "completion_unknown" if unknown else "failed",
                           {"error_type": type(exc).__name__,
                            "cancelled": cancelled,
                            "exit_code": exc.exit_code if isinstance(exc, ProfileError) else
                            130 if unknown else 6})
        raise


def execute(profile: str, action: str, params: dict) -> dict:
    """Run every preflight inside the receipt, including stale review rejection."""
    from strata_control import management

    def work() -> dict:
        if action == "measure":
            try:
                _save_progress(profile, params["job_id"], "preflight", 0, 1)
            except (OSError, ProfileError):
                pass  # Progress is supplementary; the reviewed action still runs.
        revision, digest = management._calibration_expected_source(profile, params)
        if management._boolean(params["exclusive_ack"], "exclusive_ack") is not True:
            raise ProfileError("calibration requires exclusive-access acknowledgement")
        path = management._local_report_path(params["path"], "calibration report path")
        if action == "measure":
            with _cancel_watcher(profile, params["job_id"]) as cancel_event:
                def progress(stage: str, completed: int, total: int):
                    try:
                        _save_progress(profile, params["job_id"], stage, completed, total)
                    except (OSError, ProfileError):
                        pass

                progress("preflight", 1, 1)
                report = calibration.measure(profile, path, exclusive_ack=True, say=lambda _: None,
                                             expected_profile_revision=revision,
                                             expected_source_sha256=digest, job_id=params["job_id"],
                                             progress=progress, cancel_event=cancel_event)
            return {"ok": True, "report_path": str(path), "report": report}
        _, report = management._calibration_report(profile, str(path), "calibration report path")
        expected = management._optional_revision(params["if_report_sha256"], "if_report_sha256")
        if expected is None or report["body_sha256"] != expected:
            raise ProfileError("calibration report changed after review; review again", 6)
        if action == "apply":
            return {"ok": True, "application": calibration.apply_result(
                profile, path, exclusive_ack=True, expected_report_sha256=expected)}
        return {"ok": True, "rollback": calibration.rollback_result(
            profile, path, exclusive_ack=True, expected_report_sha256=expected)}

    operation_id, outcome = run(profile, action, params, work)
    if action == "measure":
        report = outcome["report"]
        return {"ok": True, "operation_id": operation_id, "path": params["path"],
                "report_id": report["report_id"], "applicable": report["applicable"],
                "settings": report["settings"]}
    return {"ok": True, "operation_id": operation_id,
            "application" if action == "apply" else "rollback":
            outcome["application" if action == "apply" else "rollback"]}


def cancel(profile: str, job_id: str) -> dict:
    """Request a stop only for the exact active measurement worker."""
    status(profile, job_id)  # Reconcile a finished report before accepting a new request.
    record = _record(profile, job_id)
    if record["kind"] != KINDS["measure"]:
        raise ProfileError("only calibration measurement can be cancelled", 6)
    if record["state"] != "running" or _activity(profile, job_id) != "active":
        raise ProfileError("the exact calibration measurement is not active", 6)
    path = _cancel_path(profile, job_id)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                             getattr(os, "O_NOFOLLOW", 0), 0o600)
    except FileExistsError:
        _cancel_requested(profile, job_id)
    else:
        try:
            os.write(descriptor, b"1")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return {"ok": True, "job_id": job_id, "state": "cancel_requested"}


def status(profile: str, job_id: str) -> dict:
    """Report a bounded receipt; only an exact report may recover measurement success."""
    record = _record(profile, job_id)
    action = next(name for name, kind in KINDS.items() if kind == record["kind"])
    path = Path(record["intent"]["report_path"])
    report = None
    recovered = False
    if action == "measure" and record["state"] in ("running", "succeeded"):
        try:
            candidate = calibration.load_report(path)
            identity = candidate["identity"]
            if (candidate.get("operation_id") == job_id and
                    type(candidate.get("applicable")) is bool and
                    identity.get("profile_id") == profile and
                    identity.get("profile_revision") == record["profile_revision"] and
                    identity.get("source_sha256") == record["intent"]["if_source_sha256"]):
                report = candidate
        except (ProfileError, OSError):
            pass
        if report is not None and record["state"] == "running":
            try:
                journal.finish(profile, job_id, "succeeded",
                               {"ok": True, "report_path": str(path),
                                "report_id": report["report_id"],
                                "applicable": report["applicable"]})
                recovered = True
            except ProfileError:
                pass
            record = _record(profile, job_id)
    result = None
    if record["state"] == "succeeded":
        saved = record["result"]
        result = {"ok": True, "action": action, "operation_id": job_id,
                  "report_id": saved.get("report_id")}
        if action == "measure":
            result.update(path=str(path), applicable=saved.get("applicable"),
                          settings=report["settings"] if report is not None else {})
        else:
            result.update(changed=saved.get("changed"))
            if action == "apply":
                result["application_record"] = saved.get("application_record")
            else:
                result["setup_settings_removed"] = saved.get("setup_settings_removed")
    activity = _activity(profile, job_id)
    progress = _load_progress(profile, job_id) if action == "measure" else None
    cancel_requested = _cancel_requested(profile, job_id) if action == "measure" else False
    state = record["state"]
    if state == "running":
        state = "running" if activity == "active" else \
                "launching" if time.time() - record["started_at"] < 10 else "completion_unknown"
    elif state == "failed" and record["result"].get("cancelled") is True:
        state = "cancelled"
    return {"ok": True, "job": {"job_id": job_id, "action": action, "state": state,
            "started_at": round(record["started_at"] * 1000),
            "finished_at": round(record["finished_at"] * 1000) if record["finished_at"] is not None else None,
            "result": result,
            "error": "Calibration failed; inspect the saved operation and report." if state == "failed" else
                     "Calibration completion needs inspection before retrying." if state == "completion_unknown" else None,
            "outcome": "report_recovered" if recovered else "owner_journal",
            "activity": activity, "progress": progress, "cancel_requested": cancel_requested}}
