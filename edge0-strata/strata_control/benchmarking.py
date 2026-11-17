"""Reviewed Desktop launch of the same serving benchmark used by the CLI."""

from __future__ import annotations

from strata_control.private_files import private_mkdir as _private_mkdir
import hashlib
import errno
import os
import re
import stat
import time
from contextlib import contextmanager
from pathlib import Path

from strata_cli import bench
from strata_cli.cli import _launch_plan, _source
from strata_cli.client import ClientError, Endpoint
from strata_control import journal, profiles, runtime
from strata_control.profiles import ProfileError
from strata_control import posix_owner


#: Windows byte-range locks are mandatory: a locked byte cannot be read through any other handle. The lock
#: therefore sits far past the marker bytes, so another process can still read who holds the lease.
WINDOWS_LOCK_OFFSET = 1 << 16

def _lease_file(profile: str, *, create: bool) -> Path | None:
    if not profiles.PROFILE_ID.fullmatch(profile):
        raise ProfileError("benchmark lease needs a saved profile")
    directory = profiles.profile_dir()
    if create:
        _private_mkdir(directory, parents=True, exist_ok=True)
    elif not directory.exists():
        return None
    details = directory.lstat()
    if not stat.S_ISDIR(details.st_mode) or (os.name == "posix" and
            (posix_owner.owner_only_wrong(directory, details))):
        raise ProfileError(posix_owner.owner_only_text("benchmark lease directory must be owner-only", directory), 5)
    return directory / f".bench-{profile}.lock"


def _lease_descriptor(path: Path, *, create: bool) -> int:
    flags = os.O_RDWR | (os.O_CREAT if create else 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    details = os.fstat(descriptor)
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1 or details.st_size > 65 or (os.name == "posix" and
            (posix_owner.owner_only_wrong(path, details))):
        os.close(descriptor)
        raise ProfileError(posix_owner.owner_only_text("benchmark lease file must be owner-only", path), 5)
    return descriptor


def _lock(descriptor: int, *, initialize: bool = False) -> None:
    if initialize and os.fstat(descriptor).st_size == 0:
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, b"0")
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


def _path_digest(path: Path) -> bytes:
    return hashlib.sha256(os.fsencode(os.path.abspath(path))).hexdigest().encode("ascii")


@contextmanager
def benchmark_lease(profile: str, path: Path | None = None):
    """Refuse overlapping local runs for one saved profile across host processes."""
    target = _lease_file(profile, create=True)
    descriptor = _lease_descriptor(target, create=True)
    try:
        try:
            _lock(descriptor, initialize=True)
        except OSError:
            raise ProfileError("another benchmark is running for this saved profile", 6) from None
        try:
            marker = _path_digest(path) if path is not None else b"x" * 64
            os.lseek(descriptor, 1, os.SEEK_SET)
            if os.write(descriptor, marker) != len(marker):
                raise ProfileError("cannot record benchmark lease identity", 5)
            os.fsync(descriptor)
            yield
        finally:
            _unlock(descriptor)
    finally:
        os.close(descriptor)


def benchmark_activity(profile: str, path: Path) -> str:
    """Check the exact saved run's lease, without using PID files or stale timestamps."""
    target = _lease_file(profile, create=False)
    if target is None:
        return "untracked"
    try:
        descriptor = _lease_descriptor(target, create=False)
    except FileNotFoundError:
        return "untracked"
    try:
        if os.name == "nt" and os.fstat(descriptor).st_size == 0:
            return "untracked"
        try:
            _lock(descriptor)
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EACCES) and getattr(exc, "winerror", None) != 33:
                raise ProfileError("cannot inspect benchmark lease", 5) from None
            os.lseek(descriptor, 1, os.SEEK_SET)
            recorded = os.read(descriptor, 65)
            if recorded == _path_digest(path):
                return "active"
            return "other_active" if re.fullmatch(rb"[0-9a-f]{64}", recorded) else "untracked"
        else:
            try:
                os.lseek(descriptor, 1, os.SEEK_SET)
                recorded = os.read(descriptor, 65)
                return "ended" if recorded == _path_digest(path) else "untracked"
            finally:
                _unlock(descriptor)
    finally:
        os.close(descriptor)


def request_cancel(profile: str, path: Path) -> dict:
    """Request cancellation only for the exact active owner-local benchmark."""
    directory = path.parent if path.name in ("results.json", "progress.json") else path
    details = directory.lstat()
    if not stat.S_ISDIR(details.st_mode) or (os.name == "posix" and
            (posix_owner.owner_only_wrong(directory, details))):
        raise ProfileError(posix_owner.owner_only_text("benchmark output directory must be owner-only", directory), 5)
    if benchmark_activity(profile, directory) != "active":
        raise ProfileError("the exact benchmark run is not active", 6)
    if bench.progress(directory)["state"] != "running":
        raise ProfileError("benchmark already saved its final report", 6)
    marker = directory / "cancel.request"
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                             getattr(os, "O_NOFOLLOW", 0), 0o600)
    except FileExistsError:
        details = marker.lstat()
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1 or details.st_size != 1 or (os.name == "posix" and
                (posix_owner.owner_only_wrong(marker, details))):
            raise ProfileError("benchmark cancellation marker is invalid", 5)
    else:
        try:
            os.write(descriptor, b"1")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    current = bench.progress(directory)
    if current["state"] == "report_saved":
        raise ProfileError("benchmark already saved its final report", 6)
    return {"ok": True, "path": str(directory / "results.json"),
            "state": "cancelled" if current["state"] == "cancelled" else "cancel_requested"}


def cancel_requested(directory: Path) -> bool:
    """Read the private marker without treating a stale marker as process liveness."""
    marker = directory / "cancel.request"
    try:
        details = marker.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1 or details.st_size != 1 or (os.name == "posix" and
            (posix_owner.owner_only_wrong(marker, details))):
        raise ProfileError("benchmark cancellation marker is invalid", 5)
    return True


def _settings(profile: str, params: dict) -> tuple[dict, dict, Endpoint, Endpoint, Path]:
    suite, concurrency, repeat = params["suite"], params["concurrency"], params["repeat"]
    timeout, condition = params["request_timeout"], params["condition"]
    if suite not in bench.SUITES:
        raise ProfileError("unknown serving benchmark suite")
    maximum = 64 if suite == "parallel-overload-v1" else 16
    if (type(concurrency) is not int or not 1 <= concurrency <= maximum or
            type(repeat) is not int or not 1 <= repeat <= 5 or
            type(timeout) not in (int, float) or not 1 <= timeout <= 600):
        raise ProfileError("benchmark workload settings are outside supported bounds")
    if condition is not None and (not isinstance(condition, str) or
                                  not re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", condition)):
        raise ProfileError("benchmark condition must be a lowercase name up to 40 characters")
    phase_count = sum(1 if kind in ("warmup", "recovery", "cancel_active",
                                    "cancel_queued", "cancel_prefill") else repeat
                      for kind in bench.SUITES[suite])
    if phase_count * (timeout + 30) > 1700:
        raise ProfileError("benchmark workload exceeds the Desktop job limit; reduce timeout or repeats")
    path = Path(params["path"])
    if path.exists() or path.is_symlink():
        raise ProfileError("benchmark output directory already exists", 2)
    if not path.parent.is_dir():
        raise ProfileError("benchmark output parent directory does not exist", 2)
    source, saved, revision, cfg = _source(profile)
    if saved is None or revision is None:
        raise ProfileError("Desktop benchmark requires a saved profile")
    launch, _ = _launch_plan(profile)
    host = launch["effective"]["host"]
    host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    authority = f"[{host}]" if ":" in host and not host.startswith("[") else host
    url = f"http://{authority}:{launch['effective']['port']}"
    key = os.environ.get("STRATA_API_KEY")
    endpoint = Endpoint(url, key, timeout=10)
    telemetry = Endpoint(url, os.environ.get("STRATA_TELEMETRY_KEY") or key, timeout=3)
    identity = {"profile": profile, "revision": revision, "source_config": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "model_name": cfg.get("model_name"), "gpu": cfg.get("gpu")}
    return launch, identity, endpoint, telemetry, path


def plan(profile: str, params: dict) -> dict:
    launch, identity, endpoint, telemetry, path = _settings(profile, params)
    try:
        snapshot = telemetry.snapshot()
    except ClientError as exc:
        raise ProfileError("benchmark telemetry is unavailable: " + str(exc), 5) from None
    match = bench._managed_match(identity, endpoint.root, snapshot)
    if not match["matched"]:
        raise ProfileError("managed benchmark preflight failed: " + match["reason"], 5)
    owner = runtime.status(profile)
    owner_id = owner.get("owner_id")
    if (owner.get("state") != "ready" or owner.get("instance_id") != match["instance_id"] or
            owner.get("profile_revision") != identity["revision"] or
            not isinstance(owner_id, str) or
            not re.fullmatch(r"[0-9a-f]{32}", owner_id)):
        raise ProfileError("benchmark owner identity is unavailable", 5)
    if (snapshot.get("model") or {}).get("loaded") is not True:
        raise ProfileError("benchmark needs a loaded model", 5)
    work = snapshot.get("work") or {}
    suite, count, repeat = params["suite"], params["concurrency"], params["repeat"]
    if suite == "parallel-overload-v1":
        limit, deadline = work.get("admission_limit"), work.get("queue_timeout_s")
        if (type(limit) is not int or limit < 1 or type(deadline) not in (int, float) or
                deadline <= 0 or count <= limit):
            raise ProfileError("overload benchmark needs configured admission and concurrency above its limit", 5)
    if suite in ("parallel-cancel-v1", "parallel-prefill-cancel-v1"):
        slots = work.get("slots_effective")
        if (type(slots) is not int or not 1 <= slots <= 16 or
                type(work.get("client_disconnect_cancels")) is not int):
            raise ProfileError("cancellation benchmark needs 1..16 slots and disconnect telemetry", 5)
        if suite == "parallel-cancel-v1" and (count < slots or
                type(work.get("max_queue")) is not int or work["max_queue"] < 1 or
                type(work.get("queue_timeout_s")) not in (int, float) or
                work["queue_timeout_s"] < 2):
            raise ProfileError("cancellation benchmark needs a queue, two-second timeout and all slots", 5)
        if suite == "parallel-prefill-cancel-v1" and (slots < 2 or count < 2):
            raise ProfileError("prefill cancellation needs at least two slots and clients", 5)
    requests = {"serving-smoke": 1 + repeat * (2 + count),
                "parallel-mixed-v1": 1 + repeat * (2 + 2 * count),
                "parallel-overload-v1": 2 + repeat * count,
                "parallel-cancel-v1": (work.get("slots_effective") or 0) + 4,
                "parallel-prefill-cancel-v1": 4}[suite]
    return {"profile": profile, "profile_revision": identity["revision"],
            "source_sha256": identity["source_sha256"], "owner_id": owner_id,
            "instance_id": match["instance_id"], "endpoint": endpoint.root,
            "model_id": (snapshot.get("model") or {}).get("id"),
            "suite": suite, "concurrency": count, "repeat": repeat,
            "request_timeout": params["request_timeout"], "condition": params["condition"],
            "path": str(path), "phases": list(bench.SUITES[suite]),
            "planned_requests": requests, "managed_match": True,
            "api_credential_env": "STRATA_API_KEY", "telemetry_credential_env": "STRATA_TELEMETRY_KEY"}


def create_job(profile: str, params: dict) -> dict:
    """Save a bounded receipt before the Desktop launches its benchmark worker."""
    job_id = params["job_id"]
    intent = {key: value for key, value in params.items() if key != "job_id"}
    _settings(profile, intent)
    if benchmark_activity(profile, Path(intent["path"])) in ("active", "other_active"):
        raise ProfileError("another benchmark is running for this saved profile", 6)
    record = journal.begin_direct(profile, "bench.run", intent["if_profile_revision"], intent,
                                  operation_id=job_id)
    return {"ok": True, "job_id": record["operation_id"],
            "path": str(Path(intent["path"]) / "results.json")}


def _job_record(profile: str, job_id: str) -> dict:
    if not isinstance(job_id, str) or not journal.ID.fullmatch(job_id):
        raise ProfileError("benchmark job ID must be a 32-character hex value", 2)
    record = journal.load(profile, job_id)
    if record.get("kind") != "bench.run" or record.get("origin") != "direct":
        raise ProfileError("benchmark job ID does not identify a benchmark", 5)
    return record


def _receipt_result(path: Path, report: dict) -> dict:
    return {"ok": True, "report_path": str(path / "results.json"),
            "run_id": report["run_id"], "instance_id": report["instance_id"],
            "successful": report["successful"], "cancelled": report.get("cancelled", False)}


def _report_matches_job(report: dict, intent: dict, profile: str) -> bool:
    identity = report.get("profile_identity") or {}
    return (isinstance(identity, dict) and
            report.get("suite") == intent["suite"] and
            report.get("condition") == intent["condition"] and
            report.get("instance_id") == intent["if_instance_id"] and
            report.get("planned_phases") == bench._planned_phase_kinds(intent["suite"], intent["repeat"]) and
            identity.get("profile") == profile and
            identity.get("revision") == intent["if_profile_revision"] and
            identity.get("source_sha256") == intent["if_source_sha256"])


def job_status(profile: str, job_id: str) -> dict:
    """Recover a Desktop job from its journal, exact lease and private report."""
    record = _job_record(profile, job_id)
    directory = Path(record["intent"]["path"])
    path = directory / "results.json"
    if directory.exists() or directory.is_symlink():
        details = directory.lstat()
        if not stat.S_ISDIR(details.st_mode) or (os.name == "posix" and
                (posix_owner.owner_only_wrong(directory, details))):
            raise ProfileError(posix_owner.owner_only_text("benchmark output directory must be owner-only", directory), 5)
    activity = benchmark_activity(profile, directory)
    try:
        phase = bench.progress(directory)
    except ProfileError:
        phase = None
    report = None
    if phase is not None and phase["state"] in ("report_saved", "cancelled"):
        candidate = bench.load(directory)
        if _report_matches_job(candidate, record["intent"], profile):
            report = candidate
        else:
            phase = None
        if report is not None and record["state"] == "running":
            try:
                journal.finish(profile, job_id, "succeeded", _receipt_result(directory, report))
            except ProfileError:
                pass  # The benchmark worker may have finished the same receipt.
            record = _job_record(profile, job_id)
    if report is not None:
        state = "cancelled" if report.get("cancelled") else "completed"
        result = {"ok": True, "path": str(path), "successful": report["successful"],
                  "cancelled": report.get("cancelled", False), "run_id": report["run_id"],
                  "instance_id": report["instance_id"], "suite": report["suite"]}
    elif record["state"] == "succeeded" and isinstance(record.get("result"), dict):
        saved = record["result"]
        state = "cancelled" if saved.get("cancelled") else "completed"
        result = {"ok": True, "path": str(path), "successful": saved.get("successful"),
                  "cancelled": saved.get("cancelled"), "run_id": saved.get("run_id"),
                  "instance_id": saved.get("instance_id"), "suite": record["intent"]["suite"]}
    else:
        result = None
        state = ("failed" if record["state"] == "failed" else
                 "running" if activity == "active" else "completion_unknown")
        if (state == "completion_unknown" and record["state"] == "running" and phase is None and
                time.time() - record["started_at"] < 10):
            state = "launching"
    return {"ok": True, "job": {"job_id": job_id, "state": state,
            "started_at": round(record["started_at"] * 1000),
            "finished_at": round(record["finished_at"] * 1000) if record["finished_at"] is not None else None,
            "path": str(path), "result": result,
            "error": ("Benchmark failed before a complete report was saved." if state == "failed" else
                      "Benchmark completion needs inspection at the saved path."
                      if state == "completion_unknown" else None),
            "outcome": "report_recovered" if report is not None and record["state"] != "succeeded"
                       else "owner_journal", "progress": phase,
            "activity": activity, "cancel_requested": cancel_requested(directory) if directory.exists() else False}}


def run(profile: str, params: dict) -> dict:
    job_id = params.get("job_id")
    if job_id is not None:
        record = _job_record(profile, job_id)
        if record["state"] != "running" or record["intent"] != {key: value for key, value in params.items()
                                                    if key != "job_id"}:
            raise ProfileError("benchmark job no longer matches its reviewed request", 6)
    try:
        with benchmark_lease(profile, Path(params["path"])):
            planned = plan(profile, params)
            for key in ("profile_revision", "source_sha256", "owner_id", "instance_id"):
                if planned[key] != params["if_" + key]:
                    raise ProfileError("benchmark target changed after review; review again", 6)
            _, identity, endpoint, telemetry, path = _settings(profile, params)
            try:
                report = bench.run(endpoint, telemetry, path, suite=params["suite"],
                                   concurrency=params["concurrency"], repeat=params["repeat"],
                                   request_timeout=params["request_timeout"], condition=params["condition"],
                                   profile_identity=identity, require_managed_match=True)
            except ClientError as exc:
                raise ProfileError("benchmark connection failed: " + str(exc), exc.exit_code) from None
        if job_id is not None:
            saved_result = _receipt_result(path, report)
            try:
                journal.finish(profile, job_id, "succeeded", saved_result)
            except ProfileError:
                # A concurrent recovery read may have reconciled this final report.
                recorded = _job_record(profile, job_id)
                if recorded["state"] != "succeeded" or recorded["result"] != saved_result:
                    raise
        return {"ok": True, "path": str(path / "results.json"),
                "successful": report["successful"], "summary": bench.summary(report)}
    except BaseException as exc:
        if job_id is not None and _job_record(profile, job_id)["state"] == "running":
            unknown = isinstance(exc, (KeyboardInterrupt, SystemExit)) or \
                (isinstance(exc, ProfileError) and exc.exit_code == 8)
            journal.finish(profile, job_id, "completion_unknown" if unknown else "failed",
                           {"error_type": type(exc).__name__,
                            "exit_code": exc.exit_code if isinstance(exc, ProfileError) else 130 if unknown else 6})
        raise
