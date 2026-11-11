"""Owner-local, typed runtime operations shared by CLI and future Desktop adapters.

The one-shot JSON entry point is for a trusted local host process. It is not an
HTTP API and must not be forwarded from a browser or an inference credential.
All mutations still pass through the existing owner host and operation journal.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from strata_control import doctor, journal, models, profiles, runtime
from strata_control.profiles import ProfileError


MAX_REQUEST_BYTES = 64 * 1024
_ACTIONS = {
    "bench.run.plan": {"path", "suite", "concurrency", "repeat", "request_timeout", "condition"},
    "bench.run": {"path", "suite", "concurrency", "repeat", "request_timeout", "condition",
                  "if_profile_revision", "if_source_sha256", "if_owner_id", "if_instance_id"},
    "bench.job.create": {"job_id", "path", "suite", "concurrency", "repeat", "request_timeout",
                         "condition", "if_profile_revision", "if_source_sha256",
                         "if_owner_id", "if_instance_id"},
    "bench.job": {"job_id"},
    "bench.progress": {"path"},
    "bench.cancel": {"path"},
    "bench.report": {"path"},
    "bench.compare": {"baseline", "candidate"},
    "calibration.plan": set(),
    "calibration.job.create": {"job_id", "action", "path", "exclusive_ack",
                               "if_profile_revision", "if_source_sha256", "if_report_sha256"},
    "calibration.job": {"job_id"},
    "calibration.cancel": {"job_id"},
    "calibration.measure": {"path", "exclusive_ack", "if_profile_revision", "if_source_sha256"},
    "calibration.apply.plan": {"path"},
    "calibration.apply": {"path", "exclusive_ack", "if_profile_revision", "if_source_sha256",
                          "if_report_sha256"},
    "calibration.rollback.plan": {"path"},
    "calibration.rollback": {"path", "exclusive_ack", "if_profile_revision", "if_source_sha256",
                             "if_report_sha256"},
    "components.inspect": {"live"},
    "models.installed": set(),
    "models.inspect": {"model_id"},
    "models.import.plan": {"path"},
    "profile.inspect": set(),
    "profile.change.plan": {"overrides", "source_config", "if_revision"},
    "profile.change": {"overrides", "source_config", "if_revision", "if_source_sha256"},
    "runtime.start.plan": {"shutdown_timeout_s"},
    "runtime.start": {"wait_s", "shutdown_timeout_s", "if_profile_revision", "if_source_sha256"},
    "runtime.status": set(),
    "runtime.verify.plan": {"load"},
    "runtime.verify": {"wait_s", "load", "if_owner_id", "if_instance_id"},
    "runtime.stop.plan": set(),
    "runtime.stop": {"wait_s", "if_owner_id", "if_instance_id"},
    "runtime.restart.plan": {"allow_source_switch", "allow_source_migration"},
    "runtime.restart": {"wait_s", "startup_wait_s", "if_profile_revision",
                        "if_owner_id", "if_instance_id", "allow_source_switch",
                        "allow_source_migration", "if_source_sha256"},
}


def _profile_change_plan(profile: str, overrides: object, source_config: object) -> tuple[dict, dict]:
    """Validate an editable candidate without changing the saved profile."""
    current, revision = profiles.load(profile)
    if not isinstance(overrides, dict):
        raise ProfileError("profile overrides must be an object")
    profiles.validate_overrides(overrides)
    if not isinstance(source_config, str) or not Path(source_config).is_absolute():
        raise ProfileError("source_config must be an absolute path")
    if source_config == current["source_config"]:
        source = source_config
    else:
        proposed_source, source_revision = profiles.propose_source(profile, source_config)
        if source_revision != revision:
            raise ProfileError("profile changed during source preview; retry", 6)
        source = proposed_source["source_config"]
    proposed = {**current, "source_config": source, "overrides": dict(overrides)}
    profiles.validate_profile(proposed, expected_name=profile)
    path = Path(source)
    try:
        raw = path.read_bytes()
        if len(raw) > 1024 * 1024:
            raise ProfileError("run config is larger than 1 MiB", 5)
        cfg = profiles.read_run_config(path)
        if path.read_bytes() != raw:
            raise ProfileError("run config changed during profile preview; retry", 6)
    except OSError as exc:
        raise ProfileError(f"cannot read run config ({type(exc).__name__})", 5) from None
    from strata_cli.cli import _effective
    effective, origins = _effective(cfg, proposed)
    changes = {key: {"before": current["overrides"].get(key), "after": overrides.get(key)}
               for key in sorted(profiles.OVERRIDE_KEYS)
               if current["overrides"].get(key) != overrides.get(key)}
    plan = {"profile": profile, "revision": revision,
            "source_config_before": current["source_config"],
            "source_config_after": source,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "overrides_before": current["overrides"],
            "overrides_after": proposed["overrides"],
            "changes": changes, "effective": effective, "origin": origins,
            "would_save": bool(changes or current["source_config"] != source),
            "running_unchanged": True}
    return plan, proposed


def _seconds(value: object, low: float, high: float, label: str) -> float:
    if (type(value) not in (int, float) or not math.isfinite(value) or
            not low <= value <= high):
        raise ProfileError(f"{label} must be finite seconds in {low:g}..{high:g}")
    return float(value)


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ProfileError(f"{label} must be true or false")
    return value


def _optional_revision(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ProfileError(f"{label} must be a 64-character lowercase SHA-256")
    return value


def _profile_revision(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{16}", value):
        raise ProfileError("if_profile_revision must be a 16-character lowercase revision")
    return value


def _optional_identity(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ProfileError(f"{label} must be a 32-character lowercase ID")
    return value


def _local_report_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or len(value) > 1024 or not Path(value).is_absolute() or \
            any(ord(char) < 32 or ord(char) == 127 or
                0x202a <= ord(char) <= 0x202e or 0x2066 <= ord(char) <= 0x2069 for char in value):
        raise ProfileError(f"{label} must be an absolute local report path under 1024 characters")
    return Path(value)


def _calibration_source(profile: str) -> tuple[str, str]:
    source, saved, revision = profiles.resolve(profile)
    if saved is None or revision is None:
        raise ProfileError("calibration requires a saved profile", 2)
    try:
        raw = source.read_bytes()
    except OSError:
        raise ProfileError("calibration source config is unavailable", 5) from None
    if len(raw) > 1024 * 1024:
        raise ProfileError("calibration source config exceeds 1 MiB", 5)
    return revision, hashlib.sha256(raw).hexdigest()


def _calibration_expected_source(profile: str, params: dict) -> tuple[str, str]:
    revision = _profile_revision(params["if_profile_revision"])
    digest = _optional_revision(params["if_source_sha256"], "if_source_sha256")
    if revision is None or digest is None:
        raise ProfileError("calibration needs the reviewed profile revision and source digest")
    if (revision, digest) != _calibration_source(profile):
        raise ProfileError("calibration source changed after review; review again", 6)
    return revision, digest


def _calibration_report(profile: str, value: object, label: str) -> tuple[Path, dict]:
    from strata_control import calibration
    path = _local_report_path(value, label)
    report = calibration.load_report(path)
    return path, report


def runtime_action(action: str, profile: str, *, plan: bool = False,
                   wait: float = 300, shutdown_timeout: float = 30,
                   startup_wait: float = 300, load: bool = False,
                   if_profile_revision: str | None = None,
                   if_owner_id: str | None = None,
                   if_instance_id: str | None = None,
                   allow_source_switch: bool = False,
                   allow_source_migration: bool = False,
                   if_source_sha256: str | None = None,
                   die_with_pid: int | None = None) -> dict:
    """Execute one validated owner operation and return its CLI/host result."""
    runtime._name(profile)
    if type(plan) is not bool:
        raise ProfileError("plan must be true or false")
    if action == "start":
        # The same ceiling the owner runtime itself enforces, read from it rather than repeated here: a start
        # may wait for a big model longer than ten minutes (#3949). The CLI used to stop at 600 s while
        # runtime.start took 1800 s, so a caller that asked what the runtime allows was refused here first.
        wait = _seconds(wait, 0, runtime.MAX_START_WAIT, "start wait")
        shutdown_timeout = _seconds(shutdown_timeout, 0, 300, "shutdown timeout")
        if_profile_revision = _profile_revision(if_profile_revision)
        if_source_sha256 = _optional_revision(if_source_sha256, "if_source_sha256")
        if plan:
            if if_profile_revision is not None or if_source_sha256 is not None:
                raise ProfileError("start plan cannot include execution preconditions")
            runtime._supported()
            profiles.load(profile)
            launch, source_sha256 = runtime._start_candidate(profile)
            repeated_launch, repeated_digest = runtime._start_candidate(profile)
            if launch != repeated_launch or source_sha256 != repeated_digest:
                raise ProfileError("start candidate changed during preview; retry", 6)
            return {"ok": True, "plan": {"profile": profile, "launch": launch,
                                          "source_sha256": source_sha256,
                                          "shutdown_timeout": shutdown_timeout,
                                          "would_start": False}}
        revision = profiles.load(profile)[1]
        if if_profile_revision is not None or if_source_sha256 is not None:
            runtime._check_start_candidate(profile, if_profile_revision, if_source_sha256)
        operation_id, result = journal.run_direct(profile, "runtime.start", revision,
            {"wait_s": wait, "shutdown_timeout_s": shutdown_timeout,
             "expected_profile_revision": if_profile_revision,
             "expected_source_sha256": if_source_sha256},
            lambda: runtime.start(profile, wait=wait, shutdown_timeout=shutdown_timeout,
                                  expected_profile_revision=if_profile_revision,
                                  expected_source_sha256=if_source_sha256,
                                  die_with_pid=die_with_pid))
    elif action == "status":
        if plan:
            raise ProfileError("status has no plan")
        return runtime.status(profile)
    elif action == "verify":
        load = _boolean(load, "load")
        if_owner_id = _optional_identity(if_owner_id, "if_owner_id")
        if_instance_id = _optional_identity(if_instance_id, "if_instance_id")
        if plan:
            if if_owner_id is not None or if_instance_id is not None:
                raise ProfileError("verify plan cannot include execution preconditions")
            current = runtime.status(profile)
            if not current.get("ok") or current.get("state") != "ready":
                return {"ok": False, "state": current.get("state", "unavailable"),
                        "error": "runtime_not_ready", "exit_code": 6}
            owner, instance = current.get("owner_id"), current.get("instance_id")
            if _optional_identity(owner, "owner_id") is None or \
                    _optional_identity(instance, "instance_id") is None:
                raise ProfileError("runtime verification needs an owned ready instance", 6)
            return {"ok": True, "plan": {"profile": profile, "state": "ready",
                    "owner_id": owner, "instance_id": instance,
                    "model_state": current.get("model_state"),
                    "load": load, "would_load": load and current.get("model_state") == "unloaded",
                    "would_verify": False, "probe": "one_token_generation"}}
        wait = _seconds(wait, 1, 600, "verify wait")
        if (if_owner_id is None) != (if_instance_id is None):
            raise ProfileError("verify requires both reviewed owner and instance IDs")
        if if_owner_id is not None:
            current = runtime.status(profile)
            if current.get("owner_id") != if_owner_id:
                raise ProfileError("owner_changed_after_verify_plan", 6)
            if current.get("instance_id") != if_instance_id:
                raise ProfileError("instance_changed_after_verify_plan", 6)
        revision = profiles.load(profile)[1]
        operation_id, result = journal.run_direct(profile, "runtime.verify", revision,
            {"wait_s": wait, "load": load,
             "expected_owner_id": if_owner_id, "expected_instance_id": if_instance_id},
            lambda: runtime.verify(profile, timeout=wait, load=load,
                                   expected_owner_id=if_owner_id,
                                   expected_instance_id=if_instance_id))
    elif action == "stop":
        wait = _seconds(wait, 1, 600, "stop wait")
        if_owner_id = _optional_identity(if_owner_id, "if_owner_id")
        if_instance_id = _optional_identity(if_instance_id, "if_instance_id")
        if plan:
            if if_owner_id is not None or if_instance_id is not None:
                raise ProfileError("stop plan cannot include execution preconditions")
            return runtime.plan_stop(profile)
        if if_owner_id is not None:
            current = runtime.status(profile)
            if current.get("owner_id") != if_owner_id:
                raise ProfileError("owner_changed_after_stop_plan", 6)
            if current.get("instance_id") != if_instance_id:
                raise ProfileError("instance_changed_after_stop_plan", 6)
        revision = profiles.load(profile)[1]
        operation_id, result = journal.run_direct(profile, "runtime.stop", revision,
            {"wait_s": wait, "expected_owner_id": if_owner_id,
             "expected_instance_id": if_instance_id},
            lambda: runtime.stop(profile, timeout=wait,
                                 expected_owner_id=if_owner_id,
                                 expected_instance_id=if_instance_id))
    elif action == "restart":
        allow_source_switch = _boolean(allow_source_switch, "allow_source_switch")
        allow_source_migration = _boolean(allow_source_migration, "allow_source_migration")
        if_source_sha256 = _optional_revision(if_source_sha256, "if_source_sha256")
        if_profile_revision = _profile_revision(if_profile_revision)
        if_owner_id = _optional_identity(if_owner_id, "if_owner_id")
        if_instance_id = _optional_identity(if_instance_id, "if_instance_id")
        if plan:
            if any(value is not None for value in (if_profile_revision, if_source_sha256,
                                                   if_owner_id, if_instance_id)):
                raise ProfileError("restart plan cannot include execution preconditions")
            return runtime.plan_restart(profile, allow_source_switch=allow_source_switch,
                                        allow_source_migration=allow_source_migration)
        wait = _seconds(wait, 1, 1800, "restart wait")
        # Same ceiling as the restart itself takes below (#3949): a restart waits out the same big-model load a start does.
        startup_wait = _seconds(startup_wait, 1, runtime.MAX_START_WAIT, "startup wait")
        if (if_owner_id is None) != (if_instance_id is None):
            raise ProfileError("restart requires both reviewed owner and instance IDs")
        if (allow_source_switch or allow_source_migration) and not if_source_sha256:
            raise ProfileError("source change needs --if-source-sha256 from --plan")
        if if_owner_id is not None:
            current = runtime.status(profile)
            if current.get("owner_id") != if_owner_id:
                raise ProfileError("owner_changed_after_restart_plan", 6)
            if current.get("instance_id") != if_instance_id:
                raise ProfileError("instance_changed_after_restart_plan", 6)
        revision = profiles.load(profile)[1]
        expected = if_profile_revision or revision
        operation_id, result = journal.run_direct(profile, "runtime.restart", revision,
            {"wait_s": wait, "startup_wait_s": startup_wait,
             "expected_profile_revision": expected,
             "expected_owner_id": if_owner_id,
             "expected_instance_id": if_instance_id,
             "allow_source_switch": allow_source_switch,
             "allow_source_migration": allow_source_migration,
             "expected_source_sha256": if_source_sha256},
            lambda: runtime.restart(profile, timeout=wait, startup_wait=startup_wait,
                                    expected_profile_revision=expected,
                                    expected_owner_id=if_owner_id,
                                    expected_instance_id=if_instance_id,
                                    allow_source_switch=allow_source_switch,
                                    allow_source_migration=allow_source_migration,
                                    expected_source_sha256=if_source_sha256))
    else:
        raise ProfileError("unsupported runtime action")
    return {**result, "operation_id": operation_id}


def dispatch(request: object) -> dict:
    """Validate one versioned local-host request; reject unknown fields/actions."""
    if (not isinstance(request, dict) or set(request) !=
            {"schema_version", "action", "profile", "params"} or
            type(request.get("schema_version")) is not int or request["schema_version"] != 1):
        raise ProfileError("management request needs schema_version 1, action, profile and params")
    action, profile, params = request["action"], request["profile"], request["params"]
    if not isinstance(action, str) or action not in _ACTIONS:
        raise ProfileError("management action is unsupported")
    if not isinstance(profile, str) or not profiles.PROFILE_ID.fullmatch(profile):
        raise ProfileError("management profile name is invalid")
    if not isinstance(params, dict) or (set(params) != _ACTIONS[action] and
            not (action == "bench.run" and set(params) == _ACTIONS[action] | {"job_id"}) and
            not (action in ("calibration.measure", "calibration.apply", "calibration.rollback") and
                 set(params) == _ACTIONS[action] | {"job_id"})):
        raise ProfileError("management action parameters are invalid")
    if action in ("calibration.measure", "calibration.apply", "calibration.rollback") and "job_id" in params:
        from strata_control import calibration_jobs
        result = calibration_jobs.execute(profile, action.removeprefix("calibration."), params)
    elif action in ("bench.run.plan", "bench.run"):
        from strata_control import benchmarking
        checked = {**params, "path": str(_local_report_path(params["path"], "benchmark output path"))}
        if action == "bench.run":
            identities = (_profile_revision(checked["if_profile_revision"]),
                          _optional_revision(checked["if_source_sha256"], "if_source_sha256"),
                          _optional_identity(checked["if_owner_id"], "if_owner_id"),
                          _optional_identity(checked["if_instance_id"], "if_instance_id"))
            if any(identity is None for identity in identities):
                raise ProfileError("benchmark execution needs reviewed profile, source, owner and instance IDs")
            result = benchmarking.run(profile, checked)
        else:
            result = {"ok": True, "plan": benchmarking.plan(profile, checked)}
    elif action == "bench.job.create":
        from strata_control import benchmarking
        checked = {**params, "path": str(_local_report_path(params["path"], "benchmark output path"))}
        result = benchmarking.create_job(profile, checked)
    elif action == "bench.job":
        from strata_control import benchmarking
        result = benchmarking.job_status(profile, params["job_id"])
    elif action == "bench.progress":
        from strata_cli import bench
        from strata_control import benchmarking
        profiles.load(profile)
        target = _local_report_path(params["path"], "benchmark progress path")
        progress = bench.progress(target)
        directory = target.parent if target.name in ("results.json", "progress.json") else target
        result = {"ok": True, "progress": progress,
                  "activity": benchmarking.benchmark_activity(profile, directory),
                  "cancel_requested": benchmarking.cancel_requested(directory)}
    elif action == "bench.cancel":
        from strata_control import benchmarking
        profiles.load(profile)
        result = benchmarking.request_cancel(
            profile, _local_report_path(params["path"], "benchmark cancel path"))
    elif action == "bench.report":
        from strata_cli import bench
        profiles.load(profile)
        report = bench.load(_local_report_path(params["path"], "benchmark report path"))
        result = {"ok": True, "summary": bench.summary(report),
                  "markdown": bench.markdown_report(report)}
    elif action == "bench.compare":
        from strata_cli import bench
        profiles.load(profile)
        baseline = bench.load(_local_report_path(params["baseline"], "baseline report path"))
        candidate = bench.load(_local_report_path(params["candidate"], "candidate report path"))
        comparison = bench.compare(baseline, candidate)
        result = {"ok": True, "comparison": comparison,
                  "markdown": bench.markdown_comparison(comparison)}
    elif action == "calibration.plan":
        from strata_control import calibration
        result = {"ok": True, "plan": calibration.preview(profile)}
    elif action == "calibration.job.create":
        from strata_control import calibration_jobs
        if params["action"] not in calibration_jobs.KINDS:
            raise ProfileError("calibration job action is invalid", 2)
        _calibration_expected_source(profile, params)
        if _boolean(params["exclusive_ack"], "exclusive_ack") is not True:
            raise ProfileError("calibration requires exclusive-access acknowledgement")
        path = _local_report_path(params["path"], "calibration report path")
        report_sha = params["if_report_sha256"]
        if params["action"] == "measure":
            if report_sha is not None or path.exists() or path.is_symlink() or not path.parent.is_dir():
                raise ProfileError("calibration output must be a new file in an existing directory", 2)
        else:
            _, report = _calibration_report(profile, str(path), "calibration report path")
            expected = _optional_revision(report_sha, "if_report_sha256")
            if expected is None or expected != report["body_sha256"]:
                raise ProfileError("calibration report changed after review; review again", 6)
        result = calibration_jobs.create(profile, params["action"], {**params, "path": str(path)})
    elif action == "calibration.job":
        from strata_control import calibration_jobs
        result = calibration_jobs.status(profile, params["job_id"])
    elif action == "calibration.cancel":
        from strata_control import calibration_jobs
        result = calibration_jobs.cancel(profile, params["job_id"])
    elif action == "calibration.measure":
        from strata_control import calibration
        revision, digest = _calibration_expected_source(profile, params)
        if _boolean(params["exclusive_ack"], "exclusive_ack") is not True:
            raise ProfileError("calibration requires exclusive-access acknowledgement")
        path = _local_report_path(params["path"], "calibration report path")
        operation_id, measured = journal.run_direct(profile, "engine.calibrate.measure", revision,
            {"report_path": str(path), "exclusive_ack": True},
            lambda: {"ok": True, "report_path": str(path), "report": calibration.measure(
                profile, path, exclusive_ack=True, say=lambda _: None,
                expected_profile_revision=revision, expected_source_sha256=digest)})
        report = measured["report"]
        result = {"ok": True, "operation_id": operation_id, "path": str(path),
                  "report_id": report["report_id"], "applicable": report["applicable"],
                  "settings": report["settings"]}
    elif action in ("calibration.apply.plan", "calibration.rollback.plan"):
        from strata_control import calibration
        revision, digest = _calibration_source(profile)
        path, report = _calibration_report(profile, params["path"], "calibration report path")
        review = (calibration.review(profile, path) if action == "calibration.apply.plan" else
                  calibration.review_rollback(profile, path))
        result = {"ok": True, "plan": review, "path": str(path),
                  "profile_revision": revision, "source_sha256": digest,
                  "report_sha256": report["body_sha256"]}
    elif action in ("calibration.apply", "calibration.rollback"):
        from strata_control import calibration
        revision, _ = _calibration_expected_source(profile, params)
        if _boolean(params["exclusive_ack"], "exclusive_ack") is not True:
            raise ProfileError("calibration requires exclusive-access acknowledgement")
        path, report = _calibration_report(profile, params["path"], "calibration report path")
        expected_report = _optional_revision(params["if_report_sha256"], "if_report_sha256")
        if expected_report is None or report["body_sha256"] != expected_report:
            raise ProfileError("calibration report changed after review; review again", 6)
        kind = "engine.calibrate.apply" if action == "calibration.apply" else "engine.calibrate.rollback"
        operation_id, outcome = journal.run_direct(profile, kind, revision,
            {"report_path": str(path), "exclusive_ack": True},
            lambda: {"ok": True, ("application" if action == "calibration.apply" else "rollback"):
                calibration.apply_result(profile, path, exclusive_ack=True,
                    expected_report_sha256=expected_report) if action == "calibration.apply" else
                calibration.rollback_result(profile, path, exclusive_ack=True,
                    expected_report_sha256=expected_report)})
        result = {"ok": True, "operation_id": operation_id,
                  "application" if action == "calibration.apply" else "rollback":
                  outcome["application" if action == "calibration.apply" else "rollback"]}
    elif action == "components.inspect":
        live = _boolean(params["live"], "live")
        source, _, _ = profiles.resolve(profile)
        result = {"ok": True, "report": doctor.inspect(source, profile_id=profile, live=live)}
    elif action == "models.installed":
        profiles.load(profile)
        result = {"ok": True, "inventory": models.list_installed()}
    elif action == "models.inspect":
        profiles.load(profile)
        result = {"ok": True, "inspection": models.inspect(params["model_id"])}
    elif action == "models.import.plan":
        profiles.load(profile)
        source = params["path"]
        if not isinstance(source, str) or len(source) > 1024 or not Path(source).is_absolute() or \
                any(ord(char) < 32 or ord(char) == 127 for char in source):
            raise ProfileError("model import path must be an absolute path under 1024 characters")
        result = {"ok": True, "plan": models.plan_import(source)}
    elif action == "profile.inspect":
        saved, revision = profiles.load(profile)
        plan, _ = _profile_change_plan(profile, saved["overrides"], saved["source_config"])
        if plan["revision"] != revision or plan["overrides_before"] != saved["overrides"] or \
                plan["source_config_before"] != saved["source_config"]:
            raise ProfileError("profile changed during inspection; reload it", 6)
        result = {"ok": True, "profile": saved, "revision": revision,
                  "source_sha256": plan["source_sha256"],
                  "effective": plan["effective"], "origin": plan["origin"]}
    elif action == "profile.change.plan":
        expected_revision = _profile_revision(params["if_revision"])
        if expected_revision is None:
            raise ProfileError("profile preview needs its loaded revision")
        plan, _ = _profile_change_plan(profile, params["overrides"], params["source_config"])
        if plan["revision"] != expected_revision:
            raise ProfileError("profile changed since it was loaded; reload before review", 6)
        result = {"ok": True, "plan": plan}
    elif action == "profile.change":
        expected_revision = _profile_revision(params["if_revision"])
        expected_source = _optional_revision(params["if_source_sha256"], "if_source_sha256")
        if expected_revision is None or expected_source is None:
            raise ProfileError("profile save needs its reviewed revision and source digest")
        plan, proposed = _profile_change_plan(profile, params["overrides"], params["source_config"])
        if plan["revision"] != expected_revision or plan["source_sha256"] != expected_source:
            raise ProfileError("profile or source config changed after review; review again", 6)
        saved, revision = profiles.save(profile, proposed, expected_revision=expected_revision,
                                        expected_source_sha256=expected_source)
        result = {"ok": True, "profile": saved, "previous_revision": expected_revision,
                  "revision": revision, "source_sha256": expected_source,
                  "saved": plan["would_save"], "running_unchanged": True}
    elif action == "runtime.start.plan":
        result = runtime_action("start", profile, plan=True,
                                shutdown_timeout=params["shutdown_timeout_s"])
    elif action == "runtime.start":
        if params["if_profile_revision"] is None or params["if_source_sha256"] is None:
            raise ProfileError("Desktop start needs the profile revision and source digest from its plan")
        result = runtime_action("start", profile, wait=params["wait_s"],
                                shutdown_timeout=params["shutdown_timeout_s"],
                                if_profile_revision=params["if_profile_revision"],
                                if_source_sha256=params["if_source_sha256"])
    elif action == "runtime.status":
        result = runtime_action("status", profile)
    elif action == "runtime.verify.plan":
        result = runtime_action("verify", profile, plan=True, load=params["load"])
    elif action == "runtime.verify":
        if params["if_owner_id"] is None or params["if_instance_id"] is None:
            raise ProfileError("Desktop verify needs the owner and instance IDs from its plan")
        result = runtime_action("verify", profile, wait=params["wait_s"], load=params["load"],
                                if_owner_id=params["if_owner_id"],
                                if_instance_id=params["if_instance_id"])
    elif action == "runtime.stop.plan":
        result = runtime_action("stop", profile, plan=True)
    elif action == "runtime.stop":
        if params["if_owner_id"] is None:
            raise ProfileError("Desktop stop needs the owner ID from its plan")
        result = runtime_action("stop", profile, wait=params["wait_s"],
                                if_owner_id=params["if_owner_id"],
                                if_instance_id=params["if_instance_id"])
    elif action == "runtime.restart.plan":
        result = runtime_action("restart", profile, plan=True,
                                allow_source_switch=params["allow_source_switch"],
                                allow_source_migration=params["allow_source_migration"])
    else:
        if any(params[key] is None for key in ("if_profile_revision", "if_source_sha256",
                                              "if_owner_id", "if_instance_id")):
            raise ProfileError("Desktop restart needs profile, source, owner and instance IDs from its plan")
        result = runtime_action("restart", profile, wait=params["wait_s"],
                                startup_wait=params["startup_wait_s"],
                                if_profile_revision=params["if_profile_revision"],
                                if_owner_id=params["if_owner_id"],
                                if_instance_id=params["if_instance_id"],
                                allow_source_switch=params["allow_source_switch"],
                                allow_source_migration=params["allow_source_migration"],
                                if_source_sha256=params["if_source_sha256"])
    return {"schema_version": 1, **result}


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProfileError("management request has a duplicate field")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise ProfileError("management request has a nonfinite number")


def main() -> int:
    """Handle one owner-local JSON request on stdin and one JSON response on stdout."""
    try:
        payload = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if not payload or len(payload) > MAX_REQUEST_BYTES:
            raise ProfileError("management request must be 1..65536 bytes")
        request = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_pairs,
                             parse_constant=_reject_constant)
        with contextlib.redirect_stdout(sys.stderr):
            response = dispatch(request)
        code = 0 if response.get("ok") else response.get("exit_code", 6)
        if code != 0 and "exit_code" not in response:
            response = {**response, "exit_code": code}
    except (ProfileError, ValueError, UnicodeError) as exc:
        code = getattr(exc, "exit_code", 2)
        response = {"schema_version": 1, "ok": False, "error": str(exc), "exit_code": code}
    except OSError as exc:
        code = 6
        response = {"schema_version": 1, "ok": False,
                    "error": f"local operation failed ({type(exc).__name__})", "exit_code": code}
    except KeyboardInterrupt:
        code = 130
        response = {"schema_version": 1, "ok": False,
                    "error": "operation interrupted; completion must be checked in the journal",
                    "exit_code": code}
    print(json.dumps(response, ensure_ascii=False, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
