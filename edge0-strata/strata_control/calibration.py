"""Explicit measure, review and apply steps around Strata's existing calibration.

The current suite measures decode speed. It is not a four-GPU serving benchmark.
Its report binds a candidate to the run config, engine binary and observed local
hardware; applying it is a separate operation with a backup and smoke generation.
"""

from __future__ import annotations

from strata_control.private_files import private_mkdir as _private_mkdir
import copy
import hashlib
import json
import math
import os
import re
import socket
import stat
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from strata_control import profiles
from strata_control.doctor import inspect as inspect_components
from strata_control.profiles import ProfileError
from strata_control import atomic as _atomic_mod
from strata_control import posix_owner


SUITE = "legacy-decode-v1"
MAX_REPORT_BYTES = 2 * 1024 * 1024
MAX_APPLICATION_BYTES = 8 * 1024
MAX_AGE_SECONDS = 30 * 24 * 3600
ALLOWED_SETTINGS = {"--pcie-frac", "--spec-min-p", "--pool-workers",
                    "--adapt-every", "--adapt-swaps", "--adapt-decay"}


def _full_folder(name: str):
    """`setup` and `tools.calibrate` belong to a full Strata folder; the Terminal archive carries neither (it never
    installs or builds). Use them when they are importable, or when STRATA_INSTALL_DIR names a folder that has both;
    otherwise refuse in words instead of a traceback."""
    import importlib
    import sys
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    raw = os.environ.get("STRATA_INSTALL_DIR")
    if raw:
        folder = Path(raw).expanduser()
        if (folder / "setup.py").is_file() and (folder / "tools" / "calibrate.py").is_file():
            if str(folder) not in sys.path:
                sys.path.append(str(folder))
            try:
                return importlib.import_module(name)
            except ImportError:
                pass
    raise ProfileError(
        "engine calibrate needs a full Strata folder (setup.py and tools/calibrate.py); this terminal archive has "
        "neither. Set STRATA_INSTALL_DIR to such a folder, or use `strata tune run --mode api` to tune from the "
        "archive", 5) from None


def _tools_available() -> bool:
    """False in the Terminal archive without STRATA_INSTALL_DIR: `--measure` would be refused (see `_full_folder`)."""
    import importlib.util
    try:
        if importlib.util.find_spec("setup") and importlib.util.find_spec("tools.calibrate"):
            return True
    except (ImportError, ValueError):
        pass
    raw = os.environ.get("STRATA_INSTALL_DIR")
    if raw:
        folder = Path(raw).expanduser()
        return (folder / "setup.py").is_file() and (folder / "tools" / "calibrate.py").is_file()
    return False


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _target(ref: str) -> tuple[Path, dict | None, str | None, dict]:
    path, profile, revision = profiles.resolve(ref)
    return path, profile, revision, profiles.read_run_config(path)


def preview(ref: str) -> dict:
    """Inspect a proposed decode sweep without starting the engine or writing files."""
    path, profile, revision, cfg = _target(ref)
    components = inspect_components(path, profile_id=profile["id"] if profile else None)
    source_bytes = path.read_bytes()
    if json.loads(source_bytes.decode("utf-8-sig")) != cfg:
        raise ProfileError("run config changed during calibration preview; retry", 6)
    return {"profile": profile["id"] if profile else None, "profile_revision": revision,
            "source_config": str(path), "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "model_name": cfg.get("model_name"), "backend": cfg.get("backend") or "unverified",
            "devices_selected": cfg.get("gpu"), "tool": "tools/calibrate.py",
            "tool_available": _tools_available(),
            "estimated_minutes": [15, 30], "requires_exclusive_engine": True,
            "settings_measured": ["--pcie-frac", "--spec-min-p", "--pool-workers",
                                  "--adapt-every", "--adapt-swaps", "--adapt-decay"],
            "component_preflight": components, "hardware_identity": "unverified_until_measurement",
            "would_start_engine": False, "would_write_config": False}


def _resolved(raw: str, cfg: dict, path: Path) -> Path:
    from strata_control import engine_component
    if engine_component.is_reference(raw):
        return Path(engine_component.expand(raw, cfg))
    value = Path(raw).expanduser()
    if value.is_absolute():
        return value
    cwd = cfg.get("cwd")
    return (Path(cwd) if isinstance(cwd, str) and cwd else path.parent) / value


def _runtime_cfg(cfg: dict, profile: dict | None) -> dict:
    """Apply the saved engine concurrency settings without changing setup's config."""
    effective = copy.deepcopy(cfg)
    if profile is None:
        return effective
    overrides = profile["overrides"]
    if "parallel" in overrides:
        if any(flag in effective["args"] for flag in ("--batch", "--slots")):
            raise ProfileError("parallel profile override conflicts with engine args")
        effective["parallel"] = overrides["parallel"]
    if "batch_groups" in overrides:
        if "--batch-groups" in effective["args"]:
            raise ProfileError("batch_groups profile override conflicts with engine args")
        effective["args"] += ["--batch-groups", str(overrides["batch_groups"])]
    return effective


def _hardware_key(cfg: dict) -> str:
    return _full_folder("setup").hardware_key(cfg)


def _identity(path: Path, cfg: dict, profile: dict | None, revision: str | None) -> dict:
    cal = _full_folder("tools.calibrate")
    engine = _resolved(cfg["exe"], cfg, path)
    components = inspect_components(path, profile_id=profile["id"] if profile else None)
    if not components["ok"]:
        raise ProfileError("required engine/model components are missing; run strata doctor --profile first", 5)
    files = []
    for item in components["components"]:
        if item.get("required") and item.get("kind") == "file" and item.get("name") != "engine":
            file_path = Path(item["path"])
            try:
                details = file_path.stat()
            except OSError:
                raise ProfileError(f"component changed during calibration: {item['name']}", 5) from None
            files.append({"name": item["name"], "path": str(file_path),
                          "size": details.st_size, "modified_ns": details.st_mtime_ns})
    return {"source_sha256": _digest(path), "engine_sha256": _digest(engine),
            "suite_sha256": _digest(Path(cal.__file__)), "hardware_key": _hardware_key(cfg),
            "model_name": cfg.get("model_name"), "backend": cfg.get("backend"), "gpu": cfg.get("gpu"),
            "profile_id": profile["id"] if profile else None, "profile_revision": revision,
            "required_file_metadata": files, "model_integrity": "metadata_only_unverified"}


def _validate_settings(settings: object) -> dict[str, str]:
    if not isinstance(settings, dict) or set(settings) - ALLOWED_SETTINGS:
        raise ProfileError("calibration report has unsupported settings", 5)
    out = {}
    for flag, raw in settings.items():
        if not isinstance(raw, str) or len(raw) > 32:
            raise ProfileError(f"calibration setting {flag} must be a short string", 5)
        try:
            number = float(raw)
        except ValueError:
            raise ProfileError(f"calibration setting {flag} must be numeric", 5) from None
        if not number == number or number in (float("inf"), float("-inf")):
            raise ProfileError(f"calibration setting {flag} must be finite", 5)
        if flag in ("--pcie-frac", "--spec-min-p", "--adapt-decay"):
            if not 0 <= number <= 1:
                raise ProfileError(f"calibration setting {flag} must be in 0..1", 5)
        else:
            if not raw.isdecimal() or not 1 <= int(raw) <= 8192:
                raise ProfileError(f"calibration setting {flag} must be a positive whole number", 5)
        out[flag] = raw
    return out


def _port_active(cfg: dict) -> bool:
    port = cfg.get("port", 8095)
    if type(port) is not int or not 1 <= port <= 65535:
        return False
    host = cfg.get("host") or "127.0.0.1"
    address = "127.0.0.1" if host in ("0.0.0.0", "::", "localhost") else host
    try:
        with socket.create_connection((address, port), timeout=0.2):
            return True
    except OSError:
        return False


def _exclusive_preflight(cfg: dict, profile: dict | None = None) -> None:
    endpoints = [cfg]
    overrides = profile["overrides"] if profile else {}
    if "host" in overrides or "port" in overrides:
        endpoints.append({**cfg, **{key: overrides[key] for key in ("host", "port") if key in overrides}})
    if any(_port_active(endpoint) for endpoint in endpoints):
        raise ProfileError("a server is listening on this profile's port; stop it before calibration", 6)


@contextmanager
def _managed_exclusive():
    """Keep new owner hosts out for the full calibration engine action."""
    if os.name not in ("posix", "nt"):
        yield
        return
    from strata_control import runtime
    lease = runtime.acquire_activity_lease(shared=False)
    try:
        active = runtime.active_managed_instances()
        if active:
            raise ProfileError("stop managed Strata runtime instances before calibration: " +
                               ", ".join(item["profile"] for item in active), 6)
        yield
    finally:
        os.close(lease)


def _write_new(path: Path, payload: bytes) -> None:
    if not path.parent.is_dir():
        raise ProfileError("report output directory does not exist", 5)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _report_bytes(report: dict) -> bytes:
    try:
        payload = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError):
        raise ProfileError("calibration returned an invalid report", 6) from None
    if len(payload) > MAX_REPORT_BYTES:
        raise ProfileError("calibration report exceeds 2 MiB", 6)
    return payload


def _body_sha256(report: dict) -> str:
    body = {key: value for key, value in report.items() if key != "body_sha256"}
    try:
        encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
    except (TypeError, ValueError):
        raise ProfileError("calibration report has invalid values", 5) from None
    return hashlib.sha256(encoded).hexdigest()


def _lock_name(path: Path) -> str:
    return "cal" + hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:24]


def measure(ref: str, output: Path, *, exclusive_ack: bool = False, say=print, runner=None,
            expected_profile_revision: str | None = None,
            expected_source_sha256: str | None = None, job_id: str | None = None,
            progress=None, cancel_event=None) -> dict:
    """Run the existing decode sweep, leaving the run config unchanged."""
    if not exclusive_ack:
        raise ProfileError("calibration measure requires an exclusive-access acknowledgement")
    path, profile, revision, cfg = _target(ref)
    with profiles._lock(_lock_name(path)), _managed_exclusive():
        current_path, profile, revision, cfg = _target(ref)
        if current_path != path or (expected_profile_revision is not None and
                                    revision != expected_profile_revision) or \
                (expected_source_sha256 is not None and
                 _digest(current_path) != expected_source_sha256):
            raise ProfileError("calibration source changed after review; review again", 6)
        return _measure_locked(ref, path, profile, revision, cfg, output, say=say, runner=runner,
                               job_id=job_id, progress=progress, cancel_event=cancel_event)


def _measure_locked(ref: str, path: Path, profile: dict | None, revision: str | None, cfg: dict,
                    output: Path, *, say, runner, job_id: str | None = None,
                    progress=None, cancel_event=None) -> dict:
    _exclusive_preflight(cfg, profile)
    if output.exists() or output.is_symlink():
        raise ProfileError("calibration report path already exists", 2)
    if not output.parent.is_dir():                    # before the 19 minute sweep, not after it (#3187)
        raise ProfileError("report output directory does not exist", 5)
    identity = _identity(path, cfg, profile, revision)
    cal =_full_folder("tools.calibrate")
    execute = runner or cal.run
    runner_cfg = _runtime_cfg(cfg, profile)
    runner_cfg["exe"] = str(_resolved(cfg["exe"], cfg, path))
    runner_cfg["tokenizer"] = str(_resolved(cfg["tokenizer"], cfg, path))
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise cal.CalibrationCancelled("calibration cancelled")
        result = (execute(runner_cfg, say=say) if progress is None and cancel_event is None else
                  execute(runner_cfg, say=say, progress=progress, cancel_event=cancel_event))
    except cal.CalibrationCancelled:
        raise ProfileError("calibration cancelled by operator; config unchanged", 130) from None
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise ProfileError(f"calibration measurement failed ({type(exc).__name__}); config unchanged", 6) from None
    if not isinstance(result, dict) or not isinstance(result.get("report"), dict):
        raise ProfileError("calibration returned no measurement report", 6)
    settings = _validate_settings(result.get("settings"))
    try:
        end_path, end_profile, end_revision, end_cfg = _target(ref)
        unchanged = path == end_path and identity == _identity(end_path, end_cfg, end_profile, end_revision)
    except ProfileError:
        unchanged = False
    report = {"schema_version": 1, "kind": "strata.engine.calibration", "suite": SUITE,
              "report_id": uuid.uuid4().hex, "created_at": time.time(), "source_config": str(path),
              "identity": identity, "applicable": unchanged, "settings": settings,
              "measurements": result["report"]}
    if job_id is not None:
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ProfileError("calibration job ID is invalid", 5)
        report["operation_id"] = job_id
    report["body_sha256"] = _body_sha256(report)
    if progress is not None:
        progress("saving_report", 0, 1)
    _write_new(output, _report_bytes(report))
    if progress is not None:
        progress("saving_report", 1, 1)
    return report


def load_report(path: Path) -> dict:
    try:
        if not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_size > MAX_REPORT_BYTES:
            raise ProfileError("calibration report must be a regular file under 2 MiB", 5)
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, ProfileError):
            raise
        raise ProfileError(f"cannot read calibration report ({type(exc).__name__})", 5) from None
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int or \
            report["schema_version"] != 1 or \
            report.get("kind") != "strata.engine.calibration" or report.get("suite") != SUITE or \
            not isinstance(report.get("identity"), dict) or not isinstance(report.get("measurements"), dict) or \
            not isinstance(report.get("source_config"), str) or \
            type(report.get("created_at")) not in (int, float) or \
            not math.isfinite(report["created_at"]) or \
            not re.fullmatch(r"[0-9a-f]{32}", str(report.get("report_id", ""))):
        raise ProfileError("unsupported calibration report", 5)
    _validate_settings(report.get("settings"))
    if report.get("body_sha256") != _body_sha256(report):
        raise ProfileError("calibration report contents changed after measurement", 5)
    return report


def review(ref: str, report_path: Path) -> dict:
    path, profile, revision, cfg = _target(ref)
    report = load_report(report_path)
    cal = _full_folder("tools.calibrate")
    reasons = []
    if report["source_config"] != str(path):
        reasons.append("source_config_mismatch")
    if report.get("applicable") is not True:
        reasons.append("measurement_became_stale")
    age = time.time() - report["created_at"]
    if age < -300 or age > MAX_AGE_SECONDS:
        reasons.append("report_expired")
    try:
        now = _identity(path, cfg, profile, revision)
        if now != report["identity"]:
            reasons.append("source_engine_model_or_hardware_changed")
    except ProfileError as exc:
        reasons.append(f"identity_unavailable:{exc.exit_code}")
    after = cal.apply(cfg["args"], report["settings"])
    return {"schema_version": 1, "report_id": report.get("report_id"), "profile": profile["id"] if profile else None,
            "source_config": str(path), "applicable": not reasons, "reasons": reasons,
            "settings": report["settings"], "before_args": cfg["args"], "after_args": after,
            "measurements": report["measurements"], "model_integrity": "metadata_only_unverified"}


def _smoke(cfg: dict, source_path: Path) -> int:
    """Start the candidate and require at least one generated token before writing it."""
    import strata_tokenizer as st
    from serve.server import StrataEngine, child_env, engine_args
    cal = _full_folder("tools.calibrate")
    tokdir = _resolved(cfg["tokenizer"], cfg, source_path)
    vocab = json.loads((tokdir / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    tok = st.Tokenizer(tokens, (tokdir / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((tokdir / "token_type.json").read_text(encoding="utf-8")))
    engine = StrataEngine(str(_resolved(cfg["exe"], cfg, source_path)), engine_args(cfg),
                          cwd=cfg.get("cwd"), log=cfg.get("log"), env=child_env(cfg))
    try:
        ids = cal.chat_ids(tok, "Say OK.")
        count = sum(token is not None for token in engine.generate(ids, 16, {"temperature": 0}, threading.Event()))
        if count < 1:
            raise ProfileError("candidate engine produced no tokens in smoke generation", 6)
        return count
    finally:
        cal.close(engine)


def _replace_bytes(path: Path, payload: bytes) -> None:
    descriptor, temp = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_mod.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _settings_candidate(hardware_key: str, result: dict) -> tuple[Path, bytes | None, bytes]:
    """Prepare setup's remembered calibration without accepting a corrupt store."""
    target = _full_folder("setup").settings_path()
    try:
        previous = target.read_bytes()
        settings = json.loads(previous.decode("utf-8"))
        if not isinstance(settings, dict):
            raise ValueError
    except FileNotFoundError:
        previous, settings = None, {}
    except (OSError, UnicodeError, ValueError):
        raise ProfileError("setup settings could not be read safely; config unchanged", 5) from None
    if previous is not None and len(previous) > MAX_REPORT_BYTES:
        raise ProfileError("setup settings exceed the calibration rollback size limit; config unchanged", 5)
    cal = settings.get("calibration")
    if cal is None:
        cal = {}
        settings["calibration"] = cal
    if not isinstance(cal, dict):
        raise ProfileError("setup calibration store is invalid; config unchanged", 5)
    cal[hardware_key] = {"settings": result["settings"], "tok_s": result["measurements"].get("tok_s"),
                         "date": time.strftime("%Y-%m-%d")}
    try:
        proposed = (json.dumps(settings, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError):
        raise ProfileError("setup settings cannot be serialized; config unchanged", 5) from None
    if len(proposed) > MAX_REPORT_BYTES:
        raise ProfileError("updated setup settings exceed the calibration rollback size limit; config unchanged", 5)
    return target, previous, proposed


def _backup_directory(path: Path, *, create: bool = False) -> Path:
    directory = path.parent / ".strata-calibration-backups"
    try:
        if create:
            _private_mkdir(directory, exist_ok=True)
        details = directory.lstat()
    except FileNotFoundError:
        raise ProfileError("calibration backup directory is missing", 5) from None
    except OSError as exc:
        raise ProfileError(f"cannot access calibration backup directory ({type(exc).__name__})", 5) from None
    if not stat.S_ISDIR(details.st_mode) or (os.name == "posix" and
                                            (posix_owner.owner_only_wrong(directory, details))):
        raise ProfileError(posix_owner.owner_only_text("calibration backup directory must be owner-only and not a symlink", directory), 5)
    return directory


def _application_paths(path: Path, report_id: str, *, create: bool = False) -> tuple[Path, Path, Path]:
    directory = _backup_directory(path, create=create)
    return (directory / f"{path.stem}-{report_id}.json",
            directory / f"settings-{report_id}.json",
            directory / f"{path.stem}-{report_id}.apply.json")


def _sha256(payload: bytes | None) -> str | None:
    return hashlib.sha256(payload).hexdigest() if payload is not None else None


def _load_application(path: Path, report: dict) -> dict:
    backup, settings_backup, manifest = _application_paths(path, report["report_id"])
    try:
        details = manifest.lstat()
        if not stat.S_ISREG(details.st_mode) or details.st_size > MAX_APPLICATION_BYTES or \
                (os.name == "posix" and (posix_owner.owner_only_wrong(manifest, details))):
            raise ProfileError(posix_owner.owner_only_text("calibration application record is not an owner-only regular file", manifest), 5)
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ProfileError("calibration application record was not found", 5) from None
    except (OSError, ValueError, UnicodeError) as exc:
        if isinstance(exc, ProfileError):
            raise
        raise ProfileError(f"cannot read calibration application record ({type(exc).__name__})", 5) from None
    expected_keys = {"schema_version", "kind", "report_id", "report_body_sha256", "source_config",
                     "profile_revision", "before_config_sha256", "after_config_sha256",
                     "settings_path", "before_settings_sha256", "after_settings_sha256"}
    if not isinstance(data, dict) or set(data) != expected_keys or \
            type(data.get("schema_version")) is not int or data["schema_version"] != 1 or \
            data.get("kind") != "strata.engine.calibration.application" or \
            data.get("report_id") != report["report_id"] or \
            data.get("report_body_sha256") != report["body_sha256"] or \
            data.get("source_config") != str(path) or \
            data.get("profile_revision") != report["identity"].get("profile_revision") or \
            not isinstance(data.get("settings_path"), str) or \
            not Path(data["settings_path"]).is_absolute() or \
            not (data.get("before_settings_sha256") is None or
                 isinstance(data["before_settings_sha256"], str) and
                 re.fullmatch(r"[0-9a-f]{64}", data["before_settings_sha256"])) or \
            not isinstance(data.get("after_settings_sha256"), str) or \
            not re.fullmatch(r"[0-9a-f]{64}", data["after_settings_sha256"]) or \
            not all(isinstance(data.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", data[key])
                    for key in ("before_config_sha256", "after_config_sha256")) or \
            data["before_config_sha256"] != report["identity"].get("source_sha256"):
        raise ProfileError("calibration application record has an unsupported shape", 5)
    try:
        if data["settings_path"] != str(_full_folder("setup").settings_path()):
            raise ProfileError("setup settings path changed since calibration apply", 5)
    except (OSError, ValueError):
        raise ProfileError("setup settings path is unavailable", 5) from None
    return {**data, "backup": backup, "settings_backup": settings_backup, "manifest": manifest}


def apply_result(ref: str, report_path: Path, *, exclusive_ack: bool = False, smoke=None,
                 expected_report_sha256: str | None = None) -> dict:
    """Apply a reviewed result after a candidate smoke test; retain prior config."""
    if not exclusive_ack:
        raise ProfileError("calibration apply requires an exclusive-access acknowledgement")
    path, profile, revision, cfg = _target(ref)
    with profiles._lock(_lock_name(path)), _managed_exclusive():
        return _apply_locked(ref, path, profile, revision, cfg, report_path, smoke=smoke,
                             expected_report_sha256=expected_report_sha256)


def _apply_locked(ref: str, path: Path, profile: dict | None, revision: str | None, cfg: dict,
                  report_path: Path, *, smoke, expected_report_sha256: str | None) -> dict:
    _exclusive_preflight(cfg, profile)
    if expected_report_sha256 is not None and \
            load_report(report_path)["body_sha256"] != expected_report_sha256:
        raise ProfileError("calibration report changed after review; review again", 6)
    preview = review(ref, report_path)
    if not preview["applicable"]:
        raise ProfileError("calibration report is stale: " + ", ".join(preview["reasons"]), 6)
    report = load_report(report_path)
    if expected_report_sha256 is not None and report["body_sha256"] != expected_report_sha256:
        raise ProfileError("calibration report changed after review; review again", 6)
    settings_path, previous_settings, new_settings = _settings_candidate(report["identity"]["hardware_key"], report)
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != report["identity"]["source_sha256"]:
        raise ProfileError("run config changed before calibration apply", 6)
    candidate = copy.deepcopy(cfg)
    candidate["args"] = preview["after_args"]
    smoke_candidate = _runtime_cfg(candidate, profile)
    candidate_bytes = (json.dumps(candidate, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    if len(source) > MAX_REPORT_BYTES or len(candidate_bytes) > MAX_REPORT_BYTES:
        raise ProfileError("run config exceeds the calibration rollback size limit; config unchanged", 5)
    try:
        token_count = smoke(smoke_candidate) if smoke else _smoke(smoke_candidate, path)
    except ProfileError:
        raise
    except Exception as exc:
        raise ProfileError(f"candidate smoke generation failed ({type(exc).__name__}); config unchanged", 6) from None
    if type(token_count) is not int or token_count < 1:
        raise ProfileError("candidate smoke generation produced no tokens; config unchanged", 6)
    if path.read_bytes() != source:
        raise ProfileError("run config changed during smoke generation; config unchanged", 6)
    end_path, end_profile, end_revision, end_cfg = _target(ref)
    if end_path != path or _identity(end_path, end_cfg, end_profile, end_revision) != report["identity"]:
        raise ProfileError("engine, model or hardware changed during smoke generation; config unchanged", 6)
    _private_mkdir(settings_path.parent, parents=True, exist_ok=True)
    current_settings = settings_path.read_bytes() if settings_path.exists() else None
    if current_settings != previous_settings:
        raise ProfileError("setup settings changed during calibration; config unchanged", 6)
    backup, settings_backup_path, manifest = _application_paths(path, report["report_id"], create=True)
    if any(item.exists() or item.is_symlink() for item in (backup, settings_backup_path, manifest)):
        raise ProfileError("calibration result already has application records; measure again", 6)
    application = {"schema_version": 1, "kind": "strata.engine.calibration.application",
                   "report_id": report["report_id"], "report_body_sha256": report["body_sha256"],
                   "source_config": str(path), "profile_revision": revision,
                   "before_config_sha256": _sha256(source), "after_config_sha256": _sha256(candidate_bytes),
                   "settings_path": str(settings_path),
                   "before_settings_sha256": _sha256(previous_settings),
                   "after_settings_sha256": _sha256(new_settings)}
    settings_backup = settings_backup_path if previous_settings is not None else None
    written = []
    try:
        _write_new(backup, source)
        written.append(backup)
        if settings_backup is not None:
            _write_new(settings_backup, previous_settings)
            written.append(settings_backup)
        _write_new(manifest, _report_bytes(application))
        written.append(manifest)
    except (OSError, ProfileError) as exc:
        for item in written:
            item.unlink(missing_ok=True)
        raise ProfileError(f"cannot save calibration rollback evidence ({type(exc).__name__}); config unchanged", 6) from None
    try:
        if candidate_bytes != source:
            _replace_bytes(path, candidate_bytes)
        _replace_bytes(settings_path, new_settings)
    except Exception as exc:
        # Restore both stores if either write failed; backups remain for manual recovery.
        failed = False
        try:
            _replace_bytes(path, source)
        except OSError:
            failed = True
        try:
            if previous_settings is None:
                settings_path.unlink(missing_ok=True)
            else:
                _replace_bytes(settings_path, previous_settings)
        except OSError:
            failed = True
        if failed:
            raise ProfileError(f"calibration apply failed ({type(exc).__name__}); restore from {backup}", 6) from None
        raise ProfileError(f"calibration apply failed ({type(exc).__name__}); config restored", 6) from None
    return {"changed": candidate_bytes != source, "report_id": report.get("report_id"), "source_config": str(path),
            "backup": str(backup), "settings_backup": str(settings_backup) if settings_backup else None,
            "application_record": str(manifest),
            "previous_sha256": hashlib.sha256(source).hexdigest(),
            "current_sha256": hashlib.sha256(candidate_bytes).hexdigest(), "smoke_output_tokens": token_count,
            "setup_settings_persisted": True}


def _saved_bytes(path: Path, *, required: bool = True, owner_only: bool = False) -> bytes | None:
    try:
        details = path.lstat()
    except FileNotFoundError:
        if not required:
            return None
        raise ProfileError("calibration rollback file is missing", 5) from None
    if not stat.S_ISREG(details.st_mode) or details.st_size > MAX_REPORT_BYTES or \
            (owner_only and os.name == "posix" and
             (posix_owner.owner_only_wrong(path, details))):
        raise ProfileError(posix_owner.owner_only_text("calibration rollback file must be a regular owner-only file under 2 MiB", path), 5)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ProfileError(f"cannot read calibration rollback file ({type(exc).__name__})", 5) from None


def review_rollback(ref: str, report_path: Path) -> dict:
    """Preview an exact rollback without touching engine or settings."""
    path, profile, revision, cfg = _target(ref)
    report = load_report(report_path)
    application = _load_application(path, report)
    reasons: list[str] = []
    if report["source_config"] != str(path):
        reasons.append("source_config_mismatch")
    before_config = _saved_bytes(application["backup"], owner_only=True)
    if _sha256(before_config) != application["before_config_sha256"]:
        reasons.append("config_backup_changed")
    settings_backup = _saved_bytes(application["settings_backup"], required=False, owner_only=True)
    if _sha256(settings_backup) != application["before_settings_sha256"]:
        reasons.append("settings_backup_changed")
    current_config = _saved_bytes(path)
    if _sha256(current_config) != application["after_config_sha256"]:
        reasons.append("run_config_changed_after_apply")
    current_settings = _saved_bytes(Path(application["settings_path"]), required=False)
    if _sha256(current_settings) != application["after_settings_sha256"]:
        reasons.append("setup_settings_changed_after_apply")
    try:
        identity = _identity(path, cfg, profile, revision)
        expected = {**report["identity"], "source_sha256": application["after_config_sha256"]}
        if identity != expected:
            reasons.append("engine_model_hardware_or_profile_changed")
    except ProfileError as exc:
        reasons.append(f"identity_unavailable:{exc.exit_code}")
    if _port_active(cfg):
        reasons.append("profile_port_is_active")
    return {"schema_version": 1, "report_id": report["report_id"],
            "profile": profile["id"] if profile else None, "source_config": str(path),
            "applicable": not reasons, "reasons": reasons,
            "config_would_change": application["before_config_sha256"] != application["after_config_sha256"],
            "setup_settings_would_be_removed": application["before_settings_sha256"] is None,
            "backup": str(application["backup"]), "application_record": str(application["manifest"]),
            "model_integrity": "metadata_only_unverified"}


def rollback_result(ref: str, report_path: Path, *, exclusive_ack: bool = False, smoke=None,
                    expected_report_sha256: str | None = None) -> dict:
    """Smoke-test and restore the prior config/settings only if the applied state is still exact."""
    if not exclusive_ack:
        raise ProfileError("calibration rollback requires an exclusive-access acknowledgement")
    path, profile, revision, cfg = _target(ref)
    with profiles._lock(_lock_name(path)), _managed_exclusive():
        _exclusive_preflight(cfg, profile)
        if expected_report_sha256 is not None and \
                load_report(report_path)["body_sha256"] != expected_report_sha256:
            raise ProfileError("calibration report changed after review; review again", 6)
        preview = review_rollback(ref, report_path)
        if not preview["applicable"]:
            raise ProfileError("calibration rollback is stale: " + ", ".join(preview["reasons"]), 6)
        report = load_report(report_path)
        if expected_report_sha256 is not None and report["body_sha256"] != expected_report_sha256:
            raise ProfileError("calibration report changed after review; review again", 6)
        application = _load_application(path, report)
        before_config = _saved_bytes(application["backup"], owner_only=True)
        before_settings = _saved_bytes(application["settings_backup"], required=False, owner_only=True)
        current_config = _saved_bytes(path)
        settings_path = Path(application["settings_path"])
        current_settings = _saved_bytes(settings_path, required=False)
        candidate = profiles.read_run_config(application["backup"])
        smoke_candidate = _runtime_cfg(candidate, profile)
        try:
            token_count = smoke(smoke_candidate) if smoke else _smoke(smoke_candidate, path)
        except ProfileError:
            raise
        except Exception as exc:
            raise ProfileError(f"rollback smoke generation failed ({type(exc).__name__}); config unchanged", 6) from None
        if type(token_count) is not int or token_count < 1:
            raise ProfileError("rollback smoke generation produced no tokens; config unchanged", 6)
        repeated = review_rollback(ref, report_path)
        if not repeated["applicable"] or _saved_bytes(path) != current_config or \
                _saved_bytes(settings_path, required=False) != current_settings:
            raise ProfileError("config or setup settings changed during rollback smoke; config unchanged", 6)
        try:
            if before_config != current_config:
                _replace_bytes(path, before_config)
            if before_settings is None:
                settings_path.unlink(missing_ok=True)
            elif before_settings != current_settings:
                _replace_bytes(settings_path, before_settings)
        except Exception as exc:
            failed = False
            try:
                _replace_bytes(path, current_config)
            except OSError:
                failed = True
            try:
                if current_settings is None:
                    settings_path.unlink(missing_ok=True)
                else:
                    _replace_bytes(settings_path, current_settings)
            except OSError:
                failed = True
            if failed:
                raise ProfileError(f"calibration rollback failed ({type(exc).__name__}); "
                                   f"manual recovery may be needed from {application['backup']}", 6) from None
            raise ProfileError(f"calibration rollback failed ({type(exc).__name__}); applied state restored", 6) from None
        return {"changed": before_config != current_config or before_settings != current_settings,
                "report_id": report["report_id"], "source_config": str(path),
                "restored_config_sha256": _sha256(before_config),
                "setup_settings_removed": before_settings is None,
                "smoke_output_tokens": token_count, "backup": str(application["backup"])}
