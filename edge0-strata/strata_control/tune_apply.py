"""Applying a tuning recommendation: explicit, verified and reversible (checklist T07).

A sweep ends at a report, and a report changes nothing. This module is the separate, deliberate step
that turns one reviewed recommendation into a change to the active run configuration — and it is
built around the fact that the most dangerous moment in tuning is the gap between measuring a
configuration and running it for real.

Four things make that safe, and each exists because of a specific way this goes wrong:

1. **Explicit, and only from a completed sweep.** There is no default and no automatic application.
   The job must have reached a terminal state with a recommendation, and the report must still hash
   to what was reviewed — a report edited after review is refused, because the review was of
   different numbers.

2. **Identity is re-checked before anything is written.** A recommendation measured against one
   engine or one model says nothing about another. If the engine digest, model digest, backend,
   workload or context has changed since the sweep, the apply is refused *with the reason* rather
   than applied to a configuration it was never measured against.

3. **The candidate is verified by running it.** The proposed configuration is started and required
   to produce at least one token before the change is written. A configuration that boots but cannot
   generate is the exact failure a benchmark cannot see, because the benchmark never ran the real
   path.

4. **Reversal keeps the previous bytes.** The prior configuration is copied aside by digest before
   the write, and rollback restores those bytes — so "apply" is never a one-way door.

The active profile is untouched until the apply step: exploration and application are separate
commands with separate leases, and neither can be triggered by the other.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from strata_control import user_args
from strata_control import atomic as _atomic_mod

SCHEMA_VERSION = 1


class ApplyError(RuntimeError):
    """An apply was refused. The message is the reason, and nothing was written."""

    def __init__(self, message: str, code: int = 6) -> None:
        super().__init__(message)
        self.code = code


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    """Replace a file's contents in one step, so a crash cannot leave it half-written."""
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


# --------------------------------------------------------------------------- args and configs

def args_to_argv(args: dict[str, Any]) -> list[str]:
    """The recommendation's flags, as argv.

    Order is sorted so the same recommendation always produces the same bytes: an apply that
    reshuffled its own output would produce a different digest each time and break the rollback
    comparison.
    """
    from strata_control.tuning import flag_argv
    out: list[str] = []
    for flag in sorted(args):
        value = args[flag]
        if value is None or value is False:
            continue
        if value is True:
            out.append(flag)
            continue
        out.extend(flag_argv(flag, value))
    return out


def find_config_files(root: Path, *, limit: int = 200) -> list[Path]:
    """Every `strata-*.json` run configuration under `root`.

    The project keeps one configuration per installed model, and the naming is the project's own
    (`strata-<model>.json`, as `setup.py` writes them). Anything that does not parse as a JSON object
    with an `args` list is not a run configuration and is ignored rather than guessed at.
    """
    out: list[Path] = []
    for path in sorted(Path(root).rglob("strata-*.json")):
        if len(out) >= limit:
            break
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and isinstance(data.get("args"), list):
            out.append(path)
    return out


def apply_args_to_config(config: dict[str, Any], argv: Sequence[str]) -> dict[str, Any]:
    """Set a recommendation's flags on a configuration, leaving everything else alone.

    The rules are narrow on purpose:

    * a flag the configuration already carries has **all** of its occurrences replaced, because a
      half-replaced flag is a configuration that does nothing;
    * a flag it does not carry is appended;
    * no other key is touched, so unknown fields, aliases, `env` and `vision` survive — an apply
      that dropped a user's API token would be worse than no apply at all.

    `--flag value` and `--flag=value` are both produced as two argv elements, which the engine's
    parser accepts either way.
    """
    candidate = json.loads(json.dumps(config))          # deep copy, values are JSON
    existing = list(candidate.get("args") or [])

    # The recommendation's flags, paired.
    wanted: dict[str, list[str]] = {}
    i = 0
    while i < len(argv):
        flag = argv[i]
        value = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else None
        wanted[flag] = [flag] if value is None else [flag, value]
        i += 2 if value is not None else 1

    # Strip every occurrence of each recommended flag, then append the recommendation once.
    stripped: list[str] = []
    i = 0
    while i < len(existing):
        item = str(existing[i])
        if item in wanted:
            i += 2 if (i + 1 < len(existing) and not str(existing[i + 1]).startswith("--")) else 1
            continue
        stripped.append(existing[i])
        i += 1

    for flag in sorted(wanted):
        stripped.extend(wanted[flag])
    candidate["args"] = stripped
    # The person's own options (user_args, strata_control/user_args.py) are kept, except the flags this recommendation sets: a tuning
    # the person asks for is their newer choice.  The hand edits so far are recorded first, so writing "args" here loses none.
    if candidate.get(user_args.WRITTEN_KEY) is not None or candidate.get(user_args.USER_KEY):
        mine = user_args.capture(config)
        mine = user_args.flat([g for g in user_args.groups(mine) if user_args.flag_of(g) not in wanted])
        user_args.stamp(candidate, mine)
        if user_args.WRITTEN_KEY not in config:
            candidate.pop(user_args.WRITTEN_KEY, None)
    return candidate


# --------------------------------------------------------------------------- the plan

@dataclass
class ApplyPlan:
    """What an apply would do, computed before anything is written."""

    job_id: str
    config_path: Path
    before_args: list[str]
    after_args: list[str]
    added: list[str]
    removed: list[str]
    before_sha256: str
    after_sha256: str
    report_sha256: str
    reasons: list[str]                                   # non-empty means it will not apply

    @property
    def applicable(self) -> bool:
        return not self.reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "config_path": str(self.config_path),
            "before_args": self.before_args,
            "after_args": self.after_args,
            "added": self.added,
            "removed": self.removed,
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
            "report_sha256": self.report_sha256,
            "applicable": self.applicable,
            "reasons": self.reasons,
        }


def _diff(before: Sequence[str], after: Sequence[str]) -> tuple[list[str], list[str]]:
    """Which flags appeared and which disappeared, as `flag value` strings."""
    def pairs(items: Sequence[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        i = 0
        while i < len(items):
            item = str(items[i])
            if item.startswith("--") and i + 1 < len(items) and not str(items[i + 1]).startswith("--"):
                out[item] = str(items[i + 1])
                i += 2
            else:
                out[item] = ""
                i += 1
        return out

    b, a = pairs(before), pairs(after)
    added = [f"{k} {v}".strip() for k, v in sorted(a.items()) if b.get(k) != v]
    removed = [f"{k} {v}".strip() for k, v in sorted(b.items()) if a.get(k) != v]
    return added, removed


def review_apply(*, job: dict[str, Any], config_path: Path,
                 current_identity: dict[str, Any] | None,
                 report_sha256: str | None = None) -> ApplyPlan:
    """Decide whether this job's recommendation may be applied to this configuration.

    Every refusal names what is wrong. An apply that silently proceeded on a stale or incompatible
    report would be the single most damaging thing this module could do, so the checks are exhaustive
    and each one is a reason rather than a bare `False`.
    """
    reasons: list[str] = []
    report = job.get("report") or {}
    job_id = str(job.get("id") or "")

    if job.get("state") != "completed":
        reasons.append(f"the sweep is {job.get('state')}, not completed; its result is not final")
    if not report.get("recommendation_available") or not report.get("recommendation"):
        reasons.append("the sweep recommended no change: " + str(report.get("reason") or "no reason recorded"))
    if report.get("drift_material"):
        drift = report.get("drift") or {}
        reasons.append(
            f"the machine drifted by {float(drift.get('relative', 0)) * 100:+.1f}% during the sweep, "
            "so the comparison does not support a change")
    if not config_path.exists():
        reasons.append(f"no run configuration at {config_path}")
        return ApplyPlan(job_id, config_path, [], [], [], [], "", "", "", reasons)

    source = config_path.read_bytes()
    before = json.loads(source.decode("utf-8"))
    before_args = [str(x) for x in (before.get("args") or [])]

    rec = report.get("recommendation") or {}
    argv = args_to_argv(rec.get("args") or {})
    if not argv:
        reasons.append("the recommendation carries no flags, so there is nothing to apply")

    candidate = apply_args_to_config(before, argv)
    after_args = [str(x) for x in candidate["args"]]
    added, removed = _diff(before_args, after_args)
    if not added and not removed:
        reasons.append("the configuration already carries this recommendation; nothing would change")

    # The report has to be the one that was reviewed.
    if report_sha256 is not None and report_sha256 != _report_digest(job):
        reasons.append("the sweep report changed after review; review it again")

    # And it has to have been measured on what is installed now.
    if current_identity is not None:
        from strata_control import tune_history
        recorded = report.get("identity") or {}
        if not recorded:
            reasons.append("the sweep recorded no artifact identity, so it cannot be matched to what "
                           "is installed now")
        else:
            differences = tune_history.incomparable_reasons(recorded, current_identity)
            if differences:
                reasons.append("the sweep was measured against a different configuration: "
                               + ", ".join(differences))

    after_bytes = (json.dumps(candidate, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    return ApplyPlan(
        job_id=job_id, config_path=config_path, before_args=before_args, after_args=after_args,
        added=added, removed=removed, before_sha256=_sha256(source), after_sha256=_sha256(after_bytes),
        report_sha256=_report_digest(job), reasons=reasons)


def _report_digest(job: dict[str, Any]) -> str:
    """A digest of the report, so a review cannot be reused for edited numbers."""
    body = json.dumps(job.get("report") or {}, sort_keys=True, separators=(",", ":"))
    return _sha256(body.encode("utf-8"))


# --------------------------------------------------------------------------- applying

def _application_dir(root: Path) -> Path:
    d = Path(root) / "applications"
    d.mkdir(parents=True, exist_ok=True)
    return d


def apply_recommendation(*, job: dict[str, Any], config_path: Path,
                         current_identity: dict[str, Any] | None,
                         state_dir: Path,
                         verify: Callable[[dict[str, Any]], int] | None = None,
                         expected_report_sha256: str | None = None) -> dict[str, Any]:
    """Verify the candidate by running it, then write it — keeping the previous bytes.

    `verify` receives the candidate configuration and must return the number of tokens generated.
    A result below 1 is a refusal, not a warning: a configuration that cannot produce a token has not
    been verified, whatever the sweep measured.
    """
    plan = review_apply(job=job, config_path=config_path, current_identity=current_identity,
                        report_sha256=expected_report_sha256)
    if not plan.applicable:
        raise ApplyError("; ".join(plan.reasons))

    before = json.loads(config_path.read_text(encoding="utf-8"))
    candidate = apply_args_to_config(before, args_to_argv(
        (job.get("report") or {}).get("recommendation", {}).get("args") or {}))

    verified_tokens: int | None = None
    if verify is not None:
        try:
            tokens = verify(candidate)
        except Exception as exc:                                   # noqa: BLE001
            raise ApplyError(
                f"the candidate configuration failed verification ({type(exc).__name__}: {exc}); "
                "the run configuration is unchanged") from None
        if type(tokens) is not int or tokens < 1:
            raise ApplyError("the candidate configuration generated no tokens; the run "
                             "configuration is unchanged")
        verified_tokens = int(tokens)

    # Re-read after verification: a user edit during a long verification must not be overwritten.
    if _sha256(config_path.read_bytes()) != plan.before_sha256:
        raise ApplyError("the run configuration changed during verification; nothing was written")

    application_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{plan.job_id[:12]}"
    directory = _application_dir(state_dir) / application_id
    if directory.exists():
        raise ApplyError(f"an application record already exists at {directory}")
    directory.mkdir(parents=True)
    shutil.copy2(config_path, directory / "before.json")
    (directory / "candidate.json").write_bytes(
        (json.dumps(candidate, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))

    _atomic_write(config_path, (json.dumps(candidate, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))

    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": "strata.tuning.application",
        "application_id": application_id,
        "job_id": plan.job_id,
        "applied_at": time.time(),
        "config_path": str(config_path),
        "before_sha256": plan.before_sha256,
        "after_sha256": plan.after_sha256,
        "report_sha256": plan.report_sha256,
        "added": plan.added,
        "removed": plan.removed,
        "verified_tokens": verified_tokens,
        "rolled_back": False,
    }
    (directory / "application.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return {"application": record, "directory": str(directory), "plan": plan.to_dict()}


def rollback(*, application_id: str, state_dir: Path) -> dict[str, Any]:
    """Restore the configuration that was in place before an apply.

    Refuses if the configuration has been edited since the apply, because restoring would then
    discard a deliberate change: the user's edit wins, and the conflict is reported.
    """
    directory = _application_dir(state_dir) / application_id
    record_path = directory / "application.json"
    if not record_path.exists():
        raise ApplyError(f"no application record at {record_path}", 404)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("rolled_back"):
        raise ApplyError("this application has already been rolled back")
    before = directory / "before.json"
    if not before.exists():
        raise ApplyError(f"the previous configuration was not retained at {before}")
    target = Path(record["config_path"])
    if not target.exists():
        raise ApplyError(f"the run configuration is gone: {target}")
    current = _sha256(target.read_bytes())
    if current != record["after_sha256"]:
        raise ApplyError(
            "the run configuration changed after this apply, so restoring would discard a newer "
            "change; nothing was written", 409)
    _atomic_write(target, before.read_bytes())
    record["rolled_back"] = True
    record["rolled_back_at"] = time.time()
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return {"application": record, "restored": str(target)}
