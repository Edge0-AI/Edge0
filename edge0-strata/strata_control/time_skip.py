"""Real Time-skip: retain a good result, try a bounded improvement, check it, and decide honestly.

This replaces the supplied 28-second scripted flow. The contract it implements is the integration
specification's section 5:

    queued -> checkpointing -> generating -> checking -> decision -> completed
    with repeated generating/checking for bounded attempts, and the alternate terminal states
    cancelled, failed, interrupted, needs_user_review.

Design rules that are load-bearing, and the reason each one exists:

* **Nothing is fabricated.** A check that did not run is ``unknown``, never ``passed``. A run whose
  evidence is insufficient ends ``needs_user_review``; it never invents an objective score. There is
  no timer that decides anything: phases advance because work completed, not because time passed.
* **The last good version survives everything.** Checkpoints are content-addressed snapshots taken
  before any speculative work. Cancel, crash and failure leave the selected checkpoint intact.
* **Candidates are isolated.** Each attempt writes into its own directory; nothing is applied to the
  workspace until an explicit ``apply``, which refuses if the destination changed since it was
  captured. A user edit becomes a conflict, never a silent overwrite.
* **Only deterministic checks gate.** A model's opinion can advise, but it cannot assert objective
  correctness, so it is recorded as advice and never promotes a candidate on its own.
* **Applying is a separate, verified, reversible step.** It checks digests, writes a rollback copy,
  and restores on failure.

The module is deliberately stdlib-only: the terminal distribution's serving path and the App Host
both import it, and neither may depend on the agent runtime or a web framework.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from strata_control import atomic

SCHEMA_VERSION = 1

# --------------------------------------------------------------------------- phases and states

PHASES = ("queued", "checkpointing", "generating", "checking", "decision", "completed")
TERMINAL = ("completed", "cancelled", "failed", "interrupted", "needs_user_review")

#: Every phase that a run can be observed in, including terminal states.
STATES = PHASES + TERMINAL

#: Outcome vocabulary for one check. `unknown` is a real answer and must survive to the report.
CHECK_OUTCOMES = ("passed", "failed", "unknown", "error")

#: The most attempts after the baseline. The specification's default is at most two revisions.
DEFAULT_MAX_ATTEMPTS = 2


class TimeSkipError(RuntimeError):
    """A refusal the caller should surface verbatim."""


# --------------------------------------------------------------------------- checks

@dataclass(frozen=True)
class CheckSpec:
    """One declared check.

    ``kind`` is ``deterministic`` or ``advice``. Only deterministic checks can gate a decision,
    which is how a model critique is kept from asserting objective correctness.
    """

    id: str
    command: str | None = None
    #: A path whose existence and digest is the check (used when no command is declared).
    file: str | None = None
    #: Expected content digest of `file`, if the check is a content assertion.
    sha256: str | None = None
    kind: str = "deterministic"
    description: str = ""
    timeout_s: float = 300.0

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "CheckSpec":
        if not isinstance(raw, dict) or not raw.get("id"):
            raise TimeSkipError("each check needs an id")
        kind = raw.get("kind", "deterministic")
        if kind not in ("deterministic", "advice"):
            raise TimeSkipError(f"check {raw['id']}: kind must be 'deterministic' or 'advice'")
        if not raw.get("command") and not raw.get("file"):
            raise TimeSkipError(f"check {raw['id']}: needs a command or a file")
        return CheckSpec(
            id=str(raw["id"]),
            command=raw.get("command"),
            file=raw.get("file"),
            sha256=raw.get("sha256"),
            kind=kind,
            description=str(raw.get("description") or ""),
            timeout_s=float(raw.get("timeout_s") or 300.0),
        )


@dataclass
class CheckResult:
    id: str
    kind: str
    outcome: str
    detail: str = ""
    exit_code: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    sha256: str | None = None
    duration_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in ("", None) or k in ("id", "kind", "outcome")}


#: cmd.exe's own "command not found" exit code.
CMD_NOT_FOUND = 9009
#: Things cmd.exe runs itself, so there is no program to look up.
_CMD_BUILTINS = frozenset((
    "assoc call cd chdir cls copy date del dir echo endlocal erase exit for ftype goto if md mkdir mklink "
    "move path pause popd prompt pushd rd rem ren rename rmdir set setlocal shift start time title type "
    "ver verify vol").split())
_CMD_OPERATORS = set("&|<>^%")


def _shell_could_not_run(command: str, returncode: int, env: dict[str, str]) -> bool:
    """True when the shell reported that the program does not exist or cannot be executed.

    POSIX ``sh`` says so with 126/127. ``cmd.exe`` says 9009 from a batch context but often a plain 1
    from ``cmd /c`` (and the message is localised), so for exit 1 the first word is looked up on PATH
    instead of trusting the code. A real failure of an existing program stays ``failed``.
    """
    if os.name != "nt":
        return returncode in (126, 127)
    if returncode == CMD_NOT_FOUND:
        return True
    if returncode != 1:
        return False
    text = command.strip()
    if not text or _CMD_OPERATORS & set(text):
        return False  # a pipeline or redirect: the failing part is unknowable from here
    if text.startswith('"'):
        program = text[1:].split('"', 1)[0]
    else:
        program = text.split(None, 1)[0]
    if program.lower() in _CMD_BUILTINS:
        return False
    return shutil.which(program, path=env.get("PATH") or env.get("Path")) is None


def _run_shell(command: str, cwd: str, timeout: float, env: dict[str, str]) -> "subprocess.CompletedProcess[str]":
    """``subprocess.run(shell=True)`` that also stops what the shell started when the check times out.

    ``cmd.exe /c`` is a parent process: killing it leaves the real program holding the output pipes,
    so a hung check would block here until it ended on its own. On Windows the whole tree is killed.
    """
    if os.name != "nt":
        return subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, errors="replace",
                              timeout=timeout, env=env)
    with subprocess.Popen(command, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, errors="replace", env=env) as process:
        try:
            out, err = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
            process.communicate()
            raise
        return subprocess.CompletedProcess(command, process.returncode, out, err)


def run_check(spec: CheckSpec, workdir: Path, env: dict[str, str] | None = None) -> CheckResult:
    """Run one check and report what actually happened.

    A check that could not be run is ``unknown``; a check that ran and crashed is ``error``. Neither
    is ever reported as ``passed``.
    """
    started = time.time()
    if spec.command:
        try:
            proc = _run_shell(spec.command, str(workdir), spec.timeout_s, {**os.environ, **(env or {})})
        except subprocess.TimeoutExpired:
            return CheckResult(spec.id, spec.kind, "unknown",
                               detail=f"timed out after {spec.timeout_s:.0f}s",
                               duration_ms=int((time.time() - started) * 1000))
        except OSError as e:
            return CheckResult(spec.id, spec.kind, "unknown", detail=f"could not run: {e}")
        tail = lambda s: (s or "")[-2000:]  # noqa: E731 - a local shorthand
        if _shell_could_not_run(spec.command, proc.returncode, {**os.environ, **(env or {})}):
            # The shell's own "not executable" / "command not found" (126/127 on sh, 9009 on cmd).
            # The check never ran, so this is `unknown`: reporting it as `failed` would blame the
            # candidate for a declaration mistake, and reporting it as `passed` would be worse.
            return CheckResult(spec.id, spec.kind, "unknown",
                               detail=f"the check's command could not be run (shell exit "
                                      f"{proc.returncode}): {spec.command}",
                               exit_code=proc.returncode, stderr_tail=tail(proc.stderr),
                               duration_ms=int((time.time() - started) * 1000))
        outcome = "passed" if proc.returncode == 0 else "failed"
        return CheckResult(spec.id, spec.kind, outcome, exit_code=proc.returncode,
                           stdout_tail=tail(proc.stdout), stderr_tail=tail(proc.stderr),
                           duration_ms=int((time.time() - started) * 1000))

    # file assertion
    target = workdir / spec.file  # type: ignore[operator]
    if not target.exists():
        return CheckResult(spec.id, spec.kind, "failed", detail=f"{spec.file} is missing")
    digest = sha256_file(target)
    if spec.sha256 and digest != spec.sha256:
        return CheckResult(spec.id, spec.kind, "failed",
                           detail=f"{spec.file} digest {digest[:12]} does not match the expected "
                                  f"{spec.sha256[:12]}", sha256=digest)
    return CheckResult(spec.id, spec.kind, "passed", sha256=digest,
                       duration_ms=int((time.time() - started) * 1000))


def summarise(results: Sequence[CheckResult]) -> dict[str, Any]:
    """Count outcomes without collapsing `unknown` into anything friendlier."""
    counts = {k: 0 for k in CHECK_OUTCOMES}
    for r in results:
        counts[r.outcome] = counts.get(r.outcome, 0) + 1
    return {"total": len(results), **counts}


# --------------------------------------------------------------------------- digests and snapshots

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def inventory(root: Path, ignore: Iterable[str] = (".git", "__pycache__", ".strata")) -> list[dict[str, Any]]:
    """A sorted relative-path/digest inventory of a directory tree.

    Digests are what make a checkpoint an identity rather than a label. Directories that belong to
    tooling rather than to the user's work are skipped.
    """
    skip = set(ignore)
    out: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip)
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.is_symlink():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            rel = p.relative_to(root).as_posix()
            out.append({"path": rel, "bytes": st.st_size, "sha256": sha256_file(p)})
    return out


def inventory_digest(items: Sequence[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(f"{it['path']}\0{it['bytes']}\0{it['sha256']}\n".encode())
    return h.hexdigest()


def copy_tree(src: Path, dst: Path, ignore: Iterable[str] = (".git", "__pycache__", ".strata")) -> None:
    skip = set(ignore)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in skip]
        rel = Path(dirpath).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for name in filenames:
            s = Path(dirpath) / name
            if s.is_symlink():
                continue
            shutil.copy2(s, dst / rel / name, follow_symlinks=False)


# --------------------------------------------------------------------------- the run

@dataclass
class Checkpoint:
    id: str
    label: str
    path: str
    digest: str
    inventory: list[dict[str, Any]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    parent: str | None = None
    created_at: float = field(default_factory=time.time)


@dataclass
class Attempt:
    index: int
    prompt: str
    path: str
    checkpoint: str | None = None
    checks: list[dict[str, Any]] = field(default_factory=list)
    decision: str | None = None          # keep | reject | inconclusive
    reason: str = ""
    created_at: float = field(default_factory=time.time)


class TimeSkipRun:
    """One Time-skip run, durable on disk after every transition.

    The runner is injected so this module never has to know how an agent is driven: the App Host
    passes a callable that performs one bounded revision attempt. That keeps the ownership boundary
    (Harness owns the loop) while Strata Control owns the job and its evidence.
    """

    def __init__(self, root: Path, workspace: Path, prompt: str,
                 checks: Sequence[CheckSpec], max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                 run_id: str | None = None, clock: Callable[[], float] = time.time) -> None:
        self.id = run_id or uuid.uuid4().hex
        self.root = root / self.id
        self.workspace = Path(workspace)
        self.prompt = prompt
        self.checks = list(checks)
        self.max_attempts = max(0, int(max_attempts))
        self.clock = clock
        self.state = "queued"
        self.started_at = self.clock()
        self.finished_at: float | None = None
        self.checkpoints: list[Checkpoint] = []
        self.attempts: list[Attempt] = []
        self.selected_checkpoint: str | None = None
        self.events: list[dict[str, Any]] = []
        self.error: str | None = None
        self.decision: dict[str, Any] | None = None
        self._cancel_requested = False
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- persistence -----------------------------------------------------

    @property
    def state_file(self) -> Path:
        return self.root / "run.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "workspace": str(self.workspace),
            "prompt": self.prompt,
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "max_attempts": self.max_attempts,
            "checks": [c.__dict__ for c in self.checks],
            "checkpoints": [c.__dict__ for c in self.checkpoints],
            "attempts": [a.__dict__ for a in self.attempts],
            "selected_checkpoint": self.selected_checkpoint,
            "events": self.events,
            "error": self.error,
            "decision": self.decision,
            "cancel_requested": self._cancel_requested,
        }

    def save(self) -> None:
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        atomic.replace(tmp, self.state_file)

    @staticmethod
    def load(root: Path, run_id: str) -> dict[str, Any]:
        p = Path(root) / run_id / "run.json"
        if not p.exists():
            raise TimeSkipError(f"no Time-skip run {run_id}")
        return json.loads(p.read_text(encoding="utf-8"))

    # ---- cancellation ----------------------------------------------------

    def request_cancel(self) -> None:
        """Ask the run to stop at the next phase boundary.

        Cancellation is cooperative and never claims the work stopped before it did: the run reaches
        `cancelled` when the running phase observes the flag, and the last complete checkpoint stays
        selected throughout.
        """
        self._cancel_requested = True
        self.save()

    def _check_cancel(self) -> None:
        if self._cancel_requested:
            raise _Cancelled()

    # ---- events ----------------------------------------------------------

    def _enter(self, state: str, **extra: Any) -> None:
        if state not in STATES:
            raise TimeSkipError(f"unknown state {state!r}")
        self.state = state
        self.events.append({"seq": len(self.events), "at": self.clock(), "state": state, **extra})
        self.save()

    # ---- phases ----------------------------------------------------------

    def checkpoint(self, label: str, source: Path | None = None, parent: str | None = None) -> Checkpoint:
        """Snapshot a directory and record what it contained."""
        src = Path(source or (self.workspace if label == "baseline" else self.workspace))
        # `pending` holds speculative work; the workspace itself is only read here.
        cp_id = f"cp{len(self.checkpoints) + 1}-{label}"
        dest = self.root / "checkpoints" / cp_id
        copy_tree(src, dest)
        items = inventory(dest)
        cp = Checkpoint(id=cp_id, label=label, path=str(dest), digest=inventory_digest(items),
                        inventory=items, parent=parent)
        self.checkpoints.append(cp)
        self.save()
        return cp

    def run(self, revise: Callable[[Path, str, int], str | None]) -> dict[str, Any]:
        """Execute the whole state machine.

        `revise(candidate_dir, prompt, attempt_index)` performs one bounded revision attempt in an
        isolated directory and returns an optional note. It must not touch the workspace.
        """
        try:
            self._enter("checkpointing")
            # 1. Retain the current good version before any speculative work.
            baseline_dir = self.root / "baseline"
            copy_tree(self.workspace, baseline_dir)
            baseline = self.checkpoint("baseline", source=baseline_dir)
            self.selected_checkpoint = baseline.id
            self._check_cancel()

            # 2. Check the baseline, so a candidate has something real to be compared against.
            baseline.checks = [r.to_dict() for r in self._run_checks(baseline_dir)]
            self.save()
            self._check_cancel()

            # 3. Bounded attempts, each from the *selected* checkpoint, never from a rejected one.
            for index in range(1, self.max_attempts + 1):
                self._check_cancel()
                self._enter("generating", attempt=index)
                attempt_dir = self.root / "candidates" / f"attempt{index}"
                copy_tree(self._source_of(self.selected_checkpoint), attempt_dir)
                note = revise(attempt_dir, self.prompt, index)
                attempt = Attempt(index=index, prompt=self.prompt, path=str(attempt_dir),
                                  checkpoint=self.selected_checkpoint,
                                  checks=[])
                if note:
                    attempt.reason = note
                self.attempts.append(attempt)
                self.save()
                self._check_cancel()

                self._enter("checking", attempt=index)
                attempt.checks = [r.to_dict() for r in self._run_checks(attempt_dir)]
                self.save()
                self._check_cancel()

                self._enter("decision", attempt=index)
                verdict, reason = self._decide(self._checks_of(self.selected_checkpoint), attempt.checks)
                attempt.decision = verdict
                attempt.reason = reason
                if verdict == "keep":
                    kept = self.checkpoint(f"attempt{index}", source=attempt_dir,
                                           parent=self.selected_checkpoint)
                    kept.checks = attempt.checks
                    self.selected_checkpoint = kept.id
                self.save()

            self.decision = self._final_decision()
            self._enter("completed")
            self.finished_at = self.clock()
            self.save()
            return self.to_dict()

        except _Cancelled:
            self.finished_at = self.clock()
            self.state = "cancelled"
            self.events.append({"seq": len(self.events), "at": self.clock(), "state": "cancelled"})
            self.save()
            return self.to_dict()
        except Exception as e:  # noqa: BLE001 - a failed run is a recorded outcome, not a crash
            self.error = f"{type(e).__name__}: {e}"
            self.finished_at = self.clock()
            self.state = "failed"
            self.events.append({"seq": len(self.events), "at": self.clock(), "state": "failed",
                                "error": self.error})
            self.save()
            return self.to_dict()

    # ---- helpers ---------------------------------------------------------

    def _source_of(self, checkpoint_id: str | None) -> Path:
        if checkpoint_id is None:
            raise TimeSkipError("no checkpoint is selected")
        for cp in self.checkpoints:
            if cp.id == checkpoint_id:
                return Path(cp.path)
        raise TimeSkipError(f"checkpoint {checkpoint_id} is not in this run")

    def _checks_of(self, checkpoint_id: str | None) -> dict[str, dict[str, Any]]:
        """The checks recorded on a checkpoint, keyed by id.

        The gate compares a candidate against the version it was derived from, so a regression is
        measured against the retained evidence rather than against an ideal.
        """
        for cp in self.checkpoints:
            if cp.id == checkpoint_id:
                return {c["id"]: c for c in cp.checks}
        return {}

    def _run_checks(self, workdir: Path) -> list[CheckResult]:
        return [run_check(spec, workdir) for spec in self.checks]

    def _decide(self, previous: dict[str, dict[str, Any]],
                candidate: Sequence[dict[str, Any]]) -> tuple[str, str]:
        """Non-regression gate over deterministic checks only.

        The rule is the integration specification's: **every previously passing check must still
        pass**, there must be no hard errors, and there must be evidence of an intended improvement.
        Advice checks are recorded but never gate, because a model's opinion cannot assert objective
        correctness.

        Outcomes:
          * a previously passing deterministic check now fails -> ``reject``
          * a deterministic check could not be evaluated       -> ``inconclusive``
          * deterministic checks pass but nothing improved      -> ``inconclusive``
          * deterministic checks pass and something improved    -> ``keep``
        """
        det = [c for c in candidate if c.get("kind") == "deterministic"]
        if not det:
            return "inconclusive", "no deterministic checks were declared, so nothing can justify a change"

        regressed = [c["id"] for c in det
                     if c["outcome"] == "failed" and previous.get(c["id"], {}).get("outcome") == "passed"]
        if regressed:
            return "reject", f"previously passing check(s) now fail: {', '.join(regressed)}"

        hard_failed = [c["id"] for c in det if c["outcome"] == "failed"]
        if hard_failed:
            return "reject", f"deterministic check(s) failed: {', '.join(hard_failed)}"

        unevaluated = [c["id"] for c in det if c["outcome"] in ("unknown", "error")]
        if unevaluated:
            return "inconclusive", f"check(s) could not be evaluated: {', '.join(unevaluated)}"

        improved = [c["id"] for c in det
                    if c["outcome"] == "passed" and previous.get(c["id"], {}).get("outcome") != "passed"]
        if not improved:
            return "inconclusive", ("every deterministic check already passed and none changed, so "
                                    "the evidence does not show an intended improvement")
        return "keep", f"every deterministic check passed; newly passing: {', '.join(improved)}"

    def _final_decision(self) -> dict[str, Any]:
        """Compare the selected checkpoint with the baseline and say what is actually known."""
        baseline = next((c for c in self.checkpoints if c.label == "baseline"), None)
        kept = next((c for c in self.checkpoints if c.id == self.selected_checkpoint), None)
        base_checks = {c["id"]: c["outcome"] for c in (baseline.checks if baseline else [])}
        kept_checks = {c["id"]: c["outcome"] for c in (kept.checks if kept else [])}
        regressions = sorted(k for k, v in base_checks.items()
                             if v == "passed" and kept_checks.get(k) == "failed")
        improvements = sorted(k for k, v in kept_checks.items()
                              if v == "passed" and base_checks.get(k) != "passed")
        if not self.attempts:
            outcome = "inconclusive"
            why = "no revision was attempted"
        elif any(a.decision == "reject" for a in self.attempts) and not improvements:
            outcome = "inconclusive"
            why = "every candidate was rejected; the retained version is the baseline"
        elif improvements and not regressions:
            outcome = "accepted"
            why = f"improved: {', '.join(improvements)}"
        elif improvements and regressions:
            outcome = "conflict"
            why = f"improved {', '.join(improvements)} but regressed {', '.join(regressions)}"
        else:
            outcome = "inconclusive"
            why = "no check changed outcome; the evidence does not justify a change"
        return {
            "outcome": outcome,
            "reason": why,
            "retained": self.selected_checkpoint,
            "baseline": baseline.id if baseline else None,
            "improvements": improvements,
            "regressions": regressions,
            "attempts": [{"index": a.index, "decision": a.decision, "reason": a.reason} for a in self.attempts],
            "checks": {"baseline": base_checks, "retained": kept_checks},
            "needs_user_review": outcome in ("inconclusive", "conflict"),
        }

    # ---- applying --------------------------------------------------------

    def apply(self, destination: Path | None = None, dry_run: bool = False,
              attempt: int | None = None) -> dict[str, Any]:
        """Apply the retained checkpoint to the workspace, transactionally.

        ``attempt`` names a revision attempt the *person* chose after looking at it (a run whose checks
        could not decide leaves the baseline retained, and the person is asked). The same safety
        applies: a changed destination is a conflict, and a failure rolls back.

        Refuses when the destination no longer matches what was captured, which is how a concurrent
        user edit becomes a conflict instead of being overwritten. On any failure the previous
        content is restored from the rollback copy.
        """
        dest = Path(destination or self.workspace)
        if attempt is None:
            kept = self._source_of(self.selected_checkpoint)
        else:
            chosen = next((a for a in self.attempts if a.index == attempt), None)
            if chosen is None or not Path(chosen.path).is_dir():
                raise TimeSkipError(f"this run has no revision attempt {attempt} to apply")
            kept = Path(chosen.path)
        baseline = next((c for c in self.checkpoints if c.label == "baseline"), None)
        if baseline is None:
            raise TimeSkipError("this run has no baseline to compare the destination against")

        # The destination must still be what the baseline captured.
        current = inventory(dest)
        if inventory_digest(current) != baseline.digest:
            changed = _first_difference(baseline.inventory, current)
            return {"applied": False, "conflict": True,
                    "reason": "the destination changed since the run started; refusing to overwrite it",
                    "changed_path": changed}

        applied = [it["path"] for it in inventory(kept) if not _same_in(baseline.inventory, it)]
        removed = [it["path"] for it in baseline.inventory if not _exists_in(inventory(kept), it["path"])]
        if dry_run:
            return {"applied": False, "dry_run": True, "would_write": applied, "would_remove": removed}

        rollback = self.root / "rollback"
        copy_tree(dest, rollback)
        try:
            for rel in removed:
                target = dest / rel
                if target.exists():
                    target.unlink()
            for dirpath, _dirnames, filenames in os.walk(kept):
                rel = Path(dirpath).relative_to(kept)
                (dest / rel).mkdir(parents=True, exist_ok=True)
                for name in filenames:
                    shutil.copy2(Path(dirpath) / name, dest / rel / name, follow_symlinks=False)
        except Exception as e:  # noqa: BLE001 - restore and report, never leave a half-applied tree
            copy_tree(rollback, dest)
            return {"applied": False, "rolled_back": True,
                    "reason": f"apply failed and was rolled back: {type(e).__name__}: {e}"}
        return {"applied": True, "wrote": applied, "removed": removed,
                "rollback": str(rollback),
                "checkpoint": self.selected_checkpoint if attempt is None else f"attempt{attempt}"}

    def restore_original(self, destination: Path | None = None) -> dict[str, Any]:
        """Put the baseline back, whatever else has happened."""
        dest = Path(destination or self.workspace)
        baseline = next((c for c in self.checkpoints if c.label == "baseline"), None)
        if baseline is None:
            raise TimeSkipError("this run has no baseline to restore")
        copy_tree(Path(baseline.path), dest)
        self.selected_checkpoint = baseline.id
        self.decision = self._final_decision() if self.decision else None
        self.save()
        return {"restored": True, "checkpoint": baseline.id}


class _Cancelled(Exception):
    pass


# --------------------------------------------------------------------------- helpers

def _exists_in(items: Sequence[dict[str, Any]], path: str) -> bool:
    return any(i["path"] == path for i in items)


def _same_in(items: Sequence[dict[str, Any]], other: dict[str, Any]) -> bool:
    for i in items:
        if i["path"] == other["path"]:
            return i["sha256"] == other["sha256"] and i["bytes"] == other["bytes"]
    return False


def _first_difference(before: Sequence[dict[str, Any]], after: Sequence[dict[str, Any]]) -> str | None:
    b = {i["path"]: i["sha256"] for i in before}
    a = {i["path"]: i["sha256"] for i in after}
    for path in sorted(set(b) | set(a)):
        if b.get(path) != a.get(path):
            return path
    return None


# --------------------------------------------------------------------------- job store

class TimeSkipStore:
    """The runs on disk, so the UI and the CLI read the same durable truth."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict[str, Any]]:
        out = []
        for d in sorted(self.root.iterdir()):
            f = d / "run.json"
            if f.is_file():
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                out.append({k: data.get(k) for k in
                            ("id", "state", "workspace", "prompt", "started_at", "finished_at", "decision")})
        return out

    def get(self, run_id: str) -> dict[str, Any]:
        return TimeSkipRun.load(self.root, run_id)

    def load_checks(self, workspace: Path) -> list[CheckSpec]:
        """Read the declared checks for a workspace.

        `.strata/checks.json` holds ``{"checks": [...]}``. Absent file means no checks, which makes
        every run inconclusive by construction rather than accidentally successful.
        """
        path = Path(workspace) / ".strata" / "checks.json"
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw.get("checks") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            raise TimeSkipError(f"{path}: expected a list of checks")
        return [CheckSpec.from_dict(i) for i in items]
