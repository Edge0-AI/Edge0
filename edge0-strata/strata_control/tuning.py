"""Engine tuning: a bounded, evidence-driven search over runtime configuration.

This is **runtime configuration optimization**, not model training. It searches the engine's own
launch settings, measures each candidate, and produces a *recommendation* — never an automatic
change.

Why Strata owns this rather than delegating wholesale to HyperCore-Tune
-----------------------------------------------------------------------

HyperCore-Tune was evaluated as the preferred adapter and is **not sufficient as the optimizer** for
every backend. The evidence, gathered from the pinned checkout and this machine's engine:

* Its search space is expressed as engine flags: ``--vram-reserve-mib``, ``--pcie-frac``,
  ``--pool-workers``, ``--host-core``, ``--pool-affinity``, ``--ple-io``, ``--kv``, ``--kv-grow``,
  ``--prompt-cache-tail``, ``--spec-min-p``. Two of those, ``--vram-reserve-mib`` and
  ``--pcie-frac``, are NVIDIA/AMD concepts: the Apple Silicon engine does not parse them at all
  (``strata --help`` on the Metal build lists neither). On Metal their "measurements" could only be
  an engine refusal, so a HyperCore sweep there would produce ``FAIL_BOOT`` rows rather than
  evidence.
* It resolves the engine by walking up from its own directory looking for a Strata checkout, and it
  writes its deliverables next to that root. Neither is acceptable for a packaged application: the
  component registry must name an exact engine build, and the active profile must not be mutated by
  a search.
* Its README states Linux and Windows support and its tests use stub engines. That is a claim about
  the harness, not a qualification of any GPU.

So the decision, recorded here rather than in a design document alone: **Strata owns the job, the
lease, the evidence and the apply step; HyperCore-Tune remains a candidate optimizer behind that
job for the backends whose flags it actually targets.** The dimension table below is the mechanism
that makes this safe — a dimension whose flag the selected backend cannot accept is *excluded with a
reason*, so a sweep can never "measure" a configuration the engine would refuse.

What this module deliberately does not do
-----------------------------------------

* It never writes the active profile. It produces a report; applying is a separate step.
* It never invents a metric. A candidate that failed to boot, produced no tokens, or whose
  measurement could not be taken is recorded with its failure and excluded from selection.
* It never claims a winner from fewer than the required repeats, and it always re-measures the
  *baseline* at the end so drift is visible in the report.
* It never promises a speed-up. The report says what was measured, on what, and how many times.

Stdlib only, like the rest of Strata Control: the tuner is optional and its absence must leave
serving untouched.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from strata_control import atomic

SCHEMA_VERSION = 1

# --------------------------------------------------------------------------- backends

#: The backend families the product targets. The key is what the engine package manifest and the
#: profile record, not something inferred from a device name.
BACKENDS = ("metal", "cuda", "hip", "sycl", "cpu")


@dataclass(frozen=True)
class Dimension:
    """One searchable engine setting.

    ``flags`` lists the engine flags this dimension drives, and ``backends`` the backends on which
    those flags are meaningful. A dimension is offered only when *every* flag it drives is supported
    by the selected backend — a partial lever would produce a configuration the engine cannot run.
    """

    id: str
    flags: tuple[str, ...]
    values: tuple[Any, ...]
    backends: tuple[str, ...]
    help: str
    #: Benchmark-only levers that must be stripped from the exported production args.
    benchmark_only: bool = False


#: The searchable dimensions. Only entries whose flags a backend actually accepts are offered, and
#: which those are is *checked against the engine binary*, not assumed — see `validate_dimensions`.
DIMENSIONS: tuple[Dimension, ...] = (
    Dimension("ple_io", ("--ple-io",), ("direct", "mmap", "ram"),
              ("metal", "cuda", "hip", "sycl", "cpu"),
              "How the n-gram table is read: unbuffered SSD reads, mmap, or locked in RAM."),
    Dimension("kv", ("--kv",), ("fp16", "int8", "q4_0", "k8v4"),
              ("metal", "cuda", "hip", "sycl", "cpu"),
              "KV storage format. Lower precision is cheaper and slightly less exact."),
    Dimension("kv_grow", ("--kv-grow",), (False, True),
              ("metal", "cuda", "hip", "sycl", "cpu"),
              "Allocate KV only for the cells actually reached, returning VRAM to the expert cache."),
    Dimension("expert_cache", ("--expert-cache",), (0, 1024, 2048, 4096, 8192, 12288, 16384),
              ("cuda", "hip", "sycl"), "MiB of device memory for the expert cache."),
    Dimension("host_core", ("--host-core",), ("first", "last"),
              ("cuda", "hip"), "Which cores run the host-side work."),
    Dimension("pool_affinity", ("--pool-affinity",), ("all", "p-cores", "auto"),
              ("cuda", "hip"), "CPU pool affinity for the expert I/O threads."),
    Dimension("pool_workers", ("--pool-workers",), (0, 2, 4, 8, 12, 16),
              ("cuda", "hip"), "I/O workers; 0 means every physical core except the host's."),
    Dimension("spec_min_p", ("--spec-min-p",), (0.001, 0.1, 0.5, 0.9),
              ("metal", "cuda", "hip", "sycl", "cpu"),
              "Draft-acceptance floor: how confident a speculative token must be to be proposed."),
    Dimension("vram_reserve", ("--vram-reserve-mib",), (256, 512, 768, 1024, 1280, 1536),
              ("cuda", "hip"),
              "Device memory left free for other programs. NVIDIA/AMD only."),
    Dimension("pcie_frac", ("--pcie-frac",), (0.0, 0.25, 0.5, 0.75, 0.95),
              ("cuda", "hip"), "Share of PCIe bandwidth the expert streamer may use. NVIDIA/AMD only."),
    Dimension("prompt_cache_tail", ("--prompt-cache-tail",), (False, True),
              ("cuda", "hip", "sycl"),
              "Prompt-cache tail retention. BENCHMARK ONLY: measured, never exported.",
              benchmark_only=True),
)

#: Flags that take no value: on when present, off when absent. A candidate records them as "1" / "0".
BOOLEAN_FLAGS = frozenset(flag for _d in DIMENSIONS if all(isinstance(v, bool) for v in _d.values) for flag in _d.flags)


def flag_argv(flag: str, value: Any) -> list[str]:
    """One candidate setting as engine argv. A boolean flag is `--flag` when on and nothing when off; it never
    gets a value ("--kv-grow 0" is "unknown argument: 0" to the engine, and a bare "--kv-grow" meaning off enabled it)."""
    if flag in BOOLEAN_FLAGS or isinstance(value, bool):
        return [flag] if value in (True, "1", "true", "True") else []
    if value is None or value == "":
        return [flag]
    return [flag, str(value)]


#: Dimensions a backend must never be offered, with the reason kept for the report.
def _why_excluded(dim: Dimension, backend: str) -> str | None:
    if backend in dim.backends:
        return None
    if backend == "metal":
        if dim.id in ("vram_reserve", "pcie_frac", "expert_cache"):
            return ("the Apple Silicon engine does not parse this flag; it is an NVIDIA/AMD device "
                    "concept, so a sweep could only record a boot failure")
        if dim.id in ("host_core", "pool_affinity", "pool_workers"):
            return ("the Metal build's CPU-side pool is not exposed; the flag parses but its "
                    "entry point is one of the unported kernels (metal/src/engine/stubs.cpp)")
    return f"not a {backend} setting"


def validate_dimensions(backend: str,
                        engine_flags: Sequence[str] | None = None) -> tuple[list[Dimension], list[dict[str, str]]]:
    """Return the dimensions this backend may search, and why the rest were excluded.

    When ``engine_flags`` is supplied — the flags the actual engine binary reports — a dimension is
    additionally required to name only flags in that list. That is the check that keeps a sweep from
    measuring a configuration the engine would refuse: if the binary does not accept a flag, the
    dimension is excluded with the flag named, no matter what the static table says.
    """
    offered: list[Dimension] = []
    excluded: list[dict[str, str]] = []
    known = set(engine_flags) if engine_flags is not None else None
    for dim in DIMENSIONS:
        reason = _why_excluded(dim, backend)
        if reason is None and known is not None:
            missing = [f for f in dim.flags if f not in known]
            if missing:
                reason = ("the engine binary does not accept " + ", ".join(missing) +
                          ", so a candidate using this dimension could not start")
        if reason is None:
            offered.append(dim)
        else:
            excluded.append({"id": dim.id, "reason": reason})
    return offered, excluded


def engine_command(engine: Path | str) -> list[str]:
    """The argv prefix that runs an engine: the binary itself, or ``[sys.executable, script.py]``.

    A ``.py`` file stands in for the engine in tests and demos (the same READY/PP/T/DONE protocol,
    no weights). Running it through the current interpreter is what works on every OS: a ``#!`` line
    is ignored by Windows, which answers WinError 193.
    """
    import sys
    path = str(engine)
    return [sys.executable, path] if path.lower().endswith(".py") else [path]


def engine_env(engine: Path | str) -> dict[str, str] | None:
    """The environment to run an engine with, or None for the caller's own.

    An engine package (`engine-component.json` beside the binary) carries its GPU runtime libraries in its own
    folder, and `serve` puts that folder first on the library path. A tuning run that starts the same binary must
    too, or it cannot even print `--help` ("error while loading shared libraries")."""
    folder = Path(engine).resolve().parent
    if not (folder / "engine-component.json").is_file():
        return None
    var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    env = dict(os.environ)
    from strata_control import engine_component
    env[var] = os.pathsep.join(engine_component.package_lib_dirs(folder) + ([env[var]] if env.get(var) else []))
    return env


def engine_flag_inventory(engine: Path) -> list[str] | None:
    """Read the flags an engine binary documents, or None when it cannot be run here.

    Both output streams are read. The Strata engine writes its usage to **stderr**, which is easy to
    miss: a stdout-only read reports "no flags" for a perfectly good engine, and because every
    dimension is then excluded the sweep silently searches nothing. That failure is safe but
    useless, so the parser reads stderr too and only reports None when the binary could not be run
    at all.
    """
    try:
        proc = subprocess.run([*engine_command(engine), "--help"], capture_output=True, text=True, errors="replace", timeout=20,
                              env=engine_env(engine))
    except (OSError, subprocess.SubprocessError):
        return None
    import re
    text = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    flags = sorted(set(re.findall(r"^\s+(--[a-z0-9-]+)", text, re.MULTILINE)))
    # `--help` on this engine exits 0 but writes to stderr; exit codes are not a reliable signal, so
    # the presence of a usable flag list is.
    return flags or None


# --------------------------------------------------------------------------- the job

@dataclass
class Candidate:
    """One configuration and everything that was observed about it."""

    index: int
    settings: dict[str, Any]
    args: dict[str, str] = field(default_factory=dict)
    status: str = "pending"          # PASS | FAIL_BOOT | NO_TOKENS | TIMEOUT | UNMEASURED | INVALID
    reason: str = ""
    metrics: list[dict[str, float]] = field(default_factory=list)
    # `baseline` is the reference measurement; `recheck` is the same configuration measured again
    # after the sweep so machine drift is visible; `trial` is a candidate.
    kind: str = "trial"              # baseline | recheck | trial

    @property
    def is_baseline(self) -> bool:
        return self.kind == "baseline"

    @property
    def ok(self) -> bool:
        return self.status == "PASS" and bool(self.metrics)

    def median(self, metric: str) -> float | None:
        vals = [m[metric] for m in self.metrics if isinstance(m.get(metric), (int, float))]
        return statistics.median(vals) if vals else None

    def worst(self, metric: str) -> float | None:
        vals = [m[metric] for m in self.metrics if isinstance(m.get(metric), (int, float))]
        return min(vals) if vals else None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("is_baseline", None)          # derived from `kind`
        metric = "tokens_per_s"
        data["median"] = self.median(metric)
        data["worst"] = self.worst(metric)
        data["measured"] = len(self.metrics)
        return data


@dataclass
class TuningPlan:
    """What a sweep would do, before it does anything.

    A plan is the review surface: it names the exact engine, model and profile revisions, the
    dimensions and their searched values, what is held fixed, the workload, the objective, the
    limits, and the resources it needs exclusively.
    """

    id: str
    backend: str
    engine: str
    engine_flags_known: bool
    model: str
    profile_revision: str | None
    objective: str
    workload: dict[str, Any]
    dimensions: list[dict[str, Any]]
    excluded: list[dict[str, str]]
    fixed: dict[str, Any]
    max_trials: int
    total_time_s: float
    repeats: int
    devices: list[str]
    needs_exclusive: bool = True
    created_at: float = field(default_factory=time.time)
    #: What was measured, by content rather than by name. Two sweeps are comparable only when these
    #: match — see `tune_history.incomparable_reasons`.
    identity: dict[str, Any] = field(default_factory=dict)
    context: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Tuner:
    """A bounded search with durable progress.

    The measurements come from an injected ``measure`` callable, so the control flow, the selection
    rule and the retest are testable without a GPU — and, more importantly, so the module cannot
    accidentally contain a hardcoded benchmark result.

    ``measure(candidate) -> dict`` must return a dict with at least ``tokens_per_s`` and may carry
    ``ttft_ms``, ``prefill_tokens_per_s``, ``decode_tokens_per_s``, ``memory_mib``, ``power_w`` and
    ``correct`` (a correctness check the caller performs). Raising means the candidate could not be
    measured; the exception's message becomes the candidate's status.
    """

    def __init__(self, root: Path, plan: TuningPlan,
                 measure: Callable[[Candidate], dict[str, float]],
                 clock: Callable[[], float] = time.time) -> None:
        self.id = plan.id
        self.root = Path(root) / plan.id
        self.plan = plan
        self.measure = measure
        self.clock = clock
        self.candidates: list[Candidate] = []
        self.trials_run = 0
        self.history: dict[str, Any] = {}
        self.state = "queued"
        self.started_at = self.clock()
        self.finished_at: float | None = None
        self.events: list[dict[str, Any]] = []
        self.error: str | None = None
        self._cancel = False
        self._lease: Path | None = None
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- persistence -----------------------------------------------------

    @property
    def state_file(self) -> Path:
        return self.root / "job.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "state": self.state,
            "plan": self.plan.to_dict(),
            "candidates": [asdict(c) for c in self.candidates],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "events": self.events,
            "error": self.error,
            # A job that cannot compose its own report must still be persistable, or the failure
            # path would recurse and lose the reason it failed.
            "report": self._report_or_none(),
        }

    def save(self) -> None:
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        atomic.replace(tmp, self.state_file)

    def _report_or_none(self) -> dict[str, Any] | None:
        if self.state not in ("completed", "cancelled", "failed"):
            return None
        try:
            return self.report()
        except Exception as e:  # noqa: BLE001 - never lose the job record over a report bug
            return {"unavailable": f"{type(e).__name__}: {e}"}

    def _enter(self, state: str, **extra: Any) -> None:
        self.state = state
        self.events.append({"seq": len(self.events), "at": self.clock(), "state": state, **extra})
        self.save()

    def request_cancel(self) -> None:
        """Ask the sweep to stop at the next trial boundary.

        A trial already booting is allowed to finish or be killed by the measurement function; the
        point is that the sweep stops *starting* new ones, and the evidence collected so far is
        kept. Nothing is reported as a winner from a cancelled sweep.
        """
        self._cancel = True
        self.save()

    # ---- the exclusive lease --------------------------------------------

    def _acquire_lease(self) -> None:
        """Take the exclusive tuning lease for the selected devices.

        Tuning and serving must not share a device: a load beside a measurement changes what is
        measured, and the reviewed calibration's broad exclusive lease is preserved here until
        per-device leases are proven. The lease is an ``O_EXCL`` marker, so a crashed sweep leaves a
        stale one that a later run reports rather than silently ignoring.
        """
        if not self.plan.needs_exclusive:
            return
        lease = self.root / "tuning.lease"
        try:
            fd = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            other = json.loads(lease.read_text(encoding="utf-8")) if lease.exists() else {}
            raise RuntimeError(
                "another tuning job holds the exclusive device lease "
                f"({other.get('id', 'unknown')}, pid {other.get('pid', '?')}); "
                "finish or cancel it, or remove the lease if that job is gone") from None
        with os.fdopen(fd, "w") as fh:
            json.dump({"id": self.id, "pid": os.getpid(), "at": self.clock(),
                       "devices": self.plan.devices}, fh)
        self._lease = lease

    def _release_lease(self) -> None:
        if self._lease is not None:
            try:
                self._lease.unlink()
            except OSError:
                pass
            self._lease = None

    # ---- the sweep -------------------------------------------------------

    def run(self) -> dict[str, Any]:
        try:
            self._acquire_lease()
            self._enter("measuring_baseline")
            baseline = Candidate(index=0, settings=dict(self.plan.fixed), kind="baseline",
                                 args=self._args_for(dict(self.plan.fixed)))
            self._measure_candidate(baseline)
            self.candidates.append(baseline)
            self.save()

            grid = self._grid()
            if baseline.ok and not grid:
                self.events.append({"seq": len(self.events), "at": self.clock(),
                                    "state": "no_dimensions",
                                    "reason": "this backend offers no searchable dimension; "
                                              "the baseline was measured and nothing was compared"})
            if baseline.ok:
                self._enter("searching", trials=min(len(grid), self.plan.max_trials))
                started = self.clock()
                for index, settings in enumerate(grid, start=1):
                    if self._cancel:
                        break
                    if index > self.plan.max_trials:
                        break
                    if self.clock() - started > self.plan.total_time_s:
                        self.events.append({"seq": len(self.events), "at": self.clock(),
                                            "state": "time_limit", "after": index - 1})
                        break
                    candidate = Candidate(index=index, settings=settings,
                                          args=self._args_for(settings))
                    self._measure_candidate(candidate)
                    self.candidates.append(candidate)
                    self.trials_run += 1
                    self.save()
            else:
                self.events.append({"seq": len(self.events), "at": self.clock(),
                                    "state": "baseline_failed",
                                    "reason": baseline.reason or baseline.status})

            # Always re-measure the baseline last, so drift is visible in the report rather than
            # hidden inside the comparison.
            if baseline.ok and not self._cancel:
                self._enter("rechecking_baseline")
                again = Candidate(index=-1, settings=dict(self.plan.fixed), kind="recheck",
                                  args=self._args_for(dict(self.plan.fixed)))
                self._measure_candidate(again)
                again.index = len(self.candidates)
                self.candidates.append(again)

            self._enter("cancelled" if self._cancel else "completed")
            self.finished_at = self.clock()
            self._record_history()
            self.save()
            return self.to_dict()
        except Exception as e:  # noqa: BLE001 - a failed job is a recorded outcome
            self.error = f"{type(e).__name__}: {e}"
            self._enter("failed", error=self.error)
            self.finished_at = self.clock()
            self.save()
            return self.to_dict()
        finally:
            self._release_lease()

    def _measure_candidate(self, candidate: Candidate) -> None:
        """Take the required repeats, and record why a candidate could not be measured."""
        try:
            warm = self.measure(Candidate(index=candidate.index, settings=candidate.settings,
                                          args=candidate.args, kind=candidate.kind))
            if not isinstance(warm, dict):
                raise TypeError("the measurement function must return a dict")
        except Exception as e:  # noqa: BLE001
            message = str(e)
            lowered = message.lower()
            candidate.status = ("TIMEOUT" if "timeout" in lowered or "timed out" in lowered else
                                "FAIL_BOOT" if "boot" in lowered or "start" in lowered else
                                "UNMEASURED")
            candidate.reason = message[:400]
            return

        for _ in range(max(1, self.plan.repeats)):
            try:
                m = self.measure(Candidate(index=candidate.index, settings=candidate.settings,
                                           args=candidate.args, kind=candidate.kind))
            except Exception as e:  # noqa: BLE001
                candidate.status = "TIMEOUT" if "timeout" in str(e).lower() else "UNMEASURED"
                candidate.reason = str(e)[:400]
                return
            # Order matters: these two rejections must happen *before* the measurement is recorded,
            # otherwise a fast but incorrect configuration accumulates passing metrics and its
            # reason gets overwritten with PASS.
            if not isinstance(m.get("tokens_per_s"), (int, float)) or not m["tokens_per_s"]:
                # A configuration that produced no tokens is not a slow configuration; it failed.
                candidate.status = "NO_TOKENS"
                candidate.reason = "the engine produced no tokens"
                return
            if m.get("correct") is False:
                candidate.status = "INVALID"
                candidate.reason = "a correctness check failed for this configuration"
                return
            candidate.metrics.append({k: v for k, v in m.items() if isinstance(v, (int, float))})
        candidate.status = "PASS"

    def _grid(self) -> list[dict[str, Any]]:
        """A bounded, deterministic grid over the offered dimensions.

        Deterministic on purpose: a fixed seed and a fixed enumeration mean two sweeps are
        comparable, and a reviewer can reproduce any candidate from its index. An adaptive search
        (TPE) can be layered behind the same job contract later; it must log every suggestion and
        every outcome, which this structure already records.
        """
        # The plan is authoritative for *which* dimensions are searched; the static table supplies
        # the values and the flags. Reading the values from the plan as well (when it carries them)
        # keeps a caller able to narrow a dimension without editing the table.
        planned = {x.get("id"): x for x in self.plan.dimensions}
        offered: list[Dimension] = []
        for dim in DIMENSIONS:
            spec = planned.get(dim.id)
            if spec is None:
                continue
            values = tuple(spec.get("values") or dim.values)
            offered.append(dim if values == dim.values else
                           Dimension(dim.id, dim.flags, values, dim.backends, dim.help,
                                     dim.benchmark_only))
        if not offered:
            return []
        combos: list[dict[str, Any]] = [{}]
        for dim in offered:
            grown: list[dict[str, Any]] = []
            for base in combos:
                for value in dim.values:
                    grown.append({**base, dim.id: value})
            combos = grown
        return combos

    def _args_for(self, settings: dict[str, Any]) -> dict[str, str]:
        """Render settings as engine flags, dropping benchmark-only levers.

        `prompt_cache_tail` is the worked example the handoff calls out: it can be measured, but the
        production interface does not accept it, so a winner that depended on it would not survive
        export. It is dropped here and the report says so.
        """
        by_id = {d.id: d for d in DIMENSIONS}
        args: dict[str, str] = {}
        for key, value in settings.items():
            dim = by_id.get(key)
            if dim is None or dim.benchmark_only:
                continue
            flag = dim.flags[0]
            if isinstance(value, bool):
                args[flag] = "1" if value else "0"
            else:
                args[flag] = str(value)
        return args

    # ---- the report ------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """What was measured, what is recommended, and what is still unknown.

        Selection rule, in order:
          1. the candidate must have passed at least ``repeats`` measurements;
          2. it must not be worse than the baseline's *worst* measurement — comparing a candidate's
             best run against the baseline's best run is how noise gets promoted to an improvement;
          3. among those, the best median on the objective wins.

        When nothing satisfies that, the report recommends **no change**. That is a real result, not
        a failed run.
        """
        baselines = [c for c in self.candidates if c.kind == "baseline" and c.ok]
        rechecks = [c for c in self.candidates if c.kind == "recheck" and c.ok]
        trials = [c for c in self.candidates if c.kind == "trial"]
        metric = self._objective_metric()
        higher_is_better = self.plan.objective in ("generation", "prefill")

        recommendation: dict[str, Any] | None = None
        baseline_ref = baselines[0] if baselines else None
        if baseline_ref is not None:
            base_worst = baseline_ref.worst(metric)
            base_median = baseline_ref.median(metric)
            eligible = []
            for c in trials:
                if not c.ok or len(c.metrics) < self.plan.repeats:
                    continue
                worst = c.worst(metric)
                median = c.median(metric)
                if worst is None or median is None:
                    continue
                # Rule 2: the candidate's worst run must not be worse than the baseline's worst.
                if base_worst is not None and higher_is_better and worst < base_worst:
                    continue
                if base_worst is not None and not higher_is_better and worst > base_worst:
                    continue
                eligible.append((median, c))
            if eligible:
                eligible.sort(key=lambda p: p[0], reverse=higher_is_better)
                best_median, best = eligible[0]
                if base_median is not None and (
                        (higher_is_better and best_median > base_median) or
                        (not higher_is_better and best_median < base_median)):
                    recommendation = {
                        "candidate_index": best.index,
                        "settings": best.settings,
                        "args": best.args,
                        "median": best_median,
                        "baseline_median": base_median,
                    }

        drift = None
        if baselines and rechecks:
            first, last = baselines[0].median(metric), rechecks[-1].median(metric)
            if first is not None and last is not None and first:
                drift = {"metric": metric, "first": first, "last": last,
                         "relative": (last - first) / first,
                         "note": ("the baseline was measured again after the sweep; a large "
                                  "relative change means the machine drifted and the comparison is "
                                  "weaker than the repeat count suggests")}

        return {
            "objective": self.plan.objective,
            "metric": metric,
            "higher_is_better": higher_is_better,
            "repeats": self.plan.repeats,
            "baseline": baseline_ref.to_dict() if baseline_ref else None,
            "recheck": rechecks[-1].to_dict() if rechecks else None,
            "recommendation": None if self.state == "cancelled" else recommendation,
            "recommendation_available": recommendation is not None and self.state != "cancelled",
            "reason": ("the sweep was cancelled, so no comparison was completed and nothing is "
                       "recommended" if self.state == "cancelled" else
                       "no candidate beat the baseline outside its own spread; keeping the current "
                       "configuration" if recommendation is None and baseline_ref else
                       "the baseline could not be measured, so nothing could be compared"
                       if baseline_ref is None else ""),
            "candidates": [c.to_dict() for c in self.candidates],
            "trials_run": self.trials_run,
            # The grid is enumerated deterministically; a bounded sweep evaluates a *prefix* of it.
            # Reporting only "4 trials" would let a reader conclude that four was the whole space,
            # which is exactly how a partial search gets mistaken for a complete one.
            "combinations_total": self._grid_size(),
            "trials_are_a_prefix": self.trials_run < self._grid_size(),
            "failures": [{"index": c.index, "status": c.status, "reason": c.reason}
                         for c in self.candidates if not c.ok],
            "drift": drift,
            "drift_material": bool(drift and abs(drift["relative"]) >= 0.02),
            "excluded_dimensions": self.plan.excluded,
            "identity": dict(self.plan.identity),
            "history": self.history,
            "excluded_from_export": sorted(d.id for d in DIMENSIONS if d.benchmark_only),
        }

    def _grid_size(self) -> int:
        """How many combinations the offered dimensions describe, before any bound."""
        offered = {x.get("id") for x in self.plan.dimensions}
        total = 1
        for dim in DIMENSIONS:
            if dim.id in offered:
                spec = next(x for x in self.plan.dimensions if x.get("id") == dim.id)
                total *= len(spec.get("values") or dim.values)
        return total if offered else 0

    def _record_history(self) -> None:
        """Store this sweep and report what it is comparable to.

        Recording *and* comparing in one place is deliberate: a store that only wrote would let a
        report present a recommendation next to incomparable earlier numbers without saying so.
        """
        try:
            from strata_control import tune_history
        except ImportError:                                  # history is optional
            self.history = {"note": "the history store is unavailable"}
            return
        try:
            store = tune_history.HistoryStore(self.root.parent / "history")
            key = self.plan.identity or {}
            self.history = tune_history.summarise_history(store, key) if key else {
                "note": "this plan carries no artifact identity, so nothing can be compared to it"}
            report = self.report()
            rec = report.get("recommendation")
            store.record(
                key=key,
                outcome=(report.get("recommendation") or {}).get("outcome")
                        or ("cancelled" if self.state == "cancelled" else
                            "accepted" if report.get("recommendation_available") else "inconclusive"),
                recommendation_args=(rec or {}).get("args") or {},
                median=(rec or {}).get("median"),
                baseline_median=(rec or {}).get("baseline_median"),
                metric=report.get("metric", "tokens_per_s"),
                trials_run=report.get("trials_run", 0),
                combinations_total=report.get("combinations_total", 0),
                engine_path=self.plan.engine, model_path=self.plan.model,
                note=self.state)
        except Exception as e:  # noqa: BLE001 - history must never break a sweep
            self.history = {"note": f"the history store could not be used: {type(e).__name__}: {e}"}

    def _objective_metric(self) -> str:
        return {"generation": "tokens_per_s", "prefill": "prefill_tokens_per_s",
                "latency": "ttft_ms"}.get(self.plan.objective, "tokens_per_s")


# --------------------------------------------------------------------------- plan + apply

def plan_tuning(engine: Path, backend: str, model: str, objective: str,
                workload: dict[str, Any], profile_revision: str | None = None,
                hold: Sequence[str] = (), max_trials: int = 16, total_time_s: float = 1800.0,
                repeats: int = 3, devices: Sequence[str] = (),
                model_path: Path | None = None, context: int | None = None,
                full_digests: bool = False) -> TuningPlan:
    """Build the reviewable plan. Nothing is measured here; the plan is what a user approves.

    The default ``max_trials`` is the specification's short sweep. ``total_time_s`` is a displayed
    bound: a sweep stops starting new trials when it is reached, and the report says how many ran.
    """
    flags = engine_flag_inventory(engine)
    offered, excluded = validate_dimensions(backend, flags)
    held = set(hold)
    dimensions = [{"id": d.id, "values": list(d.values), "flags": list(d.flags), "help": d.help,
                   "benchmark_only": d.benchmark_only} for d in offered if d.id not in held]
    fixed = {d.id: ("held at the engine's own default" if d.benchmark_only else None)
             for d in offered if d.id in held}
    for dim_id in held:
        if dim_id not in {d.id for d in offered}:
            excluded.append({"id": dim_id, "reason": "held constant, or not a dimension of this backend"})
    if flags is not None and not dimensions:
        # Every dimension was excluded. If the reason is that the binary lacks all of them, the
        # declared backend and the engine probably disagree, which the caller should see rather
        # than discover from an empty sweep.
        excluded.append({
            "id": "(plan)",
            "reason": ("every dimension for this backend was excluded against this engine binary; "
                       "check that --backend matches the engine package you are measuring"),
        })
    # Identity by content, so a later sweep can tell whether these numbers still apply. `model_path`
    # is optional: when the caller does not know where the weights are, the model's digest is
    # `None`, which makes the record incomparable to everything rather than falsely comparable.
    from strata_control import tune_history
    identity = tune_history.comparison_key(
        engine=tune_history.file_identity(engine, full=full_digests),
        model=(tune_history.file_identity(model_path, full=full_digests) if model_path
               else {"path": None, "exists": False, "method": None, "digest": None, "bytes": None}),
        backend=backend, workload=workload, context=context)
    return TuningPlan(
        id=uuid.uuid4().hex, backend=backend, engine=str(engine), identity=identity, context=context,
        engine_flags_known=flags is not None, model=model, profile_revision=profile_revision,
        objective=objective, workload=dict(workload), dimensions=dimensions, excluded=excluded,
        fixed=fixed, max_trials=max(0, int(max_trials)), total_time_s=float(total_time_s),
        repeats=max(1, int(repeats)), devices=list(devices),
    )


def verify_exported(engine: Path, args: dict[str, str], measure: Callable[[dict[str, str]], dict[str, float]],
                    *, repeats: int = 3) -> dict[str, Any]:
    """Re-measure the configuration *as it will actually be exported*.

    This is the step that makes a recommendation honest. A sweep may measure a lever that is stripped
    from the exported args (``prompt_cache_tail`` is the documented example), so the winner must be
    measured again with exactly the flags that would be written into the profile. If the retest does
    not reproduce the result, the recommendation is withdrawn.
    """
    results: list[dict[str, float]] = []
    for _ in range(max(1, repeats)):
        try:
            m = measure(args)
        except Exception as e:  # noqa: BLE001
            return {"verified": False, "reason": f"the exported configuration failed to run: {e}",
                    "measurements": results}
        if not isinstance(m.get("tokens_per_s"), (int, float)) or not m["tokens_per_s"]:
            return {"verified": False, "reason": "the exported configuration produced no tokens",
                    "measurements": results}
        if m.get("correct") is False:
            return {"verified": False, "reason": "a correctness check failed for the exported configuration",
                    "measurements": results}
        results.append(m)
    metric = "tokens_per_s"
    return {"verified": True, "metric": metric, "median": statistics.median(r[metric] for r in results),
            "measurements": results, "args": dict(args), "repeats": len(results)}


def machine_identity() -> dict[str, Any]:
    """The machine the measurements belong to.

    Tuning history is invalidated by a change here, because a different machine's numbers are not
    comparable — which is why this is recorded with every report rather than assumed stable.
    """
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
