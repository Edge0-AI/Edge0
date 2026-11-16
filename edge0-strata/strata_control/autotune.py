"""The auto-tuner: find the fastest settings for this PC, for this model, for what the user asked for.

What it does (TUNING-SPEC.md sections 5-6, measured on five machines):

1. **Starts from the user's own configuration** (the "base") and changes one knob at a time (`tune_knobs`). It never starts
   a candidate at "the first value of every dimension".
2. Measures with the real worker over HTTP, on the engine's own clock (`tune_session`), the same fixed request set every
   time. Restart knobs are tried as **interleaved rounds**: every round runs the base and each candidate once, in a
   rotating order, so machine drift cannot favour one of them. Per-request knobs need no restart and are tried first, in
   one engine.
3. Accepts a candidate only if the **median of its per-round ratios** to the base beats the noise: at least +3%, and twice
   the drift and twice the A/A gap measured on the base runs, in every round (quick) or at least 4 of 5 (thorough
   confirm), without losing more than 2% elsewhere (prompt speed). A gain the noise can explain is not a gain: the result
   is then "your current settings are already the fastest" and nothing changes.
4. Guards before every start: free RAM, free graphics memory, other engines running, the CPU busy with something else.
   A candidate that does not fit is skipped, never started. A candidate must not use more memory than the base the user
   asked for (so "leave 20% for my work" is respected). The first crash stops the stage.
5. Only knobs whose answers are byte-identical (class exact) are chosen on their own; the others need the user's tick.

Progress and time left are reported through a callback; a cancel stops at the next run and keeps nothing half-measured.
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from strata_control import engine_component, tune_knobs, tune_machine
from strata_control.tune_session import RequestSet, RunResult, Session, StartFailed

MIN_FREE_RAM_MIB = 3584            # 3.5 GiB left for the system at every start (6.5)
EARLY_DROP = 0.05                  # a candidate this much slower than its own round's base is dropped after one round (run-to-run noise is 2-8%)
STAGE_GAP_NOISY = 0.04             # an A/A gap above this: the machine is noisy (6.5)


STAGNANT_WAIT_S = 60.0          # free memory that has not grown by STAGNANT_MIB after this long is not going to
STAGNANT_MIB = 512


class Cancelled(Exception):
    """The user stopped the tune; what was measured so far is kept in the result."""


# --------------------------------------------------------------------------- options and result

@dataclass
class Options:
    mode: str = "quick"                       # quick | thorough
    goal: str = "speed"                       # speed | room
    reserve_mib: int = 0                      # graphics memory to leave free (goal room)
    ram_wish_pct: int = 0                     # system memory to leave free, in percent (goal room, e.g. 20)
    use: str = "chat"                         # chat | coding
    allow_placement: bool = False
    allow_nonrepeatable: bool = False
    budget_s: float | None = None             # quick 20 min, thorough 2 h
    margin: float = 0.03
    rounds: int | None = None                 # quick 2, thorough 3 (screen); confirm 5 in thorough
    confirm_pairs: int | None = None          # thorough: 5
    per_request: bool = True
    min_free_ram_mib: int = MIN_FREE_RAM_MIB
    wait_ram_s: float = 600.0
    busy_pct: float = 25.0
    wait_busy_s: float = 300.0
    check_busy: bool = True

    def resolved(self) -> "Options":
        o = Options(**self.__dict__)
        thorough = o.mode == "thorough"
        o.budget_s = o.budget_s or (7200.0 if thorough else 1200.0)
        o.rounds = o.rounds or (3 if thorough else 2)
        o.confirm_pairs = o.confirm_pairs if o.confirm_pairs is not None else (5 if thorough else 0)
        return o


@dataclass
class ArmReport:
    id: str
    label: str
    settings: dict[str, Any]
    env: dict[str, Any]
    answer: str
    status: str = "pending"        # pending | tested | dropped | failed | skipped
    reason: str = ""
    runs: list[dict[str, Any]] = field(default_factory=list)       # per round: ratios vs that round's base
    ratio_decode: float | None = None
    ratio_pp: float | None = None
    pairs: int = 0
    faster: int = 0
    won: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TuneResult:
    outcome: str = "default_confirmed"   # improved | default_confirmed | inconclusive | base_failed | cancelled
    #: what a screen shows, one of: faster | already_fastest | busy_pc | failed | stopped  (see `finalize`)
    code: str = "already_fastest"
    #: why, for busy_pc: noisy | drift | not_tried (no memory or time to compare anything); for failed: start_failed | no_engine
    reason_code: str = ""
    #: the numbers a screen puts in its sentence, all plain numbers: gain_pct, base_tok_s, tuned_tok_s, noise_pct, drift_pct,
    #: minutes, pairs, faster_pairs, preset_name
    values: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    mode: str = "quick"
    seconds: float = 0.0
    waited_s: float = 0.0              # of those seconds, spent waiting (box lock, busy PC, memory)
    runs: int = 0
    base: dict[str, Any] = field(default_factory=dict)
    chosen: dict[str, Any] = field(default_factory=dict)
    gain_pct: float = 0.0
    gain_decode_pct: float = 0.0
    gain_pp_pct: float = 0.0
    settings: dict[str, Any] = field(default_factory=lambda: {"args": {}, "env": {}, "per_request": {}})
    answers: dict[str, Any] = field(default_factory=lambda: {"class": "exact", "fields": []})
    arms: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    noise: dict[str, Any] = field(default_factory=dict)
    pairs: int = 0
    faster_pairs: int = 0
    per_request: list[dict[str, Any]] = field(default_factory=list)
    per_request_confirm: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    trials: list[dict[str, Any]] = field(default_factory=list)       # every engine run, for the history store

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# --------------------------------------------------------------------------- pure decision rules

def metric(run: dict[str, Any] | RunResult, use: str) -> float | None:
    """The number a configuration is judged on (6.1): decode speed for chat; for coding the geometric mean of decode
    speed and long-prompt speed (the 4K prompt when no long one was measured)."""
    get = (lambda k: run.get(k)) if isinstance(run, dict) else (lambda k: getattr(run, k, None))
    if use == "prompt":                       # a prompt-only knob (the chunk size): long-prompt speed, the 4K prompt otherwise
        p = get("pp_long_tok_s") or get("pp_tok_s")
        return float(p) if p else None
    d = get("decode_tok_s")
    if not d:
        return None
    if use == "coding":
        p = get("pp_long_tok_s") or get("pp_tok_s")
        if p:
            return math.sqrt(d * p)
    return float(d)


def median_ratio(ratios: list[float]) -> float | None:
    return statistics.median(ratios) if ratios else None


def accept_threshold(margin: float, aa_gap: float, drift: float) -> float:
    """How far above 1.0 a median ratio has to be: the margin, or twice the noise the base runs showed (6.7 rule 3)."""
    return 1.0 + max(margin, 2.0 * aa_gap, 2.0 * drift)


def is_win(ratios: list[float], pp_ratios: list[float], *, threshold: float, need: int | None = None) -> bool:
    """A win: the median ratio beats the threshold, enough rounds were faster, and the prompt speed lost at most 2%."""
    if not ratios:
        return False
    need = len(ratios) if need is None else need
    faster = sum(1 for r in ratios if r > 1.0)
    if faster < need or statistics.median(ratios) < threshold:
        return False
    if pp_ratios and statistics.median(pp_ratios) < 0.98:
        return False
    return True


def rotate(items: list[Any], r: int) -> list[Any]:
    k = r % len(items) if items else 0
    return items[k:] + items[:k]


# --------------------------------------------------------------------------- the tuner

class Tuner:
    def __init__(self, session: Session, base_args: list[str], base_env: dict[str, str], ctx: tune_knobs.Context,
                 options: Options, *, progress: Callable[[dict[str, Any]], None] | None = None,
                 cancel: threading.Event | None = None, history_bad: set[str] | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 system: Callable[[], dict[str, Any]] | None = None, total_ram_mib: int | None = None,
                 total_vram_mib: int | None = None, quiet: Callable[[], str | None] | None = None) -> None:
        self.session = session
        self.base_args = list(base_args)
        self.base_env = dict(base_env)
        self.ctx = ctx
        self.o = options.resolved()
        self.progress = progress or (lambda p: None)
        self.cancel = cancel or threading.Event()
        self.history_bad = history_bad or set()
        self._ram_gave_up: tuple[float, float] | None = None
        self.clock = clock
        self.sleep = sleep
        self.system = system or session.system
        self.result = TuneResult(mode=self.o.mode)
        self.started = 0.0
        self.run_times: list[float] = []
        self.planned_runs = 0
        self.done_runs = 0
        self.base_runs: list[RunResult] = []
        self.total_ram = total_ram_mib or (tune_machine.memory().get("total_mib") or 0)
        self.total_vram = total_vram_mib
        self.quiet = quiet
        self.storm = False
        self.speeds: list[dict[str, Any]] = []
        self.waited_s = 0.0                # time spent waiting for the box lock, the PC to be quiet or memory: not the user's budget

    # ---- bookkeeping
    def _elapsed(self) -> float:
        return self.clock() - self.started - self.waited_s

    def _say(self, stage: str, message: str, **extra: Any) -> None:
        elapsed = self._elapsed()
        avg = statistics.mean(self.run_times) if self.run_times else 150.0
        remaining = max(0, self.planned_runs - self.done_runs)
        eta = min(max(0.0, self.o.budget_s - elapsed), remaining * avg) if self.planned_runs else None
        self.progress({"stage": stage, "message": message, "elapsed_s": round(elapsed), "runs_done": self.done_runs,
                       "runs_planned": self.planned_runs, "eta_s": None if eta is None else round(eta),
                       # what each measurement so far came to, for the live readout (label, decode tok/s as the goal judges it)
                       "speeds": list(self.speeds), **extra})

    def _check_cancel(self) -> None:
        if self.cancel.is_set():
            raise Cancelled()

    def _over_budget(self, reserve_s: float = 0.0) -> bool:
        return self._elapsed() + reserve_s > self.o.budget_s

    def _avg_run(self) -> float:
        return statistics.mean(self.run_times) if self.run_times else 150.0

    # ---- guards
    def _wait_for_quiet(self) -> None:
        """Pause while another engine runs or the CPU is busy with something else (6.5, 6.7 rule 5)."""
        if not self.o.check_busy:
            return
        waited = 0.0
        while True:
            why = self.quiet() if self.quiet else self._machine_busy()
            if not why:
                return
            if waited >= self.o.wait_busy_s:
                self.result.notes.append(f"measured while the machine was busy ({why})")
                return
            self._say("waiting", f"Waiting for the PC to be quiet ({why})")
            self._check_cancel()
            self.sleep(10.0)
            waited += 10.0

    def _machine_busy(self) -> str | None:
        eng = tune_machine.other_engines()
        if eng:
            return "another Strata engine is running"
        busy = tune_machine.cpu_busy_percent(1.0)
        if busy is not None and busy > self.o.busy_pct:
            return f"the CPU is {busy:.0f}% busy"
        return None

    def _ram_guard(self, delta_mib: int, label: str, wait_s: float | None = None) -> str | None:
        """None when a start fits; otherwise the plain reason it does not. Waiting ends early when the free memory does not
        grow (a loaded model keeps it): ten minutes per candidate made a tuning run last for hours (#3490)."""
        waited = 0.0
        first_avail: float | None = None
        while True:
            mem = self.system().get("memory", {})
            avail = mem.get("available_mib")
            if avail is None:
                return None                       # unreadable: the engine's own guard decides
            need = self.o.min_free_ram_mib + delta_mib
            base_rss = self.base_runs[0].rss_peak_mib if self.base_runs else None
            if base_rss:
                need += base_rss
            elif self.total_ram:
                need = max(need, self.o.min_free_ram_mib)
            if avail >= need:
                return None
            reason = (f"not enough free memory to start it safely ({avail / 1024:.1f} GiB free, "
                      f"{need / 1024:.1f} GiB needed)")
            prior = self._ram_gave_up
            if prior is not None and need >= prior[0] and avail <= prior[1] + STAGNANT_MIB:
                return reason                     # an earlier wait found nothing freed and this needs no less
            if first_avail is None:
                first_avail = avail
            stagnant = waited >= STAGNANT_WAIT_S and avail <= first_avail + STAGNANT_MIB
            if stagnant or waited >= (self.o.wait_ram_s if wait_s is None else wait_s):
                self._ram_gave_up = (need, avail)
                return reason
            self._say("waiting", f"Waiting for memory to be freed before trying {label} "
                                 f"({avail / 1024:.1f} GiB free, {need / 1024:.1f} GiB needed)")
            self._check_cancel()
            self.sleep(10.0)
            waited += 10.0

    # ---- one engine run
    def _run(self, args: list[str], env: dict[str, str | None], spec: RequestSet, label: str,
             points: Callable[[Any], None] | None = None, delta_mib: int = 0) -> RunResult:
        self._check_cancel()
        t0 = self.clock()

        entered: list[float] = []

        def pre_start() -> None:
            # checked again once the box lock is ours: another stream's engine may have come and gone while we waited
            lock_got = self.clock()
            self.waited_s += lock_got - t0
            self._wait_for_quiet()
            why = self._ram_guard(delta_mib, label, wait_s=60.0)
            self.waited_s += self.clock() - lock_got
            if why:
                raise StartFailed("SKIPPED", why)
            entered.append(self.clock())

        try:
            with self.session.engine(args, env, pre_start=pre_start) as eng:
                res = eng.measure(spec)
                if points is not None and res.status == "PASS":
                    points(eng)
        except StartFailed as e:
            res = RunResult(status=e.status, reason=e.reason)
        dt = self.clock() - (entered[0] if entered else t0)          # the run itself, not the waiting before it
        if res.status != "SKIPPED":
            self.run_times.append(dt)
            self.done_runs += 1
            self.result.runs += 1
        res.wall_s = round(dt, 1)
        self.result.trials.append({"label": label, "args": args, "env": env, "status": res.status, "reason": res.reason,
                                   "decode_tok_s": res.decode_tok_s, "pp_tok_s": res.pp_tok_s,
                                   "rss_peak_mib": res.rss_peak_mib, "load_s": res.load_s, "at": time.time()})
        return res

    # ---- the whole job
    def run(self) -> TuneResult:
        self.started = self.clock()
        r = self.result
        try:
            self._run_stages()
        except Cancelled:
            r.outcome, r.message = "cancelled", "Stopped. Your settings were not changed."
        r.seconds = round(self.clock() - self.started, 1)
        r.waited_s = round(self.waited_s, 1)
        self.finalize(r)
        return r

    @staticmethod
    def finalize(r: "TuneResult") -> None:
        """The structured outcome a screen shows inside the tune dialog (owner feedback, item 13): a code and plain numbers, so no
        screen has to read a sentence. faster / already_fastest / busy_pc (noisy, drifting, or nothing could be compared) /
        failed / stopped. `busy_pc` is a result, not an error: nothing was changed and trying again later is the answer."""
        r.code = {"improved": "faster", "default_confirmed": "already_fastest", "inconclusive": "busy_pc",
                  "base_failed": "failed", "cancelled": "stopped"}.get(r.outcome, "failed")
        if r.outcome == "default_confirmed" and r.noise.get("noisy"):
            r.reason_code = r.reason_code or "noisy_but_nothing_better"
        base = (r.base or {}).get("decode_tok_s")
        tuned = (r.chosen or {}).get("decode_tok_s")
        r.values = {"gain_pct": r.gain_pct, "gain_decode_pct": r.gain_decode_pct, "base_tok_s": base, "tuned_tok_s": tuned,
                    "noise_pct": round(100 * (r.noise or {}).get("spread", 0.0), 1) if r.noise else None,
                    "drift_pct": round(100 * (r.noise or {}).get("drift", 0.0), 1) if r.noise else None,
                    "minutes": round(r.seconds / 60.0 - r.waited_s / 60.0, 1), "pairs": r.pairs, "faster_pairs": r.faster_pairs}

    def _spec(self) -> RequestSet:
        long_tokens = 0
        if self.ctx.context_tokens >= 65536:
            # long contexts mean long prompts: the chunk size (spec 4.3) only shows on a long one
            long_tokens = min(98304, int(0.75 * self.ctx.context_tokens))
        elif self.o.use == "coding" and self.o.mode == "thorough":
            long_tokens = min(28000, int(0.8 * self.ctx.context_tokens))
        return RequestSet(warmups=2 if self.o.mode == "thorough" else 1, decodes=5, decode_tokens=200,
                          pp_tokens=4096 if self.ctx.context_tokens >= 6000 else 1024, long_tokens=long_tokens)

    def _run_stages(self) -> None:
        r, o = self.result, self.o
        spec = self._spec()
        self.planned_runs = 6                              # corrected once the candidates are known
        self._say("start", "Checking your PC before the first measurement")

        # ---- stage 1: the base run (the user's configuration, untouched), with the per-request guard in the same engine
        why = self._ram_guard(0, "your current settings")
        if why:
            r.outcome, r.message = "base_failed", "Not tuned: " + why
            r.reason_code = "no_memory"
            return
        pr_rows: list[dict[str, Any]] = []

        def per_request_points(eng: Any) -> None:
            if o.per_request:
                pr_rows.extend(self._per_request_stage(eng))

        self._say("base", "Measuring your current settings")
        base1 = self._run(self.base_args, {}, spec, "base", points=per_request_points)
        self.base_runs.append(base1)
        if base1.status == "PASS" and metric(base1, o.use):
            self.speeds.append({"label": "base", "tok_s": round(float(metric(base1, o.use)), 1)})
            self._say("base", "Measured your current settings")
        if base1.status != "PASS" or not metric(base1, o.use):
            r.outcome = "base_failed"
            r.reason_code = "start_failed"
            r.message = "Your current settings could not be measured: " + (base1.reason or base1.status)
            return
        r.base = base1.to_dict()
        r.facts = dict(base1.facts)
        self.ctx.facts = dict(base1.facts)
        if base1.vram_used_mib is not None and self.total_vram is None:
            cards = [g for g in self.system().get("gpus", []) if g.get("vram_total_mib")]
            self.total_vram = sum(g["vram_total_mib"] for g in cards) if cards else None
        r.per_request = pr_rows
        r.classes = tune_knobs.machine_classes(self.ctx)

        # ---- stage 2: plan the candidates for this machine; how many depends on the time chosen
        planned, skipped = tune_knobs.plan_arms(self.ctx, mode=o.mode, history_bad=self.history_bad)
        r.skipped = skipped
        per_arm = o.rounds or 2
        if o.mode == "thorough":
            n_arms = len(planned)
        else:
            left = int((o.budget_s - self._elapsed()) / max(1.0, self._avg_run()))
            fixed = (per_arm - 1) + 1                      # extra base rounds, the final base
            n_arms = min(len(planned), max(0, (left - fixed) // per_arm))
        for p in planned[n_arms:]:
            skipped.append({"id": p.arm.id, "label": p.arm.label, "reason": "left out to stay inside the time you chose"})
        planned = planned[:n_arms]
        arms = [ArmReport(p.arm.id, p.arm.label, dict(p.args), dict(p.env), p.arm.answer) for p in planned]
        by_id = {a.id: a for a in arms}
        by_use = {p.arm.id: ("prompt" if p.arm.metric == "prompt" else o.use) for p in planned}
        self.planned_runs = self.done_runs + per_arm * (len(planned) + 1) - 1 + 1 + (o.confirm_pairs or 0) * 2
        if not planned:
            self._say("plan", "No other setting is worth trying on this PC")

        # ---- stage 3: interleaved rounds. Every round runs the base and each live candidate once, in a rotating order.
        round_runs: list[dict[str, RunResult]] = [{} for _ in range(per_arm)]
        round_runs[0]["base"] = base1
        alive = list(planned)
        stop = False
        for rnd in range(per_arm):
            if stop or self.storm or (rnd > 0 and not alive):
                break
            for item in rotate(["base"] + alive, rnd):
                self._check_cancel()
                if self._over_budget(reserve_s=2 * self._avg_run()):
                    r.notes.append("the time you chose ran out before every round finished")
                    stop = True
                    break
                if item == "base":
                    if rnd == 0:
                        continue                         # round 1's base is the run above
                    if self._ram_guard(0, "your current settings"):
                        r.notes.append("not enough free memory for another measurement of your current settings; the search stopped")
                        stop = True
                        break
                    self._say("rounds", f"Measuring your current settings again (round {rnd + 1} of {per_arm})")
                    b = self._run(self.base_args, {}, spec, f"base r{rnd + 1}")
                    if b.status == "SKIPPED":
                        r.notes.append("not enough free memory for another measurement of your current settings; the search stopped")
                        self.storm = stop = True
                        break
                    if b.status != "PASS":
                        r.notes.append("your current settings failed to start a second time; the search stopped")
                        self.storm = stop = True
                        break
                    self.base_runs.append(b)
                    round_runs[rnd]["base"] = b
                    continue
                p = item
                rep = by_id[p.arm.id]
                why = self._ram_guard(p.arm.ram_delta_mib, p.arm.label)
                if why:
                    rep.status, rep.reason = "skipped", why
                    alive = [x for x in alive if x.arm.id != p.arm.id]
                    continue
                self._say("rounds", f"Trying: {p.arm.label} (round {rnd + 1} of {per_arm})", arm=p.arm.id)
                res = self._run(tune_knobs.apply_to_argv(self.base_args, p.args), dict(p.env), spec, p.arm.id,
                                delta_mib=p.arm.ram_delta_mib)
                if res.status == "SKIPPED":
                    rep.status, rep.reason = "skipped", res.reason
                    alive = [x for x in alive if x.arm.id != p.arm.id]
                    continue
                if res.status != "PASS":
                    rep.status, rep.reason = "failed", f"{res.status}: {res.reason}"
                    alive = [x for x in alive if x.arm.id != p.arm.id]
                    if res.status == "OOM":
                        self.storm = stop = True
                        r.notes.append("an engine ran out of memory; the search stopped (nothing was retried)")
                        break
                    continue
                if metric(res, o.use):
                    self.speeds.append({"label": p.arm.label, "tok_s": round(float(metric(res, o.use)), 1)})
                over = self._memory_overrun(res, base1)
                if over:
                    rep.status, rep.reason = "failed", over
                    alive = [x for x in alive if x.arm.id != p.arm.id]
                    continue
                round_runs[rnd][p.arm.id] = res
            # a candidate that lost by more than the margin against its own round's base is dropped from later rounds
            for p in list(alive):
                b, c = round_runs[rnd].get("base"), round_runs[rnd].get(p.arm.id)
                um = by_use[p.arm.id]
                m_b, m_c = (metric(b, um) if b else None), (metric(c, um) if c else None)
                if m_b and m_c and m_c / m_b < 1.0 - max(o.margin, EARLY_DROP):
                    by_id[p.arm.id].status = "dropped"
                    by_id[p.arm.id].reason = f"slower by {100 * (1 - m_c / m_b):.1f}% in round {rnd + 1}"
                    alive = [x for x in alive if x.arm.id != p.arm.id]
        data: dict[str, list[tuple[RunResult, RunResult]]] = {
            a.id: [(rr["base"], rr[a.id]) for rr in round_runs if a.id in rr and "base" in rr] for a in arms}

        # ---- noise: the A/A gap between base runs, the drift against a final base run
        end_base: RunResult | None = None
        ran_any = any(len(v) for v in data.values())
        if not self.storm and not self.cancel.is_set() and ran_any and not self._over_budget()                 and not self._ram_guard(0, "your current settings"):
            self._say("drift", "Measuring your current settings once more to check the PC did not drift")
            end_base = self._run(self.base_args, {}, spec, "base end")
            if end_base.status == "PASS":
                self.base_runs.append(end_base)
        mb = [metric(b, o.use) for b in self.base_runs if metric(b, o.use)]
        aa = abs(mb[1] / mb[0] - 1.0) if len(mb) >= 2 else 0.0
        drift = abs(mb[-1] / mb[0] - 1.0) if len(mb) >= 3 else (aa if len(mb) == 2 else 0.0)
        spread = (max(mb) / min(mb) - 1.0) if len(mb) >= 2 else 0.0
        r.noise = {"aa_gap": round(aa, 4), "drift": round(drift, 4), "spread": round(spread, 4),
                   "base_runs": [round(x, 2) for x in mb], "noisy": spread > STAGE_GAP_NOISY}
        threshold = accept_threshold(o.margin, aa, drift)

        # ---- evaluate the candidates
        winners: list[ArmReport] = []
        for p in planned:
            rep = by_id[p.arm.id]
            rows = data[p.arm.id]
            if rep.status in ("failed", "skipped") or not rows:
                continue
            ratios, pps = [], []
            um = by_use[p.arm.id]
            for b, c in rows:
                mc, mbb = metric(c, um), metric(b, um)
                if mc and mbb:
                    ratios.append(mc / mbb)
                # the other side of the trade: prompt speed for a decode knob, decode speed for a prompt knob
                if um == "prompt":
                    if c.decode_tok_s and b.decode_tok_s:
                        pps.append(c.decode_tok_s / b.decode_tok_s)
                elif c.pp_tok_s and b.pp_tok_s:
                    pps.append(c.pp_tok_s / b.pp_tok_s)
            rep.pairs = len(ratios)
            rep.faster = sum(1 for x in ratios if x > 1.0)
            rep.ratio_decode = round(statistics.median(ratios), 4) if ratios else None
            rep.ratio_pp = round(statistics.median(pps), 4) if pps else None
            rep.runs = [{"ratio": round(x, 4)} for x in ratios]
            if rep.status == "dropped":
                continue
            rep.status = "tested"
            if len(ratios) >= per_arm and is_win(ratios, pps, threshold=threshold):
                rep.won = True
                winners.append(rep)
        r.arms = [a.to_dict() for a in arms]

        if r.noise["noisy"] and not winners:
            r.notes.append(f"the PC was noisy ({spread * 100:.1f}% between identical runs)")

        # ---- stage 4: stack the winners, then (thorough) confirm against the base with more pairs
        fu = o.use
        chosen_args: dict[str, Any] = {}
        chosen_env: dict[str, str | None] = {}
        final_pairs: list[tuple[RunResult, RunResult]] = []
        if winners:
            winners.sort(key=lambda a: a.ratio_decode or 0.0, reverse=True)
            incumbent = [winners[0]]            # gains did not stack in three stacking runs (spec 1.10): the best single knob
            if len(winners) > 1:
                r.notes.append("more than one setting helped; only the best one is kept because gains did not add up in the study")
            for w in incumbent:
                chosen_args.update(w.settings)
                chosen_env.update(w.env)
            fu = by_use[incumbent[0].id]
            need_confirm = (o.confirm_pairs or 0) > 0
            if need_confirm and not self._over_budget(reserve_s=2 * self._avg_run()):
                n = o.confirm_pairs or 2
                self._say("confirm", "Confirming the combined result against your current settings")
                final_pairs = self._confirm(chosen_args, chosen_env, spec, n, fu)
                ratios = [metric(c, fu) / metric(b, fu) for b, c in final_pairs if metric(c, fu) and metric(b, fu)]
                pps = [c.decode_tok_s / b.decode_tok_s for b, c in final_pairs if c.decode_tok_s and b.decode_tok_s] if fu == "prompt" \
                    else [c.pp_tok_s / b.pp_tok_s for b, c in final_pairs if c.pp_tok_s and b.pp_tok_s]
                need = max(2, int(math.ceil(0.8 * len(ratios)))) if len(ratios) >= 5 else len(ratios)
                if not (ratios and is_win(ratios, pps, threshold=threshold, need=need)):
                    # the stack did not hold; fall back to the best single winner, which already won its own pairs
                    if len(incumbent) > 1:
                        r.notes.append("the combination did not beat the best single change; keeping the best single change")
                        best = incumbent[0]
                        chosen_args, chosen_env = dict(best.settings), dict(best.env)
                        final_pairs = [(b, c) for b, c in data[best.id]]
                    else:
                        winners, chosen_args, chosen_env, final_pairs = [], {}, {}, []
                        r.notes.append("the gain did not hold up in the confirmation, so nothing was changed")
                else:
                    pass
            else:
                best = incumbent[0]
                chosen_args, chosen_env = dict(best.settings), dict(best.env)
                final_pairs = list(data[best.id])
            if not final_pairs and winners:
                final_pairs = list(data[winners[0].id])

        fu_final = fu
        # ---- per-request winners (found in the base engine): keep only the ones that won every round
        per_request_chosen = {row["key"]: row["value"] for row in r.per_request if row.get("won")}
        if per_request_chosen and not self.storm and not self.cancel.is_set() and not self._over_budget(reserve_s=self._avg_run()):
            # a per-request win from one engine's quick points is only a candidate: confirm it in a fresh engine, 5 interleaved rounds
            self._say("confirm", "Confirming the settings that need no restart")
            r.per_request_confirm = self._confirm_per_request(per_request_chosen, spec, threshold)
            if not r.per_request_confirm["ok"]:
                r.notes.append("a per-request setting looked faster in the first test but did not hold in the confirmation, so it was not kept")
                per_request_chosen = {}
        elif per_request_chosen:
            r.notes.append("a per-request setting looked faster but could not be confirmed in the time left, so it was not kept")
            per_request_chosen = {}

        # ---- result
        if end_base is not None and end_base.status == "PASS" and drift > 0.03 and winners:
            gain_guess = statistics.median([metric(c, fu) / metric(b, fu) for b, c in final_pairs]) - 1.0 if final_pairs else 0.0
            if gain_guess < 2 * drift:
                chosen_args, chosen_env, winners, final_pairs = {}, {}, [], []
                per_request_chosen = {}
                r.outcome = "inconclusive"
                r.reason_code = "drift"
                r.message = ("The PC changed speed while measuring (about %.0f%%), more than the gain found, so nothing "
                             "was changed. Try again when nothing else is running." % (drift * 100))
        if r.outcome != "inconclusive":
            if chosen_args or chosen_env or per_request_chosen:
                r.outcome = "improved"
            elif r.noise["noisy"] and (spread > 2 * STAGE_GAP_NOISY):
                r.outcome = "inconclusive"
                r.reason_code = "noisy"
                r.message = "The PC was too noisy to tell settings apart, so nothing was changed. Try again when it is idle."
            elif arms and not any(a.status in ("tested", "dropped", "failed") for a in arms) and not r.per_request:
                # candidates were planned but none could be measured (no free memory, no time): that is not "already the fastest"
                r.outcome = "inconclusive"
                r.reason_code = "not_tried"
                r.message = ("Other settings could not be tried (not enough free memory or time), so nothing was compared and nothing was "
                             "changed. Try again when the PC is quiet.")
            else:
                r.outcome = "default_confirmed"
                r.message = "Your current settings are already the fastest we found. Nothing was changed."
                if r.noise.get("noisy"):
                    r.message += " (The PC was busy while measuring, so a small gain could be hidden.)"
        r.settings = {"args": chosen_args, "env": chosen_env, "per_request": per_request_chosen}
        cls, fields = "exact", []
        for rep in arms:
            if rep.won and any(k in chosen_args for k in rep.settings) or (rep.won and any(k in chosen_env for k in rep.env)):
                if tune_knobs.ANSWER_ORDER.index(rep.answer) > tune_knobs.ANSWER_ORDER.index(cls):
                    cls = rep.answer
                if rep.answer != "exact":
                    fields += list(rep.settings) + list(rep.env)
        for key in per_request_chosen:
            for row in tune_knobs.PER_REQUEST:
                if row["key"] == key and tune_knobs.ANSWER_ORDER.index(row["answer"]) > tune_knobs.ANSWER_ORDER.index(cls):
                    cls = row["answer"]
                if row["key"] == key and row["answer"] != "exact":
                    fields.append(key)
        r.answers = {"class": cls, "fields": sorted(set(fields)), "default_repeatable": not tune_knobs.default_unsteady(self.ctx)}
        if final_pairs:
            fu = fu_final
            ratios = [metric(c, fu) / metric(b, fu) for b, c in final_pairs if metric(c, fu) and metric(b, fu)]
            pps = [c.pp_tok_s / b.pp_tok_s for b, c in final_pairs if c.pp_tok_s and b.pp_tok_s]
            r.pairs = len(ratios)
            r.faster_pairs = sum(1 for x in ratios if x > 1.0)
            r.gain_pct = round(100 * (statistics.median(ratios) - 1.0), 1) if ratios else 0.0
            r.gain_pp_pct = round(100 * (statistics.median(pps) - 1.0), 1) if pps else 0.0
            dec = [c.decode_tok_s / b.decode_tok_s for b, c in final_pairs if c.decode_tok_s and b.decode_tok_s]
            r.gain_decode_pct = round(100 * (statistics.median(dec) - 1.0), 1) if dec else 0.0
            c_runs = [c for _, c in final_pairs]
            r.chosen = {"decode_tok_s": round(statistics.median([c.decode_tok_s for c in c_runs if c.decode_tok_s]), 2)
                        if any(c.decode_tok_s for c in c_runs) else None,
                        "pp_tok_s": round(statistics.median([c.pp_long_tok_s or c.pp_tok_s for c in c_runs if c.pp_long_tok_s or c.pp_tok_s]), 1)
                        if any(c.pp_tok_s for c in c_runs) else None,
                        "rss_peak_mib": max([c.rss_peak_mib for c in c_runs if c.rss_peak_mib] or [None]) if any(c.rss_peak_mib for c in c_runs) else None,
                        "vram_used_mib": max([c.vram_used_mib for c in c_runs if c.vram_used_mib] or [None]) if any(c.vram_used_mib for c in c_runs) else None}
        elif r.outcome == "improved" and per_request_chosen:
            r.gain_pct = r.gain_decode_pct = round(100 * ((r.per_request_confirm or {}).get("ratio", 1.0) - 1.0), 1)
            r.pairs = len((r.per_request_confirm or {}).get("ratios", []))
            r.faster_pairs = sum(1 for x in (r.per_request_confirm or {}).get("ratios", []) if x > 1.0)
            r.chosen = dict(r.base)
        else:
            r.chosen = {k: r.base.get(k) for k in ("decode_tok_s", "pp_tok_s", "rss_peak_mib", "vram_used_mib")}
        if r.outcome == "improved":
            r.message = "+%.0f%% faster answers" % r.gain_pct if r.gain_pct >= 0.5 else "slightly faster"
        self._say("done", r.message or "Done")

    # ---- helpers
    def _memory_overrun(self, res: RunResult, base: RunResult) -> str | None:
        """A candidate that takes more memory than the base the user asked for is not a candidate (their wish, 20% left
        for work, is a promise about memory)."""
        if res.rss_peak_mib and base.rss_peak_mib and res.rss_peak_mib > base.rss_peak_mib * 1.05 + 512 and self.o.goal == "room":
            return "it needs more memory than your current settings, which would eat into the memory you wanted left free"
        if res.min_avail_mib is not None and res.min_avail_mib < self.o.min_free_ram_mib and (
                base.min_avail_mib is None or res.min_avail_mib < base.min_avail_mib - 512):   # a RAM-bound base can sit below the floor itself
            return "it left the PC with too little free memory"
        if self.o.goal == "room" and self.o.ram_wish_pct and self.total_ram and res.min_avail_mib is not None and \
                base.min_avail_mib is not None:
            if res.min_avail_mib < base.min_avail_mib - max(512, 0.03 * self.total_ram):
                return "it left less memory free than your current settings"
        if self.o.goal == "room" and res.vram_used_mib is not None and base.vram_used_mib is not None and \
                res.vram_used_mib > base.vram_used_mib + 256:
            return "it uses more graphics memory than your current settings"
        if self.o.goal == "room" and self.o.reserve_mib and self.total_vram and res.vram_used_mib is not None and \
                base.vram_used_mib is not None and self.total_vram - base.vram_used_mib >= self.o.reserve_mib - 128 and \
                self.total_vram - res.vram_used_mib < self.o.reserve_mib - 128:
            return "it would not leave the graphics memory you wanted free"
        return None

    def _confirm(self, args: dict[str, Any], env: dict[str, str | None], spec: RequestSet, pairs: int, use: str | None = None) -> list[tuple[RunResult, RunResult]]:
        """Interleaved pairs of the combined settings against the base (the order alternates)."""
        out: list[tuple[RunResult, RunResult]] = []
        cand_args = tune_knobs.apply_to_argv(self.base_args, args)
        for i in range(pairs):
            self._check_cancel()
            if self._over_budget(reserve_s=2 * self._avg_run()):
                self.result.notes.append("confirmation stopped early: the time you chose ran out")
                break
            first = "cand" if i % 2 == 0 else "base"
            runs: dict[str, RunResult] = {}
            for who in (first, "base" if first == "cand" else "cand"):
                self._say("confirm", f"Confirming (pair {i + 1} of {pairs})")
                runs[who] = self._run(cand_args if who == "cand" else self.base_args, env if who == "cand" else {}, spec,
                                      f"confirm {who} {i + 1}")
            if runs["cand"].status == "PASS" and runs["base"].status == "PASS":
                use = use or self.o.use
                if metric(runs["cand"], use) and metric(runs["base"], use):
                    out.append((runs["base"], runs["cand"]))
                    self.base_runs.append(runs["base"])
            else:
                break
        return out

    def _per_request_stage(self, eng: Any) -> list[dict[str, Any]]:
        """Per-request settings against the engine's own choice, in this one engine, 3 interleaved rounds (6.3 step 2).

        A **control** point sends a per-request value equal to what the engine already uses (the configuration's own
        `--spec-min-p`), so it changes nothing: whatever it "gains" is the bias of sending a setting at all (the first run
        after an idle gap, the order). A value must beat the default by the margin AND beat the control by the margin in every
        round. A win here is only a candidate: the tuner confirms it afterwards in a fresh engine. One value per key can win."""
        points: list[tuple[str, Any]] = []
        for spec in tune_knobs.PER_REQUEST:
            if self.ctx.backend not in spec["backends"]:
                continue
            if spec["answer"] != "exact" and not tune_knobs.allowed(spec["answer"], self.ctx):
                continue
            for v in spec["values"]:
                points.append((spec["key"], v))
        if not points:
            return []
        own = tune_knobs._current(self.ctx.base_args, "--spec-min-p")
        control = ("control", float(own)) if own not in (None, "") and self.ctx.backend != "metal" else None
        rounds = 3
        default: dict[int, list[float]] = {}
        vals: dict[tuple[str, Any], dict[int, float]] = {pt: {} for pt in points + ([control] if control else [])}
        self._say("per_request", "Trying settings that need no restart")
        for rnd in range(rounds):
            order = rotate([None] + points + ([control] if control else []) + [None], rnd)
            for item in order:
                self._check_cancel()
                if item is None:
                    d = eng.decode_once(None, 200)
                    if d:
                        default.setdefault(rnd, []).append(d)
                else:
                    key = "spec_min_p" if item[0] == "control" else item[0]
                    d = eng.decode_once({key: item[1]}, 200)
                    if d:
                        vals[item][rnd] = d
        ref = {i: statistics.mean(v) for i, v in default.items() if v}
        ctl = [vals[control][i] / ref[i] for i in range(rounds) if control and i in vals[control] and i in ref] if control else []
        rows: list[dict[str, Any]] = []
        for pt in points:
            ratios = [vals[pt][i] / ref[i] for i in range(rounds) if i in vals[pt] and i in ref]
            adj = [ratios[j] / ctl[j] for j in range(min(len(ratios), len(ctl)))] if ctl else ratios
            won = len(ratios) == rounds and len(adj) == rounds and all(x >= 1.0 + self.o.margin for x in adj) \
                and all(x >= 1.0 + self.o.margin for x in ratios)
            rows.append({"key": pt[0], "value": pt[1], "ratios": [round(x, 4) for x in ratios], "control_ratios": [round(x, 4) for x in ctl],
                         "ratio": round(statistics.median(adj), 4) if adj else None, "won": won})
        best: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row["won"] and (row["key"] not in best or row["ratio"] > best[row["key"]]["ratio"]):
                best[row["key"]] = row
        for row in rows:
            row["won"] = best.get(row["key"]) is row
        return rows

    def _confirm_per_request(self, chosen: dict[str, Any], spec: RequestSet, threshold: float) -> dict[str, Any]:
        """5 interleaved rounds in a fresh engine of the chosen per-request settings against the engine's own choice (two decodes
        each, the order alternating). Kept only if the median ratio beats the threshold in at least 4 of 5 rounds."""
        ratios: list[float] = []

        def points(eng: Any) -> None:
            for i in range(5):
                self._check_cancel()
                got: dict[str, list[float]] = {"d": [], "t": []}
                for who in (("d", "t") if i % 2 == 0 else ("t", "d")):
                    for _ in range(2):
                        v = eng.decode_once(chosen if who == "t" else None, 200)
                        if v:
                            got[who].append(v)
                if got["d"] and got["t"]:
                    ratios.append(statistics.mean(got["t"]) / statistics.mean(got["d"]))

        res = self._run(self.base_args, {}, spec, "confirm per-request", points=points)
        ok = res.status == "PASS" and len(ratios) == 5 and sum(1 for x in ratios if x > 1.0) >= 4 \
            and statistics.median(ratios) >= threshold
        return {"ok": ok, "ratios": [round(x, 4) for x in ratios], "ratio": round(statistics.median(ratios), 4) if ratios else None,
                "status": res.status}


# --------------------------------------------------------------------------- building a tuner from a run configuration

def split_args(argv: list[str]) -> list[str]:
    return [str(a) for a in argv]


def detect_family(args: list[str]) -> str:
    for i, a in enumerate(args):
        if a == "--mtp" and i + 1 < len(args):
            return "qwen36" if str(args[i + 1]).isdigit() else "flash-next"
    if "--pack" in args or "--native" in args:
        return "flash-next"
    return "flash-next"


def model_gib_of(args: list[str]) -> float:
    """Total size of the model files a configuration names, shards included."""
    from pathlib import Path
    import re
    total = 0
    seen: set[str] = set()
    for i, a in enumerate(args):
        if a in ("--native", "--model", "--ple-gguf") and i + 1 < len(args):
            p = Path(str(args[i + 1]))
            if not p.is_file() or str(p) in seen:
                continue
            m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", p.name)
            group = sorted(p.parent.glob(f"{m.group(1)}-*-of-{m.group(3)}.gguf")) if m else [p]
            for f in group:
                if str(f) not in seen:
                    seen.add(str(f))
                    total += f.stat().st_size
    return round(total / 1024 ** 3, 1)


def context_from_config(cfg: dict[str, Any], *, engine_backend: str | None, opts: Options,
                        engine_flags: list[str] | None = None, system: dict[str, Any] | None = None,
                        family: str | None = None) -> tune_knobs.Context:
    args = split_args(cfg.get("args") or [])
    system = system or {"memory": tune_machine.memory(), "gpus": tune_machine.gpus(), "cpu": tune_machine.cpu()}
    gpu_cfg = cfg.get("gpu")
    wanted = gpu_cfg if isinstance(gpu_cfg, list) else ([gpu_cfg] if isinstance(gpu_cfg, int) else None)
    cards = system.get("gpus") or []
    used = [g for g in cards if not wanted or g.get("index") in wanted] or cards
    ctx_tokens = 32768
    for i, a in enumerate(args):
        if a == "--max-context" and i + 1 < len(args) and str(args[i + 1]).isdigit():
            ctx_tokens = int(args[i + 1])
    mem = system.get("memory") or {}
    cpu = system.get("cpu") or {}
    backend = engine_backend or ("cuda" if any(g.get("vendor") == "nvidia" for g in cards) else
                                 "hip" if any(g.get("vendor") == "amd" for g in cards) else "cpu")
    arch = None
    if backend == "hip":
        try:
            archs = engine_component.hip_archs_present()
            arch = archs[0] if archs else None
        except Exception:  # noqa: BLE001
            arch = None
    return tune_knobs.Context(
        backend=backend, family=family or detect_family(args), gpu_count=max(1, len(used)),
        vram_mib=min((g["vram_total_mib"] for g in used if g.get("vram_total_mib")), default=0),
        ram_gib=int((mem.get("total_mib") or 0) // 1024), model_gib=model_gib_of(args),
        hybrid_cpu=bool(cpu.get("hybrid")), logical_threads=int(cpu.get("logical") or 0), arch=arch,
        context_tokens=ctx_tokens, goal=opts.goal, use=opts.use, allow_placement=opts.allow_placement,
        allow_nonrepeatable=opts.allow_nonrepeatable, engine_flags=engine_flags, base_args=args)


# --------------------------------------------------------------------------- the short "still good?" re-check (8.3)

def recheck(session: Session, base_args: list[str], preset: dict[str, Any], *, ctx: tune_knobs.Context,
            engine_changed: bool = False, progress: Callable[[dict[str, Any]], None] | None = None,
            cancel: threading.Event | None = None, clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Is a saved preset still as fast as when it was made? One restart with the preset's settings and 30-45 s of requests.

    Pass rule (against the stored numbers, not a fresh baseline): decode and the 4K prompt at least 95% of what was stored,
    memory within the stored +5%. A failing prompt rate is measured once more with a second start before it counts (the
    prompt rate of some cards is bimodal between engine starts). When the engine changed in a way that can move speed
    (``tuning_epoch``) a baseline start is added and the preset must be at least as fast as doing nothing.
    The result is ``keep``, ``retune`` (suggested, never forced) or ``inconclusive`` (noise or a competing process)."""
    progress = progress or (lambda p: None)
    cancel = cancel or threading.Event()
    t0 = clock()
    s = preset.get("settings") or {}
    stored = preset.get("measured") or {}
    args = tune_knobs.apply_to_argv(base_args, s.get("args") or {})
    env = dict(s.get("env") or {})
    spec = RequestSet(warmups=1, decodes=3, decode_tokens=200, pp_tokens=4096 if ctx.context_tokens >= 6000 else 1024, long_tokens=0)
    if ctx.backend == "sycl":
        spec.pp_tokens = 0                         # the A750's 4K prompt takes a minute: decode only (8.3)
    out: dict[str, Any] = {"verdict": "inconclusive", "request_set": RECHECK_SET_NAME, "notes": []}

    def one(a: list[str], e: dict[str, str | None], label: str) -> RunResult:
        if cancel.is_set():
            raise Cancelled()
        progress({"stage": "recheck", "message": label, "elapsed_s": round(clock() - t0)})
        try:
            with session.engine(a, e) as eng:
                r = eng.measure(spec)
        except StartFailed as ex:
            return RunResult(status=ex.status, reason=ex.reason)
        return r

    try:
        run = one(args, env, "Checking your saved settings")
        if run.status != "PASS":
            out.update(verdict="retune", reason=f"the saved settings did not start now: {run.status} {run.reason}")
            out["seconds"] = round(clock() - t0, 1)
            return out
        out["measured"] = {"decode_tok_s": run.decode_tok_s, "pp4k_tok_s": run.pp_tok_s, "rss_peak_mib": run.rss_peak_mib,
                           "vram_peak_mib": run.vram_used_mib}
        sd, sp = stored.get("decode_tok_s"), stored.get("pp4k_tok_s")
        problems: list[str] = []
        if sd and run.decode_tok_s and run.decode_tok_s < 0.95 * sd:
            problems.append(f"answers are {100 * (1 - run.decode_tok_s / sd):.0f}% slower than when it was saved")
        if sp and run.pp_tok_s and run.pp_tok_s < 0.95 * sp:
            again = one(args, env, "Checking prompt speed once more")
            if again.status == "PASS" and again.pp_tok_s and again.pp_tok_s >= 0.95 * sp:
                out["notes"].append("prompt speed varied between starts; the second start was fine")
            else:
                problems.append("long prompts are slower than when it was saved")
        for key, got in (("rss_peak_mib", run.rss_peak_mib), ("vram_peak_mib", run.vram_used_mib)):
            ref = stored.get(key)
            if ref and got and got > 1.05 * ref:
                problems.append("it now uses more memory than when it was saved")
                break
        if engine_changed:
            base = one(base_args, {}, "Checking against the standard settings after the engine update")
            if base.status == "PASS" and base.decode_tok_s and run.decode_tok_s and run.decode_tok_s < base.decode_tok_s:
                problems.append("the standard settings are now faster than the saved ones")
            out["baseline"] = {"decode_tok_s": base.decode_tok_s, "pp4k_tok_s": base.pp_tok_s}
        out["problems"] = problems
        out["verdict"] = "retune" if problems else "keep"
        out["reason"] = ("Still as fast as when it was saved." if not problems else "; ".join(problems))
    except Cancelled:
        out.update(verdict="inconclusive", reason="Stopped.")
    out["seconds"] = round(clock() - t0, 1)
    return out


RECHECK_SET_NAME = "recheck/1"
